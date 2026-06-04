import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, parse_args, DiffAugment
import wandb
import copy
import random
from reparam_module import ReparamModule

import pandas as pd
from utils import fix_seed

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def log_eval_snapshot_artifact(eval_artifact_name, iteration, file_paths, is_best=False):
    if not file_paths:
        return

    artifact = wandb.Artifact(
        name=eval_artifact_name,
        type="distillation_eval",
        description="Synthetic distillation snapshot saved during evaluation.",
        metadata={
            "iteration": int(iteration),
            "run_id": wandb.run.id,
            "run_name": wandb.run.name,
            "dataset": args.dataset,
            "is_best_so_far": bool(is_best),
            "files": [os.path.basename(path) for path in file_paths],
        },
    )

    for file_path in file_paths:
        artifact.add_file(file_path, name=os.path.basename(file_path))

    aliases = ["latest", "iter_{}".format(int(iteration))]
    if is_best:
        aliases.append("best")

    wandb.run.log_artifact(artifact, aliases=aliases)


def log_eval_img(image_save, log_name, it, clip_val=2.5):
    std = torch.std(image_save)
    mean = torch.mean(image_save)
    upsampled = torch.clip(image_save, min=mean - clip_val * std, max=mean + clip_val * std)
    upsampled = get_loggable_images(upsampled)
    grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
    wandb.log({log_name: wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)


def get_loggable_images(images):
    if images.shape[1] in (1, 3):
        return images

    channel_idx = torch.linspace(0, images.shape[1] - 1, steps=3, device=images.device).round().long()
    return images.index_select(1, channel_idx)


def main(args):
    fix_seed(args.seed)

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model not in ['MMSConvB', 'Perceiver']:
        args.unimodal = 'model'

    eval_it_pool = np.arange(0, args.distill_steps + 1, args.eval_it).tolist()
    channel, im_size, num_classes, _, _, _, dst_train, dst_test, testloader, class_map = get_dataset(args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    args.im_size = im_size
    if args.augmentations:
        augs = DiffAugment(args.augmentations)

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    wandb.init(
        sync_tensorboard=False,
        project="DatasetDistillation",
        job_type="CleanRepo",
        config=args,
        name=args.name
    )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    # Keep a single artifact collection per run so each eval snapshot becomes a new version.
    eval_artifact_name = "distillation-eval-{}".format(wandb.run.id)
    wandb.run.summary["eval_artifact_collection"] = eval_artifact_name

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.spc

    args.distributed = torch.cuda.device_count() > 1


    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    ''' organize the real dataset '''
    images_all = []
    sensor_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        if args.unimodal == 'model':
            sample = dst_train[i]
            images_all.append(torch.unsqueeze(sample[0], dim=0))
            labels_all.append(class_map[torch.tensor(sample[1]).item()])
        else:
            sample = dst_train[i]
            images_all.append(torch.unsqueeze(sample[0][0], dim=0))
            sensor_all.append(torch.unsqueeze(sample[0][1], dim=0))
            labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to('cpu')
    sensor_all = torch.cat(sensor_all, dim=0).to('cpu') if sensor_all else torch.tensor([])
    labels_all = torch.tensor(labels_all, dtype=torch.long).to('cpu')

    if args.unimodal == 'sensor':
        images_all = torch.zeros_like(images_all)
    elif args.unimodal == 'image':
        sensor_all = torch.zeros_like(sensor_all)

    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


    ''' initialize the synthetic data '''
    label_syn = torch.tensor([np.ones(args.spc,dtype=np.int_)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
    image_syn = torch.randn(size=(num_classes * args.spc, channel, im_size[0], im_size[1]), dtype=torch.float)

    if args.unimodal != 'model':
        sensor_shape = sensor_all.shape[1:] if sensor_all.numel() > 0 else (args.n_input_features,)
        if args.unimodal != 'image':
            sensor_syn = torch.randn(size=(num_classes * args.spc, *sensor_shape), dtype=torch.float)
        else:
            sensor_syn = torch.zeros(size=(num_classes * args.spc, *sensor_shape), dtype=torch.float) # initialize with 0 for image-only

    syn_lr = torch.tensor(args.lr).to(args.device)

    if args.data_init == 'real':
        print('initialize synthetic data from random real images')
        if args.unimodal != 'model' and args.unimodal != 'image':
            print('initialize synthetic sensor data from random real sensor data')
        with torch.no_grad():
            for c in range(num_classes):
                class_slice = slice(c * args.spc, (c + 1) * args.spc)
                class_indices = np.random.permutation(indices_class[c])[:args.spc]
                image_syn[class_slice] = images_all[class_indices]
                if args.unimodal != 'model' and args.unimodal != 'image':
                    sensor_syn[class_slice] = sensor_all[class_indices]

            if args.unimodal != 'model' and args.unimodal != 'image':
                for c in range(num_classes):
                    class_slice = slice(c * args.spc, (c + 1) * args.spc)
                    sensor_syn[class_slice] = sensor_all[class_indices]
    else:
        print('initialize synthetic data from random noise')
        if args.unimodal != 'model' and args.unimodal != 'image':
            print('initialize synthetic sensor data from random noise')
            image_syn = torch.zeros_like(image_syn)


    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)    

    if args.unimodal != 'model':
        sensor_syn = sensor_syn.detach().to(args.device).requires_grad_(True)
        optimizer_sens = torch.optim.SGD([sensor_syn], lr=args.lr_img, momentum=0.5)

    optimizers = []
    if args.unimodal != 'sensor':
        optimizers.append(optimizer_img)
    if args.unimodal != 'model' and args.unimodal != 'image':
        optimizers.append(optimizer_sens)
    optimizers.append(optimizer_lr)

    criterion = nn.CrossEntropyLoss().to(args.device)

    expert_dir = os.path.join(args.buffer_path, args.dataset)
    expert_dir = os.path.join(expert_dir, args.model)
    print("Expert Dir: {}".format(expert_dir))

    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)

    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}
    
    for it in range(0, args.distill_steps+1):
        is_best = False

        kwargs = {
            'unimodal': args.unimodal if hasattr(args, 'unimodal') else '',
            'n_groups': args.n_groups if hasattr(args, 'n_groups') else 8,
            'n_sensors': args.n_sensors if hasattr(args, 'n_sensors') else None,
            'n_sensor_features': args.n_sensor_features if hasattr(args, 'n_sensor_features') else None
        }

        wandb.log({"Progress": it}, step=it)
        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                if args.augmentations:
                    print('Augmentations:\n', args.augmentations)

                accs_test = []
                accs_train = []
                for it_eval in range(args.num_eval):
                    fix_seed(args.seed + it_eval)

                    net_eval = get_network(model_eval, channel, num_classes, im_size, **kwargs).to(args.device) # get a random model

                    eval_labs = label_syn
                    with torch.no_grad():
                        image_save = image_syn
                        if args.unimodal != 'model':
                            sensor_save = sensor_syn
                    
                    # avoid any unaware modification
                    image_syn_eval = copy.deepcopy(image_save.detach())
                    label_syn_eval = copy.deepcopy(eval_labs.detach())
                    if args.unimodal != 'model':
                        sensor_syn_eval = copy.deepcopy(sensor_save.detach())
                    else:
                        sensor_syn_eval = None

                    args.lr_net_syn = syn_lr.item()
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, sensor_train=sensor_syn_eval)
                    accs_test.append(acc_test)
                    accs_train.append(acc_train)
                accs_test = np.array(accs_test)
                accs_train = np.array(accs_train)
                acc_test_mean = np.mean(accs_test)
                acc_test_std = np.std(accs_test)
                if acc_test_mean > best_acc[model_eval]:
                    best_acc[model_eval] = acc_test_mean
                    best_std[model_eval] = acc_test_std
                    is_best = True
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
                
                wandb.log({'Accuracy/': acc_test_mean}, step=it)
                wandb.log({'Max_Accuracy/': best_acc[model_eval]}, step=it)
                wandb.log({'Std/': acc_test_std}, step=it)
                wandb.log({'Max_Std/': best_std[model_eval]}, step=it)


        if it in eval_it_pool:
            with torch.no_grad():
                image_save = image_syn.to(args.device)

                save_dir = os.path.join(".", "logged_files", args.dataset, wandb.run.name)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                artifact_files = []

                image_path = os.path.join(save_dir, "images_{}.pt".format(it))
                label_path = os.path.join(save_dir, "labels_{}.pt".format(it))
                torch.save(image_save.cpu(), image_path)
                torch.save(label_syn.cpu(), label_path)
                artifact_files.extend([image_path, label_path])

                if args.unimodal != 'model':
                    sensor_tensor = sensor_syn.detach().cpu()
                    sensor_path = os.path.join(save_dir, "sensor_{}.pt".format(it))
                    torch.save(sensor_tensor, sensor_path)
                    artifact_files.append(sensor_path)

                wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)

                
                upsampled = get_loggable_images(image_save)
                grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                log_eval_img(image_save, 'Clipped_Synthetic_Images', it)                    

                if args.unimodal != 'model':
                    sensor_save = sensor_tensor.numpy()
                    sensor_save = dst_test.scaler.inverse_transform(sensor_save)
                    sensor_save_df = pd.DataFrame(sensor_save, columns=dst_test.scaler.get_feature_names_out())
                    wandb.log({'Synthetic_Sensors': wandb.Table(dataframe=sensor_save_df)}, step=it)

                log_eval_snapshot_artifact(eval_artifact_name, it, artifact_files, is_best=is_best)

        wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)

        student_net = get_network(args.model, channel, num_classes, im_size, dist=False, **kwargs).to(args.device)  # get a random model
        student_net = ReparamModule(student_net)

        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)

        student_net.train()

        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                print("loading file {}".format(expert_files[file_idx]))
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer)

        start_epoch = np.random.randint(args.min_start_epoch, args.max_start_epoch)
        starting_params = expert_trajectory[start_epoch]

        target_params = expert_trajectory[start_epoch+args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

        syn_images = image_syn
        if args.unimodal != 'model':
            syn_sensor = sensor_syn

        y_hat = label_syn.to(args.device)

        indices_chunks = []

        for step in range(args.syn_steps):

            if not indices_chunks:
                indices = torch.randperm(len(syn_images))
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()

            x = syn_images[these_indices]
            this_y = y_hat[these_indices]

            if args.augmentations:
                x = augs(x)

            if args.unimodal != 'model':
                x = (x, syn_sensor[these_indices])

            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]
            x = student_net(x, flat_param=forward_params)
            ce_loss = criterion(x, this_y)

            grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]

            student_params.append(student_params[-1] - syn_lr * grad)

        param_loss = F.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist = F.mse_loss(starting_params, target_params, reduction="sum")
        grand_loss = param_loss / param_dist

        for optimizer in optimizers:
            optimizer.zero_grad()

        grand_loss.backward()

        for optimizer in optimizers:
            optimizer.step()

        wandb.log({"Grand_Loss": grand_loss.detach().cpu(),
                   "Start_Epoch": start_epoch})

        for _ in student_params:
            del _

        if it%10 == 0:
            print('iter = %04d, loss = %.4f' % (it, grand_loss.item()))

    wandb.finish()


if __name__ == '__main__':
    args = parse_args('distill')

    main(args)
