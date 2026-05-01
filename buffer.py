import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import get_dataset, get_network, get_daparam,\
    TensorDataset, epoch, ParamDiffAug
import copy


from utils import fix_seed, MultimodalTensorDataset # added by Franz
import wandb                                        # added by Franz


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args, trial=None):
    fix_seed(args.seed)

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug.copy()

    if args.model not in ['MMSConvB', 'Perceiver']:
        args.unimodal = 'model'

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)

    # print('\n================== Exp %d ==================\n '%exp)
    print('Hyper-parameters: \n', args.__dict__)

    save_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        save_dir = os.path.join(save_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        save_dir += "_NO_ZCA"
    save_dir = os.path.join(save_dir, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #######################################################################
    # modified by Franz:
    #######################################################################
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
    images_all = torch.cat(images_all, dim=0).to("cpu")
    sensor_all = torch.cat(sensor_all, dim=0).to("cpu") if sensor_all else torch.tensor([])
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    criterion = nn.CrossEntropyLoss().to(args.device)

    trajectories = []

    if args.unimodal == 'model':
        dst_train = TensorDataset(copy.deepcopy(images_all.detach()), copy.deepcopy(labels_all.detach()))
    else:
        dst_train = MultimodalTensorDataset(copy.deepcopy(images_all.detach()), copy.deepcopy(sensor_all.detach()), copy.deepcopy(labels_all.detach()))
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
    #######################################################################

    ''' set augmentation for whole-dataset training '''
    args.dc_aug_param = get_daparam(args.dataset, args.model, args.model, None)
    args.dc_aug_param['strategy'] = 'crop_scale_rotate'  # for whole-dataset training
    print('DC augmentation parameters: \n', args.dc_aug_param)

    best_acc = 0.0

    for it in range(0, args.num_experts):
        #######################################################################
        # added by Franz:
        #######################################################################
        
        fix_seed(args.seed + it)

        run_config = dict(vars(args))
        run_config['expert_iteration'] = it
        wandb_kwargs = {
            'config': run_config,
            'reinit': True,
            'name': args.name,
            'project': 'DatasetDistillation_Training',
            'mode': args.wandb_mode
        }

        kwargs = {
            'unimodal': args.unimodal if hasattr(args, 'unimodal') else '',
            'n_groups': args.n_groups if hasattr(args, 'n_groups') else 8,
            'n_sensors': args.n_sensors if hasattr(args, 'n_sensors') else None,
            'n_sensor_features': args.n_sensor_features if hasattr(args, 'n_sensor_features') else None
        }

        with wandb.init(**wandb_kwargs):
        #######################################################################

            ''' Train synthetic data '''
            teacher_net = get_network(args.model, channel, num_classes, im_size, **kwargs).to(args.device) # get a random model
            teacher_net.train()
            lr = args.lr_teacher
            teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)  # optimizer_img for synthetic data
            teacher_optim.zero_grad()

            timestamps = []

            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

            lr_schedule = [args.train_epochs // 2 + 1]

            for e in range(args.train_epochs):

                train_loss, train_acc = epoch("train", dataloader=trainloader, net=teacher_net, optimizer=teacher_optim,
                                            criterion=criterion, args=args, aug=True)

                test_loss, test_acc = epoch("test", dataloader=testloader, net=teacher_net, optimizer=None,
                                            criterion=criterion, args=args, aug=False)

                print("Itr: {}\tEpoch: {}\tTrain Acc: {}\tTest Acc: {}".format(it, e, train_acc, test_acc))

                timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

                #######################################################################
                # added by Franz:
                #######################################################################
                
                wandb.log({
                    'train_loss': float(train_loss),
                    'train_acc': float(train_acc),
                    'test_loss': float(test_loss),
                    'test_acc': float(test_acc),
                }, step=e)

                best_acc = max(best_acc, test_acc)

                if trial is not None:
                    import optuna  # local import to avoid dependency when not using Optuna
                    trial.report(test_acc, step=e)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                #######################################################################

                if e in lr_schedule and args.decay:
                    lr *= 0.1
                    teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
                    teacher_optim.zero_grad()

        trajectories.append(timestamps)

        if len(trajectories) == args.save_interval:
            n = 0
            while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
                n += 1
            print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
            trajectories = []

    return best_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='subset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--res', type=int, default=128, help='resolution for imagenet')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--save_interval', type=int, default=10)

    #####################################################################
    # added by Franz
    #####################################################################
    parser.add_argument('--seed', type=int, default=1337, help='set random seed')
    parser.add_argument('--unimodal', type=str, default='', choices=['', 'image', 'sensor'], help='unimodal training (only for multimodal datasets)')
    parser.add_argument('--n_groups', type=int, default=8, help='group norm groups (for MMSConvB)')
    parser.add_argument('--optuna_trials', type=int, default=0, help='number of optuna trials to run (0 disables search)')
    parser.add_argument('--name', type=str, default='TrainingRun', help='name of wandb run')

    args = parser.parse_args()

    if args.model not in ['MMSConvB', 'Perceiver']:
        args.unimodal = 'model'  # used by multimodal datasets, to only provide images for unimodal models

    args.wandb_mode = os.environ.get('WANDB_MODE', 'online')
    if args.optuna_trials > 0:
        args.wandb_mode = 'disabled' # disable wandb during hyperparameter search

        import optuna
        def objective(trial):
            trial_args = copy.deepcopy(args)
            trial_args.batch_train = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
            trial_args.batch_real = trial_args.batch_train
            #trial_args.n_groups = trial.suggest_categorical('n_groups', [1, 2, 4, 8, 16, 32])
            trial_args.train_epochs = trial.suggest_categorical('epochs', [50, 100])
            trial_args.lr_teacher = trial.suggest_float('lr', 1e-7, 5e-1, log=True)
            return main(trial_args, trial)

        storage = f"sqlite:///optuna_results.db"
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=35)
        study = optuna.create_study(direction='maximize', pruner=pruner, storage=storage, load_if_exists=True, study_name=args.name)
        study.optimize(objective, n_trials=args.optuna_trials)
        print(f"Best value: {study.best_value}")
        print(f"Best params: {study.best_params}")
        print(f"Optuna results stored in {storage}")
    else:
        main(args)
