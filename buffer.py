import os
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import build_dataset, get_dataset, get_network, TensorDataset, epoch, parse_args, DiffAugment
import copy


from utils import fix_seed, MultimodalTensorDataset
import wandb


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args, trial=None):
    fix_seed(args.seed)

    aug = None
    if args.augmentations:
        aug = DiffAugment(args.augmentations)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, class_map = get_dataset(args)

    print('Hyper-parameters: \n', args.__dict__)

    save_dir = os.path.join(args.buffer_path, args.dataset)
    save_dir = os.path.join(save_dir, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images_all, sensor_all, labels_all, _ = build_dataset(dst_train, channel, num_classes, class_map, args.unimodal)

    criterion = nn.CrossEntropyLoss().to(args.device)

    trajectories = []

    if args.unimodal == 'unimodal':
        dst_train = TensorDataset(copy.deepcopy(images_all.detach()), copy.deepcopy(labels_all.detach()))
    else:
        dst_train = MultimodalTensorDataset(copy.deepcopy(images_all.detach()), copy.deepcopy(sensor_all.detach()), copy.deepcopy(labels_all.detach()))
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_size, shuffle=True, num_workers=0)

    best_acc = 0.0

    for it in range(0, args.num_experts):
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
            ''' Train synthetic data '''
            teacher_net = get_network(args.model, channel, num_classes, im_size, **kwargs).to(args.device) # get a random model
            teacher_net.train()
            lr = args.lr
            teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)  # optimizer_img for synthetic data
            teacher_optim.zero_grad()

            timestamps = []

            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

            lr_schedule = [args.train_epochs // 2 + 1]

            for e in range(args.train_epochs):

                train_loss, train_acc = epoch("train", dataloader=trainloader, net=teacher_net, optimizer=teacher_optim, criterion=criterion, args=args, augs=aug)
                test_loss, test_acc = epoch("test", dataloader=testloader, net=teacher_net, optimizer=None, criterion=criterion, args=args, augs=False)

                print(f'Itr: {it}\tEpoch: {e}\tTrain Acc: {train_acc:.4f}\tTest Acc: {test_acc:.4f}')

                timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])
                
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
    args = parse_args('buffer')

    args.wandb_mode = os.environ.get('WANDB_MODE', 'online')
    if args.optuna_trials > 0:
        args.wandb_mode = 'disabled' # disable wandb during hyperparameter search

        import optuna
        def objective(trial):
            trial_args = copy.deepcopy(args)
            trial_args.batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
            trial_args.train_epochs = trial.suggest_categorical('epochs', [50, 100])
            trial_args.lr = trial.suggest_float('lr', 1e-7, 5e-1, log=True)
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
