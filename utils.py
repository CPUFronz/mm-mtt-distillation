# adapted from
# https://github.com/VICO-UoE/DatasetCondensation

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tqdm
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import wandb
from networks import MLP, ConvNet, LeNet, AlexNet, VGG11BN, VGG11, ResNet18, ResNet18BN_AP, ResNet18_AP, Widar_CNN3D


def get_dataset(args):
    fix_seed(args.seed)

    class_map = None

    if args.dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        class_map = {x:x for x in range(num_classes)}

    elif args.dataset == 'RaspiCar':
        args.window_size = 10
        args.test_split = 0.2
        args.image_size = (32, 32)

        df = load_raspicar_data(args)

        scaler = StandardScaler()
        scaler.fit(df[args.sens_cols])

        channel = 3
        im_size = args.image_size
        # mean and std are in for all added datasets, but not used
        mean = [0.5087, 0.4848, 0.4292]
        std = [0.1729, 0.1907, 0.2188]
                                                                                
        train_data, test_data = train_test_split(df, test_size=args.test_split, random_state=args.seed)
        dst_train = RaspiCarDataset(train_data, scaler, args.sens_cols, args.image_size, args.unimodal)
        dst_test = RaspiCarDataset(test_data, scaler, args.sens_cols, args.image_size, args.unimodal)

        args.n_sensors = len(SENS_COLS_CAR)
        args.n_sensor_features = 9 # 9 = 1 sensor value + 8 statistical sensor features
        args.n_input_features = args.n_sensors * args.n_sensor_features
        args.n_output_features = NUM_STEERING_ANGLES

        num_classes = NUM_STEERING_ANGLES
        class_names = [str(i) for i in range(num_classes)]
        class_map = {x: x for x in range(num_classes)}

    elif args.dataset == 'ActionSense':
        args.window_size = 10
        args.test_split = 0.2
        args.image_size = (32, 32)

        df, images, _ = load_actionsense_data(args)
        df = (
            df.groupby('label', group_keys=False)
                .apply(lambda g: g.sample(n=min(ACTIONSENSE_SAMPLES_PER_LABEL, len(g)), random_state=0)) # always use the same subset of data
                .reset_index(drop=True)
        )

        args.sens_cols = df.columns.drop(['label', 'subject', 'video_id', 'video_frame'])

        scaler = StandardScaler()
        scaler.fit(df[args.sens_cols])

        encoder = LabelEncoder()
        encoder.fit(df['label'])
        df['label'] = encoder.transform(df['label'])

        train_data, test_data = train_test_split(df, test_size=args.test_split, random_state=args.seed)
        dst_train = ActionSenseDataset(train_data, images, scaler, args.sens_cols, args.image_size, args.unimodal)
        dst_test = ActionSenseDataset(test_data, images, scaler, args.sens_cols, args.image_size, args.unimodal)

        args.n_sensors = len(args.sens_cols)
        args.n_sensor_features = 1
        args.n_input_features = args.n_sensors * args.n_sensor_features
        args.n_output_features = df['label'].nunique()

        sample_image = dst_train[0][0][0]
        channel = sample_image.shape[0]
        im_size = sample_image.shape[1:]
        mean = [0.20614147, 0.33670203, 0.33101761]
        std = [0.15099436, 0.23058386, 0.23078493]
        num_classes = df['label'].nunique()
        class_names = encoder.classes_.tolist()
        class_map = {x: x for x in range(num_classes)}

    elif args.dataset == 'RoboMNIST':
        args.window_size = 10
        args.test_split = 0.2
        args.image_size = (32, 32)

        df, images = load_robomnist_data(args)

        args.sens_cols = df.columns.drop(['label', 'image_idx'])

        scaler = StandardScaler()
        scaler.fit(df[args.sens_cols])

        train_data, test_data = train_test_split(df, test_size=args.test_split, random_state=args.seed)

        dst_train = RoboMNISTDataset(train_data, images, scaler, args.sens_cols, args.unimodal)
        dst_test = RoboMNISTDataset(test_data, images, scaler, args.sens_cols, args.unimodal)

        args.n_sensors = len(args.sens_cols)
        args.n_sensor_features = 1
        args.n_input_features = args.n_sensors * args.n_sensor_features
        args.n_output_features = df['label'].nunique()

        sample_image = dst_train[0][0][0]
        channel = sample_image.shape[0]
        im_size = sample_image.shape[1:]
        assert im_size == args.image_size, "Image size mismatch. Expected: {}, Got: {}".format(args.image_size, im_size)
        mean = [0.359549389299777, 0.3862599337890894, 0.3577771832990368]
        std = [0.19270906559562345, 0.17443764500719838, 0.18467636776295263]
        num_classes = df['label'].nunique()
        class_names = sorted(df['label'].unique().tolist())
        class_map = {x: x for x in range(num_classes)}

    elif args.dataset == 'Widar':
        widar_root = os.path.join(args.data_path, 'Widardata2')
        dst_train = Widar_Dataset(os.path.join(widar_root, 'train'))
        dst_test = Widar_Dataset(os.path.join(widar_root, 'test'))

        channel = 22
        im_size = (20, 20)
        num_classes = len(dst_train.category)
        mean = [0.0 for _ in range(channel)]
        std = [1.0 for _ in range(channel)]
        class_names = [os.path.basename(os.path.normpath(folder)) for folder in dst_train.folder]
        class_map = {x: x for x in range(num_classes)}

    else:
        exit(f'unknown dataset: {args.dataset}')

    testloader = torch.utils.data.DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=2)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, class_map



class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]



def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling



def get_network(model, channel, num_classes, im_size=(32, 32), dist=True, **kwargs):
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == 'VGG11':
        net = VGG11( channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18_AP':
        net = ResNet18_AP(channel=channel, num_classes=num_classes)
    elif model == 'Widar_CNN3D':
        net = Widar_CNN3D(num_classes=num_classes)

    elif model == 'ConvNetD1':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=1, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD2':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=2, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD3':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=3, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD4':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD5':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=5, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD6':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=6, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD7':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=7, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD8':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=8, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)


    elif model == 'ConvNetW32':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=32, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW64':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=64, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW128':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW256':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=256, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW512':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=512, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW1024':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=1024, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)

    elif model == "ConvNetKIP":
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=1024, net_depth=net_depth, net_act=net_act,
                      net_norm="none", net_pooling=net_pooling)

    elif model == 'ConvNetAS':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='sigmoid', net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetAR':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='relu', net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetAL':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='leakyrelu', net_norm=net_norm, net_pooling=net_pooling)

    elif model == 'ConvNetNN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='none', net_pooling=net_pooling)
    elif model == 'ConvNetBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling)
    elif model == 'ConvNetLN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='layernorm', net_pooling=net_pooling)
    elif model == 'ConvNetIN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='instancenorm', net_pooling=net_pooling)
    elif model == 'ConvNetGN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='groupnorm', net_pooling=net_pooling)

    elif model == 'ConvNetNP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='none')
    elif model == 'ConvNetMP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='maxpooling')
    elif model == 'ConvNetAP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='avgpooling')

    elif model == 'MMSConvB':
        from networks import MMSConvB
        net = MMSConvB(
            n_layers_img=4,
            n_units_img=64,
            n_sensors=kwargs['n_sensors'],
            n_sensor_features=kwargs['n_sensor_features'],
            n_layers_sens=4,
            n_units_sens=512,
            n_heads_fusion=4,
            n_units_fusion=64,
            n_channels=3,
            n_classes=num_classes,
            unimodal=kwargs['unimodal'],
            im_size=im_size,
            n_groups=kwargs['n_groups']
        )
    elif model == 'Perceiver':
        from networks import MultimodalPerceiver
        net = MultimodalPerceiver(
            img_size=im_size,
            n_sensors=kwargs['n_sensors'],
            n_sensor_features=kwargs['n_sensor_features'],
            num_classes=num_classes,
            latent_dim=32,
            token_dim=128,
            num_lat=128,
            depth=3,
            cross_heads=1,
            latent_heads=8,
            seq_dropout_prob=0.2
        )

    else:
        net = None
        exit('DC error: unknown model')

    if dist:
        gpu_num = torch.cuda.device_count()
        if gpu_num>0:
            device = 'cuda'
            if gpu_num>1:
                net = nn.DataParallel(net)
        else:
            device = 'cpu'
        net = net.to(device)

    return net


def epoch(mode, dataloader, net, optimizer, criterion, args, aug):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        if args.unimodal == 'model':
            img = datum[0].float().to(args.device)
            lab = datum[1].long().to(args.device)
        else:
            img, sen = datum[0]
            img = img.float().to(args.device)
            sen = sen.float().to(args.device)
            lab = datum[1].long().to(args.device)

        if aug:
            img = aug(img)

        n_b = lab.shape[0]

        if args.unimodal == 'model':
            output = net(img)
        else:
            output = net((img, sen))
        loss = criterion(output, lab)

        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg


def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, return_loss=False, sensor_train=None, training_logs=False):
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)

    criterion = nn.CrossEntropyLoss().to(args.device)

    if sensor_train is None:
        dst_train = TensorDataset(images_train, labels_train)
    else:
        sensor_train = sensor_train.to(args.device)
        dst_train = MultimodalTensorDataset(images_train, sensor_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_size, shuffle=True, num_workers=0)

    start = time.time()
    acc_train_list = []
    loss_train_list = []

    aug = False
    if args.augmentations:
        aug = DiffAugment(args.augmentations)

    for ep in tqdm.tqdm(range(Epoch+1)):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug=aug)
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        if ep == Epoch or training_logs:
            with torch.no_grad():
                loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)

            if training_logs:
                for param_group in optimizer.param_groups:
                    current_lr = param_group["lr"]
                    break

                metric_payload = {
                    "test_acc": acc_test,
                    "test_loss": loss_test,
                    "train_acc": acc_train,
                    "train_loss": loss_train,
                    "lr": current_lr,
                }
                wandb.log(metric_payload, step=ep)

        if ep in lr_schedule:
            lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr


    time_train = time.time() - start

    print('Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    if return_loss:
        return net, acc_train_list, acc_test, loss_train_list, loss_test
    else:
        return net, acc_train_list, acc_test


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M': # multiple architectures
        # model_eval_pool = ['MLP', 'ConvNet', 'AlexNet', 'VGG11', 'ResNet18', 'LeNet']
        model_eval_pool = ['ConvNet', 'AlexNet', 'VGG11', 'ResNet18_AP', 'ResNet18']
        # model_eval_pool = ['MLP', 'ConvNet', 'AlexNet', 'VGG11', 'ResNet18']
    elif eval_mode == 'W': # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D': # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A': # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL']
    elif eval_mode == 'P': # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N': # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S': # itself
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
    elif eval_mode == 'C':
        model_eval_pool = [model, 'ConvNet']
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool

# TODO: add old Augmentations for Images

#####################################################################
# added by Shadi and Franz
#####################################################################

class DiffAugment:
    def __init__(self, param):
        self.param = param

        self.registry = {
            'flip_h_prob':              self.rand_flip_h,
            'flip_w_prob':              self.rand_flip_w,
            'velocity_scale_ratio':     self.rand_scale_velocity,
            'velocity_translate_ratio': self.rand_translate_velocity,
            'velocity_cutout_ratio':    self.rand_cutout_velocity,
            'temporal_shift_max':       self.rand_temporal_shift,
            'temporal_cutout_ratio':    self.rand_temporal_cutout,
            'temporal_flip_prob':       self.rand_temporal_flip,
            'amplitude':                self.rand_amplitude,
            'noise':                    self.rand_noise
        }

    def __call__(self, x):
        augmentation = np.random.choice(list(self.param.keys()))
        x = self.registry[augmentation](x)
        return x.contiguous()

    def rand_flip_h(self, x):
        """
        Flip x-velocity axis (H, dim=2).
        Physical meaning: mirrors the gesture left ↔ right.
        ⚠ Skip if 'slide_left' and 'slide_right' are different class labels.
        """
        prob_flip_h = self.param['flip_h_prob']  
        randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
        return torch.where(randf < prob_flip_h, x.flip(2), x)


    def rand_flip_w(self, x):
        """
        Flip y-velocity axis (W, dim=3).
        Physical meaning: mirrors the gesture forward ↔ backward.
        """
        prob_flip_w = self.param['flip_w_prob']
        randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
        return torch.where(randf < prob_flip_w, x.flip(3), x)


    def rand_scale_velocity(self,x):
        """
        Scale velocity bins via affine grid (H, W).
        Physical meaning: simulates different body-to-antenna distances
        (further away → compressed velocity distribution).
        Note: affine_grid treats T as 'channels' and scales H,W identically ✓
        """
        ratio = self.param['velocity_scale_ratio']
        sx = torch.rand(x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
        sy = torch.rand(x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
        theta = [[[sx[i], 0,    0],
                [0,    sy[i], 0]] for i in range(x.shape[0])]
        theta = torch.tensor(theta, dtype=torch.float)
        grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
        return F.grid_sample(x, grid, align_corners=True)
    

    def rand_translate_velocity(self, x):
        """
        Translate in velocity space (H, W) with wrap-around padding.
        Physical meaning: slightly different body orientation relative to antennas.
        """
        ratio = self.param['velocity_translate_ratio']
        shift_h  = int(x.size(2) * ratio + 0.5)
        shift_w  = int(x.size(3) * ratio + 0.5)
        trans_h = torch.randint(-shift_h, shift_h + 1, size=[x.size(0), 1, 1], device=x.device)
        trans_w = torch.randint(-shift_w, shift_w + 1, size=[x.size(0), 1, 1], device=x.device)

        grid_b, grid_h, grid_w = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
        )
        grid_h = torch.clamp(grid_h + trans_h + 1, 0, x.size(2) + 1)
        grid_w = torch.clamp(grid_w + trans_w + 1, 0, x.size(3) + 1)

        # pad H and W by 1 on each side, keep T (dim=1) unchanged
        x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
        # [B, T, H_pad, W_pad] → permute → [B, H_pad, W_pad, T] → index → permute back
        x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_b, grid_h, grid_w].permute(0, 3, 1, 2)
        return x
    

    def rand_cutout_velocity(self, x):
        """
        Zero out a rectangular patch in velocity space (H, W) for all T frames.
        Physical meaning: simulates partial antenna occlusion or dead velocity bins.
        """
        ratio_cutout = self.param['velocity_cutout_ratio']
        cutout_h = int(x.size(2) * ratio_cutout + 0.5)
        cutout_w = int(x.size(3) * ratio_cutout + 0.5)
        off_h = torch.randint(0, x.size(2) + (1 - cutout_h % 2),
                            size=[x.size(0), 1, 1], device=x.device)
        off_w = torch.randint(0, x.size(3) + (1 - cutout_w % 2),
                            size=[x.size(0), 1, 1], device=x.device)

        grid_b, grid_h, grid_w = torch.meshgrid(
            torch.arange(x.size(0),  dtype=torch.long, device=x.device),
            torch.arange(cutout_h,   dtype=torch.long, device=x.device),
            torch.arange(cutout_w,   dtype=torch.long, device=x.device),
        )
        grid_h = torch.clamp(grid_h + off_h - cutout_h // 2, 0, x.size(2) - 1)
        grid_w = torch.clamp(grid_w + off_w - cutout_w // 2, 0, x.size(3) - 1)

        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_b, grid_h, grid_w] = 0
        return x * mask.unsqueeze(1)  # broadcast mask over all T frames


    # ─────────────────────────────────────────────────────────────────────────────
    #  TEMPORAL AUGMENTATIONS  (operate on T=dim1)
    # ─────────────────────────────────────────────────────────────────────────────

    def rand_temporal_shift(self, x):
        """
        Key insight: reshape T to the spatial WIDTH dimension,
        then affine_grid + grid_sample gives continuous bilinear shift along T.
        Gradient flows smoothly back to x_syn through interpolated frames. ✓
        """
        max_temporal_shift = self.param['temporal_shift_max']

        B, T, H, W = x.shape

        # Normalized continuous shift in [-1, 1] space
        max_norm = max_temporal_shift / (T / 2.0)
        shifts   = (torch.rand(B, device=x.device) * 2 - 1) * max_norm  # float ✓

        # ── Reshape: [B, T, H, W] → [B, H*W, 1, T] ─────────────────────────────
        #    H*W acts as "channels", T is now the spatial width → grid_sample works ✓
        x_r = x.permute(0, 2, 3, 1).reshape(B, H * W, 1, T)

        # ── Pure translation along T (width dimension) ───────────────────────────
        theta = torch.zeros(B, 2, 3, device=x.device, dtype=x.dtype)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, 0, 2] = shifts   # shift along T axis only

        grid      = F.affine_grid(theta, x_r.shape, align_corners=True)
        x_shifted = F.grid_sample(
            x_r, grid,
            mode='bilinear',         # smooth interpolation between frames ✓
            align_corners=True,
            padding_mode='border'    # clamp at boundary frames (not wrap-around)
        )

        # ── Reshape back: [B, H*W, 1, T] → [B, T, H, W] ─────────────────────────
        return x_shifted.reshape(B, H, W, T).permute(0, 3, 1, 2)


    def rand_temporal_flip(self, x):
        """
        Reverse the T axis (time-reverse the gesture).
        Physical meaning: reversed motion — only valid if class label is symmetric
        (e.g. 'push' reversed ≠ 'push', so use carefully or only for symmetric gestures).
        """
        prob_temporal_flip = self.param['temporal_flip_prob']
        randf = torch.rand(x.size(0), device=x.device)
        # flip(1) reverses T; stack keeps gradient
        flipped = x.flip(1)
        mask = (randf < prob_temporal_flip)[:, None, None, None]
        return torch.where(mask, flipped, x)


    def rand_temporal_cutout(self, x):
        """
        Zero out a contiguous block of T frames.
        Physical meaning: missing CSI packets / person momentarily still.
        x.clone() ensures gradient flows through non-zeroed frames.
        """
        ratio_temporal_cutout = self.param['temporal_cutout_ratio']

        T          = x.size(1)
        cutout_len = max(1, int(T * ratio_temporal_cutout))
        offset = torch.randint(0, T - cutout_len + 1,
                            size=[x.size(0)], device=x.device)

        mask = torch.ones(x.size(0), T, 1, 1, dtype=x.dtype, device=x.device)
        for i in range(x.size(0)):
            mask[i, offset[i]: offset[i] + cutout_len] = 0
        return x * mask


    # ─────────────────────────────────────────────────────────────────────────────
    #  AMPLITUDE AUGMENTATIONS  (operate on all dims)
    # ─────────────────────────────────────────────────────────────────────────────

    def rand_amplitude(self, x):
        """
        Multiply entire BVP map by a per-sample random scalar.
        Physical meaning: signal strength variation due to distance / environment.
        """
        amplitude = self.param['amplitude']
        scale = (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
                * amplitude + (1.0 - amplitude / 2))
        return x * scale


    def rand_noise(self, x):
        """
        Add per-sample Gaussian noise.
        Physical meaning: multipath interference and thermal noise in CSI.
        """
        noise = self.param['noise']
        return x + noise * torch.randn_like(x)


#####################################################################
# added by Franz
#####################################################################

import os
import re
import json
import time
import random
import warnings
import pandas as pd
from glob import glob
from collections import defaultdict

import cv2
import h5py
import tqdm
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import LabelEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import f1_score
from PIL import Image


SENS_COLS_CAR = ['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z', 'tof']
NUM_STEERING_ANGLES = 11
ACTIONSENSE_SAMPLES_PER_LABEL = 2500

AUG_DEFAULTS = {
    'flip_h_prob': 0.5,                # P(flip x-velocity axis) - left/right gesture
    'flip_w_prob': 0.5,                # P(flip y-velocity axis) - fwd/back gesture
    'velocity_scale_ratio': 1.1,       # velocity bin scaling ratio
    'velocity_translate_ratio': 0.125, # velocity-space translation ratio
    'velocity_cutout_ratio': 0.5,      # velocity-space cutout size (fraction of H,W)
    # Temporal (T) params
    'temporal_shift_max': 4,           # max frames to roll +/- along T
    'temporal_cutout_ratio': 0.05,     # fraction of T frames to zero out
    'temporal_flip_prob': 0.5,         # P(reverse time axis)
    # Amplitude params
    'amplitude': 0.3,                  # uniform scale in [1 - amp/2, 1 + amp/2]
    'noise': 0.03,                     # Gaussian noise std
}


def parse_augmentations(value):
    if isinstance(value, dict):
        return value

    value = value.strip()
    if value.lower() in {"", "none", "null", "false", "{}"}:
        return {}

    try:
        augmentations = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON for --augmentations: {exc.msg}") from exc

    if augmentations is None or augmentations is False:
        return {}
    if not isinstance(augmentations, dict):
        raise argparse.ArgumentTypeError("--augmentations must be a JSON object, e.g. '{\"noise\": 0.03}'")

    unknown = sorted(set(augmentations) - set(AUG_DEFAULTS))
    if unknown:
        valid = ", ".join(sorted(AUG_DEFAULTS))
        raise argparse.ArgumentTypeError(f"unknown augmentation key(s): {', '.join(unknown)}. Valid keys: {valid}")

    for key, aug_value in augmentations.items():
        if isinstance(aug_value, bool) or not isinstance(aug_value, (int, float)):
            raise argparse.ArgumentTypeError(f"augmentation value for '{key}' must be numeric")

    return augmentations


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    os.environ.setdefault("PYTHONHASHSEED", f"{seed}")


def parse_args(mode):
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters' if mode == 'buffer' else 'initialization for synthetic learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--augmentations', type=parse_augmentations, default=AUG_DEFAULTS.copy(), metavar='JSON', help='JSON object with augmentation parameters, e.g. \'{"noise": 0.03}\'. Use "{}" or "none" to disable.')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--seed', type=int, default=42, help='set random seed')
    parser.add_argument('--unimodal', type=str, default='', choices=['', 'image', 'sensor'], help='unimodal training (only for multimodal datasets)')
    parser.add_argument('--n_groups', type=int, default=8, help='group norm groups (for MMSConvB)')
    parser.add_argument('--name', type=str, default='Run', help='name of wandb run')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=["SGD", "Adam"], help='Optimizer to use for evaluation training. Overrides the optimizer choice from the distill run config if specified.')

    if mode == "buffer":
        parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
        parser.add_argument('--train_epochs', type=int, default=50)
        parser.add_argument('--decay', action='store_true')
        parser.add_argument('--mom', type=float, default=0, help='momentum')
        parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
        parser.add_argument('--save_interval', type=int, default=10)
        parser.add_argument('--optuna_trials', type=int, default=0, help='number of optuna trials to run (0 disables search)')
    else:
        parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
        parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode, check utils.py for more info')
        parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
        parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')
        parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
        parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')
        parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
        parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
        parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
        parser.add_argument('--data_init', type=str, default='real', choices=["noise", "real"], help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
        parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
        parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
        parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')
        parser.add_argument('--load_all', action='store_true', help='only use if you can fit all expert trajectories into RAM')
        parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
        parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')
        parser.add_argument('--min_start_epoch', type=int, default=0, help='min epoch we can start at')

    return parser.parse_args()


class MultimodalTensorDataset(Dataset):
    def __init__(self, images, sensors, labels):
        self.images = images.detach().float()
        self.sensors = sensors.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return (self.images[index], self.sensors[index]), self.labels[index]

    def __len__(self):
        return self.labels.shape[0]


def load_raspicar_data(args, root='./data/raspicar/'):
    args.sens_cols = SENS_COLS_CAR

    df = pd.DataFrame()
    for dirpath, _, fnames in os.walk(root):
        for f in sorted(fnames):
            if f == 'frame_log.csv':
                log_df  = pd.read_csv(dirpath + '/frame_log.csv')
                pico_df = pd.read_csv(dirpath + '/pico_data.csv')
                # when we merge log_df = left, pico_df = right, we half the number of rows, but we use every image once, no doubles
                tmp_df = pd.merge_asof(log_df, pico_df, left_on='frame_timestamp', right_on='timestamp', direction='nearest')

                sens_cols_new = []
                for col in args.sens_cols:
                    tmp_df[col + '_mean'    ] = tmp_df[col].rolling(window=args.window_size, min_periods=1).mean()
                    tmp_df[col + '_std'     ] = tmp_df[col].rolling(window=args.window_size, min_periods=1).std()
                    tmp_df[col + '_min'     ] = tmp_df[col].rolling(window=args.window_size, min_periods=1).min()
                    tmp_df[col + '_q25'     ] = tmp_df[col].rolling(window=args.window_size, min_periods=1).quantile(0.25)
                    tmp_df[col + '_median'  ] = tmp_df[col].rolling(window=args.window_size, min_periods=1).quantile(0.5)
                    tmp_df[col + '_q75'     ] = tmp_df[col].rolling(window=args.window_size, min_periods=1).quantile(0.75)
                    tmp_df[col + '_max'     ] = tmp_df[col].rolling(window=args.window_size, min_periods=1).max()
                    tmp_df[col + '_kurtosis'] = tmp_df[col].rolling(window=args.window_size, min_periods=1).kurt()

                    sens_cols_new.extend([
                        col,
                        col + '_mean',
                        col + '_std',
                        col + '_min',
                        col + '_q25',
                        col + '_median',
                        col + '_q75',
                        col + '_max',
                        col + '_kurtosis'
                    ])

                split_path = dirpath.split('/')
                tmp_df['dataset'] = split_path[-2] + '/' + split_path[-1]

                df = pd.concat([df, tmp_df])

    bins = np.linspace(-1, 1, NUM_STEERING_ANGLES+1)
    df['steering_angle'] = pd.cut(df['steering_angle'], bins=bins, labels=False, include_lowest=True)
    df = df.dropna()

    args.sens_cols_old = args.sens_cols
    args.sens_cols = sens_cols_new

    return df


class RaspiCarDataset(Dataset):
    def __init__(self, df, scaler, sens_cols, image_size=(64, 64), unimodal=''):
        self.dataset = df
        self.n_classes = NUM_STEERING_ANGLES

        self.scaler = scaler
        self.sensor_data_scaled = self.scaler.transform(self.dataset[sens_cols])

        self.image_size = image_size        
        self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor()

        ])
        self.unimodal = unimodal

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]

        if self.unimodal != 'sensor':
            img_fn = glob(f'./data/raspicar/{row["dataset"]}/{row["frame_number"]}_*.jpg')[0]
            PIL_image = Image.open(img_fn)
            image = self.transform(PIL_image)
        else:
            image = torch.zeros((3, self.image_size[0], self.image_size[1]))

        if self.unimodal != 'image':
            sensor_data = torch.Tensor(self.sensor_data_scaled[idx , :].astype(np.float32))
        else:
            sensor_data = torch.zeros((len(self.scaler.feature_names_in_),))

        label = row['steering_angle']

        if self.unimodal == 'model':
            return image, label
        else:
            return (image, sensor_data), label
    

def load_actionsense_data(args, root='./data/actionsense/'):
    pickled_fn = f'{root}/actionsense_data_cache.pkl'
    if os.path.exists(pickled_fn):
        print('Loading existing ActionSense data from cache...')
        data = joblib.load(pickled_fn)
        assert data['metadata']['img_size'] == args.image_size, "Cached data image size does not match the specified image size."
        return data['dataframe'], data['video_frames'], data['metadata']

    full_start = time.time()
    print('Processing ActionSense data...')

    activity_df_list = []
    video_frames = defaultdict(dict)

    for fn in sorted(glob(f'{root}/*.hdf5')):
        print(f'Processing {fn}...')

        subject_id = int(fn.split('_')[-1].split('.')[0].strip('S'))
        prefix = fn.split('_')[0] + '_' + fn.split('_')[1]

        for pfx in glob(f'{root}/*.avi'):
            if pfx.startswith(prefix):
                video_fn = pfx
                break
        
        hdf_file = h5py.File(fn, 'r')
        video_ts = hdf_file['eye-tracking-video-worldGaze/frame_timestamp/time_s'][:].flatten()
        video_frame_nr = np.array([i for i in range(len(video_ts))])

        activities = hdf_file['experiment-activities/activities/data']
        for idx, activity in tqdm.tqdm(enumerate(activities), total=len(activities)):
            label, time_mark, quality, notes = activity.astype('U').tolist()
            # only use good quality data and get data between start and end
            if quality != 'Good' or time_mark == 'Stop':
                continue
            
            start_time = hdf_file['experiment-activities/activities/time_s'][idx].item()
            end_time = hdf_file['experiment-activities/activities/time_s'][idx+1].item()

            video_mask = (video_ts >= start_time) & (video_ts <= end_time)
            video_ts_clipped = video_ts[video_mask]
            video_frame_nr_clipped = video_frame_nr[video_mask]

            activity_sensors_dict = {}

            for k in hdf_file:
                # skip experiment metadata 
                if k.startswith('experiment-') or k.startswith('eye-tracking-video-'):
                    continue
                # and groups for precise timing, we use a rather coarse temporal resolution
                if k == 'eye-tracking-time' or k == 'xsens-time':
                    continue
                # skip calibration groups
                if '-calibration-' in k:
                    continue
                # skip tactile glove data, since not all subjects have this data
                if k == 'tactile-glove-left' or k == 'tactile-glove-right':
                    continue
                
                ignored_fields = [
                    # ignore fields that are not sensor measurements
                    'battery', 'emg', 'gesture', 'rssi', 'synced', 'timestamp',
                    # ignore fields, which are not available for all subjects
                    'rotation_xzy_deg', 'rotation_zxy_deg', 'orientation_euler_deg', 'orientation_quaternion', 'position_cm'
                ]
                for l in hdf_file[k]:
                    if l in ignored_fields:
                        continue
                    
                    full_key = f'{k}/{l}'
                    timestamps = hdf_file[full_key]['time_s'][:].flatten()
                    mask = (timestamps >= start_time) & (timestamps <= end_time)

                    data = hdf_file[full_key + '/data'][:][mask]

                    flattend_dim = np.prod(data.shape[1:])
                    for d in range(flattend_dim):
                        ts_index = timestamps[mask]
                        data_col = data.reshape(data.shape[0], -1)[:, d]
                        tmp_df = pd.DataFrame({'timestamp': ts_index, f'{full_key}__{d}': data_col}).set_index('timestamp')
                        tmp_df = tmp_df.groupby(level=0).mean() # some timestamps are duplicated, and have different measturements associated with them
                        tmp_df = tmp_df.reindex(video_ts_clipped, method='nearest', tolerance=0.01)  # reindex to synchronize with video
                        activity_sensors_dict[f'{full_key}_{d}'] = tmp_df
            
            activity_df = pd.concat(activity_sensors_dict.values(), axis=1).assign(label=label, subject=subject_id)
            activity_df['video_id'] = prefix
            activity_df['video_frame'] = video_frame_nr_clipped
            with warnings.catch_warnings(action="ignore"):
                activity_df = activity_df.interpolate(limit=10).dropna()
            activity_df_list.append(activity_df)

            cap = cv2.VideoCapture(video_fn)
            for frame_idx in activity_df['video_frame']:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                _, frame = cap.read()
                resized_frame = cv2.resize(frame, (args.image_size[0], args.image_size[1]),interpolation=cv2.INTER_AREA)
                video_frames[prefix][frame_idx] = resized_frame
            cap.release()

    df = pd.concat(activity_df_list, axis=0)
    data = {
        'dataframe': df,
        'video_frames': video_frames,
        'metadata': {
            'img_size': args.image_size
        }
    }
    joblib.dump(data, pickled_fn)

    print(df.shape)
    print(f'Took: {(time.time() - full_start)/60:.4f} minutes')

    return data['dataframe'], data['video_frames'], data['metadata']


class ActionSenseDataset(Dataset):
    def __init__(self, df, video_frames, scaler, sens_cols, image_size=(64, 64), unimodal=''):
        df = df.reset_index(drop=True)
        self.video_frames = video_frames
        self.image_size = image_size
        self.unimodal = unimodal

        # Keep frequently used columns/values as arrays to avoid pandas overhead in __getitem__
        self.video_ids = df['video_id'].to_numpy()
        self.frame_idxs = df['video_frame'].to_numpy(dtype=np.int64)
        self.labels = torch.as_tensor(df['label'].to_numpy(), dtype=torch.long)

        self.sens_cols = sens_cols
        sensor_np = scaler.transform(df[self.sens_cols]).astype(np.float32)
        self.sensor_data = torch.from_numpy(sensor_np)

        # Cache frame references to reduce dict lookups during iteration
        self.frames = [self.video_frames[v_id][int(f_idx)] for v_id, f_idx in zip(self.video_ids, self.frame_idxs)]

        self.zero_sensor = torch.zeros(len(self.sens_cols), dtype=torch.float32)
        self.n_classes = len(np.unique(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.unimodal != 'sensor':
            frame = self.frames[idx]
            # Frames are already resized during preprocessing; convert directly to CHW float tensor
            image = torch.from_numpy(frame).permute(2, 0, 1).float().div_(255)
        else:
            image = torch.zeros((3, *self.image_size), dtype=torch.float32)

        if self.unimodal != 'image':
            sensor_data = self.sensor_data[idx]
        else:
            sensor_data = self.zero_sensor

        return (image, sensor_data), self.labels[idx]


def approximate_rank_pooling_weights(T):
    if T < 1:
        raise ValueError("T must be >= 1")

    # Harmonic numbers H_0 ... H_T
    H = np.zeros(T + 1, dtype=np.float64)
    H[1:] = np.cumsum(1.0 / np.arange(1, T + 1, dtype=np.float64))

    t = np.arange(1, T + 1, dtype=np.float64)
    alpha = 2 * (T - t + 1) - (T + 1) * (H[T] - H[(t - 1).astype(int)])
    return alpha.astype(np.float32)


def dynamic_image_arp(frames, use_sqrt=False, output_uint8=True, eps=1e-8):
    x = np.asarray(frames)

    if x.ndim not in (3, 4):
        raise ValueError("frames must have shape [T,H,W] or [T,H,W,C]")

    if x.shape[0] < 1:
        raise ValueError("frames must contain at least one frame")

    x = x.astype(np.float32)

    # Optional nonlinearity
    if use_sqrt:
        # For image pixels usually x >= 0, but this is robust either way
        x = np.sign(x) * np.sqrt(np.abs(x))

    alpha = approximate_rank_pooling_weights(x.shape[0])

    # Weighted sum over time axis
    # [T] x [T,H,W,(C)] -> [H,W,(C)]
    dyn = np.tensordot(alpha, x, axes=(0, 0)).astype(np.float32)

    if not output_uint8:
        return dyn

    # Min-max normalization to image range [0,255]
    dyn_min = dyn.min()
    dyn_max = dyn.max()
    dyn = (dyn - dyn_min) / max(dyn_max - dyn_min, eps)
    dyn = np.clip(dyn * 255.0, 0, 255).astype(np.uint8)

    return dyn


def load_robomnist_data(args, root='./data/robomnist/'):
    cache_fn = os.path.join(root, 'robomnist_cache.pkl')

    if os.path.exists(cache_fn):
        print('Loading existing RoboMNIST data from cache...')
        data = joblib.load(cache_fn)
        return data['dataframe'], data['images']
    else:
        print('Building RoboMNIST data...')

    speed_mapping = {
        'low': 1,
        'med': 2,
        'high': 3
    }
    pattern = re.compile(
        r"(?:.*/)?(?P<name>"
        r"(?P<folder_nr>\d+)_Ar(?P<robot_nr>\d+)_V(?P<velocity>[^_]+)_Ac(?P<label>\d+)"
        r")$"
    )

    stat_fns = {
        'mean': np.mean,
        'std': np.std,
        'min': np.min,
        'q25': lambda x, axis: np.quantile(x, 0.25, axis=axis),
        'median': np.median,
        'q75': lambda x, axis: np.quantile(x, 0.75, axis=axis),
        'max': np.max,
    }

    def _flat_features(prefix, values):
        flat_values = np.asarray(values).reshape(-1)
        return {f'{prefix}_{i}': val for i, val in enumerate(flat_values)}

    def _sample_prefix(fn):
        base = os.path.basename(fn)
        prefix, sep, _ = base.rpartition('_Rx')
        return prefix if sep else os.path.splitext(base)[0]

    def _resize_robomnist_frame(rgb, robot):
        if robot == 2:
            cropped = rgb[:, 550:1270, :]
        elif robot == 1:
            cropped = rgb[:, 0:550, :]
        else:
            raise ValueError(f'Unsupported robot id: {robot}')

        return cv2.resize(cropped, (args.image_size[0], args.image_size[1]), interpolation=cv2.INTER_AREA)

    def _build_dynamic_image(video_fn, robot):
        cap = cv2.VideoCapture(video_fn)
        frames32 = []
        frame_idx = 0

        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames32.append(_resize_robomnist_frame(rgb, robot))

                frame_idx += 1
        finally:
            cap.release()

        if not frames32:
            raise ValueError(f'No usable frames found in {video_fn}')

        return dynamic_image_arp(frames32)

    def _load_csi_features(csi_json_fn):
        with open(csi_json_fn, 'r') as f:
            jf = json.load(f)

        csi_paths = jf[0]['complex_csi']
        if not csi_paths:
            raise ValueError(f'No CSI frames found in {csi_json_fn}')

        arr_frames = np.asarray([np.loadtxt(csi_path, dtype=np.complex64) for csi_path in csi_paths], dtype=np.complex64)
        arr_frames_real = arr_frames.real
        arr_frames_imag = arr_frames.imag

        features = {}
        for stat_name, stat_fn in stat_fns.items():
            features.update(_flat_features(f'csi_real_{stat_name}', stat_fn(arr_frames_real, axis=0)))
            features.update(_flat_features(f'csi_imag_{stat_name}', stat_fn(arr_frames_imag, axis=0)))

        return features

    def _process_group(group_dir):
        match = pattern.search(group_dir)
        if not match:
            return []

        robot = int(match['robot_nr'])
        label = int(match['label'])
        speed = speed_mapping[match['velocity']]

        video_files = {_sample_prefix(fn): fn for fn in sorted(glob(os.path.join(group_dir, '*Rx2_cam.mp4')))}
        csi_files   = {_sample_prefix(fn): fn for fn in sorted(glob(os.path.join(group_dir, '*_csi.json')))}

        if not video_files and not csi_files:
            return []

        missing_videos = sorted(csi_files.keys() - video_files.keys())
        missing_csi = sorted(video_files.keys() - csi_files.keys())
        if missing_videos or missing_csi:
            warnings.warn(f'Skipping unmatched RoboMNIST files in {group_dir}: {len(missing_videos)} CSI-only, {len(missing_csi)} video-only samples.')

        sample_ids = sorted(video_files.keys() & csi_files.keys())
        group_rows = []

        for sample_id in sample_ids:
            dynamic_image = _build_dynamic_image(video_files[sample_id], robot)
            row_dict = {
                'robot': robot,
                'label': label,
                'speed': speed,
            }
            row_dict.update(_load_csi_features(csi_files[sample_id]))

            group_rows.append((sample_id, row_dict, dynamic_image))

        return group_rows

    group_dirs = [g for g in sorted(glob(os.path.join(root, '*'))) if os.path.isdir(g) and pattern.search(g)]

    configured_workers = min(8, os.cpu_count() or 1)
    max_workers = min(len(group_dirs) or 1, max(1, int(configured_workers)))

    group_results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_group = {executor.submit(_process_group, group_dir): group_dir for group_dir in group_dirs}
        for future in as_completed(future_to_group):
            group_dir = future_to_group[future]
            group_results[group_dir] = future.result()

    images = []
    row_dicts = []

    # Preserve the original deterministic group/sample ordering after parallel work.
    for group_dir in group_dirs:
        for _, row_dict, dynamic_image in group_results.get(group_dir, []):
            row_dict['image_idx'] = len(images)
            images.append(dynamic_image)
            row_dicts.append(row_dict)

    if row_dicts:
        df = pd.DataFrame(row_dicts)
    else:
        raise ValueError("No valid data found in the specified directory.")

    data = {
        'dataframe': df,
        'images': images
    }
    joblib.dump(data, cache_fn)

    return df, images


class RoboMNISTDataset(Dataset):
    def __init__(self, df, images, scaler, sens_cols, unimodal=''):        
        self.unimodal = unimodal
        self.sens_cols = sens_cols
        self.scaler = scaler

        self.df = df.reset_index(drop=True)
        self.sensor_data_scaled = self.scaler.transform(df[self.sens_cols])
        self.images = images

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.unimodal != 'sensor':
            image = self.transform(self.images[int(row['image_idx'])])
        else:
            sample_image = self.images[0]
            image = torch.zeros((3, sample_image.shape[0], sample_image.shape[1]))

        if self.unimodal != 'image':
            sensor_data = torch.Tensor(self.sensor_data_scaled[idx , :].astype(np.float32))
        else:
            sensor_data = torch.zeros((len(self.scaler.feature_names_in_),))

        label = row['label']

        if self.unimodal == 'model':
            return image, label
        else:
            return (image, sensor_data), label


class Widar_Dataset(Dataset):
    def __init__(self, root_dir):
        self.data_list = glob(root_dir + '/*/*.csv')
        self.folder    = sorted(glob(root_dir + '/*/'))
        self.category  = {self.folder[i].split('/')[-2]: i for i in range(len(self.folder))}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path = self.data_list[idx]
        y    = self.category[path.split('/')[-2]]
        x    = np.genfromtxt(path, delimiter=',')
        x    = (x - 0.0025) / 0.0119
        x    = x.reshape(22, 20, 20)
        x    = np.clip(x, -3, 3) / 3.0
        return torch.FloatTensor(x), y
