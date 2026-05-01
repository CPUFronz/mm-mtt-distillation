import argparse
import numpy as np
import torch

from utils import (
    get_dataset,
    get_network,
    get_eval_pool,
    evaluate_synset,
    ParamDiffAug,
    get_daparam,
    fix_seed,
)


MULTIMODAL_MODELS = {"MMSConvB", "Perceiver"}


def _map_label(label, class_map):
    label = int(label)
    if class_map is not None and label in class_map:
        return class_map[label]
    return label


def build_class_indices(dst_train, num_classes, class_map, args):
    indices_class = [[] for _ in range(num_classes)]
    for idx in range(len(dst_train)):
        if args.unimodal == 'model':
            _, lab = dst_train[idx]
        else:
            (_, _), lab = dst_train[idx]
        lab = _map_label(lab, class_map)
        if lab < 0 or lab >= num_classes:
            raise ValueError(f"Label {lab} out of range for num_classes={num_classes}")
        indices_class[lab].append(idx)
    return indices_class


def select_random_subset(dst_train, indices_class, ipc, args):
    rng = np.random.RandomState(args.seed)
    selected = []
    selected_labels = []

    for c, idxs in enumerate(indices_class):
        if len(idxs) < ipc:
            raise ValueError(f"Not enough samples in class {c} to draw ipc={ipc} (found {len(idxs)}).")
        choice = rng.choice(idxs, size=ipc, replace=False)
        selected.extend(choice.tolist())
        selected_labels.extend([c] * ipc)

    # Shuffle to avoid class-ordered batches (DataLoader also shuffles).
    order = rng.permutation(len(selected))
    selected = [selected[i] for i in order]
    selected_labels = [selected_labels[i] for i in order]

    images = []
    sensors = []
    for idx in selected:
        if args.unimodal == 'model':
            img, _ = dst_train[idx]
            images.append(img)
        else:
            (img, sen), _ = dst_train[idx]
            images.append(img)
            sensors.append(sen)

    images_train = torch.stack(images, dim=0)
    labels_train = torch.tensor(selected_labels, dtype=torch.long)
    sensor_train = None
    if args.unimodal != 'model':
        sensor_train = torch.stack(sensors, dim=0)

    return images_train, labels_train, sensor_train


def main(args):
    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")

    if args.model not in MULTIMODAL_MODELS:
        args.unimodal = 'model'

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    fix_seed(args.seed)

    channel, im_size, num_classes, _class_names, _mean, _std, dst_train, _dst_test, testloader, _loader_train_dict, class_map, _class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)

    args.im_size = im_size
    args.dsa_param = ParamDiffAug.copy()
    if args.dsa:
        args.dc_aug_param = None
    else:
        args.dc_aug_param = get_daparam(args.dataset, args.model, args.model, args.ipc)

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    indices_class = build_class_indices(dst_train, num_classes, class_map, args)

    if args.method != 'random':
        raise ValueError(f"Unknown method: {args.method}")

    images_train, labels_train, sensor_train = select_random_subset(dst_train, indices_class, args.ipc, args)

    kwargs = {
        'unimodal': args.unimodal,
        'n_groups': args.n_groups,
        'n_sensors': getattr(args, 'n_sensors', None),
        'n_sensor_features': getattr(args, 'n_sensor_features', None),
    }

    for model_eval in model_eval_pool:
        accs_test = []
        accs_train = []
        for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, channel, num_classes, im_size, **kwargs).to(args.device)
            _, acc_train_list, acc_test = evaluate_synset(it_eval, net_eval, images_train, labels_train, testloader, args, texture=args.texture, sensor_train=sensor_train,)
            accs_test.append(acc_test)
            accs_train.append(acc_train_list[-1] if acc_train_list else 0.0)

        accs_test = np.array(accs_test)
        accs_train = np.array(accs_train)
        print(
            f"Random baseline ({model_eval}): "
            f"train acc mean={accs_train.mean():.4f} std={accs_train.std():.4f} | "
            f"test acc mean={accs_test.mean():.4f} std={accs_test.std():.4f}"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline methods for coreset comparison')

    parser.add_argument('--method', type=str, default='random', choices=['random'],
                        help='baseline method')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette',
                        help='ImageNet subset (only used when --dataset=ImageNet)')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--res', type=int, default=128, help='resolution for imagenet')

    parser.add_argument('--ipc', type=int, default=1, help='images per class to select')

    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode, check utils.py for more info')
    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--epoch_eval_train', type=int, default=1000,
                        help='epochs to train a model with selected data')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for evaluation training')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data loader')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')

    parser.add_argument('--zca', action='store_true', help='do ZCA whitening')

    parser.add_argument('--texture', action='store_true', help='train on textures instead')
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')

    parser.add_argument('--seed', type=int, default=42, help='set random seed')
    parser.add_argument('--unimodal', type=str, default='', choices=['', 'image', 'sensor'],
                        help='unimodal training (only for multimodal datasets)')
    parser.add_argument('--n_groups', type=int, default=8, help='group norm groups (for MMSConvB)')

    args = parser.parse_args()

    main(args)
