# TODO: simplify / unify with the rest of the codebase

import numpy as np
import torch

from utils import (
    parse_args,
    get_dataset,
    build_dataset,
    get_network,
    evaluate_synset,
    fix_seed,
)


def select_random_subset(images_all, sensor_all, labels_all, indices_class, spc, unimodal):
    selected_images = []
    selected_sensors = []
    selected_labels = []
    selected_indices = []

    for c, idxs in enumerate(indices_class):
        if len(idxs) < spc:
            raise ValueError(f"Not enough samples in class {c} to draw ipc={spc} (found {len(idxs)}).")
        choice = np.random.choice(idxs, size=spc, replace=False)
        selected_indices.extend(choice)

    # Shuffle to avoid class-ordered batches (DataLoader also shuffles).
    order = np.random.permutation(len(selected_indices))

    for idx in order:
        selected_labels.append(labels_all[idx])
        if unimodal != 'sensor':
            selected_images.append(images_all[idx])
        if unimodal != 'image' and unimodal != 'model':
            selected_sensors.append(sensor_all[idx])

    labels_train = torch.tensor(selected_labels, dtype=torch.long)
    images_train, sensor_train = None, None
    if unimodal != 'sensor':
        images_train = torch.stack(selected_images, dim=0)
    if unimodal != 'image' and unimodal != 'model':
        sensor_train = torch.stack(selected_sensors, dim=0)

    return images_train, sensor_train, labels_train


def main(args):
    fix_seed(args.seed)

    function_map = {
        'random': select_random_subset,
    }

    channel, im_size, num_classes, _, _, _, dst_train, _, testloader, class_map = get_dataset(args)
    images_all, sensor_all, labels_all, indices_class = build_dataset(dst_train, channel, num_classes, class_map, args.unimodal)

    kwargs = {
        'unimodal': args.unimodal,
        'n_groups': args.n_groups,
        'n_sensors': getattr(args, 'n_sensors', None),
        'n_sensor_features': getattr(args, 'n_sensor_features', None),
    }
    args.lr_net_syn = args.lr

    if args.method == 'all':
        methods = function_map.keys()
    else:
        methods = [args.method]
    
    for method in methods:
        print(f"Evaluating method: {method}")
        images_train, sensor_train, labels_train = function_map[method](images_all, sensor_all, labels_all, indices_class, args.spc, args.unimodal)

        accs_train, accs_test = [], []
        for eval_run in range(args.num_eval):            
            fix_seed(args.seed + eval_run)

            net_eval = get_network(args.model, channel, num_classes, im_size, **kwargs).to(args.device)
            _, acc_train_list, acc_test = evaluate_synset(eval_run, net_eval, images_train, labels_train, testloader, args, sensor_train=sensor_train)
            accs_train.append(acc_train_list[-1])
            accs_test.append(acc_test)

        print(accs_train, accs_test)
        print(
            f"Train Accuracy mean={np.mean(accs_train):.4f} std={np.std(accs_train):.4f} | "
            f"Test Accuracy  mean={np.mean(accs_test):.4f}  std={np.std(accs_test):.4f}"
        )


if __name__ == '__main__':
    args = parse_args('baseline')

    main(args)
