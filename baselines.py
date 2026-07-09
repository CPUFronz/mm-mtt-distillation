import os

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from dppy.finite_dpps import FiniteDPP

from utils import (
    parse_args,
    get_dataset,
    build_dataset,
    get_network,
    evaluate_synset,
    fix_seed,
)
from reparam_module import ReparamModule


def get_feature_extractor(channel, num_classes, im_size, args, kwargs):
    buffer_path = os.path.join(args.buffer_path, args.dataset, args.model, 'replay_buffer_0.pt')
    if not os.path.exists(buffer_path):
        raise FileNotFoundError(f"Buffer file not found at {buffer_path}. Please make sure buffer files exist.")

    print("Loading feature extractor weights from:", buffer_path)

    # we are using the weights of final iteraton of the first expert as feature extractor
    weights = torch.load(buffer_path, map_location=args.device)[0][-1]

    model = get_network(args.model, channel, num_classes, im_size, **kwargs)
    model = ReparamModule(model)

    named_modules = list(model.named_modules())
    idx = -2
    while True:
        feature_layer_name, feature_layer = named_modules[idx]
        if hasattr(feature_layer, 'out_features'):
            break
        idx -= 1
        if idx < -len(named_modules):
            raise RuntimeError("No suitable feature layer found in the model.")

    flat_weights = torch.cat([p.detach().reshape(-1) for p in weights], 0).to(args.device)
    model = model.to(args.device)
    for module in model.modules():
        if hasattr(module, 'device'):
            module.device = args.device
    model.eval()

    features = {}

    def save_features(_module, _inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        features["value"] = output

    feature_layer.register_forward_hook(save_features)

    def feature_extractor(x):
        if isinstance(x, tuple):
            x = tuple(item.to(args.device) for item in x)
        else:
            x = x.to(args.device)
        features.clear()
        with torch.no_grad():
            model(x, flat_param=flat_weights)
        if "value" not in features:
            raise RuntimeError(f"Layer {feature_layer_name} did not produce features.")
        output = features["value"]
        return output.flatten(1) if output.ndim > 2 else output

    feature_extractor.layer_name = feature_layer_name
    return feature_extractor


def random_select(indices_class, features_all, spc):
    selected_indices = []

    for c, idxs in enumerate(indices_class):
        if len(idxs) < spc:
            raise ValueError(f"Not enough samples in class {c} to draw ipc={spc} (found {len(idxs)}).")
        choice = np.random.choice(idxs, size=spc, replace=False)
        selected_indices.extend(choice)

    return selected_indices


def kmeans_select(indices_class, features_all, spc):
    selected_indices = []

    for c, idxs in enumerate(indices_class):
        class_indices = np.asarray(idxs, dtype=np.int64)

        if len(class_indices) <= spc:
            raise ValueError(f"Not enough samples in class {c} to draw ipc={spc} (found {len(idxs)}).")

        class_features = features_all[class_indices]
        km = KMeans(n_clusters=spc, n_init=10)
        km.fit(class_features)
        chosen = []
        for k in range(spc):
            cluster_mask = km.labels_ == k
            if cluster_mask.sum() == 0:
                continue
            cluster_features = class_features[cluster_mask]
            cluster_local_indices = np.where(cluster_mask)[0]
            distances_to_center = np.linalg.norm(cluster_features - km.cluster_centers_[k], axis=1)
            chosen.append(cluster_local_indices[distances_to_center.argmin()])
        selected_indices.extend(class_indices[chosen].tolist())

    return selected_indices


def dpp_select(indices_class, features_all, spc):
    scaler = StandardScaler()
    feat_n = scaler.fit_transform(features_all)

    selected_indices = []
    for c, idxs in enumerate(indices_class):
        class_indices = np.asarray(idxs, dtype=np.int64)
        cls_feat = feat_n[class_indices]

        if len(cls_feat) <= spc:
            raise ValueError(f"Not enough samples in class {c} to draw ipc={spc} (found {len(idxs)}).")

        pw_dist = cdist(cls_feat, cls_feat, metric='euclidean')
        sigma_vals = pw_dist[pw_dist > 0]
        sigma   = np.median(sigma_vals) if len(sigma_vals) else 1.0
        L       = np.exp(-pw_dist**2 / (2 * sigma**2))

        dpp = FiniteDPP(kernel_type='likelihood', L=L)
        dpp.sample_exact_k_dpp(size=spc)
        chosen = list(dpp.list_of_samples[0])

        selected_indices.extend(class_indices[chosen].tolist())

    return selected_indices


def kcenter_select(indices_class, features_all, spc):
    selected_indices = []
    for c, idxs in enumerate(indices_class):
        class_indices = np.asarray(idxs, dtype=np.int64)
        class_features = features_all[class_indices]

        if len(class_features) <= spc:
            raise ValueError(f"Not enough samples in class {c} to draw ipc={spc} (found {len(idxs)}).")
        
        chosen = [np.random.randint(len(class_features))]
        dists    = np.full(len(class_features), np.inf)
        for _ in range(spc - 1):
            new_d = np.linalg.norm(class_features - class_features[chosen[-1]], axis=1)
            dists = np.minimum(dists, new_d)
            dists[chosen] = -np.inf
            chosen.append(int(np.argmax(dists)))
        
        selected_indices.extend(class_indices[chosen].tolist())

    return selected_indices


def gist_select(indices_class, features_all, spc):
    eps = 0.05
    alpha = 0.95
    selected_indices = []

    features = features_all.detach().cpu().numpy() if isinstance(features_all, torch.Tensor) else np.asarray(features_all)
    features = features.reshape(features.shape[0], -1)
    features = StandardScaler().fit_transform(features)

    for c, idxs in enumerate(indices_class):
        class_indices = np.asarray(idxs, dtype=np.int64)

        if len(class_indices) < spc:
            raise ValueError(f"Not enough samples in class {c} to draw ipc={spc} (found {len(idxs)}).")
        if spc <= 0:
            continue
        if len(class_indices) == spc:
            selected_indices.extend(class_indices.tolist())
            continue

        class_features = features[class_indices]
        pw_dist = cdist(class_features, class_features, metric='euclidean')
        dmax = float(pw_dist.max())

        if dmax <= np.finfo(float).eps:
            selected_indices.extend(class_indices[:spc].tolist())
            continue

        # No utility function is passed in this baseline API, so use a linear
        # monotone utility: samples closer to the class center are more valuable.
        center = class_features.mean(axis=0, keepdims=True)
        center_dist = np.linalg.norm(class_features - center, axis=1)
        min_center_dist = float(center_dist.min())
        max_center_dist = float(center_dist.max())
        center_dist_range = max_center_dist - min_center_dist
        if center_dist_range <= np.finfo(float).eps:
            utility = np.ones(len(class_indices), dtype=np.float64)
        else:
            utility = 1.0 - (center_dist - min_center_dist) / center_dist_range
            utility = np.clip(utility, 0.0, 1.0)

        best_chosen = []
        best_score = -np.inf
        thresholds = [0.0]
        scale = 1.0
        while scale <= 2.0 / eps:
            thresholds.append(scale * eps * dmax / 2.0)
            scale *= 1.0 + eps

        for threshold in thresholds:
            chosen = []
            min_dist = np.full(len(class_indices), np.inf, dtype=np.float64)

            for _ in range(spc):
                candidate_mask = min_dist >= threshold
                if chosen:
                    candidate_mask[chosen] = False
                candidates = np.flatnonzero(candidate_mask)
                if len(candidates) == 0:
                    break

                best_local = int(candidates[np.argmax(utility[candidates])])
                chosen.append(best_local)
                min_dist = np.minimum(min_dist, pw_dist[best_local])

            if not chosen:
                continue

            if len(chosen) <= 1:
                div_value = 1.0
            else:
                chosen_dist = pw_dist[np.ix_(chosen, chosen)]
                div_value = float(chosen_dist[np.triu_indices(len(chosen), 1)].min() / dmax)
            utility_value = float(utility[chosen].sum() / spc)
            score = alpha * utility_value + (1.0 - alpha) * div_value

            if score > best_score or (np.isclose(score, best_score) and len(chosen) > len(best_chosen)):
                best_score = score
                best_chosen = chosen

        if spc >= 2:
            u, v = np.unravel_index(np.argmax(pw_dist), pw_dist.shape)
            chosen = [int(u), int(v)]
            utility_value = float(utility[chosen].sum() / spc)
            score = alpha * utility_value + (1.0 - alpha)
            if score > best_score or (np.isclose(score, best_score) and len(chosen) > len(best_chosen)):
                best_chosen = chosen

        selected_indices.extend(class_indices[best_chosen].tolist())

    return selected_indices


def main(args):
    fix_seed(args.seed)

    function_map = {
        'random': random_select,
        'kmeans': kmeans_select,
        'DPP': dpp_select,
        'kcenter': kcenter_select,
        'gist': gist_select,
    }

    channel, im_size, num_classes, _, _, _, dst_train, _, testloader, class_map = get_dataset(args)
    images_all, sensor_all, labels_all, indices_class = build_dataset(dst_train, channel, num_classes, class_map, args.unimodal)


    kwargs = {
        'unimodal': args.unimodal,
        'n_groups': args.n_groups,
        'n_sensors': getattr(args, 'n_sensors', None),
        'n_sensor_features': getattr(args, 'n_sensor_features', None),
    }
    feature_extractor = get_feature_extractor(channel, num_classes, im_size, args, kwargs)
    features_all = []
    for idx in range(len(labels_all)):
        if args.unimodal == 'unimodal':
            input_data = images_all[idx].unsqueeze(0)
        else:
            img = images_all[idx] if args.unimodal != 'sensor' else torch.zeros_like(images_all[0], dtype=torch.float32)
            sen = sensor_all[idx] if args.unimodal != 'image' else torch.zeros_like(sensor_all[0], dtype=torch.float32)
            input_data = (img.unsqueeze(0), sen.unsqueeze(0))
        feat = feature_extractor(input_data)
        features_all.append(feat.cpu().numpy())
    features_all = np.concatenate(features_all, axis=0)

    args.lr_net_syn = args.lr

    if args.method == 'all':
        methods = function_map.keys()
    else:
        methods = [args.method]
    
    results = {}

    for method in methods:
        print(f"\nEvaluating method: {method}")
        selected_indices = function_map[method](indices_class, features_all, args.spc)

        labels_train = labels_all[selected_indices].long()
        if args.unimodal == 'unimodal':
            images_train = images_all[selected_indices]
            sensor_train = None
        else:
            images_train = images_all[selected_indices] if args.unimodal != 'sensor' else torch.zeros(len(selected_indices), *images_all[0].shape)
            sensor_train = sensor_all[selected_indices] if args.unimodal != 'image' else torch.zeros(len(selected_indices), *sensor_all[0].shape)

        accs_train, accs_test = [], []
        for eval_run in range(args.num_eval):            
            fix_seed(args.seed + eval_run)

            net_eval = get_network(args.model, channel, num_classes, im_size, **kwargs).to(args.device)
            for module in net_eval.modules():
                if hasattr(module, 'device'):
                    module.device = args.device
            _, acc_train_list, acc_test = evaluate_synset(eval_run, net_eval, images_train, labels_train, testloader, args, sensor_train=sensor_train)
            accs_train.append(acc_train_list[-1])
            accs_test.append(acc_test)

        print('\n')
        print(
            f"Train Accuracy mean={np.mean(accs_train):.4f} std={np.std(accs_train):.4f} | "
            f"Test Accuracy  mean={np.mean(accs_test):.4f}  std={np.std(accs_test):.4f}"
        )
        print('\n' + '-' * 50)

        results[method] = (np.mean(accs_test), np.std(accs_test))


    if args.method == 'all':
        print("\nSummary of all methods (Test Accuracy):")
        for method, (mean_acc, std_acc) in results.items():
            print(f"{method}:".ljust(10) + f"mean={mean_acc:.3f} | std={std_acc:.3f}")


if __name__ == '__main__':
    args = parse_args('baseline')

    main(args)
