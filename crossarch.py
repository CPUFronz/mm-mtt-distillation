import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import wandb
import numpy as np

from networks import MultimodalPerceiver
from utils import evaluate_synset, fix_seed, get_dataset, get_network


def best_iteration(run):
    """Support the current accuracy metric and the name used by older runs."""
    for metric in ("Accuracy", "Accuracy/"):
        rows = [
            row for row in run.scan_history(keys=[metric, "_step"])
            if row.get(metric) is not None
        ]
        if rows:
            return int(max(rows, key=lambda row: row[metric])["_step"])
    raise ValueError(f"Run {run.id} has no recorded evaluation accuracy.")


def test_accuracy_at_iteration(run, iteration):
    """Read the original snapshot's test accuracy, rather than the running maximum."""
    rows = list(run.scan_history(min_step=iteration, max_step=iteration + 1))
    for metric, std_metric in (("Accuracy", "Std"), ("Accuracy/", "Std/")):
        accuracy_rows = [
            row for row in rows
            if row.get("_step") == iteration and row.get(metric) is not None
        ]
        if accuracy_rows:
            best_row = max(accuracy_rows, key=lambda row: row[metric])
            std = best_row.get(std_metric)
            return float(best_row[metric]), float(std) if std is not None else None
    return None, None


def load_snapshot(api, run, iteration):
    collection = run.summary.get("eval_artifact_collection", f"distillation-eval-{run.id}")
    artifact_name = f"{run.entity}/{run.project}/{collection}:iter_{iteration}"
    artifact = api.artifact(artifact_name)
    root = Path("logged_files") / "ablations" / run.id / str(iteration)
    artifact_path = Path(artifact.download(root=str(root)))

    tensors = []
    for prefix in ("images", "sensors", "labels"):
        files = sorted(artifact_path.glob(f"{prefix}*.pt"))
        if len(files) != 1:
            raise ValueError(f"Expected one {prefix} tensor in {artifact_name}, found {len(files)}.")
        tensor = torch.load(files[0], map_location="cpu", weights_only=True)
        tensors.append(tensor.long() if prefix == "labels" else tensor.float())
    images, sensors, labels = tensors
    if not (len(images) == len(sensors) == len(labels) > 0):
        raise ValueError("Snapshot images, sensors and labels must have the same nonzero length.")
    return images, sensors, labels


def build_eval_network(args, channel, num_classes, im_size):
    return get_network(
        args.model,
        channel,
        num_classes,
        im_size,
        dist=False,
        unimodal=args.unimodal,
        n_groups=getattr(args, "n_groups", 8),
        n_sensors=args.n_sensors,
        n_sensor_features=args.n_sensor_features,
        device=args.device
    )


def main(args):
    api = wandb.Api()
    run_path = args.run_id.strip("/")
    if "/" not in run_path:
        run_path = f"{api.default_entity}/DatasetDistillation/{run_path}"
    elif len(run_path.split("/")) != 3:
        raise ValueError("run_id must be a run ID or entity/project/run_id.")
    run = api.run(run_path)

    print(f'\n' + ('#' * 80))
    print(f'Loading run {run.name} ({run.id}):')
    print('#' * 80)

    iteration = args.iteration if args.iteration is not None else best_iteration(run)
    original_test_acc, original_test_std = test_accuracy_at_iteration(run, iteration)

    lr_rows = list(run.scan_history(
        keys=["Synthetic_LR"], min_step=iteration, max_step=iteration + 1,
    ))
    if not lr_rows or lr_rows[0].get("Synthetic_LR") is None:
        raise ValueError(f"Run {run.id} has no Synthetic_LR at iteration {iteration}.")

    source_config = {
        str(key): value for key, value in dict(run.config).items()
        if not str(key).startswith("_")
    }
    run_args = SimpleNamespace(**source_config)
    run_args.device = args.device
    run_args.lr_net_syn = float(lr_rows[0]["Synthetic_LR"])
    run_args.num_eval = 5
    run_args.unimodal = ""
    images, sensors, labels = load_snapshot(api, run, iteration)
    channel, im_size, num_classes, _, _, _, _, _, testloader, _ = get_dataset(run_args)
    run_args.im_size = im_size

    run_args.epoch_eval_train = args.epoch_eval_train

    variants = [("Perceiver", "")]

    print(f"Source: {run_path}, snapshot: {iteration}, epochs: {run_args.epoch_eval_train}")
    accuracies_train, accuracies_test = {}, {}
    for variant, modality in variants:
        eval_args = SimpleNamespace(**vars(run_args))
        eval_args.model = variant
        eval_args.unimodal = modality

        config = dict(source_config)
        config.update({
            "device": eval_args.device,
            "model": eval_args.model,
            "unimodal": modality,
            "lr_net_syn": eval_args.lr_net_syn,
            "epoch_eval_train": eval_args.epoch_eval_train,
            "training_logs": False,
            "variant": variant,
            "iteration": iteration,
            "source_run_id": run.id,
            "source_run_name": run.name,
            "source_run_path": run_path,
            "synthetic_images": int(labels.shape[0]),
        })

        run_accs_train, run_accs_test = [], []
        for it_eval in range(eval_args.num_eval):
            fix_seed(eval_args.seed + it_eval)
            net = build_eval_network(eval_args, channel, num_classes, im_size)
            _, acc_train_list, acc_test = evaluate_synset(
                it_eval,
                net,
                images,
                labels,
                testloader,
                eval_args,
                sensor_train=sensors,
                training_logs=False,
            )
            run_accs_train.append(float(acc_train_list[-1]))
            run_accs_test.append(float(acc_test))

        accuracies_train[variant] = run_accs_train
        accuracies_test[variant] = run_accs_test

    print("\nAccuracies:")
    for variant in accuracies_test.keys():
        train_acc = np.mean(accuracies_train[variant])
        train_std = np.std(accuracies_train[variant])
        test_acc = np.mean(accuracies_test[variant])
        test_std = np.std(accuracies_test[variant])
        print(f"  {variant:20s}: {train_acc:.4f} ± {train_std:.4f} | {test_acc:.4f} ± {test_std:.4f} \t ### ${test_acc*100:.2f}\% \pm {test_std*100:.2f}$")

    original_test_result = f"{original_test_acc:.4f}" if original_test_acc is not None else "not logged"
    if original_test_std is not None:
        original_test_result += f" ± {original_test_std:.4f}"
    print(f"  Original distill run (iteration {iteration}): test = {original_test_result}")

    print('\n' + ('#' * 80) + '\n\n')
    
    return accuracies_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="W&B DatasetDistillation run ID or entity/project/run_id.")
    parser.add_argument("--iteration", type=int, default=None, help="Distillation iteration to load. Defaults to the snapshot with the best accuracy.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epoch_eval_train", type=int, default=1000, help="Number of epochs to train the evaluation network on the synthetic data.")
    main(parser.parse_args())
