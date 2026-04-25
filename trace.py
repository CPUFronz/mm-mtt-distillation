# TODO: This file is still work in progress. My next steps would be to 
#        - add augmentations, to see how they influence the training
#        - add computation of traces, similar to Yang, William, et al. "What is dataset distillation learning?." arXiv preprint arXiv:2406.04284 (2024).

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from utils import (
    MultimodalTensorDataset,
    ParamDiffAug,
    TensorDataset,
    epoch,
    fix_seed,
    get_daparam,
    get_dataset,
    get_network
)


MULTIMODAL_MODELS = {"MMSConvB", "Perceiver"}
WANDB_PROJECT = "SyntheticTraining"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the evaluation model on a distilled snapshot from a W&B distillation run.")
    parser.add_argument("run_id", type=str, help="W&B distillation run id. You can also pass entity/project/run_id.")
    parser.add_argument("--iteration",type=int, default=None, help="Specific distillation iteration to load. Defaults to the best saved snapshot.")
    parser.add_argument("--entity", type=str, default=os.environ.get("WANDB_ENTITY"), help="Optional W&B entity when run_id is not fully qualified.")
    parser.add_argument("--project", type=str, default="DatasetDistillation", help="W&B project that contains the distillation run.")
    parser.add_argument("--epoch_eval_train", "--train_epochs", dest="train_epochs", type=int,default=None, help="Override the number of evaluation-training epochs. Defaults to epoch_eval_train from the distill run.",)
    parser.add_argument("--lr", "--train_lr", dest="train_lr", type=float, default=None, help="Override the evaluation learning rate. Defaults to Synthetic_LR at the selected iteration.")
    parser.add_argument("--batch_train", type=int, default=None, help="Override the synthetic training batch size. Defaults to batch_train from the distill run.")
    parser.add_argument("--batch_test", type=int, default=None, help="Override the real test batch size. Defaults to 128.")
    parser.add_argument("--data_path", type=str, default=None, help="Override the dataset path stored in the distill run config.")
    parser.add_argument("--seed", type=int, default=None, help="Override the seed from the distill run config.")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Device to use. Defaults to cuda when available.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for the evaluation SGD optimizer.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for the evaluation SGD optimizer.")
    parser.add_argument("--download_dir", type=str, default="./logged_files/traces",help="Directory used to download the selected W&B artifact.")
    return parser.parse_args()


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def prepare_wandb_config(values):
    config = {}
    for key, value in values.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            config[key] = value
        elif isinstance(value, Path):
            config[key] = str(value)
        elif isinstance(value, (list, tuple)):
            if all(isinstance(item, (int, float, str, bool)) or item is None for item in value):
                config[key] = list(value)
    return config


def build_run_args(run_config):
    defaults = {
        "batch_real": 256,
        "batch_train": 256,
        "canvas_samples": 1,
        "canvas_size": 2,
        "data_path": "data",
        "dataset": "CIFAR10",
        "dsa": "True",
        "dsa_strategy": "color_crop_cutout_flip_scale_rotate",
        "epoch_eval_train": 1000,
        "lr_teacher": 0.01,
        "model": "ConvNet",
        "n_groups": 8,
        "res": 128,
        "seed": 42,
        "subset": "imagenette",
        "texture": False,
        "unimodal": "",
        "zca": False,
    }

    merged = dict(defaults)
    merged.update(dict(run_config))

    args = SimpleNamespace(**merged)
    args.dsa = normalize_bool(args.dsa)
    args.texture = normalize_bool(getattr(args, "texture", False))
    args.zca = normalize_bool(getattr(args, "zca", False))
    args.unimodal = getattr(args, "unimodal", "")
    args.n_groups = int(getattr(args, "n_groups", 8))
    args.batch_real = int(getattr(args, "batch_real", 256))
    args.batch_train = int(getattr(args, "batch_train", 256))
    args.canvas_samples = int(getattr(args, "canvas_samples", 1))
    args.canvas_size = int(getattr(args, "canvas_size", 2))
    args.epoch_eval_train = int(getattr(args, "epoch_eval_train", 1000))
    args.lr_teacher = float(getattr(args, "lr_teacher", 0.01))
    args.seed = int(getattr(args, "seed", 42))
    args.res = int(getattr(args, "res", 128))
    args.dsa_param = ParamDiffAug()
    args.dc_aug_param = None if args.dsa else get_daparam(args.dataset, args.model, args.model, getattr(args, "ipc", None))
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model not in MULTIMODAL_MODELS:
        args.unimodal = "model"

    return args


def resolve_run(api, run_ref, entity=None, project="DatasetDistillation"):
    if run_ref.count("/") == 2:
        run_path = run_ref
    elif run_ref.count("/") == 1:
        resolved_entity = entity or getattr(api, "default_entity", None)
        if resolved_entity is None:
            raise ValueError(
                "W&B entity could not be inferred. Pass --entity or use a full entity/project/run_id path."
            )
        run_path = "{}/{}".format(resolved_entity, run_ref)
    else:
        resolved_entity = entity or getattr(api, "default_entity", None)
        if resolved_entity is None:
            raise ValueError(
                "W&B entity could not be inferred. Pass --entity or use a full entity/project/run_id path."
            )
        run_path = "{}/{}/{}".format(resolved_entity, project, run_ref)

    return api.run(run_path), run_path


def recover_synthetic_lr(run, iteration, fallback_lr):
    if iteration is None:
        return float(fallback_lr)

    history_rows = None
    try:
        history_rows = run.scan_history(keys=["Synthetic_LR"], min_step=iteration, max_step=iteration + 1)
    except TypeError:
        history_rows = run.scan_history(keys=["Synthetic_LR"])
    except Exception:
        history_rows = None

    if history_rows is not None:
        for row in history_rows:
            if row.get("_step") == iteration and row.get("Synthetic_LR") is not None:
                return float(row["Synthetic_LR"])

    return float(fallback_lr)


def resolve_snapshot_file(root_dir, candidates, required=True):
    for candidate in candidates:
        path = Path(root_dir) / candidate
        if path.exists():
            return path

    if required:
        raise FileNotFoundError(
            "Could not find any of the expected snapshot files: {}".format(", ".join(candidates))
        )
    return None


def load_distillation_snapshot(api, run_path, run, requested_iteration, download_root):
    entity, project, _ = run_path.split("/")
    summary = dict(getattr(run, "summary", {}) or {})
    collection = summary.get("eval_artifact_collection", "distillation-eval-{}".format(run.id))
    alias = "best_so_far" if requested_iteration is None else "iter_{}".format(int(requested_iteration))
    artifact_ref = "{}/{}/{}:{}".format(entity, project, collection, alias)

    try:
        artifact = api.artifact(artifact_ref)
    except Exception as exc:
        raise RuntimeError(
            "Unable to load artifact '{}'. Make sure the requested iteration was saved by distill.py.".format(
                artifact_ref
            )
        ) from exc

    artifact_dir = Path(artifact.download(root=str(download_root)))
    metadata = dict(getattr(artifact, "metadata", {}) or {})
    selected_iteration = requested_iteration if requested_iteration is not None else metadata.get("iteration")

    image_candidates = []
    label_candidates = []
    sensor_candidates = []

    if selected_iteration is not None:
        image_candidates.append("images_{}.pt".format(int(selected_iteration)))
        label_candidates.append("labels_{}.pt".format(int(selected_iteration)))
        sensor_candidates.append("sensor_{}.pt".format(int(selected_iteration)))

    image_candidates.append("images_best.pt")
    label_candidates.append("labels_best.pt")
    sensor_candidates.append("sensor_best.pt")

    image_path = resolve_snapshot_file(artifact_dir, image_candidates, required=True)
    label_path = resolve_snapshot_file(artifact_dir, label_candidates, required=True)
    sensor_path = resolve_snapshot_file(artifact_dir, sensor_candidates, required=False)

    images = torch.load(image_path, map_location="cpu").float()
    labels = torch.load(label_path, map_location="cpu").long()
    sensors = None if sensor_path is None else torch.load(sensor_path, map_location="cpu").float()

    if selected_iteration is None:
        selected_iteration = metadata.get("iteration")

    return {
        "artifact_ref": artifact_ref,
        "iteration": None if selected_iteration is None else int(selected_iteration),
        "images": images,
        "labels": labels,
        "sensors": sensors,
    }


def build_model_kwargs(args):
    return {
        "unimodal": getattr(args, "unimodal", ""),
        "n_groups": getattr(args, "n_groups", 8),
        "n_sensors": getattr(args, "n_sensors", None),
        "n_sensor_features": getattr(args, "n_sensor_features", None),
    }


def build_synthetic_dataset(images, labels, sensors=None):
    if sensors is None:
        return TensorDataset(images, labels)
    return MultimodalTensorDataset(images, sensors, labels)


def main():
    cli_args = parse_args()
    device = cli_args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but no CUDA device is available.")

    api = wandb.Api()
    run, run_path = resolve_run(api, cli_args.run_id, entity=cli_args.entity, project=cli_args.project)

    distill_args = build_run_args(run.config)
    if cli_args.data_path is not None:
        distill_args.data_path = cli_args.data_path

    distill_args.device = device
    distill_args.seed = distill_args.seed if cli_args.seed is None else int(cli_args.seed)
    fix_seed(distill_args.seed)

    download_root = Path(cli_args.download_dir) / run.id
    download_root.mkdir(parents=True, exist_ok=True)

    snapshot = load_distillation_snapshot(api, run_path, run, cli_args.iteration, download_root)
    distill_iteration = snapshot["iteration"]

    train_epochs = distill_args.epoch_eval_train if cli_args.train_epochs is None else int(cli_args.train_epochs)
    train_lr = cli_args.train_lr
    if train_lr is None:
        train_lr = recover_synthetic_lr(run, distill_iteration, distill_args.lr_teacher)

    distill_args.epoch_eval_train = int(train_epochs)
    distill_args.lr_net = float(train_lr)
    distill_args.batch_train = distill_args.batch_train if cli_args.batch_train is None else int(cli_args.batch_train)
    if distill_args.epoch_eval_train < 1:
        raise ValueError("epoch_eval_train must be at least 1.")
    if distill_args.batch_train < 1:
        raise ValueError("batch_train must be at least 1.")

    channel, im_size, num_classes, _, _, _, _, dst_test, testloader, _, _, _ = get_dataset(
        distill_args.dataset,
        distill_args.data_path,
        distill_args.batch_real,
        distill_args.subset,
        args=distill_args,
    )
    distill_args.im_size = im_size

    if cli_args.batch_test is not None:
        if int(cli_args.batch_test) < 1:
            raise ValueError("batch_test must be at least 1.")
        testloader = DataLoader(dst_test, batch_size=int(cli_args.batch_test), shuffle=False, num_workers=0)

    multimodal = distill_args.unimodal != "model"
    if multimodal and snapshot["sensors"] is None:
        raise ValueError("The selected artifact does not contain synthetic sensor tensors, but the run expects multimodal inputs.")

    synthetic_dataset = build_synthetic_dataset(snapshot["images"], snapshot["labels"], snapshot["sensors"])
    if len(synthetic_dataset) == 0:
        raise ValueError("Loaded synthetic dataset is empty.")

    trainloader = DataLoader(synthetic_dataset, batch_size=distill_args.batch_train, shuffle=True,num_workers=0)
    test_batch_size = getattr(testloader, "batch_size", None)

    model = get_network(distill_args.model, channel, num_classes, im_size, dist=False, init_seed=distill_args.seed, **build_model_kwargs(distill_args),).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=float(train_lr), momentum=float(cli_args.momentum), weight_decay=float(cli_args.weight_decay))
    criterion = nn.CrossEntropyLoss().to(device)
    lr_schedule = {int(train_epochs) // 2 + 1}
    current_lr = float(train_lr)

    snapshot_label = "best_so_far" if cli_args.iteration is None else "iter_{}".format(cli_args.iteration)
    print(f"Loaded distill run {run.name or run.id} ({run.id}) | dataset={distill_args.dataset} model={distill_args.model}")
    print(f"Using snapshot {snapshot_label} | resolved distill iteration={distill_iteration} | eval lr={current_lr:.6f} | epochs={train_epochs}")

    best_test_acc = float("-inf")
    best_epoch = -1

    wandb_run_name = "{}-{}".format(run.name or run.id, snapshot_label)
    wandb_config = prepare_wandb_config(
        {
            "source_run_id": run.id,
            "source_run_name": run.name,
            "source_run_path": run_path,
            "source_project": cli_args.project,
            "requested_iteration": cli_args.iteration,
            "resolved_iteration": distill_iteration,
            "dataset": distill_args.dataset,
            "subset": distill_args.subset,
            "model": distill_args.model,
            "batch_train": distill_args.batch_train,
            "batch_test": test_batch_size,
            "train_epochs": int(train_epochs),
            "train_lr": float(train_lr),
            "momentum": float(cli_args.momentum),
            "weight_decay": float(cli_args.weight_decay),
            "seed": int(distill_args.seed),
            "device": device,
            "data_path": distill_args.data_path,
            "download_dir": download_root,
        }
    )

    wandb_kwargs = {
        "config": wandb_config,
        "mode": os.environ.get("WANDB_MODE", "online"),
        "name": wandb_run_name,
        "project": WANDB_PROJECT,
        "reinit": True,
    }

    with wandb.init(**wandb_kwargs) as wandb_run:
        for ep in range(1, int(train_epochs) + 1):
            train_loss, train_acc = epoch("train", trainloader, model, optimizer, criterion, distill_args, aug=True, texture=distill_args.texture)

            with torch.no_grad():
                test_loss, test_acc = epoch("test", testloader, model, optimizer, criterion, distill_args, aug=False,texture=False)

            train_loss_value = float(train_loss)
            train_acc_value = float(train_acc)
            test_loss_value = float(test_loss)
            test_acc_value = float(test_acc)
            current_lr = float(optimizer.param_groups[0]["lr"])
            metric_payload = {
                "lr": current_lr,
                "test_acc": test_acc_value,
                "test_loss": test_loss_value,
                "train_acc": train_acc_value,
                "train_loss": train_loss_value,
            }
            wandb.log(metric_payload, step=ep)

            if test_acc_value > best_test_acc:
                best_test_acc = test_acc_value
                best_epoch = ep

            print(f"Epoch {ep:04d}/{int(train_epochs):04d} | train loss = {train_loss_value:.6f} acc = {train_acc_value:.4f} | test loss = {test_loss_value:.6f} acc = {test_acc_value:.4f}")

            if ep in lr_schedule and ep < int(train_epochs):
                current_lr *= 0.1
                optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=float(cli_args.momentum), weight_decay=float(cli_args.weight_decay))

        wandb_run.summary["best_test_acc"] = float(best_test_acc)
        wandb_run.summary["best_epoch"] = int(best_epoch)

    print(f"Finished | best test acc = {best_test_acc:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
