import os
import argparse
from pathlib import Path
from types import SimpleNamespace
from glob import glob

import torch
import wandb

from utils import evaluate_synset, fix_seed, get_dataset, get_network


def main(args):
    api = wandb.Api()

    wandb_prefix = getattr(api, "default_entity") + '/DatasetDistillation'
    run = api.run(f'{wandb_prefix}/{args.run_id}')

    metric = 'Accuracy/'
    rows = []
    for row in run.scan_history(keys=[metric, '_step']):
        if metric in row and row[metric] is not None:
            rows.append(row)
    best_run = sorted(rows, key=lambda x: x[metric], reverse=True)[0]['_step']
    iteration_used = cli_args.iteration if cli_args.iteration else best_run

    lr = list(run.scan_history(['Synthetic_LR'], min_step=iteration_used, max_step=iteration_used+1))[0]['Synthetic_LR']

    source_config = {str(key): value for key, value in dict(run.config).items() if not str(key).startswith("_")}

    run_args = SimpleNamespace(**source_config)
    run_args.run_id = cli_args.run_id
    run_args.device = cli_args.device
    run_args.lr_net_syn = lr

    artifact_collection = f'distillation-eval-{args.run_id}'
    artifact_path = str(Path("logged_files") / "traces" / run.id / str(iteration_used))
    api.artifact(f'{wandb_prefix}/DatasetDistillation/{artifact_collection}:iter_{iteration_used}').download(root=artifact_path)

    image_fn, sensor_fn = None, None
    for fn in glob(artifact_path + "/*"):
        fn = os.path.basename(fn)
        if fn.startswith("labels"):
            label_fn = fn
        elif fn.startswith("images"):
            image_fn = fn
        elif fn.startswith("sensor"):
            sensor_fn = fn

    images_train, sensors_train = None, None
    labels_train = torch.load(artifact_path + "/" + label_fn, map_location="cpu").long()
    if image_fn:
        images_train = torch.load(artifact_path + "/" + image_fn, map_location="cpu").float()
    if sensor_fn:
        sensors_train = torch.load(artifact_path + "/" + sensor_fn, map_location="cpu").float()

    fix_seed(run_args.seed)
    
    channel, im_size, num_classes, _, _, _, _, _, testloader, _ = get_dataset(run_args)
    run_args.im_size = im_size

    net = get_network(
        run_args.model,
        channel,
        num_classes,
        im_size,
        dist=False,
        unimodal=getattr(run_args, "unimodal", ""),
        n_groups=getattr(run_args, "n_groups", 8),
        n_sensors=getattr(run_args, "n_sensors", None),
        n_sensor_features=getattr(run_args, "n_sensor_features", None),
    )

    wandb_config = dict(source_config)
    wandb_config.update({
        "device": args.device,
        "lr_net_syn": float(run_args.lr_net_syn),
        "iteration": args.iteration,
        "source_run_id": run.id,
        "source_run_name": run.name,
        "synthetic_images": int(labels_train.shape[0]),
    })

    trace_name = f"{run.name}-iter_{iteration_used}"
    with wandb.init(project="SyntheticTraining", name=trace_name, config=wandb_config, reinit=True) as wandb_run:
        _, acc_train_list, acc_test = evaluate_synset(
            0,
            net,
            images_train,
            labels_train,
            testloader,
            run_args,
            sensor_train=sensors_train,
            training_logs=True,
        )
        wandb_run.summary["final_train_acc"] = float(acc_train_list[-1])
        wandb_run.summary["final_test_acc"] = float(acc_test)


if __name__ == "__main__":
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="Train the evaluation model on a distilled snapshot from a W&B distillation run.")
    parser.add_argument("run_id", type=str, help="W&B distillation run id. You can also pass entity/project/run_id.")
    parser.add_argument("--iteration",type=int, default=None, help="Specific distillation iteration to load. Defaults to the best saved snapshot.")
    parser.add_argument("--device", type=str, default=default_device, choices=["cpu", "cuda"], help="Device to use. Defaults to cuda when available.")
    cli_args = parser.parse_args()

    main(cli_args)
