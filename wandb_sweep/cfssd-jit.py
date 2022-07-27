"""Script for running wandb sweep for CFSSD over hyperparameters
- nms_threshold
- conf_threshold
"""

import os
from collections import defaultdict

import numpy as np
import torch
import torchvision  # needed to load jit models (with nms)
import wandb
from hydra import compose, initialize
from hydra.utils import instantiate
from tqdm import tqdm

from src.utils import utils
from utils.helper import mae, mae_alpha

# Helper functions


OUT_DIR = "/output/"

log = utils.get_logger(__name__)


def evaluate(config=None):
    config_name = "experiments/summer-deployment/cfvgg16/base.yaml"

    # read config file
    with initialize(config_path="../configs/", job_name="codebase run"):
        hydra_config = compose(config_name=config_name, overrides=[])

    wandb.init(
        name=config_name,
        config=config,
        project="pest-monitoring-new",
        settings=wandb.Settings(console="off"),
        dir=None,
        notes="Wandb: Hyperparameter Sweep",
    )
    config = wandb.config

    # instantiate and load model with checkpoint
    jit_checkpoint_path = (
        "/output/compress/summer-deployment/cfvgg19/basev3.yaml/",
        "prune/epoch=382.ckpt/jit-checkpoint.pt",
    )
    assert os.path.exists(jit_checkpoint_path), "Checkpoint not found"
    log.info(f"Instantiating JIT model <{hydra_config.model._target_}>")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(jit_checkpoint_path).to(device)

    # Make sure jit has @torch.jit.export over conf_threshold and nms_threshold
    model.conf_threshold = config.conf_threshold
    model.nms_threshold = config.nms_threshold

    # load datamodule
    datamodule = instantiate(
        hydra_config.datamodule,
        data_config=hydra_config.datamodule,
        batch_size=64,
        _recursive_=False,
    )
    datamodule.setup()

    # get val dataloader
    dataloader = datamodule.val_dataloader()

    # store info
    gt_boxes = []
    gt_image_labels = []
    gt_image_sizes = []
    gt_image_ids = []

    outputs = []
    for _, batch in enumerate(tqdm(dataloader)):
        img = batch["img"].to(device)

        # store actual size
        gt_image_ids.extend(batch["img_id"])
        gt_boxes.extend(batch["bbox_class"])
        gt_image_labels.extend(batch["label_value"].tolist())
        gt_image_sizes.extend(batch["img_size"])

        # store outputs
        outputs.extend(model.predict(img))

    metrics_dict = defaultdict(list)

    for i, _ in tqdm(enumerate(gt_image_ids)):
        abw_gt = len([x for x in gt_boxes[i] if x == 0.0]) if gt_boxes[i] is not None else 0
        pbw_gt = len([x for x in gt_boxes[i] if x == 1.0]) if gt_boxes[i] is not None else 0

        # predictions for all images
        abw_pd = len([x for x in outputs[i]["labels"] if x == 1])
        pbw_pd = len([x for x in outputs[i]["labels"] if x == 2])

        # image label
        img_label = int(gt_image_labels[i])
        pred_label = int(outputs[i]["validation_scores"].detach().tolist()[1] > 0.5)

        metrics_dict["validation_acc"].append(img_label == pred_label)

        # compute metrics
        metrics_dict["MAE-Alpha/ABW"].append(mae_alpha(abw_gt, abw_pd))
        metrics_dict["MAE-Alpha/PBW"].append(mae_alpha(pbw_gt, pbw_pd))
        metrics_dict["MAE/ABW"].append(mae(abw_gt, abw_pd))
        metrics_dict["MAE/PBW"].append(mae(pbw_gt, pbw_pd))

        # Ground Truth Trap only
        if img_label == 1:
            metrics_dict["MAE-Alpha/ABW (Ground Truth Trap Only)"].append(mae_alpha(abw_gt, abw_pd))
            metrics_dict["MAE-Alpha/PBW (Ground Truth Trap Only)"].append(mae_alpha(pbw_gt, pbw_pd))
            metrics_dict["MAE/ABW (Ground Truth Trap Only)"].append(mae(abw_gt, abw_pd))
            metrics_dict["MAE/PBW (Ground Truth Trap Only)"].append(mae(pbw_gt, pbw_pd))

        # Predicted trap only
        if pred_label == 1:
            metrics_dict["MAE-Alpha/ABW (Predicted Trap Only)"].append(mae_alpha(abw_gt, abw_pd))
            metrics_dict["MAE-Alpha/PBW (Predicted Trap Only)"].append(mae_alpha(pbw_gt, pbw_pd))
            metrics_dict["MAE/ABW (Predicted Trap Only)"].append(mae(abw_gt, abw_pd))
            metrics_dict["MAE/PBW (Predicted Trap Only)"].append(mae(pbw_gt, pbw_pd))

    for k, v in metrics_dict.items():
        metrics_dict[k] = np.nanmean(v)

    wandb.log({"MAE-Alpha/PBW": metrics_dict["MAE-Alpha/PBW"]})
    for k, v in metrics_dict.items():
        if k != "MAE-Alpha/PBW":
            wandb.log({k: metrics_dict[k]})


def main():
    """Main function"""
    log.info("Instantiating Wandb Logger")
    sweep_config = {
        "name": "HyperOpt for CFSSD (Jit Checkpoint)",
        "method": "bayes",
        "metric": {"name": "MAE-Alpha/PBW", "goal": "minimize"},
        "parameters": {
            "nms_threshold": {"distribution": "q_uniform", "min": 0.2, "max": 0.5, "q": 0.01},
            "conf_threshold": {"distribution": "q_uniform", "min": 0.2, "max": 0.5, "q": 0.01},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="pest-monitoring-new")
    wandb.agent(sweep_id, evaluate)


if __name__ == "__main__":
    main()
