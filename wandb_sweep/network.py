"""Script for running wandb sweep for object detection over hyperparameters
- nms_threshold
- conf_threshold

To run the script, run the following command:
python wandb_sweep/network.py -cn config_path +ckpt=checkpoint_path +n=100

Arguments
---------
-cn (str): Path of the experiment config used for training or evaluation (make sure to include
wandb sweep config in this config file which is at "configs/wandb_sweep/". As an example, look
at config files in "configs/defaults/ssd/")
+ckpt (str): Path of the checkpoint to use
+n (int): Limit on number of evaluation runs in the sweep
"""
from collections import defaultdict
from functools import partial

import hydra
import numpy as np
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf as oc
from tqdm import tqdm

from src.models.utils import detach
from src.utils import utils
from utils.helper import mae, mae_alpha

OUT_DIR = "/output/"

log = utils.get_logger(__name__)


def evaluate(expt_config, model, dataloader, device, config=None):
    wandb.init(
        name=f"eval-sweep-{expt_config.name}",
        config=expt_config,
        project=expt_config.wandb_sweep.project,
        settings=wandb.Settings(console="off"),
        dir=None,
        notes="Wandb: Hyperparameter Sweep",
    )
    config = wandb.config

    # Set hyperparameter value to model.network
    for hyperparameter, value in config.items():
        setattr(model.network, hyperparameter, value)

    # store info
    gt_boxes = []
    gt_image_sizes = []
    gt_image_ids = []

    outputs = []
    for _, batch in enumerate(tqdm(dataloader)):
        img = batch["img"].to(device)

        # store actual size
        gt_image_ids.extend(batch["img_id"])
        gt_boxes.extend(batch["bbox_class"])
        gt_image_sizes.extend(batch["img_size"])

        # store outputs
        outputs.extend(detach(model.predict(img)))

    metrics_dict = defaultdict(list)

    for i, _ in tqdm(enumerate(gt_image_ids)):
        abw_gt = len([x for x in gt_boxes[i] if x == 0.0]) if gt_boxes[i] is not None else 0
        pbw_gt = len([x for x in gt_boxes[i] if x == 1.0]) if gt_boxes[i] is not None else 0

        # predictions for all images
        abw_pd = len([x for x in outputs[i]["labels"] if x == 1])
        pbw_pd = len([x for x in outputs[i]["labels"] if x == 2])

        # compute metrics
        metrics_dict["MAE-Alpha/ABW"].append(mae_alpha(abw_gt, abw_pd))
        metrics_dict["MAE-Alpha/PBW"].append(mae_alpha(pbw_gt, pbw_pd))
        metrics_dict["MAE/ABW"].append(mae(abw_gt, abw_pd))
        metrics_dict["MAE/PBW"].append(mae(pbw_gt, pbw_pd))

    for k, v in metrics_dict.items():
        metrics_dict[k] = np.nanmean(v)

    wandb.log({"MAE-Alpha/PBW": metrics_dict["MAE-Alpha/PBW"]})
    for k, v in metrics_dict.items():
        if k != "MAE-Alpha/PBW":
            wandb.log({k: metrics_dict[k]})


@hydra.main(config_path="../configs/")
def main(config: DictConfig):
    """Main function"""
    # get sweep config
    assert "wandb_sweep" in config, "Add sweep config in the config file."
    sweep_config = oc.to_container(config.wandb_sweep)

    # initialize model
    log.info(f"Instantiating model <{config.model._target_}>")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = (
        instantiate(config.model, model_config=config["model"], _recursive_=False)
        .load_from_checkpoint(config.ckpt, model_config=config["model"])
        .to(device)
    )

    # check if hyperparameter attributes exist for model.network
    for hyperparameter, _ in sweep_config["parameters"].items():
        log.info(f"Checking if {hyperparameter} exists in model.network")
        if not hasattr(model.network, hyperparameter):
            raise AttributeError(f"Hyperparameter {hyperparameter} does not exist in model.network")

    # load datamodule
    datamodule = instantiate(
        config.datamodule,
        data_config=config.datamodule,
        batch_size=64,
        _recursive_=False,
    )
    datamodule.setup()

    # get val dataloader
    dataloader = datamodule.val_dataloader()

    partial_evaluate = partial(
        evaluate, expt_config=config, model=model, dataloader=dataloader, device=device
    )
    sweep_id = wandb.sweep(sweep_config, project=config.wandb_sweep.project)
    wandb.agent(sweep_id, partial_evaluate, count=config.n)


if __name__ == "__main__":
    main()
