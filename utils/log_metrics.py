"""This script is responsible for logging metrics given a json file.

This code is very specific to the Pest Monitoring dataset and requires a Weights and Biases
account to be created.
TODO: Current Work in Progress to make it generic

Things that are included in the table,
- COCO metrics
- MAE and MAE-Alpha Numbers
- Validation Metrics
- Condessa Metrics

Usage:
    python log_metrics.py \
        --file_path=<file_path> \
        --metrics val mae coco condessa \
        --split <split> \

    Parameters
    ---------
    file_path: str
        Path to the json file to be logged
    metrics:
        metrics to log, default: [val, coco, mae, condessa]
    split: str
        Split to log, default: val
"""

import argparse
import json
from os.path import basename
from pathlib import Path

import numpy as np
import wandb

# Import all helper functions
# TODO: get_coco_metrics_class calculates on a fixed nms th of 0.5,
# in future, if required, parameterize
from helper import (
    get_coco_metrics,
    get_coco_metrics_class,
    get_condessa_capped_regression_metrics,
    get_mae_metrics,
    get_mae_metrics_system,
    get_validation_model_metrics,
)

from src.utils import utils

log = utils.get_logger(__name__)

all_metrics = {
    "mae": get_mae_metrics,
    "mae-sys": get_mae_metrics_system,
    "coco": get_coco_metrics,
    "coco-class": get_coco_metrics_class,  # get class specific coco metrics, on 0.5 nms iou
    "val": get_validation_model_metrics,
    "condessa": get_condessa_capped_regression_metrics,
}

if __name__ == "__main__":
    # Setup Argparse
    parser = argparse.ArgumentParser(description="Runs the script on the split specified")
    parser.add_argument(
        "-f",
        "--file_path",
        type=str,
        help="Path to the json file containing the outputs",
    )
    parser.add_argument(
        "-m",
        "--metrics",
        nargs="*",
        help="Metrics to log",
        default=["val", "coco", "mae-sys", "condessa"],
    )
    parser.add_argument("-s", "--split", type=str, default="val", help="Split to be used")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "-d",
        "--dump_metrics_json",
        action="store_true",
        help="Dump metrics to json file",
    )
    parser.add_argument(
        "--metric_json_loc",
        type=str,
        default="",
        help=(
            "Location to dump the metrics json file. Defaults to the location of the input"
            "json file, with name metrics.json"
        ),
    )
    parser.add_argument(
        "--wandb_dir",
        type=str,
        default="/output/",
        help="Path to the wandb directory",
    )
    args = parser.parse_args()

    # if the location of metric json is not specified, use the location of the input json file
    if args.metric_json_loc == "":
        args.metric_json_loc = Path(Path(args.file_path).parent, "metrics.json")

    # Load the json file
    output = json.load(open(args.file_path, "r"))
    split = args.split

    # log info on split
    log.info(f"Running metrics for {split}")

    # init wandb
    if not args.disable_wandb:
        run_name = basename(args.file_path).split(".json")[0]
        wandb.init(
            name=f"Metrics on {run_name}",
            project="pest-monitoring-new",
            dir=args.wandb_dir,
            notes=f"Output of the {run_name} run on split {split}",
        )
    else:
        log.info("Wandb logging disabled.")

    # choose the metrics
    metrics_to_be_logged = []
    for metric in args.metrics:
        metrics_to_be_logged.append(all_metrics[metric])

    # creating a metric dump in case we wanna dump into a json
    metric_json = {}

    # helper function
    def to_list(data: dict):
        """a function to convert all numpy arrays (np.ndarray) into lists recursively in a dict.

        Parameters:
            data (dict): the dictionary to search and convert any numpy list to lists.

        Returns:
            converted_data (dict)"""
        converted_data = {}
        # data can be dict
        if isinstance(data, dict):
            if len(data) == 0:
                return {}
            else:  # if dict is filled, recurse into
                for key, value in data.items():
                    converted_data[key] = to_list(value)
                return converted_data
        elif isinstance(data, (np.ndarray, np.generic)):  # or a numpy thing
            return data.tolist()  # convert to list
        else:  # or anything else
            return data

    # Run the functions
    for func in metrics_to_be_logged:
        log.info(f"Running {func.__name__}")
        try:
            metrics_dict = func(output, split)
            for k, v in metrics_dict.items():
                if not args.disable_wandb:
                    wandb.log({k: v})
                metric_json[k] = to_list(v)
        except Exception as e:
            log.info(f"{func.__name__} failed with {e}")
            log.info("Skipping this function")

    # if asked to dump metric json, dumping
    if args.dump_metrics_json:
        log.info(f"Dumping metrics to {args.metric_json_loc}")
        with open(args.metric_json_loc, "w") as fp:
            json.dump(metric_json, fp)
