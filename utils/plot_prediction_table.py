"""This script is responsible for plotting an image-wise prediction on Weights and Biases.

TODO: This code is very specific to the Pest Monitoring dataset and requires a Weights and Biases
account to be created. Current Work in Progress to make it generic

Things that are included in the table,
- Counts on ABW and PBW pests
- MAE and MAE-Alpha Numbers
- Images are plotted
- Ground Truth and Predicted Bounding Boxes are plotted

Usage:
    python plot_prediction_table.py --base_path=<base_path> --file_name=<file_name>

    Arguments
    ---------
    base_path (Default: '/output/jsons/'):
    Path to the base directory
    file_name: Name of the merged json file containing the outputs
    image_size: Size of the image to be logged
"""

import argparse
import json
from os.path import basename

import wandb
from helper import get_prediction_df

if __name__ == "__main__":
    # Setup Argparse
    parser = argparse.ArgumentParser(description="Runs the script on the split specified")
    parser.add_argument(
        "-fn",
        "--file_path",
        type=str,
        default="/output/jsons/cfssd-resnet18-base-val-001.00.000.json",
        help="Name of the merged json file to be used",
    )
    parser.add_argument("-s", "--split", type=str, default="val", help="Split to be used")
    parser.add_argument(
        "--wandb_dir",
        type=str,
        default="/output/",
        help="Path to the wandb directory",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Size of the images to be used",
    )
    args = parser.parse_args()

    # Load the json file
    output = json.load(open(args.file_path))
    split = args.split

    # Get the Dataframe
    image_size = (args.image_size, args.image_size)
    df = get_prediction_df(output, split, image_size)

    # Initialize wandb
    run_name = basename(args.file_path).split(".json")[0]
    run = wandb.init(
        name=f"Image-wise Prediction on {run_name}",
        project="pest-monitoring-new",
        dir=args.wandb_dir,
        notes=f"Output of the {run_name} run on split {split}",
    )

    # Log the data frame on wandb
    table = wandb.Table(dataframe=df, columns=df.columns.tolist())
    run.log({f"Prediction on {split} split for {run_name}": table})
