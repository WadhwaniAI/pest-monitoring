import argparse
import json
import os
from shutil import rmtree

import torch
from combined_model import CombinedModel


def main():
    """Run the deployment packaging pipeline."""
    # Get args
    parser = argparse.ArgumentParser(
        description="Package counting and validation jit models into a single jit model"
    )
    parser.add_argument(
        "-vj", "--validation_jit", type=str, required=True, help="validation model jit file path"
    )
    parser.add_argument(
        "-cj", "--counting_jit", type=str, required=True, help="counting model jit file path"
    )
    parser.add_argument("-nt", "--nms_th", type=float, default=0.4, help="NMS Threshold")
    parser.add_argument("-mn", "--max_num", type=int, default=200, help="Max boxes allowed")
    parser.add_argument("-m", "--metadata", type=str, required=True, help="metadata file")
    parser.add_argument("-bd", "--boxdata", type=str, required=True, help="default box data file")
    parser.add_argument(
        "-o", "--out", type=str, required=True, help="output directory to save package"
    )
    parser.add_argument(
        "-ow",
        "--overwrite",
        action="store_true",
        help="overwrite the existing version in out directory",
    )
    parser.add_argument(
        "-oj",
        "--only_jit",
        action="store_true",
        help="save only combined jit otherwise the whole zip package will be saved",
    )
    # Parse args
    args = parser.parse_args()
    # Parse metadata
    md = json.load(open(args.metadata))
    # Parse boxdata
    bd = json.load(open(args.boxdata))
    # Log input information
    print("Parameters\n----------")
    print("args")
    for arg in vars(args):
        print("\t", arg, ":", getattr(args, arg))
    print("metadata")
    for k, v in md.items():
        print("\t", k, ":", v)
    print("boxdata")
    for k, v in md.items():
        print("\t", k, ":", v)
    print("__________\n")

    # Initialize Model
    print("Initializing Model ...", end=" ")
    combined_model = CombinedModel(
        args.validation_jit,
        args.counting_jit,
        args.nms_th,
        args.max_num,
        md["input_size"],
        md["threshold"],
        bd,
    )
    print("done")

    # Trace Model
    print("Tracing Model ...", end=" ")
    x = torch.randn((1, 3, md["input_size"], md["input_size"]))
    with torch.no_grad():
        traced_model = torch.jit.trace(combined_model, x)
    print("done")

    # Prepare Output Directory Structure
    print("Preparing Output Directory ...", end=" ")
    model_dirname = f"v_{md['version']}"
    save_dir = os.path.join(args.out, model_dirname)
    if os.path.exists(save_dir):
        if args.overwrite:
            rmtree(save_dir)
    try:
        os.mkdir(save_dir)
    except Exception as e:
        print(f"Could not create output directory {save_dir}")
        raise e
    print("done")

    # Save jit model
    print("Saving JIT Model ...", end=" ")
    jit_out_path = os.path.join(save_dir, "model.pt")
    traced_model.save(jit_out_path)
    print("done")

    # If not only jit
    if not args.only_jit:
        # Save metadata
        print("Saving metadata ...", end=" ")
        md_out_path = os.path.join(save_dir, "metadata.json")
        with open(os.path.join(md_out_path), "w+") as f:
            json.dump(md, f)
        print("done")
        # Create zip package
        print("creating ZIP package ...")
        cmd = f"cd {args.out} && zip -r {model_dirname}/{model_dirname}.zip {model_dirname}"
        _ = os.system(cmd)
        print("done")
        # Save mar file
        print("Saving MAR Model ...", end=" ")
        cmd = f"""\
        torch-model-archiver \
        --model-name cfssd_{md["version"]} \
        --version {md["version"]} \
        --serialized-file {jit_out_path} \
        --handler /workspace/pest-monitoring-new/deployment/handler.py \
        --export-path {save_dir} \
        --extra-files {md_out_path}
        """
        _ = os.system(cmd)
        print("done")


if __name__ == "__main__":
    main()
