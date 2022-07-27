# Description:
# Script downloads sample coco image data required by data/jsons/experiments/ssd/sample.json
# Usage:
# python prepare_sample_coco_data.py [path to input json]
# Example:
# >>> python prepare_sample_coco_data.py jsons/experiments/ssd/sample.json

import json
import os
import sys

import wget
from tqdm import tqdm


def download_coco(json_path):
    images = json.load(open(json_path))["images"]
    for image in tqdm(images, desc="Downloading..."):
        if os.path.exists(image["file_path"]):
            continue
        wget.download(image["s3_url"], image["file_path"])


def main():
    if len(sys.argv) <= 1:
        json_path = "/workspace/pest-monitoring-new/data/jsons/experiments/ssd/sample.json"
    else:
        json_path = sys.argv[1]
    download_coco(json_path)


if __name__ == "__main__":
    main()
