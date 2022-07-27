"""File ported from pest-monitoring"""

import io
import json
import logging
import os
import time

import numpy as np
import torch
from PIL import Image

from src.data import transforms


class CFSSDHandler(object):
    """Handler class for creating MAR file."""

    def __init__(self):
        """Setup handler information."""
        self.model = None
        self.device = None
        self.input_size = None
        self.image = None
        self.initialized = False

    def initialize(self, context):
        """Initialize the handler information."""

        properties = context.system_properties
        self.manifest = context.manifest

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu"
        )
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_path = os.path.join(model_dir, serialized_file)

        self.model = torch.jit.load(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        logging.info("Model loaded from {}".format(model_path))

        metadata_file = os.path.join(model_dir, "metadata.json")
        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)

        self.input_size = int(self.metadata["input_size"])
        self.trap_label = self.metadata["trap_label"]

        self.initialized = True

    def preprocess(self, data):
        """Pre-process the input data"""

        transform = transforms.Compose(
            [
                transforms.ConvertFromInts(),
                transforms.Resize(self.input_size, self.input_size),
                transforms.ToTensor(),
            ]
        )

        images = []
        for row in data:
            image = row.get("data") or row.get("body")
            image = Image.open(io.BytesIO(image))
            image = np.asarray(image)
            image = transform(image)[0]
            images.append(image)

        return torch.stack(images, 0)

    def inference(self, data):
        """Perform forward on the JIT model."""
        with torch.no_grad():
            return self.model.forward(data.to(self.device))

    def postprocess(self, data):
        """Post-process the output of the JIT Model."""
        return {
            "trap": self.trap_label[str(torch.argmax(data[0]).item())],
            "num_abw": data[1].size(0),
            "num_pbw": data[2].size(0),
            "trap_conf": str(data[0].tolist()),
            "abw_bbox": str(data[1].tolist()),
            "pbw_bbox": str(data[2].tolist()),
            "abw_bbox_conf": str(data[3].tolist()),
            "pbw_bbox_conf": str(data[4].tolist()),
        }

    def handle(self, data, context):
        """Handler Pipeline."""
        if data is None:
            return None

        tick = time.time()
        if not self.initialized:
            self.initialize(context)
        init_time = time.time() - tick

        tick = time.time()
        data = self.preprocess(data)
        preprocess_time = time.time() - tick

        tick = time.time()
        data = self.inference(data)
        inference_time = time.time() - tick

        tick = time.time()
        data = self.postprocess(data)
        postprocess_time = time.time() - tick

        time_log = {
            "initialize_time": init_time,
            "preprocess_time": preprocess_time,
            "inference_time": inference_time,
            "postprocess_time": postprocess_time,
            "response_time": init_time + preprocess_time + inference_time + postprocess_time,
        }

        return [{**data, **time_log}]


service = CFSSDHandler()


def handle(data, context):
    return service.handle(data, context)
