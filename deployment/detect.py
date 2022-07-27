"""Code used for Pest Management Model Deployment using CFSSD
------------------------ TODO: HardCoded for our codebase ------------------------

Currently, predictions are a list of predictions per image. This function uses the
predict() function to get the predictions and returns them in a different format.
Desired format is a list of tuples where each tuple is,
(validation_out, abw_boxes, pbw_boxes, abw_scores, pbw_scores)

Each tuple contains the following:
1. validation_out: 2 length tensor, where the values are
(class 1 confidence, class 0 confidence)
2. abw_boxes: Nx4 torch tensor of abw box coordinates,
where the columns are - top_left_x, top_left_y, bottom_right_x, bottom_right_y
3. pbw_boxes: Nx4 torch tensor of pbw box coordinates,
where the columns are - top_left_x, top_left_y, bottom_right_x, bottom_right_y
3. abw_scores: Nx1 torch tensor of abw box scores
5. pbw_scores: Nx1 torch tensor of pbw box scores

Basically, use this code to generate the final jit version to be handed off to
engineering.
```python
import torch
import torchvision
from deployment.detect import DetectModel

model = DetectModel(
    model = torch.jit.load(<path to cfssd model>)
)
m = torch.jit.script(model)
m.save(<path to save cfssd model>)
```

"""
from typing import List, Tuple

import torch
from torch import Tensor


class DetectModel(torch.nn.Module):
    def __init__(self, model):
        super(DetectModel, self).__init__()
        self.model = model
        # Check if model has .detect() method
        if not hasattr(self.model, "predict"):
            raise AttributeError("Model does not have .predict() method. Please Implement it.")

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        predictions = self.model.predict(x)
        image_detection = predictions[0]

        # Flip scores
        val_out = torch.flip(image_detection["validation_scores"], [0])

        abw_indices: List[int] = []
        pbw_indices: List[int] = []
        labels: List[int] = image_detection["labels"].tolist()
        # TODO: Currently Hardcoded
        for i, val in enumerate(labels):
            if val == 2:
                pbw_indices.append(i)
            elif val == 1:
                abw_indices.append(i)
            else:
                raise ValueError(f"Invalid label {val}")

        abw_boxes: Tensor = image_detection["boxes"][abw_indices]
        pbw_boxes: Tensor = image_detection["boxes"][pbw_indices]

        # Normalize by image size
        abw_boxes = abw_boxes / x.shape[-2]
        pbw_boxes = pbw_boxes / x.shape[-1]

        abw_scores: Tensor = image_detection["scores"][abw_indices]
        pbw_scores: Tensor = image_detection["scores"][pbw_indices]

        return (val_out, abw_boxes, pbw_boxes, abw_scores, pbw_scores)
