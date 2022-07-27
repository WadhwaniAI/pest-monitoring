from collections import defaultdict
from typing import Dict, Optional, Union

import torch


def dict_collate(batch: Dict[str, Optional[Union[str, torch.Tensor]]]) -> Dict:
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Parameters
    ----------
    batch : Dict[str, Optional[Union[str, torch.Tensor]]]
        A single record's Dict with image and associated ground truth
        information after applying transforms. Contains "img_id", "img",
        "bbox_class", "bbox_coord", "label_class" and
        "label_value" keys.

    Returns
    -------
    Dict
        Dict of the whole batch with image and associated ground truth information
        after applying transforms. Every key contains the information of the whole
        batch. Contains "img_ids", "img", "bbox_class", "bbox_coord", "label_class"
        and "label_value" keys.
    """

    records = defaultdict(list)

    for record in batch:
        for key, val in record.items():
            records[key].append(val)

    # special stacking for vector type stuff.
    keys = [
        "img",
        "label_class_cat",
        "label_value_cat",
        "label_class_reg",
        "label_value_reg",
        "label_class",
        "label_value",
    ]
    for key in keys:
        if key in records:
            # if an attribute of frist datapoint is None, expect all are like that.
            if records[key][0] is not None:
                # when we have multiple elements, we stack.
                if len(records[key][0]) > 1:
                    records[key] = torch.stack(records[key], 0)
                # len(records[key]) == 1: when we have single element, we don't need to stack.
                else:
                    records[key] = torch.cat(records[key])
            else:
                records[key] = None

    return records


def torchvision_dict_collate(
    batch: Dict[str, Optional[Union[str, torch.Tensor]]], include_background: bool = False
):
    """Torchvision based collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Parameters
    ----------
    batch : Dict[str, Optional[Union[str, torch.Tensor]]]
        A single record's Dict with image and associated ground truth
        information after applying transforms. Contains "img_id", "img",
        "bbox_class", "bbox_coord", "label_class" and
        "label_value" keys.
    include_background: bool
        Some models require box labelling to start from 1

    Returns
    -------
    images: List[Tensor]
        A list of images converted to tensors of the form (C, H, W).
    targets: List[Dict[str, Tensor]]
        each element in targets consists of a dict with "boxes" and "labels" keys.
    """

    # set image_ids, images and targets
    img_ids, images, targets = [], [], []
    for record in batch:
        img_ids.append(record["img_id"])
        images.append(record["img"])
        if record["bbox_coord"] is not None:
            # since bounding boxes are currently start from 0 to 1
            targets.append(
                {
                    "boxes": record["bbox_coord"],
                    "labels": (record["bbox_class"] + (1 if include_background else 0)).long(),
                }
            )
        else:
            targets.append(
                {"boxes": torch.empty(0, 4, dtype=torch.float32), "labels": torch.zeros((0,))}
            )

    return {"img_ids": img_ids, "images": images, "targets": targets}
