import itertools as it
import os
import subprocess as sp
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

import numpy as np
import pandas as pd
import wandb
from PIL import Image
from scipy.stats import sem
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

from src.metrics.bounding_box import BoundingBox
from src.metrics.coco_evaluator import get_coco_metrics as get_coco_summary_classwise
from src.metrics.coco_evaluator import get_coco_summary
from src.metrics.enumerators import BBFormat, BBType, CoordinatesType


# Utilities
def mae(gt_count: int, pd_count: int):
    """Calculates mean absolute error if
    gt_count and pd_count are provided

    Parameters
    ----------
    gt_count : int
        Ground truth count
    pd_count : int
        Predicted count
    """
    return abs(gt_count - pd_count)


def alpha_capped_error(gt_count: int, pd_count: int, alpha: int = 30) -> int:
    """Calculates alpha capped error if gt_count and pd_count are provided

    Parameters
    ----------
    gt_count : int
        Ground truth count
    pd_count : int
        Predicted count
    alpha : int, optional
        Upper bound on error, by default 30

    Returns
    -------
    int
        alpha capped absolute error
    """
    return min(alpha, abs(pd_count - gt_count))


def mae_alpha(gt_count: int, pd_count: int, alpha: int = 30):
    """Calculates mean absolute error (alpha) if
    gt_count and pd_count are provided

    Parameters
    ----------
    gt_count : int
        Ground truth count
    pd_count : int
        Predicted count
    """
    if pd_count == 0 and gt_count == 0:  # cases of True Negatives
        mae = np.nan
    elif pd_count > alpha and gt_count > alpha:
        mae = 0
    else:
        mae = min(alpha, abs(pd_count - gt_count))

    return mae


def non_rejected_error(systems_df: pd.DataFrame, object: str) -> float:
    """Calculates condessa regression port rejection weakness on alpha capped error

    Parameters
    ----------
    systems_df : pd.DataFrame
        DataFrame with rejection and regression errors
    object : str
        class on which metrics required

    Returns
    -------
    float
        condessa regression ported non rejected error
    """
    reject_df = systems_df[systems_df["reject"]]
    non_reject_df = systems_df[~systems_df["reject"]]
    nre = non_reject_df[f"{object}_error"].sum() / (len(systems_df) - len(reject_df))
    return nre


def regression_weakness(systems_df: pd.DataFrame, object: str, alpha: int = 30) -> float:
    """Calculates condessa regression port regression weakness on alpha capped error

    Parameters
    ----------
    systems_df : pd.DataFrame
        DataFrame with rejection and regression errors
    object : str
        class on which metrics required
    alpha : int
        alpha used in calculating alpha capped error, by default 30

    Returns
    -------
    float
        condessa regression ported regression weakness
    """
    reject_df = systems_df[systems_df["reject"]]
    non_reject_df = systems_df[~systems_df["reject"]]
    reg_w_num = (
        non_reject_df[f"{object}_error"].sum() + (alpha - reject_df[f"{object}_error"]).sum()
    )
    reg_w_den = len(non_reject_df) + len(reject_df)
    reg_w = reg_w_num / reg_w_den
    return reg_w


def rejection_weakness(systems_df: pd.DataFrame, object: str, alpha: int = 30) -> float:
    """Calculates condessa regression port rejection weakness on alpha capped error

    Parameters
    ----------
    systems_df : pd.DataFrame
        DataFrame with rejection and regression errors
    object : str
        class on which metrics required
    alpha : int
        alpha used in calculating alpha capped error, by default 30

    Returns
    -------
    float
        condessa regression ported rejection weakness
    """
    reject_df = systems_df[systems_df["reject"]]

    if len(reject_df[f"{object}_error"]) == 0:
        return 1.0

    if reject_df[f"{object}_error"].sum() == 0:
        return np.inf

    rej_w_num = (alpha - reject_df[f"{object}_error"]).sum() / reject_df[f"{object}_error"].sum()
    rej_w_den = (alpha - systems_df[f"{object}_error"]).sum() / systems_df[f"{object}_error"].sum()
    rej_w = rej_w_num / rej_w_den
    return rej_w


def convert_to_relative_xyx2y2(coords, im_height, im_width):
    """Converts absolute x,y,w,h to relative x,y,w,h"""
    x, y, w, h = coords  # currently absolute
    x2, y2 = x + w, y + h
    return x / im_width, y / im_height, x2 / im_width, y2 / im_height  # converting to relative


#
# Obtain metrics from pm-model-tools pipeline
#

# It's more straightforward to run the pm-model-tools pipeline from
# within that repo. This class manages changing into that location,
# and changing back once the pipeline is run
class DirectoryManager:
    def __init__(self, target):
        self.target = target
        self.source = Path.cwd()

    def __enter__(self):
        os.chdir(self.target)

    def __exit__(self, exc_type, exc_value, traceback):
        os.chdir(self.source)


# Handles running the pm-model-tools pipeline
class ModelToolsEvaluator:
    # Environment variable containing the path to the pm-model-tools
    # repository.
    _envkey = "PM_MODEL_TOOLS"

    # pm-model-tools jobs to run; see the tools documentation for
    # details, and for other job options.
    _tasks = (
        "basics",
        "condessa",
    )

    def __init__(self, output, split, wd):
        self.output = output
        self.split = split
        self.wd = wd

    def __iter__(self):
        # Switch into the repo directory and run the pipeline.
        with DirectoryManager(self.wd):
            script = Path("bin", "run").with_suffix(".sh")
            if not script.exists():
                raise FileNotFoundError(self.wd.joinpath(script))

            yield from self.run(script)

    def run(self, script):
        with TemporaryDirectory() as tmp:
            io = {x: Path(mkdtemp(dir=tmp)) for x in ("r", "o")}

            # All JSON files for evaluation are assumed to exist in a
            # single directory.
            io.get("r").joinpath(self.output.name).symlink_to(self.output)

            # Generate the CLI arguments and run the script.
            params = [
                str(script),
                f"-s {self.split}",
            ]
            params.extend(it.starmap("-{} {}".format, io.items()))
            params.extend(map("-d {}".format, self._tasks))
            sp.run(" ".join(params), shell=True)

            # Return each PNG that is produced. Their paths provide
            # some illumination of their meaning.
            dst = io["o"]
            for i in dst.rglob("*.png"):
                key = i.relative_to(dst).with_suffix("")
                value = wandb.Image(str(i))
                yield (key, value)


# Functional entry point for running the pm-model-tools pipeline
def get_pm_model_tools_metrics(output, split):
    wd = os.environ.get(ModelToolsEvaluator._envkey)
    if wd is None:
        raise LookupError(f"{ModelToolsEvaluator._envkey} not in env")
    (output, wd) = map(Path, (output, wd))
    mte = ModelToolsEvaluator(output, split, wd)

    return dict(mte)


# function to get accuracy/precision/recall metrics from json file
def get_validation_model_metrics(output, split):
    """Helper function get accuracy/precision/recall metrics from json file

    Parameters
    ----------
    output : dict
        Output dictionary from json file
    split : str
        Split name
    """
    img_ids = [image["image_id"] for image in output["splits"] if image["split"] == split]
    caption_annotations = pd.DataFrame(
        [caption for caption in output["caption_annotations"] if caption["image_id"] in img_ids]
    )
    gts, preds, confs = [], [], []
    for img_id in img_ids:
        gt = int(
            caption_annotations[
                (caption_annotations.image_id == img_id) & (caption_annotations.category_id == 2)
            ]["caption"].item()
        )
        pred = int(
            caption_annotations[
                (caption_annotations.image_id == img_id) & (caption_annotations.category_id == 5)
            ]["caption"].item()
        )
        conf = caption_annotations[
            (caption_annotations.image_id == img_id) & (caption_annotations.category_id == 5)
        ]["conf"].item()

        gts.append(gt)
        preds.append(pred)
        confs.append(conf if pred == 1 else 1 - conf)

    # calculate all the metrics using sklearn.metrics
    return_dict = {
        "Accuracy": accuracy_score(gts, preds),
        "Precision": precision_score(gts, preds),
        "Recall": recall_score(gts, preds),
        "roc_auc_score": roc_auc_score(gts, confs),
    }

    return return_dict


# function to get mae metrics for counting model from json file
def get_mae_metrics(output, split):
    """Helper function to get MAE Metrics for object counting given a prediction json.


    Parameters
    ----------
    output : dict
        Prediction json
    split : str
        Split name
    """
    img_ids = [image["image_id"] for image in output["splits"] if image["split"] == split]
    boxes = [box for box in output["box_annotations"] if box["image_id"] in img_ids]

    categories = output["categories"]

    gt_box_id_list = [
        category["id"] for category in categories if category["supercategory"] == "bounding box"
    ]
    pd_box_id_list = [
        category["id"]
        for category in categories
        if category["supercategory"] == "predicted bounding box"
    ]

    gt_boxes = [box for box in boxes if box["category_id"] in gt_box_id_list]
    pd_boxes = [box for box in boxes if box["category_id"] in pd_box_id_list]

    # Convert to pandas dataframe to get numbers quick
    gt_boxes_df = pd.DataFrame(gt_boxes)
    pd_boxes_df = pd.DataFrame(pd_boxes)

    # Store MAE Numbers
    metrics_dict = defaultdict(list)

    for img_id in tqdm(img_ids):
        abw_gt = sum(gt_boxes_df[gt_boxes_df.image_id == img_id]["category_id"] == 0)
        pbw_gt = sum(gt_boxes_df[gt_boxes_df.image_id == img_id]["category_id"] == 1)
        # predictions for all images
        abw_pd = sum(pd_boxes_df[pd_boxes_df.image_id == img_id]["category_id"] == 3)
        pbw_pd = sum(pd_boxes_df[pd_boxes_df.image_id == img_id]["category_id"] == 4)

        metrics_dict["MAE-Alpha/ABW"].append(mae_alpha(abw_gt, abw_pd))
        metrics_dict["MAE-Alpha/PBW"].append(mae_alpha(pbw_gt, pbw_pd))
        metrics_dict["MAE/ABW"].append(mae(abw_gt, abw_pd))
        metrics_dict["MAE/PBW"].append(mae(pbw_gt, pbw_pd))

    for k, v in metrics_dict.items():
        metrics_dict[k] = np.nanmean(v)

    return metrics_dict


# function to get mae metrics for a system level model from json file
def get_mae_metrics_system(output, split):
    """Helper function to get MAE Metrics for a system given a prediction json


    Parameters
    ----------
    output : dict
        Prediction json
    split : str
        Split name
    """
    img_ids = [image["image_id"] for image in output["splits"] if image["split"] == split]
    boxes = [box for box in output["box_annotations"] if box["image_id"] in img_ids]
    caption_annotations = [
        caption for caption in output["caption_annotations"] if caption["image_id"] in img_ids
    ]

    categories = output["categories"]

    gt_box_id_list = [
        category["id"] for category in categories if category["supercategory"] == "bounding box"
    ]
    pd_box_id_list = [
        category["id"]
        for category in categories
        if category["supercategory"] == "predicted bounding box"
    ]

    gt_boxes = [box for box in boxes if box["category_id"] in gt_box_id_list]
    pd_boxes = [box for box in boxes if box["category_id"] in pd_box_id_list]

    # True trap boxes
    trap_img_ids = np.unique([box["image_id"] for box in gt_boxes]).tolist()
    true_trap_boxes = [box for box in pd_boxes if box["image_id"] in trap_img_ids]

    # get predicted trap img_ids
    pd_trap_img_ids = np.unique(
        [
            caption["image_id"]
            for caption in caption_annotations
            if caption["category_id"] == 5 and caption["caption"] == 1.0
        ]
    )
    pred_trap_boxes = [box for box in pd_boxes if box["image_id"] in pd_trap_img_ids]

    # Convert to pandas dataframe to get numbers quick
    gt_boxes_df = pd.DataFrame(gt_boxes)
    pd_boxes_df = pd.DataFrame(pd_boxes)
    true_trap_boxes_df = pd.DataFrame(true_trap_boxes)
    pred_trap_boxes_df = pd.DataFrame(pred_trap_boxes)

    # Store MAE Numbers
    metrics_dict = defaultdict(list)

    for img_id in tqdm(img_ids):
        abw_gt = sum(gt_boxes_df[gt_boxes_df.image_id == img_id]["category_id"] == 0)
        pbw_gt = sum(gt_boxes_df[gt_boxes_df.image_id == img_id]["category_id"] == 1)
        # predictions for all images
        abw_pd = sum(pd_boxes_df[pd_boxes_df.image_id == img_id]["category_id"] == 3)
        pbw_pd = sum(pd_boxes_df[pd_boxes_df.image_id == img_id]["category_id"] == 4)

        # predictions for true trap only
        abw_pd_trap = sum(
            true_trap_boxes_df[true_trap_boxes_df.image_id == img_id]["category_id"] == 3
        )
        pbw_pd_trap = sum(
            true_trap_boxes_df[true_trap_boxes_df.image_id == img_id]["category_id"] == 4
        )

        # predictions for predicted trap only
        abw_pd_pred_trap = sum(
            pred_trap_boxes_df[pred_trap_boxes_df.image_id == img_id]["category_id"] == 3
        )
        pbw_pd_pred_trap = sum(
            pred_trap_boxes_df[pred_trap_boxes_df.image_id == img_id]["category_id"] == 4
        )

        metrics_dict["MAE-Alpha/ABW"].append(mae_alpha(abw_gt, abw_pd))
        metrics_dict["MAE-Alpha/PBW"].append(mae_alpha(pbw_gt, pbw_pd))
        metrics_dict["MAE/ABW"].append(mae(abw_gt, abw_pd))
        metrics_dict["MAE/PBW"].append(mae(pbw_gt, pbw_pd))

        # Ground Truth Trap only
        metrics_dict["MAE-Alpha/ABW (Ground Truth Trap Only)"].append(
            mae_alpha(abw_gt, abw_pd_trap)
        )
        metrics_dict["MAE-Alpha/PBW (Ground Truth Trap Only)"].append(
            mae_alpha(pbw_gt, pbw_pd_trap)
        )
        metrics_dict["MAE/ABW (Ground Truth Trap Only)"].append(mae(abw_gt, abw_pd_trap))
        metrics_dict["MAE/PBW (Ground Truth Trap Only)"].append(mae(pbw_gt, pbw_pd_trap))

        # Predicted trap only
        metrics_dict["MAE-Alpha/ABW (Predicted Trap Only)"].append(
            mae_alpha(abw_gt, abw_pd_pred_trap)
        )
        metrics_dict["MAE-Alpha/PBW (Predicted Trap Only)"].append(
            mae_alpha(pbw_gt, pbw_pd_pred_trap)
        )
        metrics_dict["MAE/ABW (Predicted Trap Only)"].append(mae(abw_gt, abw_pd_pred_trap))
        metrics_dict["MAE/PBW (Predicted Trap Only)"].append(mae(pbw_gt, pbw_pd_pred_trap))

    final_dict = {}
    for k, v in metrics_dict.items():
        final_dict[k] = np.nanmean(v)
        final_dict[f"S{k[2:]}"] = sem(v, nan_policy="omit")

    return final_dict


# COCO Metrics
# Utility Function
def getBoundingBoxList(boxes, class_mapping, img_size_dict, ground_truth=False):
    """Converts the bounding box list to a list of BoundingBox objects

    Parameters
    ----------
    boxes : list
        List of bounding boxes in the format [x, y, w, h]
    class_mapping : dict
        Dictionary of class mappings
    img_size_dict : dict
        Dictionary of image sizes
    ground_truth : bool
        Whether the bounding box is ground truth or not
    """
    ret = []
    bb_type = BBType.GROUND_TRUTH if ground_truth else BBType.DETECTED
    for box in boxes:
        ret.append(
            BoundingBox(
                image_name=str(box["image_id"]),
                class_id=class_mapping[box["category_id"]],
                coordinates=box["bbox"],
                type_coordinates=CoordinatesType.ABSOLUTE,
                img_size=img_size_dict[box["image_id"]],
                bb_type=bb_type,
                confidence=box["bbox_score"] if "bbox_score" in box else None,
                format=BBFormat.XYWH,
            )
        )
    return ret


def get_coco_metrics_class(output, split, iou_th=0.5):
    """Helper function to get COCO Metrics given a prediction json classwise

    Parameters
    ----------
    output : dict
        Prediction json
    split : str
        Split name
    """

    img_ids = [image["image_id"] for image in output["splits"] if image["split"] == split]
    boxes = [box for box in output["box_annotations"] if box["image_id"] in img_ids]
    categories = output["categories"]

    gt_box_id_list = [
        category["id"] for category in categories if category["supercategory"] == "bounding box"
    ]
    pd_box_id_list = [
        category["id"]
        for category in categories
        if category["supercategory"] == "predicted bounding box"
    ]

    gt_boxes = [box for box in boxes if box["category_id"] in gt_box_id_list]
    pd_boxes = [box for box in boxes if box["category_id"] in pd_box_id_list]

    img_size_dict = {im["id"]: (im["height"], im["width"]) for im in output["images"]}
    class_mapping = {category["id"]: category["name"] for category in categories}

    metrics_dict = {}

    # Get the ground truth / pred boxes
    gt_bbs = getBoundingBoxList(gt_boxes, class_mapping, img_size_dict, True)
    pd_bbs = getBoundingBoxList(pd_boxes, class_mapping, img_size_dict, False)

    result_all = get_coco_summary_classwise(
        gt_bbs, pd_bbs, iou_threshold=iou_th
    )  # not changing the default values

    for k, v in result_all.items():
        metrics_dict[k] = v

    return metrics_dict


def get_coco_metrics(output, split):
    """Helper function to get COCO Metrics given a prediction json

    Parameters
    ----------
    output : dict
        Prediction json
    split : str
        Split name
    """

    img_ids = [image["image_id"] for image in output["splits"] if image["split"] == split]
    boxes = [box for box in output["box_annotations"] if box["image_id"] in img_ids]
    categories = output["categories"]

    gt_box_id_list = [
        category["id"] for category in categories if category["supercategory"] == "bounding box"
    ]
    pd_box_id_list = [
        category["id"]
        for category in categories
        if category["supercategory"] == "predicted bounding box"
    ]

    gt_boxes = [box for box in boxes if box["category_id"] in gt_box_id_list]
    pd_boxes = [box for box in boxes if box["category_id"] in pd_box_id_list]

    img_size_dict = {im["id"]: (im["height"], im["width"]) for im in output["images"]}
    class_mapping = {category["id"]: category["name"] for category in categories}

    metrics_dict = {}

    # Get the ground truth / pred boxes
    gt_bbs = getBoundingBoxList(gt_boxes, class_mapping, img_size_dict, True)
    pd_bbs = getBoundingBoxList(pd_boxes, class_mapping, img_size_dict, False)

    result_all = get_coco_summary(gt_bbs, pd_bbs)

    for k, v in result_all.items():
        metrics_dict[k] = v

    return metrics_dict


# For plotting Image wise predictions on Weights and Biases


def get_box_data(df, img_size_dict, class_id_to_label, ground_truth=False):
    """Returns a list of dictionaries containing the bounding box data to be used
    by Weights and Biases wandb.Image(images, boxes)

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the predictions
    img_size_dict : dict
        Dictionary containing the image sizes
    class_id_to_label : dict
        Dictionary containing the class id to label mapping
    ground_truth : bool
        If True, the ground truth bounding boxes are returned
    """

    box_data = []
    for i, row in df.iterrows():
        img_size = img_size_dict[row["image_id"]]
        minX, minY, maxX, maxY = convert_to_relative_xyx2y2(row["bbox"], img_size[0], img_size[1])
        if ground_truth:
            box_data.append(
                {
                    "position": {
                        "minX": minX,
                        "maxX": maxX,
                        "minY": minY,
                        "maxY": maxY,
                    },
                    "class_id": row["category_id"],
                    "box_caption": class_id_to_label[row["category_id"]],
                }
            )
        else:
            box_data.append(
                {
                    "position": {
                        "minX": minX,
                        "maxX": maxX,
                        "minY": minY,
                        "maxY": maxY,
                    },
                    "class_id": row["category_id"] - 3,
                    "box_caption": class_id_to_label[row["category_id"] - 3],
                    "scores": {
                        "confidence": row["bbox_score"],
                    },
                }
            )
    return box_data


class_id_to_label = {
    0: "ABW",
    1: "PBW",
}

class_set = wandb.Classes(
    [
        {"name": "ABW", "id": 0},
        {"name": "PBW", "id": 1},
    ]
)


# function to get the dataframe
def get_prediction_df(output, split, image_size=(512, 512)):
    """Function to get the dataframe from the output json file

    Parameters
    ----------
    output : dict
        The output json file
    split : str
        The split of the data
    """

    img_ids = [image["image_id"] for image in output["splits"] if image["split"] == split]

    # get id to path dict
    id_to_path = {im["id"]: im["file_path"] for im in output["images"]}

    # get id to size dict
    img_size_dict = {im["id"]: (im["height"], im["width"]) for im in output["images"]}

    # get necessary gt and pd boxes
    boxes = [box for box in output["box_annotations"] if box["image_id"] in img_ids]
    gt_boxes = [box for box in boxes if box["category_id"] in [0, 1]]
    pd_boxes = [box for box in boxes if box["category_id"] in [3, 4]]

    # Convert to pandas dataframe to get numbers quick
    gt_boxes_df = pd.DataFrame(gt_boxes)
    pd_boxes_df = pd.DataFrame(pd_boxes)

    # Caption Annotations #TODO Might change based on Apoorv's PR
    captions_df = [
        caption for caption in output["caption_annotations"] if caption["image_id"] in img_ids
    ]
    captions_df = pd.DataFrame(captions_df)

    # Store MAE Numbers
    df_dict = {
        "file_path": [],
        "image": [],
        "Trap": [],
        "Trap Prediction": [],
        "Trap Accuracy": [],
        "Ground Truth Count (ABW)": [],
        "Ground Truth Count (PBW)": [],
        "Predicted Count (ABW)": [],
        "Predicted Count (PBW)": [],
        "MAE-Alpha (ABW)": [],
        "MAE-Alpha (PBW)": [],
        "MAE (ABW)": [],
        "MAE (PBW)": [],
    }

    for img_id in tqdm(img_ids):
        df_dict["file_path"].append(id_to_path[img_id])

        abw_gt = sum(gt_boxes_df[gt_boxes_df.image_id == img_id]["category_id"] == 0)
        pbw_gt = sum(gt_boxes_df[gt_boxes_df.image_id == img_id]["category_id"] == 1)
        # predictions for all images
        abw_pd = sum(pd_boxes_df[pd_boxes_df.image_id == img_id]["category_id"] == 3)
        pbw_pd = sum(pd_boxes_df[pd_boxes_df.image_id == img_id]["category_id"] == 4)

        pd_box_data = get_box_data(
            pd_boxes_df[pd_boxes_df.image_id == img_id], img_size_dict, class_id_to_label, False
        )
        gt_box_data = get_box_data(
            gt_boxes_df[gt_boxes_df.image_id == img_id], img_size_dict, class_id_to_label, True
        )

        # Trap Non-Trap Accuracy
        trap_prediction = captions_df[
            (captions_df.image_id == img_id) & (captions_df.category_id == 5)
        ]["caption"].item()
        trap_label = captions_df[(captions_df.image_id == img_id) & (captions_df.category_id == 2)][
            "caption"
        ].item()
        trap_accuracy = (int(trap_prediction) == int(trap_label)) * 1

        boxes = {
            "predictions": {
                "box_data": pd_box_data,
                "class_labels": class_id_to_label,
            },
            "ground_truth": {
                "box_data": gt_box_data,
                "class_labels": class_id_to_label,
            },
        }
        im = Image.open(id_to_path[img_id])
        new_size = image_size

        df_dict["image"].append(wandb.Image(im.resize(new_size), boxes=boxes, classes=class_set))
        df_dict["Trap"].append(trap_label)
        df_dict["Trap Prediction"].append(trap_prediction)
        df_dict["Trap Accuracy"].append(trap_accuracy)
        df_dict["Ground Truth Count (ABW)"].append(abw_gt)
        df_dict["Ground Truth Count (PBW)"].append(pbw_gt)
        df_dict["Predicted Count (ABW)"].append(abw_pd)
        df_dict["Predicted Count (PBW)"].append(pbw_pd)
        df_dict["MAE-Alpha (ABW)"].append(mae_alpha(abw_gt, abw_pd))
        df_dict["MAE-Alpha (PBW)"].append(mae_alpha(pbw_gt, pbw_pd))
        df_dict["MAE (ABW)"].append(mae(abw_gt, abw_pd))
        df_dict["MAE (PBW)"].append(mae(pbw_gt, pbw_pd))

    return pd.DataFrame.from_dict(df_dict)


# function to get mae metrics from json file
def get_condessa_capped_regression_metrics(output, split):
    """Helper function to get MAE Metrics given a prediction json

    Parameters
    ----------
    output : dict
        Prediction json
    split : str
        Split name
    """
    img_ids = [image["image_id"] for image in output["splits"] if image["split"] == split]
    boxes = [box for box in output["box_annotations"] if box["image_id"] in img_ids]
    validations = [
        caption for caption in output["caption_annotations"] if caption["image_id"] in img_ids
    ]
    categories = output["categories"]

    gt_box_id_list = [
        category["id"] for category in categories if category["supercategory"] == "bounding box"
    ]
    pd_box_id_list = [
        category["id"]
        for category in categories
        if category["supercategory"] == "predicted bounding box"
    ]

    gt_boxes = [box for box in boxes if box["category_id"] in gt_box_id_list]
    pd_boxes = [box for box in boxes if box["category_id"] in pd_box_id_list]

    # Convert to pandas dataframe to get numbers quick
    gt_boxes_df = pd.DataFrame(gt_boxes)
    pd_boxes_df = pd.DataFrame(pd_boxes)
    validations_df = pd.DataFrame(validations)

    # Put all system metric reqs in one df
    systems_df = validations_df[validations_df["category_id"] == 5][["image_id", "caption"]]
    systems_df["reject"] = systems_df["caption"].apply(lambda x: x == 0)
    systems_df.drop(columns=["caption"], inplace=True)

    for idx, row in systems_df.iterrows():
        img_id = row["image_id"]
        abw_gt = sum(gt_boxes_df[gt_boxes_df.image_id == img_id]["category_id"] == 0)
        pbw_gt = sum(gt_boxes_df[gt_boxes_df.image_id == img_id]["category_id"] == 1)
        # predictions for all images
        abw_pd = sum(pd_boxes_df[pd_boxes_df.image_id == img_id]["category_id"] == 3)
        pbw_pd = sum(pd_boxes_df[pd_boxes_df.image_id == img_id]["category_id"] == 4)

        systems_df.loc[idx, "abw_error"] = alpha_capped_error(abw_gt, abw_pd)
        systems_df.loc[idx, "pbw_error"] = alpha_capped_error(pbw_gt, pbw_pd)

    metrics_dict = {
        "Condessa/Regression/Non-Rejected-Error-ABW": non_rejected_error(systems_df, "abw"),
        "Condessa/Regression/Non-Rejected-Error-PBW": non_rejected_error(systems_df, "pbw"),
        "Condessa/Regression/Regression-Weakness-ABW": regression_weakness(systems_df, "abw"),
        "Condessa/Regression/Regression-Weakness-PBW": regression_weakness(systems_df, "pbw"),
        "Condessa/Regression/Rejection-Weakness-ABW": rejection_weakness(systems_df, "abw"),
        "Condessa/Regression/Rejection-Weakness-PBW": rejection_weakness(systems_df, "pbw"),
    }

    return metrics_dict
