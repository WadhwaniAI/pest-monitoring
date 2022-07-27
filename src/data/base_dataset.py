import json
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset

from .transforms import Compose


class BaseDataset(Dataset):
    """BaseDataset Class for Object Detection (+ Image Validation) Tasks"""

    def __init__(
        self,
        dataset_config: DictConfig,
        mode: str,
        transforms: Optional[Compose] = None,
        **kwargs,
    ) -> None:
        """Initialize the dataset based on the dataset_config, mode and transforms.

        Parameters
        ----------
        dataset_config : DictConfig
            Hydra config
        mode : str
            Dataset split to load. Takes one of the strings: "train", "val", "test"
        transforms : Optional[Compose], optional
            Compose of transforms, defaults to None, by default None
        """
        super().__init__()
        self.mode = mode
        self.dataset_config = dataset_config

        self.prepare_data()

        self.transforms = transforms

    def __len__(self) -> int:
        """Length of the dataset as informed by unique image IDs

        Returns
        -------
        len : int
            Length of the Dataset
        """
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Dict[str, Optional[Union[str, torch.Tensor]]]:
        """Assemble and transform a single record of the dataset as expected in the interable.

        Parameters
        ----------
        idx : int
            Index number of the record in the dataset

        Returns
        -------
        record : Dict[str, Optional[Union[str, torch.Tensor]]]
            Dict with image and associated ground truth information after applying transforms.
            Contains "img_id", "img", "bbox_class", "bbox_coord", "label_class" and "label_value"
            keys.
        """
        img_id, img, bbox_class, bbox_coord, label_class, label_value = self.pull_item(idx)
        if self.transforms is not None:
            img, bbox_class, bbox_coord, label_class, label_value = self.transforms(
                img, bbox_class, bbox_coord, label_class, label_value
            )
        record = {
            "img_id": img_id,
            "img": img,
            "bbox_class": bbox_class,
            "bbox_coord": bbox_coord,
            "label_class": label_class,
            "label_value": label_value,
            "img_size": self.img_size_dict[img_id],
        }
        return record

    def pull_item(
        self, idx: int
    ) -> Tuple[
        str,
        np.array,
        Optional[np.array],
        Optional[np.array],
        Optional[np.array],
        Optional[np.array],
    ]:
        """Collect a single record containing image and associated ground truth
        information

        Parameters
        ----------
        idx : int
            Index number of the record in the input json


        Returns
        -------
        img_id : str
            image UID at ``idx`` in input json. image UID is required to index dataframes prepared
            in ``self.prepare_data`` corresponding to ``idx``.
        img : np.array
            numpy array of the image with image UID as ``img_id``.
        bbox_class : Optional[np.array]
            numpy array of the classes of bounding boxes of the image with image UID as ``img_id``
            if bounding boxes present, else None
        bbox_coord : Optional[np.array]
            numpy array of the coordinates of bounding boxes of the image with image UID as
            ``img_id`` if bounding boxes present, else None. Bounding box corrdinates are in
            [x, y, w, h] absolute pixel format. Where (x,y) is the top-left corner of the image and
            (w,h) are the width and height of the image respectively.
        label_class : Optional[np.array]
            numpy array of the label class of the image with image UID as ``img_id`` if labels
            present, else None.
        label_value : Optional[np.array]
            numpy array of the label value corresponding to label class of the image with image UID
            as ``img_id`` if labels present, else None.
        """
        img_id = self.img_ids[idx]
        img = self.get_img(self.img_df.loc[img_id]["file_path"])
        bbox_class, bbox_coord = (
            (None, None) if self.bbox_ann_df is None else self.get_bbox_anns(img_id)
        )
        label_class, label_value = (
            (None, None) if self.caption_ann_df is None else self.get_label_anns(img_id)
        )

        return img_id, img, bbox_class, bbox_coord, label_class, label_value

    def prepare_data(self, *args, **kwargs) -> None:
        """Custom function to read data or metadata from disk. This function should populate
        ``self.img_ids``: List of image ids in the current mode
        ``self.img_df``: pandas DataFrame with image ids set as index and a column "file_path"
        containing absolute path or equivalent for each image
        ``self.bbox_ann_df``: pandas DataFrame with columns "id" and "image_id" if bbox information
        is relevant, else None
        ``self.caption_ann_df``: pandas DataFrame with columns "id" and "image_id" if caption
        information is relevant, else None
        ``self.category_df``: pandas DataFrame with columns "id" and "name"
        """
        with open(self.dataset_config.data_file, "r") as file:
            self.data = json.load(file)
        self.img_size_dict = self.get_img_size_dict()
        img_ids = pd.DataFrame(self.data["splits"])
        self.img_ids = (
            img_ids[img_ids["split"] == self.mode].sort_values(by="image_id")["image_id"].tolist()
        )

        img_df = pd.DataFrame(self.data["images"]).set_index("id")
        self.img_df = img_df.loc[self.img_ids]
        if "box_annotations" in self.data and len(self.data["box_annotations"]) > 0:
            bbox_ann_df = pd.DataFrame(self.data["box_annotations"]).set_index("id")
            self.bbox_ann_df = bbox_ann_df[bbox_ann_df["image_id"].isin(self.img_ids)]
        else:
            self.bbox_ann_df = None
        if "caption_annotations" in self.data and len(self.data["caption_annotations"]) > 0:
            caption_ann_df = pd.DataFrame(self.data["caption_annotations"]).set_index("id")
            self.caption_ann_df = caption_ann_df[caption_ann_df["image_id"].isin(self.img_ids)]
        else:
            self.caption_ann_df = None
        self.category_df = pd.DataFrame(self.data["categories"]).set_index("id")

    def get_img_size_dict(self):
        """Utils function to map image id to image size"""
        images = self.data["images"]
        return {image["id"]: (image["height"], image["width"]) for image in images}

    def get_img(self, path: str, *args, **kwargs) -> np.array:
        """Custom function to get an image using "file_path" in ``self.img_df``

        Parameters
        ----------
        path : str
            Absolute image path

        Returns
        -------
        im : np.array
            Image as a numpy array

        """
        im = Image.open(path)
        im = np.asarray(im).copy()
        return im

    def get_bbox_anns(
        self, img_id: str, *args, **kwargs
    ) -> Tuple[Optional[np.array], Optional[np.array]]:
        """Parse bounding box coordinates from the input file. Override this function to change how
        to get bounding box information.

        Parameters
        ----------
        img_id : str
            UID of the image in input file

        Returns
        -------
        bbox_class : Optional[np.array]
            numpy array of the classes of bounding boxes of the image with image UID as ``img_id``
            if bounding boxes present, else None
        bbox_coord : Optional[np.array]
            numpy array of the coordinates of bounding boxes of the image with image UID as
            ``img_id`` if bounding boxes present, else None. Bounding box corrdinates are in
            [x, y, w, h] absolute pixel format. Where (x,y) is the top-left corner of the image and
            (w,h) are the width and height of the image respectively.
        """
        ann_df = self.bbox_ann_df[self.bbox_ann_df["image_id"] == img_id][["category_id", "bbox"]]
        if not len(ann_df):
            return None, None
        bbox_class = np.array(ann_df["category_id"].tolist())
        bbox_coord = np.vstack(ann_df["bbox"].apply(np.array).tolist())
        return bbox_class, bbox_coord

    def get_label_anns(
        self, img_id: Union[int, str], *args, **kwargs
    ) -> Tuple[Optional[np.array], Optional[np.array]]:
        """Parse image level labels from the input file. Override this function to change how to get
         image level information.

        Parameters
        ----------
        img_id : Union[int, str]
            UID of the image in input file

        Returns
        -------
        label_class : Optional[np.array]
            numpy array of the label class of the image with image UID as ``img_id`` if labels
            present, else None.
        label_value : Optional[np.array]
            numpy array of the label value corresponding to label class of the image with image UID
            as ``img_id`` if labels present, else None.
        """
        ann_df = self.caption_ann_df[self.caption_ann_df["image_id"] == img_id][
            ["category_id", "caption"]
        ]
        if not len(ann_df):
            return None, None
        label_class = np.array(ann_df["category_id"].tolist())
        label_caption = ann_df["caption"].tolist()
        label_value = np.array(list(map(self.eval_caption, label_caption)))
        return label_class, label_value

    def eval_caption(self, caption: str) -> Union[float, int, list]:
        """Resolve string type image level caption into value from the input file. Override this
        function to change how to resolve image level string caption into value.

        Parameters
        ----------
        caption : str
            Image level information's caption

        Returns
        -------
        value : Union[float, int, list]
            Interpreted value of input caption

        """
        value = eval(caption)  # Unsafe operation
        return value
