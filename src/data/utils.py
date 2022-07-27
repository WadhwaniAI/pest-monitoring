from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops.boxes import box_convert, box_iou

from src.metrics.bounding_box import BoundingBox
from src.metrics.enumerators import BBFormat, BBType, CoordinatesType


class Encoder(object):
    """
    Inspired by https://github.com/kuangliu/pytorch-src
    Transform between (bboxes, labels) <-> SSD output

    dboxes: default boxes in size 8732 x 4,
        encoder: input ltrb format, output xywh format
        decoder: input xywh format, output ltrb format
    encode:
        input  : bboxes_in (Tensor nboxes x 4), labels_in (Tensor nboxes)
        output : bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
        criteria : IoU threshold of bboexes
    decode:
        input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
        output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
        criteria : IoU threshold of bboexes
        max_output : maximum number of output bboxes

    Parameters
    ----------
    dboxes : DefaultBoxes
        default boxes
    """

    def __init__(self, dboxes):
        self.dboxes = dboxes(order="ltrb")
        self.dboxes_xywh = dboxes(order="xywh").unsqueeze(dim=0)
        self.nboxes = self.dboxes.size(0)
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

    def encode(self, bboxes_in, labels_in, criteria=0.5):
        """Encode bounding boxes for an image sample.

        Parameters
        ----------
        bboxes_in : torch.Tensor
            Input bounding box locations
        labels_in : torch.Tensor
            Input bounding box labels
        criteria : float, optional
            IOU threshold, by default 0.5

        Returns
        -------
        bboxes_out : torch.Tensor
            Encoded boxes
        labels_out : torch.Tensor
            Encoded labels

        """
        ious = box_iou(bboxes_in, self.dboxes)
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)
        best_bbox_ious, best_bbox_idx = ious.max(dim=1)

        # set best ious 2.0
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)
        device = bboxes_in.device
        idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64).to(device)
        best_dbox_idx[best_bbox_idx[idx]] = idx

        # filter IoU > 0.5
        masks = best_dbox_ious > criteria
        labels_out = torch.zeros(self.nboxes, dtype=torch.float).to(device)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]
        bboxes_out = self.dboxes.clone()
        bboxes_out[masks, :] = bboxes_in[best_dbox_idx[masks], :]
        bboxes_out = box_convert(bboxes_out, in_fmt="xyxy", out_fmt="cxcywh")
        return bboxes_out, labels_out

    def encode_batch(self, bboxes_in_batch, labels_in_batch, criteria=0.5):
        """Encode bounding boxes for a batch of images.

        Parameters
        ----------
        bboxes_in_batch : torch.Tensor
            Input batch of bounding box locations
        labels_in_batch : torch.Tensor
            Input batch of bounding box labels
        criteria : float, optional
            IOU Threshold, by default 0.5

        Returns
        -------
        bboxes_out_batch : torch.Tensor
            Encoded batch of boxes
        labels_out_batch : torch.Tensor
            Encoded batch of labels
        """
        device = bboxes_in_batch[0].device
        self.move_to_correct_device(device)
        bboxes_out_batch, labels_out_batch = [], []
        for bboxes_in, labels_in in zip(bboxes_in_batch, labels_in_batch):
            if labels_in.numel() != 0:
                bboxes_out, labels_out = self.encode(bboxes_in, labels_in)
                bboxes_out_batch.append(bboxes_out)
                labels_out_batch.append(labels_out)
            else:
                bboxes_out = self.dboxes.clone()
                bboxes_out = box_convert(bboxes_out, in_fmt="xyxy", out_fmt="cxcywh").to(device)
                labels_out = torch.zeros(self.nboxes, dtype=torch.long).to(device)
                bboxes_out_batch.append(bboxes_out)
                labels_out_batch.append(labels_out)

        return torch.stack(bboxes_out_batch, 0), torch.stack(labels_out_batch, 0)

    def scale_back_batch(self, bboxes_in, scores_in):
        """Do scale and transform from xywh to ltrb suppose input Nx4xnum_bbox Nxlabel_numxnum_bbox

        Parameters
        ----------
        bboxes_in : torch.Tensor
            Input batch of bounding boxes
        scores_in : torch.Tensor
            Input batch of bounding box scores

        Returns
        -------
        bboxes_out : torch.Tensor
            Scaled and transformed bounding box
        scores_out : torch.Tensor
            Scaled and transformed bounding box scores
        """
        self.dboxes = self.dboxes
        self.dboxes_xywh = self.dboxes_xywh

        bboxes_in = bboxes_in.permute(0, 2, 1)  # (batch, 8732, 4)
        scores_in = scores_in.permute(0, 2, 1)  # (batch, 8732, 2)

        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]

        bboxes_in[:, :, :2] = (
            bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        )
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]
        bboxes_in = box_convert(bboxes_in, in_fmt="cxcywh", out_fmt="xyxy")

        return bboxes_in, F.softmax(scores_in, dim=-1)

    def decode_batch(self, bboxes_in, scores_in, nms_threshold=0.45, max_output=200):
        """Decode bounding boxes for a batch of images.

        Parameters
        ----------
        bboxes_in : torch.Tensor
            Input batch of bounding boxes
        scores_in : torch.Tensor
            Input batch of bounding box scores
        nms_threshold : float, optional
            NMS Threshold, by default 0.45
        max_output : int, optional
            Max output number of boxes, by default 200

        Returns
        -------
        output : List[torch.Tensor]
            Decoded bounding box locations, labels and scores for each image in batch
        """
        self.move_to_correct_device(bboxes_in.device)
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)
        output = []
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            output.append(self.decode_single(bbox, prob, nms_threshold, max_output))
        return output

    def decode_single(self, bboxes_in, scores_in, nms_threshold, max_output, max_num=200):
        """Decode bounding boxes for a single images.

        Parameters
        ----------
        bboxes_in : torch.Tensor
            input bounding boxes
        scores_in : torch.Tensor
            input bounding box scores
        nms_threshold : float
            Threshold for NMS
        max_output : int
            Maxiumum output bounding boxes allowed
        max_num : int, optional
            Max number of boxes to be selected, by default 200

        Returns
        -------
        bboxes_out : torch.Tensor
            Decoded bounding boxes
        labels_out : torch.Tensor
            Decoded bounding box labels
        scores_out : torch.Tensor
            Decoded bounding box scores
        """
        bboxes_out = []
        scores_out = []
        labels_out = []

        for i, score in enumerate(scores_in.split(1, 1)):
            if i == 0:
                continue

            score = score.squeeze(1)
            mask = score > 0.2

            bboxes, score = bboxes_in[mask, :], score[mask]
            if score.size(0) == 0:
                continue

            score_sorted, score_idx_sorted = score.sort(dim=0)

            # select max_output indices
            score_idx_sorted = score_idx_sorted[-max_num:]
            candidates = []

            while score_idx_sorted.numel() > 0:
                idx = score_idx_sorted[-1].item()
                bboxes_sorted = bboxes[score_idx_sorted, :]
                bboxes_idx = bboxes[idx, :].unsqueeze(dim=0)
                iou_sorted = box_iou(bboxes_sorted, bboxes_idx).squeeze()
                # we only need iou < nms_threshold
                score_idx_sorted = score_idx_sorted[iou_sorted < nms_threshold]
                candidates.append(idx)

            bboxes_out.append(bboxes[candidates, :])
            scores_out.append(score[candidates])
            labels_out.extend([i] * len(candidates))

        if not bboxes_out:
            return [torch.tensor([]) for _ in range(3)]

        bboxes_out, labels_out, scores_out = (
            torch.cat(bboxes_out, dim=0),
            torch.tensor(labels_out, dtype=torch.long),
            torch.cat(scores_out, dim=0),
        )

        _, max_ids = scores_out.sort(dim=0)
        max_ids = max_ids[-max_output:]
        return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]

    def move_to_correct_device(self, device):
        self.dboxes = self.dboxes.to(device)
        self.dboxes_xywh = self.dboxes_xywh.to(device)


class BoxTransform:
    """This Class can be used to convert ground truth/predicted boxes for a particular image from
    torch.Tensor/np.array/list to List[BoundingBox]

    Parameters
    ----------
    bbox_locs: torch.Tensor/np.array/list
        Bounding box locations
    bbox_labels: torch.Tensor/np.array/list
        Bounding box labels
    bbox_scores: torch.Tensor/np.array/list
        Bounding box scores
    img_id: str
        Image id
    img_size: tuple
        Image size
    bbox_format: str
        Bounding box format, can be one of,
        - 'XYX2Y2'
        - 'XYWH'
        - 'YOLO'
    coordinates_type: str
        Coordinates type, can be 'RELATIVE' or 'ABSOLUTE'
    bbox_type: str
        Bounding box type, can be one of,
        - 'GROUND_TRUTH'
        - 'DETECTED'
    box_label_mapping: dict
        Mapping of labels to their index in the list of classes
    """

    def __init__(
        self,
        bbox_locs: Any = None,
        bbox_labels: Any = None,
        bbox_scores: Any = None,
        img_id: str = None,
        img_size: Tuple[int, int] = None,
        bbox_format: str = "XYX2Y2",
        coordinates_type: str = "RELATIVE",
        bbox_type: str = "GROUND_TRUTH",
        box_label_mapping: Dict[int, str] = None,
    ) -> None:
        # Store the input parameters
        self.bbox_locs = bbox_locs
        self.bbox_labels = bbox_labels
        self.bbox_scores = bbox_scores

        # if the inputs are tensors, shift to cpu and convert to numpy
        if isinstance(self.bbox_locs, torch.Tensor):
            self._convert_from_tensor_to_numpy()

        self.img_id = img_id
        self.img_size = img_size
        self.box_label_mapping = box_label_mapping

        # check if inputs are in the correct format
        self._check_locs_labels_scores(bbox_type)
        self._check_box_label_mapping()

        # convert to right format
        self.bbox_format = self._convert_bbox_format(bbox_format)
        self.coordinates_type = self._convert_coordinates_type(coordinates_type)
        self.bbox_type = self._convert_bbox_type(bbox_type)

        # convert to list of BoundingBox
        self.bboxes = self._convert_to_bboxes()

    def _check_box_label_mapping(self):
        """Check bounding box label mapping

        Raises
        ------
        ValueError
            If bounding box label not provided
        ValueError
            If bounding box label mapping does not contain all bounding box labels
        """
        if self.box_label_mapping is None:
            raise ValueError("box_label_mapping is not provided")

        # check if bbox_labels are in the box_label_mapping
        if not set(self.bbox_labels).issubset(set(self.box_label_mapping.keys())):
            raise ValueError("box_label_mapping does not contain all the bbox_labels")

    def _convert_to_bboxes(self):
        """Converts to list of BoundingBox
        Function looks at type(bbox_locs), if
            - type is numpy.ndarray, calls _convert_from_numpy()
            - type is list, calls _convert_from_list()

        Returns
        -------
        out : List[BoundingBox]
            Converted list of bounding boxes into BoundingBox class

        Raises
        ------
        TypeError
            Bounding Box locations not numpy.ndarray or list
        """
        if isinstance(self.bbox_locs, np.ndarray):
            return self._convert_from_numpy()
        elif isinstance(self.bbox_locs, list):
            return self._convert_from_list()
        else:
            raise TypeError("bbox_locs should be of type numpy.ndarray or list")

    def _check_locs_labels_scores(self, bbox_type: str):
        """Checks if inputs to this class are in the right format

        Parameters
        ----------
        bbox_type : str
            Type of bounding box

        Raises
        ------
        TypeError
            If image id is not a string
        TypeError
            If bounding box locations not numpy array or list
        TypeError
            If bounding box labels not numpy array or list
        TypeError
            If bounding box scores not numpy array or list
        ValueError
            If bounding box scores not None for ground truth
        ValueError
            If bounding box type not ground truth or predicted
        """
        # Checks if img_id is a String
        if not isinstance(self.img_id, str):
            raise TypeError("img_id must be a string")

        # Checks if bbox_locs is one of np.ndarray or list
        if not isinstance(self.bbox_locs, (np.ndarray, list)):
            raise TypeError(
                f"bbox_locs must be one of np.ndarray or list, but found {type(self.bbox_locs)}"
            )

        # Checks if bbox_labels is one of np.ndarray or list
        if not isinstance(self.bbox_labels, (np.ndarray, list)):
            raise TypeError(
                f"bbox_labels must be one of np.ndarray or list, but found {type(self.bbox_labels)}"
            )

        # If bbox_type is Predicted, checks if bbox_scores is one of
        # np.ndarray or list
        if bbox_type == "DETECTED":
            if not isinstance(self.bbox_scores, (np.ndarray, list)):
                raise TypeError(
                    "bbox_scores must be one of np.ndarray or list, "
                    f"but found {type(self.bbox_scores)}"
                )
        elif bbox_type == "GROUND_TRUTH":
            if self.bbox_scores is not None:
                raise ValueError(
                    "bbox_scores must be None for bbox_type GROUND_TRUTH, "
                    f"but found {self.bbox_scores}"
                )
        else:
            raise ValueError(
                f"bbox_type must be one of GROUND_TRUTH or DETECTED, but found {bbox_type}"
            )

    def _convert_bbox_type(self, bbox_type: str):
        """Converts the bbox_type to BBType format

        Parameters
        ----------
        bbox_type : str
            bounding box type

        Returns
        -------
        out : int
            Bounding box type ID

        Raises
        ------
        ValueError
            If bounding box not ground truth or detected
        """
        # Check if bbox_type is one of the allowed values
        allowed_bbox_types = ["GROUND_TRUTH", "DETECTED"]
        if bbox_type not in allowed_bbox_types:
            raise ValueError(f"bbox_type must be one of {allowed_bbox_types}")
        if bbox_type == "GROUND_TRUTH":
            return BBType.GROUND_TRUTH
        else:
            return BBType.DETECTED

    def _convert_coordinates_type(self, coordinates_type: str):
        """Converts the coordinates_type to CoordinatesType format

        Parameters
        ----------
        coordinates_type : str
            type of coordinart format

        Returns
        -------
        int
            Coordinate Type ID

        Raises
        ------
        ValueError
            if coorindate type not allowed
        """
        # Check if coordinates_type is one of the allowed values
        allowed_coordinates_types = ["RELATIVE", "ABSOLUTE"]
        if coordinates_type not in allowed_coordinates_types:
            raise ValueError(f"coordinates_type must be one of {allowed_coordinates_types}")
        if coordinates_type == "RELATIVE":
            return CoordinatesType.RELATIVE
        else:
            return CoordinatesType.ABSOLUTE

    def _convert_bbox_format(self, bbox_format: str):
        """Converts the bbox_format to BBFormat format

        Parameters
        ----------
        bbox_format : str
            Format of bounding boxes

        Returns
        -------
        int
            Bounding box format ID

        Raises
        ------
        ValueError
            if bounding box format not allowed
        """
        # Check if bbox_format is one of the allowed values
        allowed_bbox_formats = ["XYX2Y2", "XYWH", "YOLO"]
        if bbox_format not in allowed_bbox_formats:
            raise ValueError(f"bbox_format must be one of {allowed_bbox_formats}")
        if bbox_format == "XYX2Y2":
            return BBFormat.XYX2Y2
        elif bbox_format == "XYWH":
            return BBFormat.XYWH
        else:
            return BBFormat.YOLO

    def _convert_from_numpy(self):
        """Converts a numpy array of boxes to a list[BoundingBox]

        Returns
        -------
        bbox_list : List[BoundingBox]
            Converted list of bounding boxes into BoundingBox class
        """
        bbox_list = []
        for i in range(self.bbox_locs.shape[0]):
            bbox_list.append(
                BoundingBox(
                    image_name=self.img_id,
                    class_id=self.box_label_mapping[self.bbox_labels[i]],
                    coordinates=self.bbox_locs[i],
                    type_coordinates=self.coordinates_type,
                    img_size=self.img_size,
                    bb_type=self.bbox_type,
                    confidence=self.bbox_scores[i] if self.bbox_scores is not None else None,
                    format=self.bbox_format,
                )
            )
        return bbox_list

    def _convert_from_tensor_to_numpy(self):
        """Converts a tensor of boxes to a list[BoundingBox]"""
        self.bbox_locs = self.bbox_locs.cpu().numpy()
        self.bbox_labels = self.bbox_labels.cpu().numpy()
        if self.bbox_scores is not None:
            self.bbox_scores = self.bbox_scores.cpu().numpy()

    def _convert_from_list(self):
        """Converts a list of boxes to a list[BoundingBox]

        Returns
        -------
        bbox_list : List[BoundingBox]
            Converted list of bounding boxes into BoundingBox class
        """
        bbox_list = []
        for i in range(len(self.bbox_locs)):
            bbox_list.append(
                BoundingBox(
                    image_name=self.img_id,
                    class_id=self.box_label_mapping[self.bbox_labels[i]],
                    coordinates=self.bbox_locs[i],
                    type_coordinates=self.coordinates_type,
                    img_size=self.img_size,
                    bb_type=self.bbox_type,
                    confidence=self.bbox_scores[i] if self.bbox_scores is not None else None,
                    format=self.bbox_format,
                )
            )
        return bbox_list
