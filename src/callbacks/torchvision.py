"""Callbacks to be used with torchvision models"""
from collections import defaultdict
from typing import Any, Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from src.data.utils import BoxTransform
from src.metrics.coco_evaluator import get_coco_summary


class BaseObjectDetectionMetricPlot(pl.Callback):
    def __init__(self, box_label_mapping: dict = None):
        super().__init__()
        if not box_label_mapping:
            raise ValueError("box_label_mapping is required")
        self.box_label_mapping = (
            {int(k): v for k, v in box_label_mapping.items()} if box_label_mapping else None
        )

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.outputs = defaultdict(list)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.outputs["img_ids"].extend(outputs["img_ids"])
        self.outputs["detections"].extend(outputs["detections"])
        self.outputs["targets"].extend(outputs["targets"])

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pd_boxes, gt_boxes = self._get_boxes(self.outputs)
        metrics = get_coco_summary(gt_boxes, pd_boxes)
        self._log_metrics(pl_module, metrics)

    def _log_metrics(self, pl_module, metrics):
        # log AP values
        for k, v in metrics.items():
            pl_module.log(f"val/{k}", v, prog_bar=True, logger=True, on_epoch=True)

    def _get_boxes(self, outputs: Any):
        """Converts detections to List[BoundingBox] using BoxTransform

        Parameters
        ----------
        outputs: Storing img_ids and detections

        Returns
        -------
        pd_boxes : list
            transformed predicted boxes
        gt_boxes : list
            transformed ground truth boxes
        """
        detections, targets, img_ids = (
            outputs["detections"],
            outputs["targets"],
            outputs["img_ids"],
        )

        # Predictions
        boxes = [x["boxes"] for x in detections]
        labels = [x["labels"] for x in detections]
        scores = [x["scores"] for x in detections]

        # Ground Truth
        target_boxes = [x["boxes"] for x in targets]
        target_labels = [x["labels"] for x in targets]

        # Length of the list of predictions
        n = len(boxes)

        pd_boxes = []
        gt_boxes = []
        for i in range(n):
            if boxes[i].shape[0] > 0:
                pd_boxes.extend(
                    BoxTransform(
                        bbox_locs=boxes[i],
                        bbox_scores=scores[i],
                        bbox_labels=labels[i],
                        img_id=str(img_ids[i]),
                        bbox_format="XYX2Y2",
                        coordinates_type="ABSOLUTE",
                        bbox_type="DETECTED",
                        box_label_mapping=self.box_label_mapping,
                    ).bboxes
                )
            if target_boxes[i].shape[0] > 0:
                gt_boxes.extend(
                    BoxTransform(
                        bbox_locs=target_boxes[i],
                        bbox_labels=target_labels[i],
                        img_id=str(img_ids[i]),
                        bbox_format="XYX2Y2",
                        coordinates_type="ABSOLUTE",
                        bbox_type="GROUND_TRUTH",
                        box_label_mapping=self.box_label_mapping,
                    ).bboxes
                )
        return pd_boxes, gt_boxes
