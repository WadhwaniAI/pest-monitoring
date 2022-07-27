"""Generic LightningModule Class for TorchVision Models"""
from typing import Any

from .object_detect_net import ObjectDetectNet
from .utils import detach


class ObjectDetectNetTVFormat(ObjectDetectNet):
    """Object Detection Models that follow TorchVision format"""

    def training_step(self, batch: Any, batch_idx: int):
        """Training step for the training phase"""
        images, targets = batch["images"], batch["targets"]
        loss_dict = self.network(images, targets)
        step_output = {
            "img_ids": batch["img_ids"],
            "loss": sum(loss_dict.values()),
        }
        step_output.update({f"loss_{k}": detach(v) for k, v in loss_dict.items()})
        self._log_loss(step_output, "train")
        return step_output

    def validation_step(self, batch: Any, batch_idx: int):
        """Validation step for the validation phase"""
        images, targets = batch["images"], batch["targets"]
        detections = self.network(images)
        return {"img_ids": batch["img_ids"], "detections": detections, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        """Test step for the test phase"""
        images, targets = batch["images"], batch["targets"]
        detections = self.network(images)
        return {"img_ids": batch["img_ids"], "detections": detections, "targets": targets}

    def predict_step(self, batch: Any, batch_idx: int):
        """Prediction step for the test phase"""
        return self.predict(batch["images"])
