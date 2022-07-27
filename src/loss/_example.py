import sys
import logging

import torch
import pandas as pd

from src.loss import (
    BatchInfo,
    CompositeLoss,
    BasicValidationLoss,
    DetectionSmoothL1Loss,
    DetectionLossAggregator,
    DetectionHardMinedCELoss,
)

#
# Example usage. May need to set the PYTHONPATH correctly; from Bash:
#
# $> export PYTHONPATH=`git rev-parse --show-toplevel`:$PYTHONPATH
#
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create a box aggregator for detection losses
    aggregator = DetectionLossAggregator()

    # Compose the various loss "functions"
    losses = (
        (DetectionSmoothL1Loss(aggregator), 0.4),
        (DetectionHardMinedCELoss(aggregator), 0.4),
        (BasicValidationLoss(), 0.2),
    )
    composite = None
    for i in losses:
        composite = CompositeLoss(*i, composite)

    # Establish prediction and box coordinates, and calculate the loss
    (images, coordinates, classes, boxes) = (4, 4, 10, 8732)
    pr = BatchInfo(
        torch.rand(images, coordinates, boxes),
        torch.rand(images, classes, boxes),
        torch.rand(3, 3),
    )
    gt = BatchInfo(
        torch.rand(images, coordinates, boxes),
        torch.zeros(images, boxes).to(torch.long),
        torch.argmax(pr.labels, axis=1),
    )
    result = composite(pr, gt)

    # Output the loss to a log
    logging.info(f"Loss: {result.loss}")

    # Create a data frame from the result
    df = pd.DataFrame.from_records(result.to_records())

    # Use the dataframe to calculate the loss
    loss = df.filter(items=["loss", "weight"]).product(axis="columns").sum()
    logging.info("Pandas matches LossTrail: {}!".format("yes" if loss == result.loss else "no"))

    # Output the loss chain to CSV. Changing the type prior to writing
    # makes the output more aestheically pleasing. It works in this
    # case because the Tensor's are a single dimension.
    df.astype({"loss": float}).to_csv(sys.stdout, index=False)
