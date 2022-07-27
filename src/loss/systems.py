from . import (
    AbstractLoss,
    BasicValidationLoss,
    BatchInfo,
    BoxPreserver,
    BoxTransformer,
    CompositeLoss,
    DetectionHardMinedCELoss,
    DetectionLossAggregator,
    DetectionSmoothL1Loss,
    LossTrail,
    ModifiedDetectionHardMinedCELoss,
)


class AbstractSystemLoss(AbstractLoss):
    """Composition of two detection losses that also transforms boxes

    Notes
    -----
    This is a special composition of detection losses that can act as
    a single loss. It exists because box transformation is coupled
    with loss calculation. If that heuristic went away, so could this
    class: CompositeLoss would be good enough.
    """

    def __init__(
        self,
        composition: tuple,
        transformer: BoxTransformer = None,
    ):
        """

        Parameters
        ----------
        location_weight: float, optional
            Weight applied to localization loss. Defaults to 1
        confidence_weight: float, optional
            Weight applied to localization loss. Defaults to 1
        transformer: BoxTransformer, optional
            Box transformation class. `None` is converted to `BoxPreserver`
        """

        super().__init__()

        self.btrans = BoxPreserver() if transformer is None else transformer
        self.composition = composition

        # Aggregation should take place within the loss calculator. In
        # fact there is logic within the loss calculator type for
        # doing so. However, that logic cannot be used here because
        # aggregation takes place on a different ground truth than
        # what is passed to the calculator. What is done here is an
        # abuse of the functionality that should not be repeated once
        # transformation takes place elsewhere.
        self.aggregate = DetectionLossAggregator()

    def forward(self, pr: BatchInfo, gt: BatchInfo):
        """

        Parameters
        ----------
        pr: BatchInfo
            Predicted information
        gt: BatchInfo
            Ground truth information

        Returns
        -------
        LossTrail
        """

        trail = LossTrail()
        gt_ = self.btrans.expand(gt)

        for (i, j) in self.composition:
            loss = self.aggregate(pr, gt, i(pr, gt_))
            trail += loss._replace(weight=j)

        return trail


class StandardDetectionLoss(AbstractSystemLoss):
    def __init__(
        self,
        location_weight: float = 1.0,
        confidence_weight: float = 1.0,
        transformer: BoxTransformer = None,
    ):
        """

        Parameters
        ----------
        location_weight: float, optional
            Weight applied to localization loss. Defaults to 1
        confidence_weight: float, optional
            Weight applied to localization loss. Defaults to 1
        transformer: BoxTransformer, optional
            Box transformation class. `None` is converted to `BoxPreserver`
        """

        composition = (
            (DetectionSmoothL1Loss(), location_weight),
            (DetectionHardMinedCELoss(), confidence_weight),
        )
        super().__init__(composition, transformer)


class ModifiedDetectionLoss(AbstractSystemLoss):
    """Modified Detection Loss that uses ModifiedHardMinedLoss
    This modification allows for background images to be passed to
    the object detection model

    """

    def __init__(
        self,
        location_weight: float = 1.0,
        confidence_weight: float = 1.0,
        neg_pos_ratio: float = 3.0,
        min_neg_num: int = 5,
        transformer: BoxTransformer = None,
    ):
        """

        Parameters
        ----------
        location_weight: float, optional
            Weight applied to localization loss. Defaults to 1
        confidence_weight: float, optional
            Weight applied to localization loss. Defaults to 1
        neg_pos_ratio: float, optional
            Ratio of negative to positive samples. Defaults to 3
        min_neg_num: int, optional
            Minimum number of negative samples. Defaults to 5
        transformer: BoxTransformer, optional
            Box transformation class. `None` is converted to `BoxPreserver`
        """

        ModifiedDetectionHardMinedCELoss_ = ModifiedDetectionHardMinedCELoss(
            neg_pos_ratio=neg_pos_ratio,
            min_neg_num=min_neg_num,
        )
        composition = (
            (DetectionSmoothL1Loss(), location_weight),
            (ModifiedDetectionHardMinedCELoss_, confidence_weight),
        )
        super().__init__(composition, transformer)


class DefaultSystemLoss(AbstractLoss):
    """A basic "system" (detection plus validation) loss

    Notes
    -----
    This exists to make it easy to have a vanilla loss that requires
    very little from the Hydra configuration framework. If
    StandardDetectionLoss goes away, this can be constructed using
    detection losses individually.
    """

    def __init__(
        self,
        detection_weight: float = 1.0,
        validation_weight: float = 1.0,
        transformer: BoxTransformer = None,
    ):
        """

        Parameters
        ----------
        detection_weight: float, optional
            Weight applied to detection loss. Defaults to 1
        validation_weight: float, optional
            Weight applied to validation loss. Defaults to 1
        transformer: BoxTransformer, optional
            Passed directly to `StandardDetectionLoss`
        """

        super().__init__()

        iterable = (
            (StandardDetectionLoss(transformer=transformer), detection_weight),
            (BasicValidationLoss(), validation_weight),
        )
        self.composite = None
        for i in iterable:
            self.composite = CompositeLoss(*i, self.composite)

    def forward(self, pr: BatchInfo, gt: BatchInfo):
        """

        Parameters
        ----------
        pr: BatchInfo
            Predicted information
        gt: BatchInfo
            Ground truth information

        Returns
        -------
        LossTrail
            See `CompositeLoss::_forward`
        """

        return self.composite(pr, gt)
