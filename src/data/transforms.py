import types
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from numpy import random
from torchvision import transforms as tv_transforms

# Augmentations written by us


class Compose:
    """Composes several transforms together.

    Parameters
    ----------
        transforms: list, list of transforms to use.

    Examples
    --------
        >>> transforms.Compose([
        >>>     transforms.Resize(300, 300),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms: Sequence) -> None:
        """Initialize the transform with an iterable of transforms to compose.

        Parameters
        ----------
        transforms : Sequence
            Iterable of transforms to compose.
        """
        self.transforms = transforms

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Apply all the transforms passed in the given order.

        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes, by default None

        Returns
        -------
        img : np.ndarray
            transformed numpy array of the image.
        bbox_class : Optional[np.ndarray]
            transformed numpy array of the classes of bounding boxes of the image if bounding boxes
            present, else None
        bbox_coord : Optional[np.ndarray]
            transformed numpy array of the coordinates of bounding boxes of the image if bounding
            boxes present, else None. Bounding box coordinates are in [x1, y1, x2, y2] absolute
            pixel format. Where (x1,y1) is the top-left corner of the box and (x2,y2) is the
            bottom-right corner of the box.
        label_class : Optional[np.ndarray]
            transformed numpy array of the label class of the image if labels present, else None.
        label_value : Optional[np.ndarray]
            transformed numpy array of the label value corresponding to label class of the image if
            labels present, else None.
        """
        for transform in self.transforms:
            img, bbox_class, bbox_coord, label_class, label_value = transform(
                img, bbox_class, bbox_coord, label_class, label_value
            )
        return img, bbox_class, bbox_coord, label_class, label_value


class BaseImageOnlyAlbumentationsAug:
    def __init__(self):
        self.transform = None

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Apply the transform.

        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes. Bounding box corrdinates are in [x, y, w, h] absolute
            pixel format. Where (x,y) is the top-left corner of the image and (w,h) are the width
            and height of the image respectively, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes, by default
            None

        Returns
        -------
        img : np.array
            transformed image.
        bbox_class : Optional[np.array]
            transformed classes of bounding boxes of the image if bounding boxes present,
            else None
        bbox_coord : Optional[np.array]
            transformed coordinates of bounding boxes of the image if bounding boxes present, else
            None.
        label_class : Optional[np.array]
            label class of the image if labels present, else None. This is not altered.
        label_value : Optional[np.array]
            label value corresponding to label class of the image if labels present, else None. This
             is not altered.
        """
        transformed_img = self.transform(image=img)["image"]
        return transformed_img, bbox_class, bbox_coord, label_class, label_value


class BaseImageBboxAlbumentationsAug:
    def __init__(self):
        self.transform = None
        self.bbox_to_percent = None
        self.bbox_to_absolute = None

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Apply the transform.

        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes. Bounding box corrdinates are in [x, y, w, h] absolute
            pixel format. Where (x,y) is the top-left corner of the image and (w,h) are the width
            and height of the image respectively, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes, by default
            None

        Returns
        -------
        img : np.array
            transformed image.
        bbox_class : Optional[np.array]
            transformed classes of bounding boxes of the image if bounding boxes present,
            else None
        bbox_coord : Optional[np.array]
            transformed coordinates of bounding boxes of the image if bounding boxes present, else
            None.
        label_class : Optional[np.array]
            label class of the image if labels present, else None. This is neither required not
            altered.
        label_value : Optional[np.array]
            label value corresponding to label class of the image if labels present, else None. This
             is not altered.
        """
        if bbox_coord is None:
            transformed_img = self.transform(image=img)["image"]
            return transformed_img, bbox_class, bbox_coord, label_class, label_value

        _, _, percent_bbox_coord, _, _ = self.bbox_to_percent(img=img, bbox_coord=bbox_coord)
        percent_bbox_coord_class = np.hstack((percent_bbox_coord, np.expand_dims(bbox_class, 1)))

        transformed = self.transform(image=img, bboxes=percent_bbox_coord_class)
        transformed_img = transformed["image"]
        transformed_percent_bbox_coord_class = np.array(transformed["bboxes"])
        transformed_bbox_class = transformed_percent_bbox_coord_class[:, -1]
        transformed_percent_bbox_coord = transformed_percent_bbox_coord_class[:, :4]
        _, _, transformed_bbox_coord, _, _ = self.bbox_to_absolute(
            img=transformed_img, bbox_coord=transformed_percent_bbox_coord
        )
        return (
            transformed_img,
            transformed_bbox_class,
            transformed_bbox_coord,
            label_class,
            label_value,
        )


class ToTensor:
    """Converts inputs to Tensor. The numpy HWC image is converted to pytorch CHW tensor."""

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Transform call to convert the inputs to torch Tensor.

        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes, by default
            None

        Returns
        -------
        img : torch.Tensor
            transformed torch tensor of the image.
        bbox_class : Optional[torch.Tensor]
            transformed torch tensor of the classes of bounding boxes of the image if bounding boxes
             present, else None
        bbox_coord : Optional[torch.Tensor]
            transformed torch tensor of the coordinates of bounding boxes of the image if bounding
            boxes present, else None.
        label_class : Optional[torch.Tensor]
            transformed torch tensor of the label class of the image if labels present, else None.
        label_value : Optional[torch.Tensor]
            tranformed torch tensor of the label value corresponding to label class of the image if
            labels present, else None.
        """
        transform = ToTensorV2()
        transformed_img = transform(image=img)["image"]
        bbox_class = None if bbox_class is None else torch.Tensor(bbox_class)
        bbox_coord = None if bbox_coord is None else torch.FloatTensor(bbox_coord)
        label_class = None if label_class is None else torch.IntTensor(label_class)
        label_value = None if label_value is None else torch.Tensor(label_value)
        return transformed_img, bbox_class, bbox_coord, label_class, label_value


class ToNumpy:
    """Converts inputs to Numpy. The torch CHW tensor is converted to numpy HWC array."""

    def __call__(
        self,
        img: torch.Tensor,
        bbox_class: Optional[torch.Tensor] = None,
        bbox_coord: Optional[torch.Tensor] = None,
        label_class: Optional[torch.Tensor] = None,
        label_value: Optional[torch.Tensor] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Converts all inputs to numpy array
        Parameters
        ----------
        img : torch.Tensor
            The input image
        bbox_class : Optional[torch.Tensor], optional
            classes of the bounding boxes, by default None
        bbox_coord : Optional[torch.Tensor], optional
            location of the bounding boxes, by default None
        label_class : Optional[torch.Tensor], optional
            classes of the image labels, by default None
        label_value : Optional[torch.Tensor], optional
            value of the image label classes, by default None
        Returns
        -------
        transformed_img : np.ndarray
            transformed numpy array of the image.
        bbox_class : Optional[np.ndarray]
            transformed numpy array of the classes of bounding boxes of the image if bounding boxes
            present, else None.
        bbox_coord : Optional[np.ndarray]
            transformed numpy array of the coordinates of bounding boxes of the image if bounding
            boxes present, else None.
        label_class : Optional[np.ndarray]
            transformed numpy array of the label class of the image if labels present, else None.
        label_value : Optional[np.ndarray]
            transformed numpy array of the label value corresponding to label class of the image if
            labels present, else None.
        """
        transformed_img = img.permute(1, 2, 0).numpy()
        bbox_class = None if bbox_class is None else bbox_class.numpy()
        bbox_coord = None if bbox_coord is None else bbox_coord.numpy()
        label_class = None if label_class is None else label_class.numpy()
        label_value = None if label_value is None else label_value.numpy()
        return transformed_img, bbox_class, bbox_coord, label_class, label_value


class XYWHToXYX2Y2:
    """Converts bbox from XYWH format to XYX2Y2 format. Opposite of XYX2Y2ToXYWH transform."""

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Transform call to convert boxes from XYWH format to X2Y2 format.

        Parameters
        ----------
        img : np.ndarray
            The input image not used or altered
        bbox_class : Optional[np.ndarray], optional
            location of the bounding boxes not used or altered, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels this is not used or altered, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes this is not used or altered, by default None

        Returns
        -------
        img : np.ndarray
            unaltered numpy array of the image.
        bbox_class : Optional[np.ndarray]
            unaltered classes of the bounding boxes.
        bbox_coord : Optional[np.ndarray]
            transformed numpy array of the coordinates of bounding boxes of the image if bounding
            boxes present, else None
        label_class : Optional[np.ndarray]
            unaltered classes of image labels.
        label_value : Optional[np.ndarray]
            unaltered values of image labels.
        """
        if bbox_coord is None:
            return img, bbox_class, bbox_coord, label_class, label_value
        bbox_coord[:, 2] = bbox_coord[:, 0] + bbox_coord[:, 2]
        bbox_coord[:, 3] = bbox_coord[:, 1] + bbox_coord[:, 3]
        return img, bbox_class, bbox_coord, label_class, label_value


class XYX2Y2ToXYWH:
    """Converts bbox from XYX2Y2 format to XYWH format. Opposite of XYWHToXYX2Y2 transform."""

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Transform call to convert boxes from X2Y2 format to XYWH format.

        Parameters
        ----------
        img : np.ndarray
            The input image not used or altered.
        bbox_class : Optional[np.ndarray], optional
            location of the bounding boxes not used or altered, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels this is not used or altered, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes this is not used or altered, by default None

        Returns
        -------
        img : np.ndarray
            unaltered numpy array of the image.
        bbox_class : Optional[np.ndarray]
            unaltered classes of the bounding boxes.
        bbox_coord : Optional[np.ndarray]
            transformed numpy array of the coordinates of bounding boxes of the image if bounding
            boxes present, else None.
        label_class : Optional[np.ndarray]
            unaltered classes of image labels.
        label_value : Optional[np.ndarray]
            unaltered values of image labels.
        """
        if bbox_coord is None:
            return img, bbox_class, bbox_coord, label_class, label_value
        bbox_coord[:, 2] = bbox_coord[:, 2] - bbox_coord[:, 0]
        bbox_coord[:, 3] = bbox_coord[:, 3] - bbox_coord[:, 1]
        return img, bbox_class, bbox_coord, label_class, label_value


class Resize(BaseImageBboxAlbumentationsAug):
    """Resize images and boxes in absolute coordinates"""

    def __init__(self, height: int = 512, width: int = 512):
        """Initalize the transform with the height and width to resize to.

        Parameters
        ----------
        height : int, optional
            target height, by default 512
        width : int, optional
            target width, by default 512
        """
        self.transform = A.Resize(height, width)
        self.bbox_to_percent = ToPercentCoords()
        self.bbox_to_absolute = ToAbsoluteCoords()


class RandomPerspective(BaseImageBboxAlbumentationsAug):
    """Applies albumentations.augmentations.geometric.transforms.Perspective"""

    def __init__(self, p: float = 0.5, *args, **kwargs) -> None:
        """Initilizing the transform with application propbability ``p``.

        Parameters
        ----------
        p : float
            The probability of applying the transform to the input, by default 0.5

        Returns
        -------
        """
        self.transform = A.augmentations.geometric.transforms.Perspective(
            p=p, keep_size=True, *args, **kwargs
        )
        self.bbox_to_percent = ToPercentCoords()
        self.bbox_to_absolute = ToAbsoluteCoords()


class RandomAffine:
    """Applies torchvision.transforms.RandomAffine"""

    def __init__(self, degrees: int = 90, *args, **kwargs) -> None:
        """Initialize the transform to apply random affine transform

        Parameters
        ----------
        degrees : int, optional
            Range of degrees to select from, by default 90
        """
        self.transform = tv_transforms.RandomAffine(degrees, *args, **kwargs)
        self.ToTensor = ToTensor()
        self.ToNumpy = ToNumpy()

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Transform call to randomly apply affine transform.

        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes; this is not used or altered, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes; this is not used or altered, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels; this is not used or altered, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes; this is not used or altered, by default None

        Returns
        -------
        numpy_img : np.ndarray
            trannsformed numpy array of the image.
        bbox_class : Optional[np.ndarray]
            unaltered classes of the bounding boxes.
        bbox_coord : Optional[np.ndarray]
            unaltered coords of the bounding boxes.
        label_class : Optional[np.ndarray]
            unaltered classes of image labels.
        label_value : Optional[np.ndarray]
            unaltered values of image labels.
        """
        tensor_img, _, _, _, _ = self.ToTensor(img)
        transformed_img = self.transform(tensor_img / 255.0) * 255.0
        numpy_img, _, _, _, _ = self.ToNumpy(transformed_img)
        return numpy_img, bbox_class, bbox_coord, label_class, label_value


class RandomInvert:
    """Applies torchvision.transforms.RandomInvert"""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize the transform to apply random invert transform

        Parameters
        ----------
        p : float, optional
            probability of the image being color inverted, by default 0.5
        """
        self.transform = tv_transforms.RandomInvert(p)
        self.ToTensor = ToTensor()
        self.ToNumpy = ToNumpy()

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Transform call to invert color

        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes; this is not used or altered, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes; this is not used or altered, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels; this is not used or altered, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes; this is not used or altered, by default None

        Returns
        -------
        numpy_img : np.ndarray
            trannsformed numpy array of the image.
        bbox_class : Optional[np.ndarray]
            unaltered classes of the bounding boxes.
        bbox_coord : Optional[np.ndarray]
            unaltered coords of the bounding boxes.
        label_class : Optional[np.ndarray]
            unaltered classes of image labels.
        label_value : Optional[np.ndarray]
            unaltered values of image labels.
        """
        tensor_img, _, _, _, _ = self.ToTensor(img)
        transformed_img = self.transform(tensor_img / 255.0) * 255.0
        numpy_img, _, _, _, _ = self.ToNumpy(transformed_img)
        return numpy_img, bbox_class, bbox_coord, label_class, label_value


class RandomAutocontrast:
    """Applies torchvision.transforms.RandomAutocontrast"""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize the transform to apply random auto-contrast transform

        Parameters
        ----------
        p : float, optional
            probability of the image being autocontrasted, by default 0.5
        """
        self.transform = tv_transforms.RandomAutocontrast(p)
        self.ToTensor = ToTensor()
        self.ToNumpy = ToNumpy()

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Transform call to randomly apply auto contrast

        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes; this is not used or altered, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes; this is not used or altered, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels; this is not used or altered, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes; this is not used or altered, by default None

        Returns
        -------
        numpy_img : np.ndarray
            trannsformed numpy array of the image.
        bbox_class : Optional[np.ndarray]
            unaltered classes of the bounding boxes.
        bbox_coord : Optional[np.ndarray]
            unaltered coords of the bounding boxes.
        label_class : Optional[np.ndarray]
            unaltered classes of image labels.
        label_value : Optional[np.ndarray]
            unaltered values of image labels.
        """
        tensor_img, _, _, _, _ = self.ToTensor(img)
        transformed_img = self.transform(tensor_img / 255.0) * 255.0
        numpy_img, _, _, _, _ = self.ToNumpy(transformed_img)
        return numpy_img, bbox_class, bbox_coord, label_class, label_value


class RandomRotation(BaseImageBboxAlbumentationsAug):
    """Applies albumentations.augmentations.geometric.rotate.Rotate"""

    def __init__(self, p: float = 0.5, *args, **kwargs) -> None:
        """Initilizing the transform with application propbability ``p``.

        Parameters
        ----------
        p : float
            The probability of applying the transform to the input, by default 0.5

        Returns
        -------
        """
        self.transform = A.augmentations.geometric.rotate.Rotate(
            p=p, border_mode=cv2.BORDER_CONSTANT, *args, **kwargs
        )
        self.bbox_to_percent = ToPercentCoords()
        self.bbox_to_absolute = ToAbsoluteCoords()


class RandomApply:
    """Randomly apply a list of transforms."""

    def __init__(self, transforms: List[Callable], p: float = 0.5):
        """Initialize with te list of transforms to be applied and probability with which to apply.

        Parameters
        ----------
        transforms : List[Callable]
            List of transforms
        p : float, optional
            probablity of applying the transforms, by default 0.5
        """
        self.transforms = transforms
        self.p = p

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Transform call to randomly select whether to apply transforms

        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes, by default None
        Returns
        -------
        img : np.ndarray
            transformed numpy array of the image.
        bbox_class : Optional[np.ndarray]
            transformed numpy array of the classes of bounding boxes of the image if bounding boxes
            present, else None.
        bbox_coord : Optional[np.ndarray]
            transformed numpy array of the coordinates of bounding boxes of the image if bounding
            boxes present, else None.
        label_class : Optional[np.ndarray]
            transformed numpy array of the label class of the image if labels present, else None.
        label_value : Optional[np.ndarray]
            transformed numpy array of the label value corresponding to label class of the image if
            labels present, else None.
        """
        if random.random() < self.p:
            for transform in self.transforms:
                img, bbox_class, bbox_coord, label_class, label_value = transform(
                    img, bbox_class, bbox_coord, label_class, label_value
                )
        return img, bbox_class, bbox_coord, label_class, label_value


class Normalize(BaseImageOnlyAlbumentationsAug):
    """Normalizes images. Uses albumentations.augmentations.transforms.Normalize. Mean and Std are
    supposed to be between 0 and 1. Normalization is applied by the formula:
    ``img = (img - mean * max_pixel_value) / (std * max_pixel_value) where max_pixel_value = 255``
    """

    def __init__(
        self,
        mean: Tuple[float, float, float] = [0.485, 0.456, 0.406],
        std: Tuple[float, float, float] = [0.229, 0.224, 0.225],
        max_pixel_value: float = 255.0,
    ):
        """Initialize the transform with the mean and standard deviation to normalize to.

        Parameters
        ----------
        mean : Tuple[float, float, float], optional
            Mean for the three channels in 0 to 1 pixel value, by default [0.485, 0.456, 0.406]
        std : Tuple[float, float, float], optional
            Standard Deviation for the three channels in 0 to 1 pixel value, by default
            [0.229, 0.224, 0.225]
        max_pixel_value : float, optional
            Maximum pixel value, by default 255.0
        """
        self.transform = A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value)


class StandardizeImage0to1(BaseImageOnlyAlbumentationsAug):
    """Converts images from 0 to 255 to 0 to 1."""

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        return (img / 255.0), bbox_class, bbox_coord, label_class, label_value


class GaussNoise(BaseImageOnlyAlbumentationsAug):
    """Adds noise to images. Uses albumentations.augmentations.transforms.GaussNoise."""

    def __init__(self, var_limit: Union[Tuple[float, float], float], mean: float, p: float = 0.1):
        """Initialize the transform with variance range, mean and the probability.

        Parameters
        ----------
        var_limit : Union[Tuple[float, float], float]
            variance range for noise. If var_limit is a single float, the range will be ``(0,
            var_limit)``.
        mean : float
            mean of the noise
        p : float, optional
            probability of applying the transform, by default 0.1
        """
        self.transform = A.GaussNoise(var_limit=var_limit, mean=mean, p=p)


class RandomBlur(BaseImageOnlyAlbumentationsAug):
    """Choose one out of Blur, GaussianBlur, MotionBlur to randomly apply blur to the image."""

    def __init__(self, strength: int = 1, p: float = 0.1):
        """Initialize transform with blur strength and probability to apply.

        Parameters
        ----------
        strength : int, optional
            strength of blur augmentation. Should be an odd integer, by default 1
        p : float, optional
            probability of applying the transform, by default 0.1
        """
        self.transform = A.OneOf(
            [
                A.Blur(blur_limit=7 * strength, p=1),
                A.GaussianBlur(blur_limit=(3 * strength, 7 * strength), p=1),
                A.MotionBlur(blur_limit=(3 * strength, 7 * strength), p=1),
            ],
            p=p,
        )


class CLAHE(BaseImageOnlyAlbumentationsAug):
    """CLAHE: Apply Contrast Limited Adaptive Histogram Equalization to the input image."""

    def __init__(
        self,
        clip_limit: Optional[float] = 4.0,
        tile_grid_size: Optional[tuple] = (8, 8),
        always_apply: Optional[bool] = False,
        p: Optional[float] = 0.5,
    ):
        """Initialize the transform with clip limit, tile grid size and probability to apply.

        Args:
            clip_limit : float, optional
                upper threshold value for contrast limiting. If clip_limit is a single float
                value, the range will be (1, clip_limit). Default: (1, 4). Defaults to 4.0.
            tile_grid_size : tuple, optional
                size of grid for histogram equalization. Default: (8, 8). Defaults to (8, 8).
            always_apply : bool, optional
                if True, apply the transform on each image.
            p : float, optional
                probability of applying the transform, by default 0.5.
        """
        self.transform = A.CLAHE(
            clip_limit=clip_limit, tile_grid_size=tile_grid_size, always_apply=always_apply, p=p
        )


class ColorJitter(BaseImageOnlyAlbumentationsAug):
    """Color Jitter from Albumentation.

    Randomly changes the brightness, contrast, and saturation of an image. Compared to ColorJitter
    from torchvision, this transform gives a little bit different results because Pillow
    (used in torchvision) and OpenCV (used in Albumentations) transform an image to HSV
    format by different formulas. Another difference - Pillow uses uint8 overflow, but we
    use value saturation.
    """

    def __init__(
        self,
        brightness: Optional[float] = 0.2,
        contrast: Optional[float] = 0.2,
        saturation: Optional[float] = 0.2,
        hue: Optional[float] = 0.2,
        always_apply: Optional[float] = False,
        p: Optional[float] = 0.5,
    ):
        """Initialize the transform with brightness, contrast, saturation, hue,
            always_apply and probability.

        Parameters
        ----------
        brightness : float, optional
            How much to jitter brightness. brightness_factor is chosen uniformly
            from [max(0, 1 - brightness), 1 + brightness] or the given [min, max].
            Should be non negative numbers.
        contrast : float, optional
            How much to jitter contrast. contrast_factor is chosen uniformly from
            [max(0, 1 - contrast), 1 + contrast] or the given [min, max]. Should be
            non negative numbers.
        saturation : float, optional
            How much to jitter saturation. saturation_factor is chosen uniformly
            from [max(0, 1 - saturation), 1 + saturation] or the given
            [min, max]. Should be non negative numbers.
        hue : float, optional
            How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue] or
            the given [min, max]. Should have 0 <= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
        always_apply : bool, optional
            If True, apply the transform on every image. If False, only apply the transform
        p : float, optional
            probability of applying the transform, by default 0.5
        """
        self.transform = A.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            always_apply=always_apply,
            p=p,
        )


class Equalize(BaseImageOnlyAlbumentationsAug):
    """Transform to equalize the image histogram. Taken from albumentation."""

    def __init__(
        self,
        mode: Optional[str] = "cv",
        by_channels: Optional[bool] = True,
        mask: Optional[np.ndarray] = None,
        mask_params: Union[None, list, str] = (),
        always_apply: Optional[bool] = False,
        p: Optional[float] = 0.5,
    ):
        """Initiliaze the transform with the mode, by_channels, mask, mask_params and probability.

        Parameters
        ----------
        mode : str, optional
            {'cv', 'pil'}. Use OpenCV or Pillow equalization method.
        by_channels : bool, optional
            If True, use equalization by channels separately, else convert
            image to YCbCr representation and use equalization by Y channel.
        mask : np.ndarray, optional
            If given, only the pixels selected by the mask are included in the analysis.
            Maybe 1 channel or 3 channel array or callable. Function signature must
            include image argument.
        mask_params : Union[list, str], optional
            Params for mask function.
        always_apply : bool, optional
            If True, apply transform on every image.
        p : float, optional
            probability of applying the transform, by default 0.5
        """
        self.transform = A.Equalize(
            mode=mode,
            by_channels=by_channels,
            mask=mask,
            mask_params=mask_params,
            always_apply=always_apply,
            p=p,
        )


class FancyPCA(BaseImageOnlyAlbumentationsAug):
    """Augment RGB image using FancyPCA from Krizhevsky's paper
    'ImageNet Classification with Deep Convolutional Neural Networks'"""

    def __init__(
        self,
        alpha: Optional[float] = 0.1,
        always_apply: Optional[bool] = False,
        p: Optional[float] = 0.5,
    ):
        """Initialize the tranform with probability to apply.

        Parameters
        ----------
        alpha : float, optional
            how much to perturb/scale the eigen vecs and vals. scale is
            samples from gaussian distribution (mu=0, sigma=alpha)
        always_apply : bool, optional
            if True, apply the transform on each image, by default False
        p : float, optional
            probability of applying the transform, by default 0.5
        """
        self.transform = A.FancyPCA(alpha=alpha, always_apply=always_apply, p=p)


class HorizontalFlip(BaseImageBboxAlbumentationsAug):
    """Transform to flip the image along the vertical axis."""

    def __init__(self, p: float = 0.25):
        """Initialize the transform with probability to apply.

        Parameters
        ----------
        p : float, optional
            probability of applying the transform, by default 0.25
        """
        self.transform = A.HorizontalFlip(p=p)
        self.bbox_to_percent = ToPercentCoords()
        self.bbox_to_absolute = ToAbsoluteCoords()


class VerticalFlip(BaseImageBboxAlbumentationsAug):
    """Transform to flip the image along the horizontal axis."""

    def __init__(self, p: float = 0.25):
        """Initialize the transform with probability to apply.

        Parameters
        ----------
        p : float, optional
            probability of applying the transform, by default 0.25
        """
        self.transform = A.VerticalFlip(p=p)
        self.bbox_to_percent = ToPercentCoords()
        self.bbox_to_absolute = ToAbsoluteCoords()


class BlurGray:
    """Operates only on the image. Converts it to single channel grayscale and applies gaussian blur
    ``self.n`` times. Not to be used independently in the experiment transform list as this alters
    the number of output channels and may break transforms following it."""

    def __init__(self, n: int = 1):
        """Initialise transform with the number of blur cycles ``n``.

        Parameters
        ----------
        n : int, optional
            number of blur cycles, by default 1
        """
        self.n = n

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """transform call to only modify the image to single channel grayscale and apply gausian
        blur ``self.n`` times.

        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes this is not required, used or altered, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes this is not required, used or altered, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels this is not required, used or altered, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes this is not required, used or altered, by default None

        Returns
        -------
        img : np.array
            transformed numpy array of the image.
        bbox_class : Optional[np.array]
            numpy array of the classes of bounding boxes of the image if bounding boxes present,
            else None
        bbox_coord : Optional[np.array]
            numpy array of the coordinates of bounding boxes of the image if bounding boxes
            present, else None. Bounding box corrdinates are in [x, y, w, h] absolute pixel
            format. Where (x,y) is the top-left corner of the image and (w,h) are the width and
            height of the image respectively.
        label_class : Optional[np.array]
            numpy array of the label class of the image if labels present, else None.
        label_value : Optional[np.array]
            numpy array of the label value corresponding to label class of the image if labels
            present, else None.
        """

        transformed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for i in range(self.n):
            transformed_img = cv2.GaussianBlur(transformed_img, (5, 5), cv2.BORDER_DEFAULT)
        return transformed_img, bbox_class, bbox_coord, label_class, label_value


class ClassicSegMean:
    """Transform that uses mean classical segmentation to add a segmentation mask as a new last
    channel to the image. ``self.blur_n`` can be set to inform the number of blur cycles to be used
    before segmenting the image. This transform has to be the last transform before ``ToTensor`` in
    the experiment transforms list."""

    def __init__(self, blur_n: int = 10):
        """Initialise transform with the number of blur cycles ``blur_n``.

        Parameters
        ----------
        blur_n : int, optional
            number of blur cycles, by default 10
        """
        self.blur_n = blur_n
        self.blur_gray = BlurGray(self.blur_n)

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """transform call to only modify the image to claculate and add the segmentation mask as the
         new last channel.

        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes this is not required, used or altered, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes this is not required, used or altered, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels this is not required, used or altered, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes this is not required, used or altered, by default None

        Returns
        -------
        img : np.array
            transformed numpy array of the image.
        bbox_class : Optional[np.array]
            numpy array of the classes of bounding boxes of the image if bounding boxes present,
            else None
        bbox_coord : Optional[np.array]
            numpy array of the coordinates of bounding boxes of the image if bounding boxes
            present, else None. Bounding box corrdinates are in [x, y, w, h] absolute pixel
            format. Where (x,y) is the top-left corner of the image and (w,h) are the width and
            height of the image respectively.
        label_class : Optional[np.array]
            numpy array of the label class of the image if labels present, else None.
        label_value : Optional[np.array]
            numpy array of the label value corresponding to label class of the image if labels
            present, else None.
        """
        # convert to grayscale and apply blur
        gray_img, _, _, _, _ = self.blur_gray(img.copy())
        # find threshold
        th = np.mean(gray_img)
        # apply threshold
        _, seg_channel = cv2.threshold(gray_img, th, 255, cv2.THRESH_BINARY_INV)
        # add channel
        transformed_img = np.concatenate((img, np.expand_dims(seg_channel, axis=-1)), axis=-1)
        return transformed_img, bbox_class, bbox_coord, label_class, label_value


class ClassicSegOtsu:
    """Transform that uses Otsu classical segmentation to add a segmentation mask as a new last
    channel to the image. ``self.blur_n`` can be set to inform the number of blur cycles to be used
    before segmenting the image. This transform has to be the last transform before ``ToTensor`` in
    the experiment transforms list."""

    def __init__(self, blur_n: int = 10):
        """Initialise transform with the number of blur cycles ``blur_n``.

        Parameters
        ----------
        blur_n : int, optional
            number of blur cycles, by default 10
        """
        self.blur_n = blur_n
        self.blur_gray = BlurGray(self.blur_n)

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """transform call to only modify the image to claculate and add the segmentation mask as the
         new last channel.

        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes this is not required, used or altered, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes this is not required, used or altered, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels this is not required, used or altered, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes this is not required, used or altered, by default None

        Returns
        -------
        img : np.array
            transformed numpy array of the image.
        bbox_class : Optional[np.array]
            numpy array of the classes of bounding boxes of the image if bounding boxes present,
            else None
        bbox_coord : Optional[np.array]
            numpy array of the coordinates of bounding boxes of the image if bounding boxes
            present, else None. Bounding box corrdinates are in [x, y, w, h] absolute pixel
            format. Where (x,y) is the top-left corner of the image and (w,h) are the width and
            height of the image respectively.
        label_class : Optional[np.array]
            numpy array of the label class of the image if labels present, else None.
        label_value : Optional[np.array]
            numpy array of the label value corresponding to label class of the image if labels
            present, else None.
        """
        # convert to grayscale and apply blur
        gray_img, _, _, _, _ = self.blur_gray(img.copy())
        # find and apply threshold
        _, seg_channel = cv2.threshold(
            gray_img.astype(np.uint8), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        # add channel
        seg_channel = seg_channel.astype(np.float32)
        transformed_img = np.concatenate((img, np.expand_dims(seg_channel, axis=-1)), axis=-1)
        return transformed_img, bbox_class, bbox_coord, label_class, label_value


# Augmentations taken from
# https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py


def intersect(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    """Calculate the area under intersection of paiwise boxes of ``box_a`` from ``box_b``.
    Parameters
    ----------
    box_a : np.ndarray
        multiple bounding boxes, Shape: [num_boxes,4]
    box_b : np.ndarray
        single bounding box, Shape: [4]

    Returns
    -------
    out : np.ndarray
        intersection area
    """
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    """Compute the jaccard overlap of two sets of boxes. The jaccard overlap is simply the
    intersection over union of two boxes.
    ``A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)``

    Parameters
    ----------
    box_a : np.ndarray
        multiple bounding boxes, Shape: [num_boxes,4]
    box_b : np.ndarray
        single bounding box, Shape: [4]

    Returns
    -------
    out : np.ndarray
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])  # [A,B]
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd: Callable):
        """Initialize with the lambda function to be used as a transform.

        Parameters
        ----------
        lambd : Callable
            lambda function
        """
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Transform call to apply the lambda function as a transform.

        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes, by default None
        Returns
        -------
        img : np.ndarray
            transformed numpy array of the image.
        bbox_class : Optional[np.ndarray]
            transformed numpy array of the classes of bounding boxes of the image if bounding boxes
            present, else None
        bbox_coord : Optional[np.ndarray]
            transformed numpy array of the coordinates of bounding boxes of the image if bounding
            boxes present, else None. Bounding box coordinates are in [x1, y1, x2, y2] absolute
            pixel format. Where (x1,y1) is the top-left corner of the box and (x2,y2) is the
            bottom-right corner of the box.
        label_class : Optional[np.ndarray]
            transformed numpy array of the label class of the image if labels present, else None.
        label_value : Optional[np.ndarray]
            transformed numpy array of the label value corresponding to label class of the image if
            labels present, else None.
        """
        return self.lambd(img, bbox_class, bbox_coord, label_class, label_value)


class ConvertFromInts(object):
    """Converts all inputs to have numeric values of type np.float32"""

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Transform call to convert inputs to float
        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes, by default None
        Returns
        -------
        img : np.ndarray
            transformed numpy array of the image.
        bbox_class : Optional[np.ndarray]
            transformed numpy array of the classes of bounding boxes of the image if bounding boxes
            present, else None
        bbox_coord : Optional[np.ndarray]
            transformed numpy array of the coordinates of bounding boxes of the image if bounding
            boxes present, else None. Bounding box coordinates are in [x1, y1, x2, y2] absolute
            pixel format. Where (x1,y1) is the top-left corner of the box and (x2,y2) is the
            bottom-right corner of the box.
        label_class : Optional[np.ndarray]
            transformed numpy array of the label class of the image if labels present, else None.
        label_value : Optional[np.ndarray]
            transformed numpy array of the label value corresponding to label class of the image if
            labels present, else None.
        """
        bbox_class = None if bbox_class is None else np.float32(bbox_class)
        bbox_coord = None if bbox_coord is None else np.float32(bbox_coord)
        label_class = None if label_class is None else np.float32(label_class)
        label_value = None if label_value is None else np.float32(label_value)
        return img.astype(np.float32), bbox_class, bbox_coord, label_class, label_value


class SubtractMeans(object):
    """Subtracts mean from every channel of the image"""

    def __init__(self, mean: Union[int, Tuple[int, int, int]]):
        """Initializes mean

        Parameters
        ----------
        mean : Union[int, Tuple[int, int, int]]
            mean which is subtracted from each channel of the image
        """
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Transform call to subtract means from the image channels.

        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes this is not used or altered, by default
            None
        label_class : Optional[np.ndarray], optional
            classes of the image labels this is not used or altered, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes this is not used or altered, by default None

        Returns
        -------
        img : np.ndarray
            transformed numpy array of the image.
        bbox_class : Optional[np.ndarray]
            unaltered numpy array of the classes of bounding boxes of the image if bounding boxes
            present, else None
        bbox_coord : Optional[np.ndarray]
            unaltered numpy array of the coordinates of bounding boxes of the image if bounding
            boxes present, else None.
        label_class : Optional[np.ndarray]
            unaltered numpy array of the label class of the image if labels present, else None.
        label_value : Optional[np.ndarray]
            unaltered numpy array of the label value corresponding to label class of the image if
            labels present, else None.
        """
        img = img.astype(np.float32)
        img -= self.mean
        return img.astype(np.float32), bbox_class, bbox_coord, label_class, label_value


class ToAbsoluteCoords(object):
    """Converts bounding boxes from normalized (percent) coordinates to absolute coordinates"""

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Transform call to convert bounding box coordinates to absolute pixel values.

        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels this is not used or altered, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes this is not used or altered, by default None

        Returns
        -------
        img : np.ndarray
            unaltered numpy array of the image.
        bbox_class : Optional[np.ndarray]
            unaltered numpy array of the classes of bounding boxes of the image if bounding boxes
            present, else None
        bbox_coord : Optional[np.ndarray]
            transformed numpy array of the coordinates of bounding boxes of the image if bounding
            boxes present, else None. Bounding box coordinates are in [x1, y1, x2, y2] absolute
            pixel format. Where (x1,y1) is the top-left corner of the box and (x2,y2) is the
            bottom-right corner of the box.
        label_class : Optional[np.ndarray]
            unaltered numpy array of the label class of the image if labels present, else None.
        label_value : Optional[np.ndarray]
            unaltered numpy array of the label value corresponding to label class of the image if
            labels present, else None.
        """
        if bbox_coord is None:
            return img, bbox_class, bbox_coord, label_class, label_value
        height, width, channels = img.shape
        bbox_coord[:, 0] *= width
        bbox_coord[:, 2] *= width
        bbox_coord[:, 1] *= height
        bbox_coord[:, 3] *= height

        return img, bbox_class, bbox_coord, label_class, label_value


class ToPercentCoords(object):
    """Converts bounding boxes from absolute coordinates to normalized coordinates"""

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Transform call to convert bounding box coordinates to normalized values.

        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels this is not used or altered, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes this is not used or altered, by default None

        Returns
        -------
        img : np.ndarray
            unaltered numpy array of the image.
        bbox_class : Optional[np.ndarray]
            unaltered numpy array of the classes of bounding boxes of the image if bounding boxes
            present, else None
        bbox_coord : Optional[np.ndarray]
            transformed numpy array of the coordinates of bounding boxes of the image if bounding
            boxes present, else None. Bounding box coordinates are in [x1, y1, x2, y2] relative (to
            image width and height) format. Where (x1,y1) is the top-left corner of the box and (x2,
            y2) is the bottom-right corner of the box.
        label_class : Optional[np.ndarray]
            unaltered numpy array of the label class of the image if labels present, else None.
        label_value : Optional[np.ndarray]
            unaltered numpy array of the label value corresponding to label class of the image if
            labels present, else None.
        """
        if bbox_coord is None:
            return img, bbox_class, bbox_coord, label_class, label_value
        height, width, channels = img.shape
        bbox_coord[:, 0] /= width
        bbox_coord[:, 2] /= width
        bbox_coord[:, 1] /= height
        bbox_coord[:, 3] /= height

        return img, bbox_class, bbox_coord, label_class, label_value


class RandomSaturation(object):
    """Applies random saturation transform to an HSV image."""

    def __init__(self, lower: float = 0.5, upper: float = 1.5):
        """Initialize the transform with the minimum and maximum scale saturation value.

        Parameters
        ----------
        lower : float, optional
            lower bound for saturation scale value, by default 0.5
        upper : float, optional
            upper bound for saturation scale value, by default 1.5
        """
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
        self.RGB2HSV = ConvertColor(current="RGB", transform="HSV")
        self.HSV2RGB = ConvertColor(current="HSV", transform="RGB")

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Transform call for applying random saturation to an image.
        Parameters
        ----------
        img : np.ndarray
            The input image in HSV colorspace.
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes this is not used or altered, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes this is not used or altered, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels this is not used or altered, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes this is not used or altered, by default None

        Returns
        -------
        img : np.ndarray
            transformed numpy array of the image.
        bbox_class : Optional[np.ndarray]
            unaltered numpy array of the classes of bounding boxes of the image if bounding boxes
            present, else None
        bbox_coord : Optional[np.ndarray]
            unaltered numpy array of the coordinates of bounding boxes of the image if bounding
            boxes present, else None. Bounding box coordinates are in [x1, y1, x2, y2] absolute
            pixel format. Where (x1,y1) is the top-left corner of the box and (x2,y2) is the
            bottom-right corner of the box.
        label_class : Optional[np.ndarray]
            unaltered numpy array of the label class of the image if labels present, else None.
        label_value : Optional[np.ndarray]
            unaltered numpy array of the label value corresponding to label class of the image if
            labels present, else None.
        """
        hsv_img, _, _, _, _ = self.RGB2HSV(img=img)
        if random.randint(2):
            hsv_img[:, :, 1] *= random.uniform(self.lower, self.upper)
        rgb_img, _, _, _, _ = self.HSV2RGB(img=hsv_img)
        return rgb_img, bbox_class, bbox_coord, label_class, label_value


class RandomHue(object):
    """Applies random hue transform to an HSV image."""

    def __init__(self, delta: float = 18.0):
        """Initialize the transform with the strength of random hue.

        Parameters
        ----------
        delta : float, optional
            strength of random hue, by default 18.0
        """
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta
        self.RGB2HSV = ConvertColor(current="RGB", transform="HSV")
        self.HSV2RGB = ConvertColor(current="HSV", transform="RGB")

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Transform call to apply random hue to an image
        Parameters
        ----------
        img : np.ndarray
            The input image in HSV colorspace
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes this is not used or altered, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels this is not used or altered, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes this is not used or altered, by default None
        Returns
        -------
        img : np.ndarray
            transformed numpy array of the image.
        bbox_class : Optional[np.ndarray]
            unaltered numpy array of the classes of bounding boxes of the image if bounding boxes
            present, else None
        bbox_coord : Optional[np.ndarray]
            unaltered numpy array of the coordinates of bounding boxes of the image if bounding
            boxes present, else None. Bounding box coordinates are in [x1, y1, x2, y2] absolute
            pixel format. Where (x1,y1) is the top-left corner of the box and (x2,y2) is the
            bottom-right corner of the box.
        label_class : Optional[np.ndarray]
            unaltered numpy array of the label class of the image if labels present, else None.
        label_value : Optional[np.ndarray]
            unaltered numpy array of the label value corresponding to label class of the image if
            labels present, else None.
        """
        hsv_img, _, _, _, _ = self.RGB2HSV(img=img)
        if random.randint(2):
            hsv_img[:, :, 0] += random.uniform(-self.delta, self.delta)
            hsv_img[:, :, 0][hsv_img[:, :, 0] > 360.0] -= 360.0
            hsv_img[:, :, 0][hsv_img[:, :, 0] < 0.0] += 360.0
        rgb_img, _, _, _, _ = self.HSV2RGB(img=hsv_img)
        return rgb_img, bbox_class, bbox_coord, label_class, label_value


class RandomLightingNoise(object):
    """Randomly changes the image lighting"""

    def __init__(self):
        """Initialize the transform with channel swap orders."""
        self.perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
        self.RGB2BGR = ConvertColor(current="RGB", transform="BGR")
        self.BGR2RGB = ConvertColor(current="BGR", transform="RGB")

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Transform call to randomly change image lighting
        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes, this is not used or altered, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes this is not used or altered, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels this is not used or altered, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes this is not used or altered, by default None

        Returns
        -------
        img : np.ndarray
            transformed numpy array of the image.
        bbox_class : Optional[np.ndarray]
            unaltered numpy array of the classes of bounding boxes of the image if bounding boxes
            present, else None
        bbox_coord : Optional[np.ndarray]
            unaltered numpy array of the coordinates of bounding boxes of the image if bounding
            boxes present, else None.
        label_class : Optional[np.ndarray]
            unaltered numpy array of the label class of the image if labels present, else None.
        label_value : Optional[np.ndarray]
            unaltered numpy array of the label value corresponding to label class of the image if
            labels present, else None.
        """
        bgr_img, _, _, _, _ = self.RGB2BGR(img=img)
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            bgr_img, _, _, _, _ = shuffle(bgr_img)
        rgb_img, _, _, _, _ = self.BGR2RGB(img=bgr_img)
        return rgb_img, bbox_class, bbox_coord, label_class, label_value


class ConvertColor(object):
    """Changes colorspace of the image. Currently, BGR <-> HSV, RGB <-> BGR and RGB <-> HSV
    conversion are supported"""

    def __init__(self, current: str = "BGR", transform: str = "HSV"):
        """Initializes current and desired colorspace

        Parameters
        ----------
        current : str, optional
            current colorspace of the image, by default "BGR"
        transform : str, optional
            final colorspace of the image, by default "HSV"
        """
        self.transform = transform
        self.current = current

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Changes image colorspace

        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes, this is not used or altered, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes this is not used or altered, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels this is not used or altered, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes this is not used or altered, by default None

        Returns
        -------
        img : np.ndarray
            transformed numpy array of the image.
        bbox_class : Optional[np.ndarray]
            unaltered numpy array of the classes of bounding boxes of the image if bounding boxes
            present, else None
        bbox_coord : Optional[np.ndarray]
            unaltered numpy array of the coordinates of bounding boxes of the image if bounding
            boxes present, else None.
        label_class : Optional[np.ndarray]
            unaltered numpy array of the label class of the image if labels present, else None.
        label_value : Optional[np.ndarray]
            unaltered numpy array of the label value corresponding to label class of the image if
            labels present, else None.
        """
        if self.current == "BGR" and self.transform == "HSV":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.current == "HSV" and self.transform == "BGR":
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        elif self.current == "RGB" and self.transform == "BGR":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif self.current == "BGR" and self.transform == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.current == "RGB" and self.transform == "HSV":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif self.current == "HSV" and self.transform == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return img, bbox_class, bbox_coord, label_class, label_value


class RandomContrast(object):
    """Randomly changes the image contrast"""

    def __init__(self, lower: float = 0.5, upper: float = 1.5):
        """Initialize the transform with the range of contrast scale.

        Parameters
        ----------
        lower : float, optional
            lower bound for contrast scale value, by default 0.5
        upper : float, optional
            upper bound for contrast scale value, by default 1.5
        """
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
        self.RGB2BGR = ConvertColor(current="RGB", transform="BGR")
        self.BGR2RGB = ConvertColor(current="BGR", transform="RGB")

    # expects float image
    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Transform call to randomly change the contrast.

        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes this is not used or altered, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels this is not used or altered, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes this is not used or altered, by default None

        Returns
        -------
        img : np.ndarray
            transformed numpy array of the image.
        bbox_class : Optional[np.ndarray]
            unaltered numpy array of the classes of bounding boxes of the image if bounding boxes
            present, else None
        bbox_coord : Optional[np.ndarray]
            unaltered numpy array of the coordinates of bounding boxes of the image if bounding
            boxes present, else None.
        label_class : Optional[np.ndarray]
            unaltered numpy array of the label class of the image if labels present, else None.
        label_value : Optional[np.ndarray]
            unaltered numpy array of the label value corresponding to label class of the image if
            labels present, else None.
        """
        bgr_img, _, _, _, _ = self.RGB2BGR(img=img)
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            bgr_img *= alpha
        rgb_img, _, _, _, _ = self.BGR2RGB(img=bgr_img)
        return rgb_img, bbox_class, bbox_coord, label_class, label_value


class RandomBrightness(object):
    """Randomly changes the image brightness"""

    def __init__(self, delta: int = 32):
        """Initializes the strength of random brightness transform

        Parameters
        ----------
        delta : int, optional
            strength with which brightness is altered, by default 32
        """
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
        self.RGB2BGR = ConvertColor(current="RGB", transform="BGR")
        self.BGR2RGB = ConvertColor(current="BGR", transform="RGB")

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Transform call to randomly change brightness of the image

        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes, this is not used or altered, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes this is not used or altered, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels this is not used or altered, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes this is not used or altered, by default None

        Returns
        -------
        img : np.ndarray
            transformed numpy array of the image.
        bbox_class : Optional[np.ndarray]
            unaltered numpy array of the classes of bounding boxes of the image if bounding boxes
            present, else None
        bbox_coord : Optional[np.ndarray]
            unaltered numpy array of the coordinates of bounding boxes of the image if bounding
            boxes present, else None.
        label_class : Optional[np.ndarray]
            unaltered numpy array of the label class of the image if labels present, else None.
        label_value : Optional[np.ndarray]
            unaltered numpy array of the label value corresponding to label class of the image if
            labels present, else None.
        """
        bgr_img, _, _, _, _ = self.RGB2BGR(img=img)
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            bgr_img += delta
        rgb_img, _, _, _, _ = self.BGR2RGB(img=bgr_img)
        return rgb_img, bbox_class, bbox_coord, label_class, label_value


class RandomSampleCrop(object):
    """Randomly crops the input image"""

    def __init__(self):
        """Initialize the transform with sampling options. Sampling is done wrt jaccard w/
        object."""
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Transform call to randomly sample and crop the image.

        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels this is not used or altered, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes this is not used or altered, by default None

        Returns
        -------
        img : np.ndarray
            transformed numpy array of the image.
        bbox_class : Optional[np.ndarray]
            transformed numpy array of the classes of bounding boxes of the image if bounding boxes
            present, else None
        bbox_coord : Optional[np.ndarray]
            transformed numpy array of the coordinates of bounding boxes of the image if bounding
            boxes present, else None. Bounding box coordinates are in [x1, y1, x2, y2] absolute
            pixel format. Where (x1,y1) is the top-left corner of the box and (x2,y2) is the
            bottom-right corner of the box.
        label_class : Optional[np.ndarray]
            unaltered numpy array of the label class of the image if labels present, else None.
        label_value : Optional[np.ndarray]
            unaltered numpy array of the label value corresponding to label class of the image if
            labels present, else None.
        """
        height, width, _ = img.shape
        while True:
            # randomly choose a mode
            mode = self.sample_options[random.choice(len(self.sample_options))]
            if mode is None:
                return img, bbox_class, bbox_coord, label_class, label_value

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float("-inf")
            if max_iou is None:
                max_iou = float("inf")

            # max trails (50)
            for _ in range(50):
                current_img = img

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                if bbox_coord is not None:
                    # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                    overlap = jaccard_numpy(bbox_coord, rect)

                    # is min and max overlap constraint satisfied? if not try again
                    if overlap.min() < min_iou and max_iou < overlap.max():
                        continue

                # cut the crop from the image
                current_img = current_img[rect[1] : rect[3], rect[0] : rect[2], :]

                if bbox_coord is None:
                    resize = Resize(height=height, width=width)
                    resized_img, _, _, _, _ = resize(img=current_img)
                    return resized_img, bbox_class, bbox_coord, label_class, label_value

                # keep overlap with gt box IF center in sampled patch
                centers = (bbox_coord[:, :2] + bbox_coord[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_bbox_coord = bbox_coord[mask, :].copy()

                # take only matching gt labels
                current_bbox_class = bbox_class[mask]

                # should we use the box left and top corner or the crop's
                current_bbox_coord[:, :2] = np.maximum(current_bbox_coord[:, :2], rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_bbox_coord[:, :2] -= rect[:2]

                current_bbox_coord[:, 2:] = np.minimum(current_bbox_coord[:, 2:], rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_bbox_coord[:, 2:] -= rect[:2]

                resize = Resize(height=height, width=width)
                resized_img, _, resized_bbox_coord, _, _ = resize(
                    img=current_img, bbox_class=current_bbox_class, bbox_coord=current_bbox_coord
                )
                return resized_img, current_bbox_class, resized_bbox_coord, label_class, label_value


class Expand(object):
    """Embeds the input image inside a larger uniform (constant valued) image."""

    def __init__(self, mean: Iterable[int]):
        """[summary]

        Parameters
        ----------
        mean : Iterable[int]
            mean of ecah channel of the image. Is becomes the value of the constant image in which
            the input image is embedded.
        """
        self.mean = mean

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Transform call to apply expand on image and bboxes

        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes this is not used or altered, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels this is not used or altered, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes this is not used or altered, by default None

        Returns
        -------
        img : np.ndarray
            transformed numpy array of the image.
        bbox_class : Optional[np.ndarray]
            unaltered numpy array of the classes of bounding boxes of the image if bounding boxes
            present, else None
        bbox_coord : Optional[np.ndarray]
            transformed numpy array of the coordinates of bounding boxes of the image if bounding
            boxes present, else None. Bounding box coordinates are in [x1, y1, x2, y2] absolute
            pixel format. Where (x1,y1) is the top-left corner of the box and (x2,y2) is the
            bottom-right corner of the box.
        label_class : Optional[np.ndarray]
            unaltered numpy array of the label class of the image if labels present, else None.
        label_value : Optional[np.ndarray]
            unaltered numpy array of the label value corresponding to label class of the image if
            labels present, else None.
        """
        if random.randint(2):
            return img, bbox_class, bbox_coord, label_class, label_value

        height, width, depth = img.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        expand_img = np.zeros((int(height * ratio), int(width * ratio), depth), dtype=img.dtype)
        expand_img[:, :, :] = self.mean
        expand_img[int(top) : int(top + height), int(left) : int(left + width)] = img
        img = expand_img

        if bbox_coord is None:
            resize = Resize(height=height, width=width)
            resized_img, _, _, _, _ = resize(img=img)
            return resized_img, bbox_class, bbox_coord, label_class, label_value
        bbox_coord = bbox_coord.copy()
        bbox_coord[:, :2] += (int(left), int(top))
        bbox_coord[:, 2:] += (int(left), int(top))

        resize = Resize(height=height, width=width)
        resized_img, _, resized_bbox_coord, _, _ = resize(
            img=img, bbox_class=bbox_class, bbox_coord=bbox_coord
        )

        return resized_img, bbox_class, resized_bbox_coord, label_class, label_value


class RandomMirror(object):
    """Applies random mirror transform on input image and bboxes"""

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Transform call to mirror the image and bboxes

        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels this is not used or altered, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes this is not used or altered, by default None

        Returns
        -------
        img : np.ndarray
            transformed numpy array of the image.
        bbox_class : Optional[np.ndarray]
            transformed numpy array of the classes of bounding boxes of the image if bounding boxes
            present, else None
        bbox_coord : Optional[np.ndarray]
            transformed numpy array of the coordinates of bounding boxes of the image if bounding
            boxes present, else None. Bounding box coordinates are in [x1, y1, x2, y2] absolute
            pixel format. Where (x1,y1) is the top-left corner of the box and (x2,y2) is the
            bottom-right corner of the box.
        label_class : Optional[np.ndarray]
            unaltered numpy array of the label class of the image if labels present, else None.
        label_value : Optional[np.ndarray]
            unaltered numpy array of the label value corresponding to label class of the image if
            labels present, else None.
        """
        _, width, _ = img.shape
        if random.randint(2):
            img = img[:, ::-1]
            if bbox_coord is None:
                return img, bbox_class, bbox_coord, label_class, label_value
            bbox_coord = bbox_coord.copy()
            bbox_coord[:, 0::2] = width - bbox_coord[:, 2::-2]
        return img, bbox_class, bbox_coord, label_class, label_value


class SwapChannels(object):
    """Transforms an image by swapping the channels in the order specified in the swap tuple."""

    def __init__(self, swaps: Iterable[int]):
        """Initializes order in which the channels will be swapped

        Parameters
        ----------
        swaps : Iterable[int]
            final order of channels, eg: (2, 1, 0)
        """
        self.swaps = swaps

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Transform call ti swap image channels

        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes, this is not used or altered, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes this is not used or altered, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels this is not used or altered, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes this is not used or altered, by default None

        Returns
        -------
        img : np.ndarray
            transformed numpy array of the image.
        bbox_class : Optional[np.ndarray]
            unaltered numpy array of the classes of bounding boxes of the image if bounding boxes
            present, else None
        bbox_coord : Optional[np.ndarray]
            unaltered numpy array of the coordinates of bounding boxes of the image if bounding
            boxes present, else None.
        label_class : Optional[np.ndarray]
            unaltered numpy array of the label class of the image if labels present, else None.
        label_value : Optional[np.ndarray]
            unaltered numpy array of the label value corresponding to label class of the image if
            labels present, else None.
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        img = img[:, :, self.swaps]
        return img, bbox_class, bbox_coord, label_class, label_value


class PhotometricDistort(object):
    """Applies photometric distort transformation on the image"""

    def __init__(self):
        """Instantiates other transforms"""
        self.pd = [
            RandomContrast(),
            RandomSaturation(),
            RandomHue(),
            RandomContrast(),
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Transform call to apply photometric disort on the image
        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray] this is not used or altered, optional
            classes of the bounding boxes, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes this is not used or altered, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels this is not used or altered, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes this is not used or altered, by default None

        Returns
        -------
        img : np.ndarray
            transformed numpy array of the image.
        bbox_class : Optional[np.ndarray]
            unaltered numpy array of the classes of bounding boxes of the image if bounding boxes
            present, else None
        bbox_coord : Optional[np.ndarray]
            unaltered numpy array of the coordinates of bounding boxes of the image if bounding
            boxes present, else None.
        label_class : Optional[np.ndarray]
            unaltered numpy array of the label class of the image if labels present, else None.
        label_value : Optional[np.ndarray]
            unaltered numpy array of the label value corresponding to label class of the image if
            labels present, else None.
        """
        im = img.copy()
        im, bbox_class, bbox_coord, label_class, label_value = self.rand_brightness(
            im, bbox_class, bbox_coord, label_class, label_value
        )
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, bbox_class, bbox_coord, label_class, label_value = distort(
            im, bbox_class, bbox_coord, label_class, label_value
        )
        return self.rand_light_noise(im, bbox_class, bbox_coord, label_class, label_value)


class SSDAugmentation(object):
    """Applies a predefined set of transforms, which are expected to be useful for SSD model, to
    inputs"""

    def __init__(
        self,
        size: Iterable[int] = [300, 300],
        mean: Iterable[int] = [104, 117, 123],
        eval_: bool = False,
    ):
        """Initializes parameters of transforms

        Parameters
        ----------
        size : Iterable[int], optional
            (height, width) to which the image will be resized, by default [300, 300]
        mean : Iterable[int], optional
            mean of each channel of the image. This is used by SubtractMeans and Expand, by default
            [104, 117, 123]
        eval_ : bool, optional
            flag to identify eval mode. Transforms applied depend on the mode, by default False
        """
        self.mean = tuple(mean)
        self.size = tuple(size)
        if eval_:
            self.augment = Compose(
                [
                    XYWHToXYX2Y2(),
                    ConvertFromInts(),
                    Resize(self.size[0], self.size[1]),
                    SubtractMeans(self.mean),
                    ToPercentCoords(),
                    ToTensor(),
                ]
            )
        else:
            self.augment = Compose(
                [
                    XYWHToXYX2Y2(),
                    ConvertFromInts(),
                    Resize(self.size[0], self.size[1]),
                    RandomBlur(),
                    PhotometricDistort(),
                    Expand(self.mean),
                    RandomSampleCrop(),
                    RandomMirror(),
                    HorizontalFlip(),
                    VerticalFlip(),
                    SubtractMeans(self.mean),
                    ToPercentCoords(),
                    ToTensor(),
                ]
            )

    def __call__(
        self,
        img: np.ndarray,
        bbox_class: Optional[np.ndarray] = None,
        bbox_coord: Optional[np.ndarray] = None,
        label_class: Optional[np.ndarray] = None,
        label_value: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Transform call to apply SSD augmentations on inputs

        Parameters
        ----------
        img : np.ndarray
            The input image
        bbox_class : Optional[np.ndarray], optional
            classes of the bounding boxes, by default None
        bbox_coord : Optional[np.ndarray], optional
            location of the bounding boxes, by default None
        label_class : Optional[np.ndarray], optional
            classes of the image labels this is not used or altered, by default None
        label_value : Optional[np.ndarray], optional
            value of the image label classes this is not used or altered, by default None

        Returns
        -------
        img : np.ndarray
            transformed numpy array of the image.
        bbox_class : Optional[np.ndarray]
            transformed numpy array of the classes of bounding boxes of the image if bounding boxes
            present, else None
        bbox_coord : Optional[np.ndarray]
            transformed numpy array of the coordinates of bounding boxes of the image if bounding
            boxes present, else None. Bounding box coordinates are in [x1, y1, x2, y2] relative (to
            image width and height) format. Where (x1,y1) is the top-left corner of the box and (x2,
            y2) is the bottom-right corner of the box.
        label_class : Optional[np.ndarray]
            transformed numpy array of the label class of the image if labels present, else None.
        label_value : Optional[np.ndarray]
            transformed numpy array of the label value corresponding to label class of the image if
            labels present, else None.
        """
        return self.augment(img, bbox_class, bbox_coord, label_class, label_value)
