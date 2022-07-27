import torch
from torch import Tensor


def resize_boxes(boxes, original_size, new_size) -> Tensor:
    """Resize boxes from original_size to new_size.

    Parameters
    ----------
    boxes: Tensor
        Boxes to resize.
    original_size: Tuple[int, int]
        Size of the image before resizing.
    new_size: Tuple[int, int]
        Size of the image after resizing.
    """
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)
