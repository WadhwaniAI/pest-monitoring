from typing import Dict, List, Union

from torch import Tensor


def detach(x: Union[Tensor, List[Tensor], List[Dict[str, Tensor]]]):
    # if x is of Tensor type, detach directly
    if isinstance(x, Tensor):
        return x.detach()

    # if x is of list type, detach each element
    if isinstance(x, list):
        return [detach(e) for e in x]

    # if x is of dict type,
    if isinstance(x, dict):
        return {k: detach(v) for k, v in x.items()}
