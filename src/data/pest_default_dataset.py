import numpy as np
from PIL import Image, ImageFile

from .base_dataset import BaseDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class PestDefaultDataset(BaseDataset):
    """Extension of the BaseDataset class for the Pest Monitoring/Management Project

    Changes over BaseDataset:
    - Contains valid/invalid indices corresponding to images with/without bounding boxes
    - Performs an additional `.convert("RGB")` in the `get_img` function
    """

    def __init__(self, **kwargs):
        """Initializes the dataset class.

        Parameters
        ----------
        dataset_config : DictConfig
            Hydra config
        mode : str
            Dataset split to load. Takes one of the strings: "train", "val", "test"
        transforms : Optional[Compose], optional
            Compose of transforms, defaults to None, by default None
        """
        super(PestDefaultDataset, self).__init__(**kwargs)
        self._get_valid_invalid_indices()

    def _get_valid_invalid_indices(self):
        self.valid_indices = []
        self.invalid_indices = []

        for i, id_ in enumerate(self.img_ids):
            bbox_class, _ = self.get_bbox_anns(id_)
            if bbox_class is None:
                self.invalid_indices.append(i)
            else:
                self.valid_indices.append(i)

    # Copy needed :
    # https://stackoverflow.com/questions/39554660/np-arrays-being-immutable-assignment-destination-is-read-only/39554807
    def get_img(self, path, *args, **kwargs):
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

        im = Image.open(path).convert("RGB")
        im = np.asarray(im)
        return im.copy()

    def eval_caption(self, caption):
        """Resolve string type image level caption into value from the input file. Override this
        function to change how to resolve image level string caption into value.

        Parameters
        ----------
        caption : str
            Image level information's caption

        Returns
        -------
        value : int
            Interpreted value of input caption

        """
        value = int(caption)
        return value
