from typing import Dict, Optional, Sequence, Union

import torch

from .pest_default_dataset import PestDefaultDataset


class PestMultiHeadDataset(PestDefaultDataset):
    """This Dataset class takes in a list of category names to be included
    in the output of the dataset. The rest of the heads are suppressed.

    Note: This dataset doesn't work for cases when we want to have bboxes as well as
    image level labels.

    Parameters
    ----------
    heads : Sequence
        Sequence of category names to be included in the dataset. If "bboxes" is in the list, then
        the bboxes will be included in the output of the dataset.

    """

    def __init__(self, heads: Sequence, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heads = list(heads)

    def __getitem__(self, idx: int) -> Dict[str, Optional[Union[str, torch.Tensor]]]:
        """Gets those items were requested by the initilization.

        Parameters
        ----------
        idx : int
            training index of the sample

        Returns
        -------
        Dict[str, Optional[Union[str, torch.Tensor]]]
            Dict with image and associated ground truth information after applying transforms.
            Contains "img_id", "img", "bbox_class", "bbox_coord", "label_class_cat",
            "label_value_cat", "label_class_reg", "label_value_reg" keys.

        Raises
        ------
        NotImplementedError
            if number of heads is not 1
        ValueError
            If unknow supercategory used
        ValueError
            If any datapoint doesn't have the required heads
        """
        img_id, img, bbox_class, bbox_coord, label_class, label_value = self.pull_item(idx)
        if "bboxes" not in self.heads:
            bbox_class = None
            bbox_coord = None

        if self.transforms is not None:
            img, bbox_class, bbox_coord, label_class, label_value = self.transforms(
                img, bbox_class, bbox_coord, label_class, label_value
            )

        # since we are only interested in `self.heads` label, we will remove all other details
        out_label_class_cat = []
        out_label_value_cat = []
        out_label_class_reg = []
        out_label_value_reg = []
        for label, val in zip(label_class.tolist(), label_value.tolist()):
            label = int(label)
            # print ("label:", label)
            row = self.category_df.loc[label]
            category_name = row["name"]
            supercategory = row["supercategory"]

            # TODO currently, it's not clear if this is correct, since it does'nt force
            # an error in case a certain head is missing
            # TODO, make sure that the values are always in order as requested in self.heads
            # raise not implemented error if len of heads > 1
            if len(self.heads) != 1:
                raise NotImplementedError

            if category_name in self.heads:
                if supercategory == "Image Level Categorical Label":
                    out_label_class_cat.append(label)
                    out_label_value_cat.append(int(val))
                elif supercategory == "Image Level Regressional Label":
                    out_label_class_reg.append(label)
                    out_label_value_reg.append(val)
                else:
                    raise ValueError(
                        f"Unknown supercategory: {supercategory} of head: {category_name}"
                    )

        # checking if all the heads were present for this datapoint
        num_tasks = len(out_label_class_cat) + len(out_label_class_reg)
        if "bboxes" in self.heads:  # if bboxes were requested, increase num_tasks
            num_tasks += 1

        if num_tasks != len(self.heads):
            all_heads = {head for head in self.heads if head != "bboxes"}
            heads_retrieved = set()
            for cat in out_label_class_cat + out_label_class_reg:
                heads_retrieved.add(self.category_df.loc[cat]["name"])
            missing_heads = all_heads - heads_retrieved
            raise ValueError(
                f"The datapoint: {img_id}, is missing following heads: {missing_heads}"
            )

        record = {
            "img_id": img_id,
            "img": img,
            "bbox_class": bbox_class,
            "bbox_coord": bbox_coord,
            "label_class_cat": torch.IntTensor(out_label_class_cat)
            if len(out_label_class_cat) > 0
            else None,
            "label_value_cat": torch.LongTensor(out_label_value_cat)
            if len(out_label_value_cat) > 0
            else None,
            "label_class_reg": torch.IntTensor(out_label_class_reg)
            if len(out_label_class_reg) > 0
            else None,
            "label_value_reg": torch.FloatTensor(out_label_value_reg)
            if len(out_label_value_reg) > 0
            else None,
        }
        return record
