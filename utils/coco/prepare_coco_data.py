import argparse
import copy
import json

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Starting Run")

    parser.add_argument(
        "-ogt",
        "--coco_original_train_path",
        type=str,
        default="/data/COCO/annotations_trainval2017/annotations/instances_train2017.json",
        help="path to original train coco json",
    )
    parser.add_argument(
        "-ogv",
        "--coco_original_val_path",
        type=str,
        default="/data/COCO/annotations_trainval2017/annotations/instances_val2017.json",
        help="path to original val coco json",
    )
    parser.add_argument(
        "-d",
        "--destination_path",
        type=str,
        default="/data/COCO/version_files/000_intern_instance_mod.json",
        help="path at which to store the json",
    )
    parser.add_argument(
        "-c",
        "--classes",
        type=str,
        default="dog-cat-bicycle",
        help=(
            "classes to be used, separated by - \n the first class will be used in training and all"
            " the classes will be used in validation\nif classes = all, all classes will be used"
        ),
    )
    args = parser.parse_args()
    return args


def get_data(
    args, coco_train, wanted_image_id_set, wanted_category_id_set, mode="train", new_categories={}
):
    coco_train.pop("licenses")

    img_id_list = []
    for ann in coco_train["images"]:
        img_id_list.append(ann["id"])

    coco_train["info"].pop("year")

    coco_train["box_annotations"] = []
    for i, ann in tqdm(enumerate(coco_train["annotations"])):
        if (ann["category_id"] in wanted_category_id_set) and (
            ann["image_id"] in wanted_image_id_set
        ):
            if ann["category_id"] not in new_categories:
                new_categories[ann["category_id"]] = len(new_categories)
            ann["category_id"] = new_categories[ann["category_id"]]
            rem_list = ["segmentation", "area", "iscrowd"]
            for k in rem_list:
                ann.pop(k)
            if ann["bbox"][2] > 0 and ann["bbox"][3] > 0:
                coco_train["box_annotations"].append(ann)
    coco_train.pop("annotations")
    for i, img in tqdm(enumerate(coco_train["images"])):
        ann = copy.deepcopy(img)
        rem_list = ["license", "coco_url", "flickr_url"]
        for k in rem_list:
            ann.pop(k)

        ann["file_path"] = f"/data/COCO/{mode}2017/" + ann["file_name"]
        ann["s3_url"] = None
        ann.pop("file_name")
        coco_train["images"][i] = ann

    coco_train["caption_annotations"] = []
    coco_train["splits"] = []
    return coco_train, new_categories


def load(path):
    with open(path) as f:
        coco_train = json.load(f)
    return coco_train


def dump(path, coco_train):
    with open(path, "w") as f:
        json.dump(coco_train, f)


def get_classes_list(args):
    return args.classes.split("-")


def get_category_name_id_map(coco_train):
    name_id_map = {}
    for i in range(len(coco_train["categories"])):
        name_id_map[coco_train["categories"][i]["name"]] = coco_train["categories"][i]["id"]
    return name_id_map


def get_wanted_set(args, coco_train, classes_list):
    name_id_map = get_category_name_id_map(coco_train)
    id_set = set([name_id_map[name] for name in classes_list])
    wanted_set = set()
    for i, ann in tqdm(enumerate(coco_train["annotations"])):
        if ann["category_id"] in id_set:
            wanted_set.add(ann["image_id"])
    return wanted_set, id_set


def merge(coco_train, coco_val, train_wanted_set, val_wanted_set):
    for k in coco_train:
        if not k == "info":
            coco_train[k].extend(coco_val[k])
    for image_id in train_wanted_set:
        d = {}
        d["image_id"] = image_id
        d["split"] = "train"
        coco_train["splits"].append(d)
    for image_id in val_wanted_set:
        d = {}
        d["image_id"] = image_id
        d["split"] = "val"
        coco_train["splits"].append(d)
    return coco_train


def main(args):
    classes_list = get_classes_list(args)
    coco_train = load(args.coco_original_train_path)
    classes_list_train = classes_list[:1]
    train_wanted_image_id_set, train_wanted_category_id_set = get_wanted_set(
        args, coco_train, classes_list_train
    )
    coco_train, new_categories = get_data(
        args, coco_train, train_wanted_image_id_set, train_wanted_category_id_set, mode="train"
    )
    coco_val = load(args.coco_original_val_path)
    classes_list_val = classes_list[:1]
    val_wanted_image_id_set, val_wanted_category_id_set = get_wanted_set(
        args, coco_val, classes_list_val
    )
    coco_val, new_categories = get_data(
        args,
        coco_val,
        val_wanted_image_id_set,
        val_wanted_category_id_set,
        mode="val",
        new_categories=new_categories,
    )
    print("old to new category map", new_categories)
    coco_train = merge(coco_train, coco_val, train_wanted_image_id_set, val_wanted_image_id_set)
    dump(args.destination_path, coco_train)
    return coco_train


if __name__ == "__main__":
    args = get_args()
    main(args)
