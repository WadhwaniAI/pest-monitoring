# Sample data

The directory houses sample input jsons in `jsons` directory and the
images used in the jsons in `images` directory.

To download the images required for the sample jsons run the following command
```bash
python prepare_sample_coco_data.py jsons/experiments/ssd/sample.json
```

## Data format

### Input JSON Format

The basic building blocks for our custom JSON annotation files look like the following:

- **info**: contains high-level information about the dataset.
- **images**: contains all the image information in the dataset without bounding box or segmentation information. image ids need to be unique
- **box_annotations**: list of every individual box level annotation for every image in the dataset
- **caption_annotations**: empty list
- **categories**: list of categories used in annotations. Every category belongs to a single supercategory: [image, bounding box, …]
- **splits**: list of split information for images in the dataset

#### Definition

``` JSON
{
"info"                  : info,
"images"                : [image],
"box_annotations"       : [box_annotation],
"caption_annotations"   : [],
"categories"            : [category],
"splits"                : [split]
}

info{
"version"               : str, # data version
"description"           : str, # basic description about this data version
"contributor"           : str, # the poc
"url"                   : str, # local absolute path of this file
"date_created"          : datetime, # timestamp of this data version
}

image{
"id"                    : int, # unique (in `images` block) image id
"width"                 : int, # width of the image in pixels
"height"                : int, # height of the image in pixels
"file_path"             : str, # local absolute path of the image
"s3_url"                : str, # s3 link of the image
"date_captured"         : datetime, # date exif tag on the image
}

box_annotation{
"id"                    : int, # unique (in `box_annotations` block) annotation id
"image_id"              : int, # image id of the annotation [maps to `images` block]
"category_id"           : int, # category id of the annotation [maps to `categories` block]
"bbox"                  : [x1, y1, w, h], # absolute values (in pixels/int) of the annotation.
                                          # image origin is on top-left.
                                          # (x1, y1) is top-left corner of annotation
}

category{
"id"                    : int, # unique (in `categories` block) category id
"name"                  : str, # name of the category eg. pbw, trap, etc.
"supercategory"         : str, # name of the supercategory eg. image, bounding box, etc.
}

split{
"image_id"              : int, # image id of the annotation [maps to `images` block]
"split"                 : str, # name of the split the image belongs to eg. train, val, test, etc.
}
```

#### Sample File

```JSON
"info": {
"version"               : "002.010.003",
"description"           : "sample data file",
"contributor"           : "anmol",
"url"                   : "/data/input/002.010.003.json",
"date_created"          : "2020-08-10",
},

"images": [
{"id"                   : 0
"width"                 : 1024
"height"                : 720
"file_path"             : "/data/images/img0.jpg"
"s3_url"                : "s3://data/images/img0.jpg",
"date_captured"         : "2019-03-23:13:20:30.34"},

{"id"                   : 1
"width"                 : 1080
"height"                : 720
"file_path"             : "/data/images/img1.jpg"
"s3_url"                : "s3://data/images/img1.jpg",
"date_captured"         : "2019-04-13:15:40:30.14"},
]

"box_annotations": [
{"id"                   : 0
"image_id"              : 0
"category_id"           : 0
"bbox"                  : [200, 130, 120, 100]},

{"id"                   : 1
"image_id"              : 0
"category_id"           : 1
"bbox"                  : [500, 530, 100, 150]},
]

"caption_annotations": [
]

"categories": [
{"id"                   : 0
"name"                  : "pbw",
"supercategory"         : "bounding box"},

{"id"                   : 1
"name"                  : "abw",
"supercategory"         : "bounding box"},

{"id"                   : 2
"name"                  : "is_trap",
"supercategory"         : "image"},
]

"splits": [
{"image_id"             : 0,
"split"                 : "train"},

{"image_id"             : 1,
"split"                 : "val"},
]
```

#### Conventions

We will refer to list of dictionaries (images, categories, box_annotations, caption_annotationsand splits) as tables where keys become the columns of the table and each dictionary becomes a row.

#### Indexes

Indexes (`id` key in `images`, `categories` and `box_annotations`) are defined to make referencing within the JSON convenient. They have no physical significance and hence it would be arbitrary to impose that indexes should remain consistent across different JSON versions. Thus, **indexes should not be used outside of a JSON**. Instead, we should use a combination of keys that physically make sense.

_For example, instead of referring to a bounding box as belonging to `image_id` 123 and `category_id` 3, we should identify it as belonging to image ‘/abc/xyz.jpg’ and category (abw, predicted bounding box)._

#### Allowed Categories

Category **name** and **supercategory** should be picked from the following table so as to ensure that they remain consistent across different JSONs.

| Category `name` | Category `supercategory` |
| --------------- | ------------------------ |
| abw             | bounding box             |
| pbw             | bounding box             |
| _abw_           | _predicted bounding box_ |
| _pbw_           | _predicted bounding box_ |

This table can be appended when new categories are needed but older categories should not be modified.

Also, _italicised_ rows should be used to distinguish Predictions from Ground Truth only when both Predictions and Ground Truth are present in the JSON.

#### Duplicates

Following tables should not have duplicate rows based on mentioned keys:

| table             | unique keys                                           |
| ----------------- | ----------------------------------------------------- |
| `images`          | `file_path`                                           |
| `categories`      | `name`, `supercategory`                               |
| `box_annotations` | `image_id`, `category_id`, `bbox_coord`, `bbox_score` |

Things to note:

1. `splits` table can have duplicate rows as user might want to oversample some examples.
2. Above mentioned tables **can be allowed** to have duplicate rows if it makes sense to do so. For example, if a model predicts identical bounding box twice, because of a bug in NMS code, both bounding boxes can be stored in separate rows.
