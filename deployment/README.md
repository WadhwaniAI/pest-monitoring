# Usage
```
$ python package.py -h
usage: package.py [-h] -vj VALIDATION_JIT -cj COUNTING_JIT [-nt NMS_TH] [-mn MAX_NUM] -m METADATA -bd BOXDATA -o OUT [-ow] [-oj]

Package counting and validation jit models into a single jit model

optional arguments:
  -h, --help            show this help message and exit
  -vj VALIDATION_JIT, --validation_jit VALIDATION_JIT
                        validation model jit file path
  -cj COUNTING_JIT, --counting_jit COUNTING_JIT
                        counting model jit file path
  -nt NMS_TH, --nms_th NMS_TH
                        NMS Threshold
  -mn MAX_NUM, --max_num MAX_NUM
                        Max boxes allowed
  -m METADATA, --metadata METADATA
                        metadata file
  -bd BOXDATA, --boxdata BOXDATA
                        default box data file
  -o OUT, --out OUT     output directory to save package
  -ow, --overwrite      overwrite the existing version in out directory
  -oj, --only_jit       save only combined jit otherwise the whole zip package will be saved
```

# Example
```
$ python package.py -vj ./validation.jit -cj ./counting.jit -m ./sample_metadata.json -bd ./boxdata/ssd_300.json -o ./out
```

# Metadata File
```
{
    "input_size": 300,                                          # The height=width of the image tensor, the jit models expect
    "trap_label": {"0": "pheromone", "1": "non-trap"},          # Output to class mapping of the validation jit model
    "threshold": [0.2, 0.2],                                    # Confidence threshold for boxes. First element is for abw and the second for pbw
    "research_version": "configs/experiments/default.yaml",     # Training config of the model
    "version": "2.5"                                            # Deployment version
}
```

# Boxdata File
Information required to generate default boxes for SSD
```
{
    "feat_size" : [38, 19, 10, 5, 3, 1],
    "steps" : [8, 16, 32, 64, 100, 300],
    "scales" : [21, 45, 99, 153, 207, 261, 315],
    "aspect_ratios" : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    "scale_xy" : 0.1,
    "scale_wh" : 0.2
}
```

# Output Directory Structure
After a successful run of the following command
```
$ python package.py -vj ./validation.jit -cj ./counting.jit -m ./sample_metadata.json -bd ./boxdata/ssd_300.json -o ./out
```

the output directory will look like this

```txt
out
└── v2.5                       # Here "2.5" is the version mentioned in the metadata file passed in -m flag
    ├── model.pt               # JIT model
    ├── metadata.json          # copy of metadata file
    ├── cfssd_2.5.mar          # MAR model. Here "cfssd" means combined model and "2.5" is the version mentioned in the metadata file
    └── v2.5.zip               # Zip package. Here "2.5" is the version mentioned in the metadata file
        ├── model.pt           # Copy of out/v2.5/model.pt
        └── metadata.json      # Copy of out/v2.5/metadata.json
```

# JIT Model
## Input
Batch with a single transformed image tensor. This should be exactly same as the input validation and counting jit models require.

## Output
The output of the mdoel is a 5-length tuples.
`(validation_out, abw_boxes, pbw_boxes, abw_scores, pbw_scores)`
- `validation_out`: 2 length tensor, where the values are (trap_confidence, non_trap_confidence). This is the exact output of the validation jit model and is returned as is.
- `abw_boxes`: Nx4 torch tensor of abw box coordinates, where the columns are - top_left_x, top_left_y, bottom_right_x, bottom_right_y. The values are as a fraction of the height and width of the image.
- `pbw_boxes`: Nx4 torch tensor of pbw box coordinates, where the columns are - top_left_x, top_left_y, bottom_right_x, bottom_right_y. The values are as a fraction of the height and width of the image.
- `abw_scores`: N length tensor, where the values are the confidence scores of the corresponding boxes in `abw_boxes`.
- `pbw_scores`: N length tensor, where the values are the confidence scores of the corresponding boxes in `pbw_boxes`.

# Mar Model
## Input
*todo*

## Output
*todo*
