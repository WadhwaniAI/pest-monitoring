## 🚀 Model Zoo

Note: Reading any config described [here](configs).

Our repository currently supports models (or networks) for 2 kinds of ML tasks,
1. Object Detection
2. Object Detection + Classification (Jointly Trained over 2 tasks)

We mostly use networks either from standard packages like (torchvision, mmdetection, etc) or modify the source code of networks for our tasks in this codebase.

### Networks modified from Source Code

#### SSD or Single Shot Detector [ [Configs](configs/defaults/ssd) / [Code](src/networks/ssd/) ]
We use the torchvision implementation of SSD as described [here](https://pytorch.org/blog/torchvision-ssd-implementation/). This is because they allow for variable image size (300 and 512 commonly used) and allow for different backbones. We modify it for the following,
- To allow us to track validation losses
- To allow us to get predictions in both train and val mode
- Create a new Object detection + Classification network by extending the feature network for image level classification

P.S. Configs for object detection + classification are present as `cfvgg`/`cfresnet` while object detection are present as `vgg`/`resnet`

#### Retina-Net [ [Configs](configs/defaults/retina-net) / [Code](src/networks/retine_net/) ]
We use the torchvision implementation of RetinaNet as described. We modify it for the following,
- To allow us to track validation losses
- To allow us to get predictions in both train and val mode
- Create a new Object detection + Classification network by extending the feature network for image level classification

P.S. Configs for object detection + classification are present as `cfresnet` while object detection are present as `resnet`

### Networks From Standard Libararies

We also allow extending support to directly use standard object detection libraries in our pytorch lightning framework.

#### Torchvision [ [Code](https://pytorch.org/vision/0.8/models.html#object-detection-instance-segmentation-and-person-keypoint-detection) / [Configs](configs/defaults/faster_rcnn/) ]
For networks from torchvision, we can directly import the network as follows,
```yaml
_target_: torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn
progress: True
num_classes: 3
pretrained_backbone: True
trainable_backbone_layers: 6
```
We have kept different network based on torchvision [here](configs/model/network/torchvision/faster_rcnn)

Note: Currently supporting networks from `torchvision==0.8`

#### YoloRT [ [Link](https://github.com/zhiqwang/yolov5-rt-stack) / [Configs](configs/defaults/yolov5/)]
We have kept the different networks based on YoloRT [here](configs/model/network/yolov5/)
