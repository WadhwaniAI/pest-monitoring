### Hydra Configs

This is the directory from which all the hydra configs are read. All the major components (networks, models, callbacks, data) as mentioned in the [src](https://github.com/WadhwaniAI/pest-monitoring-new/tree/main/src), have a corresponding config mentioned in this directory.

#### Example Config
To put it simply, every experiment is a config containing information about different components. For example, if we take the following config, Example taken from this [config](https://github.com/WadhwaniAI/pest-monitoring-new/blob/main/configs/experiments/ssd/cfvgg16-size512.yaml).
```yaml
seed: 12345

defaults:
  - /hydra: default.yaml
  - /logger: wandb.yaml
  - /callbacks: val-loss.yaml
  - /datamodule: object-detection/base-size-512-w-norm-000.yaml
  - /trainer: single-gpu.yaml
  - /model/network: ssd/cfvgg-size512.yaml
  - /model/optimizer: AdamW/wd-1e-3.yaml
  - /model/lr_scheduler: stepLR.yaml

model:
  _target_: src.models.CFNet
```

In the above config, we use,
- logger: We use wandb logger, but you can use your own logger like [this](https://github.com/ashleve/lightning-hydra-template/tree/main/configs/logger). The logger structure is as follows,
```yaml
wandb:
  _target_: pytorch_lightning.loggers.wandb.WandbLogger
  project: "pest-monitoring-new"
  notes: "Wandb Description: Experiments from Pest-Monitoring-New"
```
- callback: We use the ModelCheckpoint class to save checkpoints based on `val/loss`. All the callbacks options used in the codebase exists in the [callbacks configs directory](configs/callbacks/).
```yaml
model_checkpoint:
  _target_: pytorch_lightning.callbacks.ModelCheckpoint
  monitor: "val/loss"
  mode: min
  dirpath: "checkpoints/"
  filename: "{epoch:02d}"
```
- datamodule: For the object-detection + classification task on our pest monitoring dataset, we use the have stored some default experiments in the [datamodule configs directory](configs/datamodule/).
```yaml
defaults:
  - dataset: object-detection/v1.0.0.yaml
  - sampler_config: cfsampler.yaml
  - override dataset/transforms: object-detection/base-w-norm.yaml

_target_: src.data.base_datamodule.BaseDataModule
num_workers: 10
batch_size: 64
drop_last: False
pin_memory: True
```
- model/network: In this example, we use the CFSSD network architecture, basically (Object Detection SSD + Classification).
```yaml
_target_: src.networks.ssd.CFSSD
feature_extractor:
  _target_: src.networks.ssd._vgg_extractor
  backbone_name: vgg16
  highres: False
  pretrained_backbone: True
anchor_generator:
  _target_: src.networks.ssd.DefaultBoxGenerator
  aspect_ratios: [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
  steps: [8, 16, 32, 64, 100, 300]
  scales: [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
rejection_head:
  _target_: src.networks.ssd.CFSSDVGGRejectionHead
loss:
  _target_: src.networks.ssd.CFSSDDefaultLoss
img_size: [512, 512]
num_classes: 3
```
- model/optimizer: In this example, we use the `AdamW` optimizer.
```yaml
_target_: torch.optim.AdamW
lr: 0.0001
weight_decay: 0.0005
```
- model/lr_scheduler: We use the `stepLR` learning rate scheduler.
```yaml
_target_: torch.optim.lr_scheduler.StepLR
step_size: 10
gamma: 0.95
```
