<div align="center">


<h1 style="font-size:10vw"> Pest Monitoring Codebase</h1>

[🚀 Model Zoo](model_zoo.md)
[📗 Configs](configs/)
[💻 Setup Instructions](setup)
[🤚 Contributing](CONTRIBUTING.md)

</div>

Pest Monitoring is a project at the Wadhwani Institute for Artificial
Intelligence (Wadhwani AI). The project helps cotton farmers make
better pest management decisions by providing pesticide advice based
on photos of pests caught in specialised traps. To provide the correct
advice, specific pests must be identified within a photo and then
counted.  This code in this respository is focused on accomplishing
that task.

## Background

At its core, this repository packages an object detection
implementation. It does so in way that is conducive to the specific
aspects of the project overall:

* Being able to quickly experiment with project-informed new
  ideas. For example, easily adding new components at unexepected
  places in the training or evaluation pipeline, or using a
  configuration or evaluation system that is common across other
  projects at Wadhwani AI.

* Gracefully handling images that do not meet our expectations. Images
  commonly taken when users are getting to know the product, for
  example.

* Taking a trained model to a size and format that is appropriate for
  mobile application in which users interface. This involves both
  model serialization and compression.

While there are several object detection implementations, and even
implementation aggregations, there are none that completely solve for
the challenges we face.

## Documentation

* [Setup](./setup): get started using this codebase.
* [Configuration](./configs): alter the default components of this
  codebase to accomplish non-default tasks.
* [Data I/O formatting and manipulation](./sample-data): understand
  the I/O format to do extended evaluation, or to work with custom
  datasets.

### Further reading

* [Pest Management for Cotton Farming](https://www.wadhwaniai.org/programs/pest-management/)
* [Pest management in cotton farms: an AI-system case study from the
  global South](https://dx.doi.org/10.1145/3394486.3403363)
* [How Wadhwani AI Uses PyTorch To Empower Cotton Farmers](https://medium.com/pytorch/how-wadhwani-ai-uses-pytorch-to-empower-cotton-farmers-14397f4c9f2b)

## Usage basics

### Training

Use the following command, with the config name relative to the configs folder (`configs/`):

```bash
python train.py -cn defaults/object-detection-ssd.yaml ++name=experiment_name

Arguments
---------
-cn (str): Path of the config file wrt the configs/ directory.
++name (str): Name of the experiment (Compulsory to pass if not passing test_run=True)
+test_run (bool): If set to True, the name check will be skipped.

Additional Information
----------------------
+ : Appending key to a config
++ : Overriding key in a config
```

### Evaluating given a Checkpoint

Since we use hydra CLI instead of the usual CLI (with argparse), when we need to pass additional arguments to the CLI, we use the `+key=value` syntax. For example, to evaluate a certain model, we need the checkpoint path and the split name. We do this by passing (`+ckpt` and `+split`). The following command can be used to evaluate a certain model.

```bash
python eval.py -cn config_path +ckpt='checkpoint_path' +split=val ++name=experiment_name

Arguments
---------
-cn (str): Path of the config file wrt the configs/ directory.
+ckpt (str): Absolute path to checkpoint file
+split (str): One of ["val", "test", "train"]
++name (str): Name of the experiment
+test_run (bool): If set to True, the name check will be skipped.

Additional Information
----------------------
+ : Appending key to a config
++ : Overriding key in a config
ckpt: If Checkpoint has an = in the path, then consider using `/=` instead of the `=` OR wrap it like '+ckpt="checkpoint_path"'
```

## Contributing

Contributions to this codebase are welcome. Please do so by forking
the repository and making pull requests.

### Git Pre-commit Hooks

Please peform the following steps when cloning the repository for the first time. This would install the necessary packages for installing pre-commit hooks to make the code we write prettier. :)

```bash
pip install pre-commit
pre-commit install
```
Ref: https://levelup.gitconnected.com/raise-the-bar-of-code-quality-in-python-projects-7c49743f004f
