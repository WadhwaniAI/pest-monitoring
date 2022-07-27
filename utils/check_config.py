""" Config Checker

    Config files will have different requirements in train/eval/compress mode.
    This script will check the config file and make sure it is correct.

    Called in train.py, eval.py and compress.py before calling the respective functions

"""
import json
from os.path import exists

from omegaconf import DictConfig
from tqdm import tqdm

from src.utils import utils

log = utils.get_logger(__name__)


def check_train_config(config: DictConfig = None):
    """Check the train config file.

    Parameters
    ----------
    config : DictConfig
        Config file
    """
    # Check if config is passed
    if config is None:
        log.error("Config file is not passed")
        raise FileExistsError("Config file is not passed")

    # Perform config name check
    check_config_name(config)

    # Check model config
    if "model" not in config:
        log.error("Config file does not contain model")
        raise KeyError("Config file does not contain model")
    else:
        _check_model_config(config.model)

    # Check datamodule config
    if "datamodule" not in config:
        log.error("Config file does not contain datamodule")
        raise KeyError("Config file does not contain datamodule")
    else:
        _check_datamodule_config(config.datamodule)

    # Check trainer config
    if "trainer" not in config:
        log.error("Config file does not contain trainer")
        raise KeyError("Config file does not contain trainer")
    else:
        _check_trainer_config(config.trainer)

    # Check auto_lr_find and optimizer
    _check_auto_lr_optimizer(config.trainer, config.model)


def check_eval_config(config: DictConfig = None, split: str = "val", ckpt_PATH: str = None):
    """Check the eval config file.

    Parameters
    ----------
    config : DictConfig
        Config file
    split : str
        Split to evaluate on, default is val
    ckpt_PATH : str
        Path to the checkpoint to evaluate on
    """
    # Check if config is passed
    if config is None:
        log.error("Config file is not passed")
        raise FileExistsError("Config file is not passed")

    # Check config name
    check_config_name(config)

    # Check if split key is present in config and is one of val/test/train
    if "split" not in config:
        log.error("Config file does not contain split")
        raise KeyError("Config file does not contain split")

    split = config.split
    if split not in ["val", "test", "train"]:
        raise ValueError(f"Evaluate on {split} is not supported")

    # Check if ckpt exists as a key in config
    if "ckpt" not in config:
        log.error("Config file does not contain ckpt")
        raise KeyError("Config file does not contain ckpt")

    # Check if ckpt is not None or does not exist
    ckpt = config.ckpt
    if ckpt is None or not exists(ckpt):
        log.error("ckpt does not exist")
        raise ValueError("ckpt does not exist")

    # Check model config
    if "model" not in config:
        log.error("Config file does not contain model")
        raise KeyError("Config file does not contain model")
    else:
        _check_model_config(config.model)

    # Check datamodule config
    if "datamodule" not in config:
        log.error("Config file does not contain datamodule")
        raise KeyError("Config file does not contain datamodule")
    else:
        _check_datamodule_config(config.datamodule)

    # Check trainer config
    if "trainer" not in config:
        log.error("Config file does not contain trainer")
        raise KeyError("Config file does not contain trainer")
    else:
        _check_trainer_config(config.trainer)


def check_compress_config(config: DictConfig):
    """Check the compression config file.

    Parameters
    ----------
    config : DictConfig
        Config file
    """
    # Check if config exists
    if config is None:
        log.error("Config file is not passed")
        raise FileExistsError("Config file is not passed")

    # Perform config name check
    check_config_name(config)

    # Check if ckpt exists as a key in config
    if "ckpt" not in config:
        log.error("Config file does not contain ckpt")
        raise KeyError("Config file does not contain ckpt")

    # Check if ckpt is not None or does not exist
    ckpt = config.ckpt
    if ckpt is None or not exists(ckpt):
        log.error("ckpt does not exist")
        raise ValueError("ckpt does not exist")

    # Check model config
    if "model" not in config:
        log.error("Config file does not contain model")
        raise KeyError("Config file does not contain model")
    else:
        _check_model_config(config.model)

    # Check datamodule config
    if "datamodule" not in config:
        log.error("Config file does not contain datamodule")
        raise KeyError("Config file does not contain datamodule")
    else:
        _check_datamodule_config(config.datamodule)

    # Check trainer config
    if "trainer" not in config:
        log.error("Config file does not contain trainer")
        raise KeyError("Config file does not contain trainer")
    else:
        _check_trainer_config(config.trainer)

    # Check pruner config
    if "pruner" not in config:
        log.error("Config file does not contain pruner")
        raise KeyError("Config file does not contain pruner")
    else:
        _check_pruner_settings(config.pruner)


# Utility functions
def _check_datamodule_config(datamodule_config: DictConfig = None):
    """Check the datamodule config file.

    Parameters
    ----------
    datamodule_config : DictConfig
        Config file
    """
    # Check if datamodule_config is not none
    if datamodule_config is None:
        log.error("No datamodule config found")
        raise ValueError("No datamodule config found")

    # Check if config contains dataset
    if "dataset" in datamodule_config:
        _check_dataset_config(datamodule_config.dataset)
    else:
        log.error("Config file does not contain dataset")
        raise AttributeError("Config file does not contain dataset attribute")

    # Check if other arguments for dataloader are passed, if not log about using defaults
    # Batch Size
    if "batch_size" in datamodule_config:
        log.info(f"Using batch_size {datamodule_config.batch_size}")
    else:
        log.info("Using default batch size : 1")

    # num_workers
    if "num_workers" in datamodule_config:
        log.info(f"Using num_workers {datamodule_config.num_workers}")
    else:
        log.info("Using default num_workers : 0")

    # pin_memory
    if "pin_memory" in datamodule_config:
        log.info(f"Using pin_memory {datamodule_config.pin_memory}")
    else:
        log.info("Using default pin_memory : False")


def _check_dataset_config(dataset_config: DictConfig = None):
    """Check dataset component inside datamodule config

    Parameters
    ----------
    dataset_config : DictConfig
        Config file
    """
    # Check if dataset_config is not none
    if dataset_config is None:
        log.error("No dataset config found")
        raise ValueError("No dataset config found")

    # Check if data_file inside dataset_config exists
    if "data_file" in dataset_config:
        path = dataset_config.data_file
        if not exists(path):
            log.error(f"Datafile {path} does not exist")
            raise FileExistsError(f"Datafile {path} does not exist")

        # Check the json data file
        _check_data_file_json(path)
    else:
        log.error("Config file does not contain data_file")
        raise ValueError("Config file does not contain data_file")

    # Check if transforms
    if "transforms" in dataset_config:
        _check_transforms_config(dataset_config.transforms)
    else:
        log.error("Config file does not contain transforms")
        raise ValueError("Config file does not contain transforms")


def _check_data_file_json(json_path: str):
    """Checks the structure of json data_file

    Parameters
    ----------
    json_path : str
        Path to the json data_file
    """
    log.info(f"Checking json file {json_path}")

    # read the json file
    with open(json_path, "r") as f:
        data = json.load(f)

    # Check if all the minimum keys are present in the json file and type is correct
    keys = ["info", "images", "categories", "splits"]
    for key in keys:
        if key not in data:
            log.error(f"{key} not found in json file")
            raise KeyError(f"{key} not found in json file")

        # Check if info is a dictionary
        if type(data["info"]) is not dict:
            raise TypeError("info is not a dictionary")

        # Check if images, categories and splits are lists of dictionaries
        if key in ["images", "categories", "splits"]:
            if type(data[key]) is not list:
                raise KeyError(f"{key} is not a list")

    # Check all images in the list for keys and structure
    keys = ["id", "width", "height", "file_path", "s3_url", "date_captured"]
    for image in tqdm(data["images"]):
        for key in keys:
            if key not in image:
                log.error(f"{key} not found in image dict")
                raise KeyError(f"{key} not found in image dict")

    # Check if atleast one of 'box_annotations', 'caption_annotations' is present,
    # if not raise error
    if "box_annotations" not in data and "caption_annotations" not in data:
        raise KeyError("box_annotations or caption_annotations not found in json file")

    # If both box annotations and caption annotations are present, check if both are not empty
    if "box_annotations" in data and "caption_annotations" in data:
        if len(data["box_annotations"]) == 0 and len(data["caption_annotations"]) == 0:
            log.error("Both box_annotations and caption_annotations are empty")
            raise ValueError("Both box_annotations and caption_annotations are empty")

    # Check if box_annotations is present and is a list of dictionaries and have necessary keys
    if "box_annotations" in data:
        if type(data["box_annotations"]) is not list:
            raise TypeError("box_annotations is not a list")

        keys = ["id", "image_id", "category_id", "bbox"]
        for box_annotation in tqdm(data["box_annotations"]):
            if type(box_annotation) is not dict:
                raise TypeError("box_annotations is not a list of dictionaries")
            for key in keys:
                if key not in box_annotation:
                    raise KeyError(f"{key} not found in box_annotation")
    else:
        log.info("box_annotations not found in json file")

    # Check if caption_annotations is present and is a list of dictionaries and have necessary keys
    if "caption_annotations" in data:
        if type(data["caption_annotations"]) is not list:
            raise TypeError("caption_annotations is not a list")

        keys = ["id", "image_id", "category_id", "caption"]
        for caption_annotation in tqdm(data["caption_annotations"]):
            if type(caption_annotation) is not dict:
                raise TypeError("caption_annotations is not a list of dictionaries")
            for key in keys:
                if key not in caption_annotation:
                    raise KeyError(f"{key} not found in caption_annotation")
    else:
        log.info("caption_annotations not found in json file")

    # Check if category elements are dictionaries and have all the keys
    if len(data["categories"]) > 0:
        keys = ["id", "name", "supercategory"]
        for category in data["categories"]:
            for key in keys:
                if key not in category:
                    raise KeyError(f"{key} not found in category dict")
            break
    else:
        raise ValueError("No categories present")

    # Check if splits are dictionaries and have all the keys
    if len(data["splits"]) > 0:
        keys = ["image_id", "split"]
        for row in data["splits"]:
            for key in keys:
                if key not in row:
                    raise KeyError(f"{key} not found in split dict")
            break
    else:
        raise ValueError("No splits present")


def _check_transforms_config(transforms_config: DictConfig = None, mode: str = "train"):
    """Check the transforms config file.

    Parameters
    ----------
    transforms_config : DictConfig
        Config file
    mode : str
        Mode of the dataset, train, val or test, default is train
    """
    if transforms_config is None:
        log.error("No transforms config found")
        raise ValueError("No transforms config found")

    # mode_list provides which transforms need to be present for a particular mode
    mode_list = {
        "train": ["train", "val"],
        "val": ["train", "val", "test"],
        "compress": ["train", "val"],
    }

    # Make sure for a particular mode the transforms corresponding to the list elements are present
    for split_ in mode_list[mode]:
        if split_ not in transforms_config:
            raise KeyError(f"{split_} not found in transforms config")


def _check_model_config(model_config: DictConfig = None):
    """Check the model config file.

    Parameters
    ----------
    model_config : DictConfig
        Config file
    """
    # Check if model_config is present
    if model_config is None:
        log.error("Config file does not contain model_config")
        raise ValueError("Config file does not contain model_config")

    # Check if network is present
    if "network" not in model_config:
        raise KeyError("Model config does not contain network")

    # Check if loss is present
    if "loss" not in model_config:
        log.info("Model config does not contain loss")

    # Check if optimizer is present
    if "optimizer" not in model_config:
        raise KeyError("Model config does not contain optimizer")

    # If Optimizer is present and learning rate is not present, pass a default value
    # Default value of optimizer learning rate is needed in the case of auto_lr_find
    if "lr" not in model_config["optimizer"]:
        log.info("Learning rate not present in optimizer config. Using default value : 0.001")
        model_config["optimizer"]["lr"] = 0.001

    # Check if lr_scheduler is present, if not just log about it and continue
    if "lr_scheduler" not in model_config:
        log.info("Model config does not contain lr_scheduler")


def _check_trainer_config(trainer_config: DictConfig = None):
    """Check the trainer config

    Parameters
    ----------
    trainer_config : DictConfig
        Config file
    """
    # Check if trainer is present
    if trainer_config is None:
        log.error("Trainer config not found")
        raise ValueError("Trainer config not found")

    # check if _target_ is present
    if "_target_" not in trainer_config:
        log.error("Trainer config does not contain _target_")
        raise KeyError("Trainer config does not contain _target_")

    # log about other keys and their values, if any
    keys_other_than_target = trainer_config.keys() - {"_target_"}
    for key in keys_other_than_target:
        log.info(f"Non default value of {key} has been used, and been set to {trainer_config[key]}")


def _check_pruner_settings(pruner_config: DictConfig = None):
    """Check the compression config

    Parameters
    ----------
    pruner_config : DictConfig
        Config file
    """
    # Check if compression config is present
    if pruner_config is None:
        log.info("pruner_config not present")
        raise KeyError("pruner_config not present")


def _check_auto_lr_optimizer(trainer_config: DictConfig = None, model_config: DictConfig = None):
    """Checks if auto_lr_find is False in trainer, optimizer needs to have the lr key

    Parameters
    ----------
    trainer_config : DictConfig
        Config file
    model_config : DictConfig
        Config file
    """
    # if either of the configs is None, return
    if trainer_config is None or model_config is None:
        raise ValueError("Either trainer_config or model_config is None")

    # if auto_lr_find is present and is False, check if optimizer has lr key
    if "auto_lr_find" in trainer_config and not trainer_config["auto_lr_find"]:
        if "lr" not in model_config["optimizer"]:
            raise KeyError("lr not found in optimizer config")


def check_config_name(config):
    """Check if the config.name has been passed and changed from default
    value.

    Parameters
    ----------
    config : DictConfig
        Config file
    """
    if config.name == "default":
        if "test_run" in config:
            log.info(
                "Config name is default. As test_run key is passed",
                "in config, we will test_run this",
            )
        else:
            log.critical(
                "Since experiment name isn't specified in command line,"
                f"using default name: {config.name}."
                "It's strongly advisable to manually set experiment name."
            )
            raise ValueError("Experiment name is not specified")
