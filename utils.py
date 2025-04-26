import os
import logging
import wandb
import time
import torch

def load_model(model_name: str, checkpoint: str):
    """
    Load the model based on the model name and checkpoint path.
    
    Args:
        model_name (str): The name of the model to load.
        checkpoint (str): The path to the model checkpoint.
        
    Returns:
        model: The loaded model.
    """
    if checkpoint is not None and not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint file {checkpoint} does not exist.")
    
    if model_name == "VideoLLaVA":
        from models.videollava import VideoLlava
        model = VideoLlava(checkpoint_path=checkpoint)
    elif model_name == "SmolVLM":
        from models.smolvlm import SmolVLM
        model = SmolVLM(checkpoint=checkpoint)
    elif model_name == "TimeSformer":
        from models.timesformer import TimeSformer
        model = TimeSformer()
    else:
        from models.basemodel import BaseModel
        model = BaseModel(checkpoint=checkpoint)
    
    return model


def setup_logging(model_name: str):
    """
    Set up logging for the model training and evaluation.
    """
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler = logging.FileHandler(f"logs/{model_name}.log")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    return logger

def setup_wandb(model_name: str, config: dict):
    """
    Set up Weights & Biases for logging.
    """
    run = wandb.init(
        project = "NewbornActivityRecognition",
        name = f'{model_name}_{time.strftime("%Y%m%d-%H%M%S")}',
        config = config,
        resume = "allow")
    
    return run

def collate_fn(batch):
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}