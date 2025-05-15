import os
from argparse import ArgumentParser
from utils import load_model, setup_logging, setup_wandb, collate_fn, set_global_seed
from config import CONFIG
from torch.utils.data import DataLoader
import torch

def main():
    
    parser = ArgumentParser(description = "Train and test models for Newborn Activity Recognition")
    parser.add_argument("--model", type = str, default = "baseModel", help = "Model name")
    parser.add_argument("--checkpoint", type = str, default = None, help = "Path to the model checkpoint")
    parser.add_argument("--train", action = "store_true", help = "Train the model from token dataset")
    parser.add_argument("--test", action = "store_true", help = "Test the model from token datset")
    
    args = parser.parse_args()
    model_name = args.model
    checkpoint = args.checkpoint
    train = args.train
    test = args.test
    
    logger = setup_logging(model_name)
    logger.info("-" * 20)
    logger.info(f"Starting the main function with model: {model_name}")
    logger.info(f"Checkpoint: {checkpoint}")
    logger.info(f"Train: {train}")
    logger.info(f"Test: {test}")
    logger.info("-" * 20)
    
    set_global_seed(42)
    
    wandb_run = setup_wandb(model_name, CONFIG)
    logger.info(f"WandB run initialized: {wandb_run}")
    # logger.info(f"Config: {CONFIG}")
    logger.info("-" * 20)