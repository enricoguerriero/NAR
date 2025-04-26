'''
Script to test the NAR model with 0-shot prompting.
(Hopefully) lightweight and dummy.
'''
import os
from argparse import ArgumentParser
from utils import setup_logging, setup_wandb, load_model


def main():
    
    # Parse command line arguments
    parser = ArgumentParser(description="Test NAR model with 0-shot prompting.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to test.")
    
    args = parser.parse_args()
    model_name = args.model_name
    
    # Set up logging
    logger = setup_logging(model_name)
    logger.info("Starting 0 shot test script.")
    logger.info(f"Model name: {model_name}")
    
    # Set up Weights & Biases
    config = {
        "model_name": model_name,
        "test_type": "0-shot",
        "data": "test data",
    }
    wandb_run = setup_wandb(model_name, config)
    logger.info("Weights & Biases setup complete.")
    
    # Load the model
    model = load_model(model_name, None)
    logger.info(f"Model {model_name} loaded successfully.")
    logger.info(f"Model device: {model.device}")
    
    