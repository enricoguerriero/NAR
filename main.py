import os
from argparse import ArgumentParser
from utils import load_model, setup_logging, setup_wandb
from config import CONFIG
from torch.utils.data import DataLoader

def main():
    
    parser = ArgumentParser(description = "Train and test models for Newborn Activity Recognition")
    parser.add_argument("--model", type = str, default = "baseModel", help = "Model name")
    parser.add_argument("--checkpoint", type = str, default = None, help = "Path to the model checkpoint")
    parser.add_argument("--train", action = "store_true", help = "Train the model")
    parser.add_argument("--test", action = "store_true", help = "Test the model")
    parser.add_argument("--export_tokens", action = "store_true", help = "Export tokens")
    parser.add_argument("--export_features", action = "store_true", help = "Export features")
    
    args = parser.parse_args()
    model_name = args.model
    checkpoint = args.checkpoint
    train = args.train
    test = args.test
    export_tokens = args.export_tokens
    export_features = args.export_features
    
    logger = setup_logging(model_name)
    logger.info("-" * 20)
    logger.info(f"Starting the main function with model: {model_name}")
    logger.info(f"Checkpoint: {checkpoint}")
    logger.info(f"Train: {train}")
    logger.info(f"Test: {test}")
    logger.info(f"Export tokens: {export_tokens}")
    logger.info(f"Export features: {export_features}")
    logger.info("-" * 20)
    
    wandb_run = setup_wandb(model_name, CONFIG)
    logger.info(f"WandB run initialized: {wandb_run}")
    logger.info(f"Config: {CONFIG}")
    logger.info("-" * 20)
    
    model = load_model(model_name, checkpoint)
    logger.info(f"Model loaded: {model}")
    logger.info("-" * 20)
    
    if export_tokens:
        logger.info("Exporting tokens ...")
        for split in ["train", "eval", "test"]:
            logger.info(f"Exporting tokens for {split} ...")
            model.export_tokens(video_folder = os.path.join(CONFIG["video_folder"], model.model_name, split),
                                annotation_folder = os.path.join(CONFIG["annotation_folder"], model.model_name, split),
                                output_folder = os.path.join(CONFIG["token_dir"], model.model_name, split),
                                clip_length = CONFIG["clip_length"],
                                overlapping = CONFIG["overlapping"],
                                frame_per_second = CONFIG["frame_per_second"],
                                prompt = CONFIG["prompt"],
                                system_message = CONFIG["system_message"],
                                logger = logger)
            logger.info(f"{split} tokens exported successfully.")
        logger.info("-" * 20)
        
    if export_features:
        logger.info("Exporting features ...")
        logger.info("Token dataset creation ...")
        from data.token_dataset import TokenDataset
        for split in ["train", "eval", "test"]:
            logger.info(f"Creating token dataset for {split} ...")
            dataset = TokenDataset(data_dir = os.path.join(CONFIG["token_dir"], model.model_name, split))
            data_loader = DataLoader(dataset, batch_size = CONFIG["batch_size_feature"], shuffle = False, num_workers = 4)
            logger.info(f"Token dataset for {split} created successfully.")
            logger.info(f"Exporting features for {split} ...")
            model.save_features(dataloader = data_loader,
                                output_dir = os.path.join(CONFIG["feature_dir"], model.model_name, split))
            logger.info(f"{split} features exported successfully.")
        logger.info("-" * 20)
         
    