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
    parser.add_argument("--train_from_tokens", action = "store_true", help = "Train the model from token dataset")
    parser.add_argument("--test_from_tokens", action = "store_true", help = "Test the model from token datset")
    parser.add_argument("--export_tokens", action = "store_true", help = "Export tokens")
    parser.add_argument("--export_features", action = "store_true", help = "Export features")
    parser.add_argument("--train_from_features", action = "store_true", help = "Train from features")
    parser.add_argument("--test_from_features", action = "store_true", help = "Test from features")
    
    args = parser.parse_args()
    model_name = args.model
    checkpoint = args.checkpoint
    export_tokens = args.export_tokens
    export_features = args.export_features
    train_from_tokens = args.train_from_tokens
    test_from_tokens = args.test_from_tokens
    train_from_features = args.train_from_features
    test_from_features = args.test_from_features
    
    logger = setup_logging(model_name)
    logger.info("-" * 20)
    logger.info(f"Starting the main function with model: {model_name}")
    logger.info(f"Checkpoint: {checkpoint}")
    logger.info(f"Export tokens: {export_tokens}")
    logger.info(f"Export features: {export_features}")
    logger.info(f"Train from tokens: {train_from_tokens}")
    logger.info(f"Test from tokens: {test_from_tokens}")
    logger.info(f"Train from features: {train_from_features}")
    logger.info(f"Test from features: {test_from_features}")
    logger.info("-" * 20)
    
    set_global_seed(42)
    
    wandb_run = setup_wandb(model_name, CONFIG)
    logger.info(f"WandB run initialized: {wandb_run}")
    # logger.info(f"Config: {CONFIG}")
    logger.info("-" * 20)
    
    model = load_model(model_name, checkpoint)
    logger.info(f"Model loaded: {model}")
    logger.info("-" * 20)
    
    # POS WEIGHT AND CLASS PRIOR PROBABILITY HARDCODED SINCE TRAINING DATA DO NOT CHANGE
    pos_weight = torch.tensor([0.19311390817165375, 2.532083511352539, 7.530612468719482, 6.510387420654297])
    prior_probability = torch.tensor([0.8381429314613342, 0.2831190228462219, 0.11722487956285477, 0.1331489235162735])
    
    if export_tokens:
        logger.info("Exporting tokens ...")
        for split in ["train", "validation", "test"]:
            logger.info(f"Exporting tokens for {split} ...")
            model.export_tokens(video_folder = os.path.join(CONFIG["video_folder"], split),
                                annotation_folder = os.path.join(CONFIG["annotation_folder"], split),
                                output_folder = os.path.join(CONFIG["token_dir"], model.model_name, split),
                                clip_length = CONFIG["clip_length"],
                                overlapping = CONFIG["overlapping"],
                                frame_per_second = CONFIG.get(f"{model_name}_frame_per_second", CONFIG["frame_per_second"]),
                                prompt = CONFIG["prompt"],
                                system_message = CONFIG["system_message"],
                                logger = logger)
            logger.info(f"{split} tokens exported successfully.")
        logger.info("-" * 20)
        
    if export_features:
        logger.info("Exporting features ...")
        logger.info("Token dataset creation ...")
        from data.token_dataset import TokenDataset
        for split in ["train", "validation", "test"]:
            logger.info(f"Creating token dataset for {split} ...")
            dataset = TokenDataset(data_dir = os.path.join(CONFIG["token_dir"], 
                                                           model.model_name,
                                                           split, 
                                                           f'{CONFIG["clip_length"]}sec_{CONFIG.get(f"{model_name}_frame_per_second" ,CONFIG["frame_per_second"])}fps'))
            data_loader = DataLoader(dataset, 
                                     batch_size = CONFIG["batch_size_feature"],
                                     shuffle = False,
                                     num_workers = CONFIG["num_workers"])
            logger.info(f"Token dataset for {split} created successfully.")
            logger.info(f"Exporting features for {split} ...")
            model.save_features(dataloader = data_loader,
                                output_dir = os.path.join(CONFIG["feature_dir"],
                                                          model.model_name, 
                                                          split,
                                                          f'{CONFIG["clip_length"]}sec_{CONFIG.get(f"{model_name}_frame_per_second" ,CONFIG["frame_per_second"])}fps'))
            logger.info(f"{split} features exported successfully.")
        logger.info("-" * 20)
         
    if train_from_tokens:
        from data.token_dataset import TokenDataset
        logger.info("Training model ...")
        logger.info("Token dataset creation ...")
        train_dataset = TokenDataset(data_dir = os.path.join(CONFIG["token_dir"], 
                                                             model.model_name, 
                                                             "train", 
                                                             f'{CONFIG["clip_length"]}sec_{CONFIG.get(f"{model_name}_frame_per_second" ,CONFIG["frame_per_second"])}fps'))
        train_dataloader = DataLoader(train_dataset, batch_size = CONFIG.get(f"{model_name}_batch_size", CONFIG["batch_size"]), 
                                      shuffle = True, num_workers = CONFIG["num_workers"], 
                                      collate_fn = model.collate_fn_tokens)
        validation_dataset = TokenDataset(data_dir = os.path.join(CONFIG["token_dir"], 
                                                                  model.model_name,
                                                                  "validation", 
                                                                  f'{CONFIG["clip_length"]}sec_{CONFIG.get(f"{model_name}_frame_per_second" ,CONFIG["frame_per_second"])}fps'))
        validation_dataloader = DataLoader(validation_dataset, 
                                           batch_size = CONFIG.get(f"{model_name}_batch_size", CONFIG["batch_size"]), 
                                           shuffle = False, num_workers = CONFIG["num_workers"], 
                                           collate_fn = model.collate_fn_tokens)
        logger.info("Token dataset created successfully.")
        logger.info("Training model ...")
        logger.info("Using the following parameters:")
        logger.info(f"Batch size: {CONFIG.get(f"{model_name}_batch_size", CONFIG["batch_size"])}")
        logger.info(f"Learning rate: {CONFIG.get(f"{model_name}_learning_rate", CONFIG["learning_rate"])}")
        logger.info(f"Optimizer: {CONFIG.get(f"{model_name}_optimizer", CONFIG["optimizer"])}")
        logger.info(f"Criterion: {CONFIG['criterion']}")
        logger.info(f"Scheduler: {CONFIG.get(f"{model_name}_scheduler", CONFIG["scheduler"])}")
        logger.info(f"Patience: {CONFIG.get(f"{model_name}_patience", CONFIG["patience"])}")
        logger.info(f"Epochs: {CONFIG.get(f"{model_name}_epochs", CONFIG["epochs"])}")
        logger.info(f"Threshold: {CONFIG.get(f"{model_name}_threshold", CONFIG["threshold"])}")
        logger.info(f"Freezing condition: {CONFIG.get(f"{model_name}_freezing_condition", CONFIG["freezing_condition"])}")
        model.train_from_tokens(train_dataloader = train_dataloader,
                    val_dataloader = validation_dataloader,
                    epochs = CONFIG.get(f"{model_name}_epochs", CONFIG["epochs"]),
                    optimizer_name = CONFIG.get(f"{model_name}_optimizer", CONFIG["optimizer"]),
                    learning_rate = CONFIG.get(f"{model_name}_learning_rate", CONFIG["learning_rate"]),
                    momentum = CONFIG["momentum"],
                    weight_decay = CONFIG["weight_decay"],
                    criterion_name = CONFIG["criterion"],
                    pos_weight = pos_weight,
                    threshold = CONFIG.get(f"{model_name}_threshold", CONFIG["threshold"]),
                    scheduler_name = CONFIG.get(f"{model_name}_scheduler", CONFIG["scheduler"]),
                    scheduler_patience = CONFIG["scheduler_patience"],
                    patience = CONFIG.get(f"{model_name}_patience", CONFIG["patience"]),
                    show_progress = True,
                    prior_probability = prior_probability,
                    wandb_run = wandb_run,
                    logger = logger,
                    freezing_condition= CONFIG.get(f"{model_name}_freezing_condition", CONFIG["freezing_condition"]))
        logger.info("Model trained successfully.")
        logger.info("-" * 20)
    
    if test_from_tokens:
        logger.info("Testing model ...")
        logger.info("Token dataset creation ...")
        test_dataset = TokenDataset(data_dir = os.path.join(CONFIG["token_dir"],
                                                            model.model_name,
                                                            "test",
                                                            f'{CONFIG["clip_length"]}sec_{CONFIG.get(f"{model_name}_frame_per_second" ,CONFIG["frame_per_second"])}fps'))
        test_dataloader = DataLoader(test_dataset, 
                                     batch_size = CONFIG.get(f"{model_name}_batch_size", CONFIG["batch_size"]), 
                                     shuffle = False, 
                                     num_workers = CONFIG["num_workers"],
                                     collate_fn = model.collate_fn_tokens)
        logger.info("Token dataset created successfully.")
        logger.info("Testing model ...")
        model.test_from_tokens(test_dataloader = test_dataloader,
                   criterion_name = CONFIG["criterion"],
                   pos_weight = pos_weight,
                   threshold = CONFIG.get(f"{model_name}_threshold", CONFIG["threshold"]),
                   wandb_run = wandb_run)
        logger.info("Model tested successfully.")
        logger.info("-" * 20)
    
    if train_from_features:
        logger.info("Training classifier ...")
        from data.feature_dataset import FeatureDataset
        train_dataset = FeatureDataset(data_dir = os.path.join(CONFIG["feature_dir"], 
                                                               model.model_name, 
                                                               "train",
                                                               f'{CONFIG["clip_length"]}sec_{CONFIG.get(f"{model_name}_frame_per_second" ,CONFIG["frame_per_second"])}fps'))
        train_dataloader = DataLoader(train_dataset, 
                                      batch_size = CONFIG.get(f"{model_name}_batch_size", CONFIG["batch_size"]),
                                      shuffle = True, 
                                      num_workers = CONFIG["num_workers"],
                                      drop_last = True)
        validation_dataset = FeatureDataset(data_dir = os.path.join(CONFIG["feature_dir"],
                                                                    model.model_name, "validation",
                                                                    f'{CONFIG["clip_length"]}sec_{CONFIG.get(f"{model_name}_frame_per_second" ,CONFIG["frame_per_second"])}fps'))
        validation_dataloader = DataLoader(validation_dataset, 
                                           batch_size = CONFIG.get(f"{model_name}_batch_size", CONFIG["batch_size"]),
                                           shuffle = False,
                                           num_workers = CONFIG["num_workers"],
                                           drop_last = True)
        logger.info("Classifier dataset created successfully.")
        logger.info("Training classifier ...")
        logger.info("Using the following parameters:")
        logger.info(f"Batch size: {CONFIG['batch_size']}")
        logger.info(f"Learning rate: {CONFIG.get(f"{model_name}_learning_rate", CONFIG["learning_rate"])}")
        logger.info(f"Optimizer: {CONFIG.get(f"{model_name}_optimizer", CONFIG.get(f"{model_name}_optimizer", CONFIG["optimizer"]))}")
        logger.info(f"Criterion: {CONFIG['criterion']}")
        logger.info(f"Scheduler: {CONFIG.get(f"{model_name}_scheduler", CONFIG.get(f"{model_name}_scheduler", CONFIG["scheduler"]))}")
        logger.info(f"Patience: {CONFIG['patience']}")
        logger.info(f"Epochs: {CONFIG.get(f"{model_name}_epochs", CONFIG["epochs"])}")
        logger.info(f"Threshold: {CONFIG.get(f"{model_name}_threshold", CONFIG["threshold"])}")
        optimizer = model.define_optimizer(optimizer_name = CONFIG.get(f"{model_name}_optimizer", CONFIG["optimizer"]),
                                           learning_rate = CONFIG.get(f"{model_name}_learning_rate", CONFIG["learning_rate"]),
                                           momentum = CONFIG["momentum"],
                                           weight_decay = CONFIG["weight_decay"])
        criterion = model.define_criterion(criterion_name = CONFIG["criterion"],
                                           pos_weight = pos_weight)
        scheduler = model.define_scheduler(scheduler_name = CONFIG.get(f"{model_name}_scheduler", CONFIG["scheduler"]),
                                           optimizer = optimizer,
                                           epochs= CONFIG.get(f"{model_name}_epochs", CONFIG["epochs"]),
                                           patience = CONFIG.get(f"{model_name}_patience", CONFIG["patience"]))
        model.train_classifier(train_dataloader = train_dataloader,
                               val_dataloader = validation_dataloader,
                               epochs = CONFIG.get(f"{model_name}_epochs", CONFIG["epochs"]),
                               optimizer = optimizer,
                               criterion = criterion,
                               threshold = CONFIG.get(f"{model_name}_threshold", CONFIG["threshold"]),
                               scheduler = scheduler,
                               patience = CONFIG.get(f"{model_name}_patience", CONFIG["patience"]),
                               show_progress = True,
                               wandb_run = wandb_run)
        logger.info("Classifier trained successfully.")
        logger.info("-" * 20)
    
    if test_from_features:
        logger.info("Testing classifier ...")
        from data.feature_dataset import FeatureDataset
        test_dataset = FeatureDataset(data_dir = os.path.join(CONFIG["feature_dir"],
                                                              model.model_name,
                                                              "test",
                                                              f'{CONFIG["clip_length"]}sec_{CONFIG.get(f"{model_name}_frame_per_second" ,CONFIG["frame_per_second"])}fps'))
        test_dataloader = DataLoader(test_dataset, 
                                     batch_size = CONFIG.get(f"{model_name}_batch_size", CONFIG["batch_size"]),
                                     shuffle = False,
                                     num_workers = CONFIG["num_workers"],
                                     drop_last = True)
        logger.info("Classifier dataset created successfully.")
        logger.info("Testing classifier ...")
        model.test_classifier(test_dataloader = test_dataloader,
                               threshold = CONFIG.get(f"{model_name}_threshold", CONFIG["threshold"]),
                               wandb_run = wandb_run)
        logger.info("Classifier tested successfully.")
        logger.info("-" * 20)
        
    wandb_run.finish()
    logger.info("WandB run finished.")
    logger.info("Code finished.")
    logger.info("Bye bye!")
    logger.info("-" * 20)
    
    
if __name__ == "__main__":
    main()