import os
from argparse import ArgumentParser
from utils import load_model, setup_logging, setup_wandb, collate_fn
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
    parser.add_argument("--train_classifier", action = "store_true", help = "Train the classifier")
    parser.add_argument("--test_classifier", action = "store_true", help = "Test the classifier")
    
    args = parser.parse_args()
    model_name = args.model
    checkpoint = args.checkpoint
    train = args.train
    test = args.test
    export_tokens = args.export_tokens
    export_features = args.export_features
    train_classifier = args.train_classifier
    test_classifier = args.test_classifier
    
    logger = setup_logging(model_name)
    logger.info("-" * 20)
    logger.info(f"Starting the main function with model: {model_name}")
    logger.info(f"Checkpoint: {checkpoint}") 
    logger.info(f"Train: {train}")
    logger.info(f"Test: {test}")
    logger.info(f"Train classifier: {train_classifier}")
    logger.info(f"Test classifier: {test_classifier}")
    logger.info(f"Export tokens: {export_tokens}")
    logger.info(f"Export features: {export_features}")
    logger.info("-" * 20)
    
    wandb_run = setup_wandb(model_name, CONFIG)
    logger.info(f"WandB run initialized: {wandb_run}")
    # logger.info(f"Config: {CONFIG}")
    logger.info("-" * 20)
    
    model = load_model(model_name, checkpoint)
    logger.info(f"Model loaded: {model}")
    logger.info("-" * 20)
    
    if export_tokens:
        logger.info("Exporting tokens ...")
        for split in ["train", "validation", "test"]:
            logger.info(f"Exporting tokens for {split} ...")
            model.export_tokens(video_folder = os.path.join(CONFIG["video_folder"], split),
                                annotation_folder = os.path.join(CONFIG["annotation_folder"], split),
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
        for split in ["train", "validation", "test"]:
            logger.info(f"Creating token dataset for {split} ...")
            dataset = TokenDataset(data_dir = os.path.join(CONFIG["token_dir"], model.model_name, split, f'{CONFIG["clip_length"]}sec_{CONFIG["frame_per_second"]}fps'))
            data_loader = DataLoader(dataset, batch_size = CONFIG["batch_size_feature"], shuffle = False, num_workers = CONFIG["num_workers"])
            logger.info(f"Token dataset for {split} created successfully.")
            logger.info(f"Exporting features for {split} ...")
            model.save_features(dataloader = data_loader,
                                output_dir = os.path.join(CONFIG["feature_dir"], model.model_name, split))
            logger.info(f"{split} features exported successfully.")
        logger.info("-" * 20)
         
    if train:
        from data.token_dataset import TokenDataset
        logger.info("Training model ...")
        logger.info("Token dataset creation ...")
        train_dataset = TokenDataset(data_dir = os.path.join(CONFIG["token_dir"], model.model_name, "train", f'{CONFIG["clip_length"]}sec_{CONFIG["frame_per_second"]}fps'))
        train_dataloader = DataLoader(train_dataset, batch_size = CONFIG["batch_size"], shuffle = True, num_workers = CONFIG["num_workers"], collate_fn = collate_fn)
        validation_dataset = TokenDataset(data_dir = os.path.join(CONFIG["token_dir"], model.model_name, "validation", f'{CONFIG["clip_length"]}sec_{CONFIG["frame_per_second"]}fps'))
        validation_dataloader = DataLoader(validation_dataset, batch_size = CONFIG["batch_size"], shuffle = False, num_workers = CONFIG["num_workers"], collate_fn = collate_fn)
        logger.info("Token dataset created successfully.")
        logger.info("Training model ...")
        logger.info("Using the following parameters:")
        logger.info(f"Batch size: {CONFIG['batch_size']}")
        logger.info(f"Learning rate: {CONFIG['learning_rate']}")
        logger.info(f"Optimizer: {CONFIG['optimizer']}")
        logger.info(f"Criterion: {CONFIG['criterion']}")
        logger.info(f"Scheduler: {CONFIG['scheduler']}")
        logger.info(f"Patience: {CONFIG['patience']}")
        logger.info(f"Epochs: {CONFIG['epochs']}")
        logger.info(f"Threshold: {CONFIG['threshold']}")
        optimizer = model.define_optimizer(optimizer_name = CONFIG["optimizer"],
                                           learning_rate = CONFIG["learning_rate"],
                                           momentum = CONFIG["momentum"],
                                           weight_decay = CONFIG["weight_decay"])
        pos_weight = train_dataset.weight_computation()
        criterion = model.define_criterion(criterion_name = CONFIG["criterion"],
                                           pos_weight = pos_weight)
        scheduler = model.define_scheduler(scheduler_name = CONFIG["scheduler"],
                                           optimizer = optimizer,
                                           epochs= CONFIG["epochs"],
                                           patience = CONFIG["patience"])
        model.train_model(train_dataloader = train_dataloader,
                    val_dataloader = validation_dataloader,
                    epochs = CONFIG["epochs"],
                    optimizer = optimizer,
                    criterion = criterion,
                    threshold = CONFIG["threshold"],
                    scheduler = scheduler,
                    patience = CONFIG["patience"],
                    show_progress = True,
                    wandb_run = wandb_run)
        logger.info("Model trained successfully.")
        logger.info("-" * 20)
    
    if test:
        logger.info("Testing model ...")
        logger.info("Token dataset creation ...")
        test_dataset = TokenDataset(data_dir = os.path.join(CONFIG["token_dir"], model.model_name, "test", f'{CONFIG["clip_length"]}sec_{CONFIG["frame_per_second"]}fps'))
        test_dataloader = DataLoader(test_dataset, batch_size = CONFIG["batch_size"], shuffle = False, num_workers = CONFIG["num_workers"], collate_fn = collate_fn)
        logger.info("Token dataset created successfully.")
        logger.info("Testing model ...")
        model.test_model(test_dataloader = test_dataloader,
                   threshold = CONFIG["threshold"],
                   wandb = wandb_run)
        logger.info("Model tested successfully.")
        logger.info("-" * 20)
    
    if train_classifier:
        logger.info("Training classifier ...")
        from data.feature_dataset import FeatureDataset
        train_dataset = FeatureDataset(data_dir = os.path.join(CONFIG["feature_dir"], model.model_name, "train"))
        pos_weight = train_dataset.weight_computation()
        train_dataloader = DataLoader(train_dataset, batch_size = CONFIG["batch_size"], shuffle = True, num_workers = CONFIG["num_workers"])
        validation_dataset = FeatureDataset(data_dir = os.path.join(CONFIG["feature_dir"], model.model_name, "validation"))
        validation_dataloader = DataLoader(validation_dataset, batch_size = CONFIG["batch_size"], shuffle = False, num_workers = CONFIG["num_workers"])
        logger.info("Classifier dataset created successfully.")
        logger.info("Training classifier ...")
        logger.info("Using the following parameters:")
        logger.info(f"Batch size: {CONFIG['batch_size']}")
        logger.info(f"Learning rate: {CONFIG['learning_rate']}")
        logger.info(f"Optimizer: {CONFIG['optimizer']}")
        logger.info(f"Criterion: {CONFIG['criterion']}")
        logger.info(f"Scheduler: {CONFIG['scheduler']}")
        logger.info(f"Patience: {CONFIG['patience']}")
        logger.info(f"Epochs: {CONFIG['epochs']}")
        logger.info(f"Threshold: {CONFIG['threshold']}")
        optimizer = model.define_optimizer(optimizer_name = CONFIG["optimizer"],
                                           learning_rate = CONFIG["learning_rate"],
                                           momentum = CONFIG["momentum"],
                                           weight_decay = CONFIG["weight_decay"])
        criterion = model.define_criterion(criterion_name = CONFIG["criterion"],
                                           pos_weight = pos_weight)
        scheduler = model.define_scheduler(scheduler_name = CONFIG["scheduler"],
                                           optimizer = optimizer,
                                           epochs= CONFIG["epochs"],
                                           patience = CONFIG["patience"])
        model.train_classifier(train_dataloader = train_dataloader,
                               val_dataloader = validation_dataloader,
                               epochs = CONFIG["epochs"],
                               optimizer = optimizer,
                               criterion = criterion,
                               threshold = CONFIG["threshold"],
                               scheduler = scheduler,
                               patience = CONFIG["patience"],
                               show_progress = True,
                               wandb = wandb_run)
        logger.info("Classifier trained successfully.")
        logger.info("-" * 20)
    
    if test_classifier:
        logger.info("Testing classifier ...")
        from data.feature_dataset import FeatureDataset
        test_dataset = FeatureDataset(data_dir = os.path.join(CONFIG["feature_dir"], model.model_name, "test"))
        test_dataloader = DataLoader(test_dataset, batch_size = CONFIG["batch_size"], shuffle = False, num_workers = CONFIG["num_workers"])
        logger.info("Classifier dataset created successfully.")
        logger.info("Testing classifier ...")
        model.test_classifier(test_dataloader = test_dataloader,
                               threshold = CONFIG["threshold"],
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