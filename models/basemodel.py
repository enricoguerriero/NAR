import torch.nn as nn
import logging
import os
from tqdm import tqdm
import cv2
import numpy as np
import torch
import glob
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from torch import Tensor
from torch.optim import lr_scheduler
import wandb

class BaseModel(nn.Module):
    """
    An abstract base class for video models.
    Provides a common interface for training, inference, and last-layer modifications.
    """
    def __init__(self, device = "cuda", model_name: str = "baseModel"):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.pos_weights = None
        self.processor = None
        self.classifier = None
        self.num_classes = None
        
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement the forward method.")

    def modify_last_layer(self, new_layer_config):
        """
        Modify the last layer(s) of the model.
        :param new_layer_config: A new layer or sequential block to replace the final layer(s).
        """
        raise NotImplementedError("Subclasses must implement modify_last_layer().")
    
    def process_input(self, frame_list = None, prompt = None, system_message = None):
        """
        To be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement process_input().")
    
    def feature_extraction(self, pixel_values=None, input_ids=None, attention_mask=None):
        """
        Extract features from the model.
        :param pixel_values: Input pixel values (images).
        :param input_ids: Input token IDs (text).
        :param attention_mask: Attention mask for the input.
        :return: Extracted features.
        """
        raise NotImplementedError("Subclasses must implement feature_extraction().")
    
    def save_features(self, dataloader: DataLoader, output_dir: str):
        """
        Save features to the output directory.
        """
        self.eval()
        os.makedirs(output_dir, exist_ok=True)
        
        with torch.no_grad():
            for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Extracting features"):
                pixel_values = batch.get("pixel_values_videos")
                input_ids = batch.get("input_ids")
                attention_mask = batch.get("attention_mask")
                
                if pixel_values is not None:
                    pixel_values = pixel_values.squeeze(1).to(self.device)
                if input_ids is not None:
                    input_ids = input_ids.squeeze(1).to(self.device)
                    attention_mask = attention_mask.squeeze(1).to(self.device)

                features = self.feature_extraction(pixel_values, input_ids, attention_mask)
                labels = batch["labels"]
                data = {
                    "features": features.cpu(),
                    "labels": labels.cpu()
                }
                
                torch.save(data, os.path.join(output_dir, f"features_{step}.pt"))
    
    def set_weights(self, weights):
        """
        Set the model weights from a given path.
        """
        self.pos_weights = weights
        
    def save_model(self):
        """
        Save the model to a file.
        """
        model_path = os.path.join("models/saved", self.model_name)
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_path, "model.pth"))   
    
    def save_checkpoint(self, epoch, optimizer, scheduler):
        """
        Save the model checkpoint.
        """
        checkpoint_path = os.path.join("models/saved", self.model_name, "checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(checkpoint_path, f"checkpoint_{epoch}.pth"))     
        
    def define_optimizer(self, 
                         optimizer_name: str, 
                         learning_rate: float,
                         momentum: float = None,
                         weight_decay: float = None):
        """
        Defines the optimizer for the model.
        By now you can choose between Adam and SGD.
        """
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                         lr=learning_rate)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                        lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                          lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer {optimizer_name} not available")
        return optimizer
    
    def define_criterion(self, 
                         criterion_name: str, 
                         pos_weight: torch.Tensor = None):
        """
        Defines the criterion for the model.
        By now you can choose between BCE and CrossEntropy.
        """
        if criterion_name == "bce":
            criterion = torch.nn.BCEWithLogitsLoss()
        elif criterion_name == "crossentropy":
            criterion = torch.nn.CrossEntropyLoss()
        elif criterion_name == "wbce":
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
        else:
            raise ValueError(f"Criterion {criterion_name} not available")
        return criterion
    
    def define_scheduler(self, 
                         scheduler_name: str, 
                         optimizer: torch.optim.Optimizer = None,
                         epochs: int = None, 
                         patience: int = None,
                         step_size: int = 5, 
                         gamma: float = 0.1,
                         eta_min: float = 0,
                         factor: float = 0.1,
                         mode: str = "min"):
        """
        Defines the scheduler for the model.
        By now you can choose between StepLR and CosineAnnealingLR.
        """
        if scheduler_name == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == "cosineannealinglr":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
        elif scheduler_name == "reduceonplateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
        else:
            raise ValueError(f"Scheduler {scheduler_name} not available")
        return scheduler

    def export_tokens(self, 
                      video_folder: str, 
                      annotation_folder: str, 
                      output_folder: str,
                      clip_length: int,
                      overlapping: float,
                      frame_per_second: int,
                      prompt: str = "Describe the scene",
                      system_message: str = "You are a helpful assistant.",
                      logger: logging.Logger = None):
        """
        Export processed tokens ready to be taken as input for the model.
        """
        logger.info(f"Exporting tokens for model {self.model_name}")
        logger.info(f"Video folder: {video_folder}")
        logger.info(f"Annotation folder: {annotation_folder}")
        logger.info(f"Output folder: {output_folder}")
        logger.info(f"Clip length: {clip_length}")
        logger.info(f"Overlapping: {overlapping}")
        logger.info(f"Frame per second: {frame_per_second}")
        
        # sort the video and annotation files
        video_files = sorted(glob.glob(os.path.join(video_folder, "*.mp4")))
        annotation_files = sorted(glob.glob(os.path.join(annotation_folder, "*.txt")))
        logger.info(f"Found {len(video_files)} video files and {len(annotation_files)} annotation files.")
        
        folder_name = f'{clip_length}sec_{frame_per_second}fps'
        output_folder = os.path.join(output_folder, folder_name)
        logger.info(f"Output folder: {output_folder}")
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"Created output folder: {output_folder}")
        logger.info(f"Exporting tokens to {output_folder}")
        
        for i, _ in enumerate(video_files):
            
            logger.info("-" * 20)
            logger.info(f"Processing video {i + 1}/{len(video_files)}")
            logger.info("-" * 20)
            
            video_file = video_files[i]
            annotation_file = annotation_files[i]
            logger.info(f"Processing {video_file} and {annotation_file}")
            annotation = self.read_annotations(annotation_file)
            
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                logger.error(f"Error opening video file {video_file}")
                continue
        
            fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"Video FPS: {fps}")
            logger.info(f"Expected FPS: {frame_per_second}")
            frame_interval = int(fps / frame_per_second)
            if frame_interval < 1:
                frame_interval = 1
            logger.info(f"Frame interval: {frame_interval}")
            
            frame_per_clip = int(clip_length * frame_per_second)
            overlapping_frames = int(overlapping * frame_per_clip)
            logger.info(f"Frame per clip: {frame_per_clip}")
            logger.info(f"Overlapping frames: {overlapping_frames}")
            
            frame_index = 0
            frames_list = []
            clip_index = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_index % frame_interval == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_list.append(frame)
                    
                    if len(frames_list) == frame_per_clip:
                        label = self.label_clip((frame_index / fps) * 1000, clip_length * 1000, annotation)
                        label_str = "_".join(str(x) for x in label)
                        tokens = self.process_input(frames_list, prompt, system_message)
                        file_name = "video_" + str(i) + "_clip_" + str(clip_index) + "_" + label_str + ".pt"
                        torch.save(tokens, os.path.join(output_folder, file_name))
                        slide = frame_per_clip - overlapping_frames
                        frames_list = frames_list[slide:]
                        
                        clip_index += 1
                frame_index += 1
                
            cap.release()
            logger.info(f"Finished processing {video_file}")
            logger.info(f"Exported {clip_index} clips from {video_file}")
            logger.info("-" * 20)
    
    def read_annotations(self, file_path):
        """
        Reads an annotation .txt file and returns a list of tuples:
        (label: str, start: int, end: int, length: int)

        Assumes each line has a variable-length label followed by three integer fields.
        """
        annotations = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                # Find first numeric token; tokens before that form the label
                idx = next((i for i, p in enumerate(parts) if p.isdigit()), None)
                if idx is None or idx + 2 >= len(parts):
                    continue  # skip malformed lines
                label = ' '.join(parts[:idx])
                ann_start = int(parts[idx])
                ann_end = int(parts[idx + 1])
                ann_length = int(parts[idx + 2])
                annotations.append((label, ann_start, ann_end, ann_length))
        return annotations
            
    def label_clip(self, clip_start, clip_length, annotations):
        """
        Label the clip based on the annotations.
        """
        # Initialize labels and overlap accumulators
        labels = [0, 0, 0, 0]
        overlap = [0, 0, 0, 0]

        clip_end = clip_start + clip_length

        # Mapping of annotation labels to indices
        category_map = {
            'Baby visible': 0,
            'CPAP': 1, 'PPV': 1,
            'Stimulation trunk': 2, 'Stimulation back/nates': 2,
            'Suction': 3
        }

        for ann in annotations:
            ann_label, ann_start, ann_end, _ = ann
            if ann_label not in category_map:
                continue
            # Compute overlap duration
            start_overlap = max(clip_start, ann_start)
            end_overlap = min(clip_end, ann_end)
            dur = end_overlap - start_overlap
            if dur > 0:
                idx = category_map[ann_label]
                overlap[idx] += dur

        # Determine labels based on >50% coverage
        threshold = clip_length / 2
        for i in range(len(labels)):
            if overlap[i] > threshold:
                labels[i] = 1

        return labels
    
    def metric_computation(self,
                        logits: torch.Tensor,
                        labels: torch.Tensor,
                        threshold: float = 0.5):
        """
        Compute TP/FP/FN/TN and derived metrics for a multi-label task.
        """

        # make everything boolean
        preds = (logits.sigmoid() >= threshold)
        truths = labels.bool()

        # now logical ops do what you expect
        TP =   (preds &  truths).sum(dim=0).cpu().numpy()
        FP =   (preds & ~truths).sum(dim=0).cpu().numpy()
        FN =  (~preds &  truths).sum(dim=0).cpu().numpy()
        TN =  (~preds & ~truths).sum(dim=0).cpu().numpy()

        # per-class accuracy
        total = TP + FP + FN + TN
        acc_per_class = (TP + TN) / total.clip(min=1)

        # precision/recall/f1 via sklearn
        p, r, f1, _ = precision_recall_fscore_support(
            truths.cpu().numpy().astype(int),
            preds.cpu().numpy().astype(int),
            average=None,
            zero_division=0
        )

        metrics = {
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "accuracy":      acc_per_class,
            "precision":     p,
            "recall":        r,
            "f1":            f1,
            "accuracy_macro":  acc_per_class.mean().item(),
            "precision_macro": p.mean().item(),
            "recall_macro":    r.mean().item(),
            "f1_macro":        f1.mean().item()
        }
        return metrics

    def log_wandb(self, wandb_run, epoch, train_loss, train_metrics, val_loss=None, val_metrics=None):
        """
        Log the training and validation metrics to Weights & Biases.
        """
        class_names = ["Baby visible", "Ventilation", "Stimulation", "Suction"]
        final_log = {}
        
        for i, class_name in enumerate(class_names):
            matrix = np.array([
                [train_metrics["TN"][i], train_metrics["FP"][i]],
                [train_metrics["FN"][i], train_metrics["TP"][i]]
            ])
            final_log[f"train/{class_name}/Confusion_matrix"] = wandb.Image(matrix, caption=f"Confusion matrix for {class_name}")
            final_log[f"train/{class_name}/accuracy"] = train_metrics["accuracy"][i]
            final_log[f"train/{class_name}/precision"] = train_metrics["precision"][i]
            final_log[f"train/{class_name}/recall"] = train_metrics["recall"][i]
            final_log[f"train/{class_name}/f1"] = train_metrics["f1"][i]
            
            if val_metrics is not None:
                matrix = np.array([
                    [val_metrics["TN"][i], val_metrics["FP"][i]],
                    [val_metrics["FN"][i], val_metrics["TP"][i]]
                ])
                final_log[f"val/{class_name}/Confusion_matrix"] = wandb.Image(matrix, caption=f"Confusion matrix for {class_name}")
                final_log[f"val/{class_name}/accuracy"] = val_metrics["accuracy"][i]
                final_log[f"val/{class_name}/precision"] = val_metrics["precision"][i]
                final_log[f"val/{class_name}/recall"] = val_metrics["recall"][i]
                final_log[f"val/{class_name}/f1"] = val_metrics["f1"][i]
        final_log["epoch"] = epoch
        final_log["train/loss"] = train_loss
        if val_loss is not None:
            final_log["val/loss"] = val_loss
        final_log["train/accuracy_macro"] = train_metrics["accuracy_macro"]
        final_log["train/precision_macro"] = train_metrics["precision_macro"]
        final_log["train/recall_macro"] = train_metrics["recall_macro"]
        final_log["train/f1_macro"] = train_metrics["f1_macro"]
        if val_metrics is not None:
            final_log["val/accuracy_macro"] = val_metrics["accuracy_macro"]
            final_log["val/precision_macro"] = val_metrics["precision_macro"]
            final_log["val/recall_macro"] = val_metrics["recall_macro"]
            final_log["val/f1_macro"] = val_metrics["f1_macro"]
        wandb_run.log(final_log, step=epoch)

    def log_test_wandb(self, wandb_run, test_metrics, test_loss=None):
        """
        Log the test metrics to Weights & Biases.
        """
        class_names = ["Baby visible", "Ventilation", "Stimulation", "Suction"]
        final_log = {}
        for i, class_name in enumerate(class_names):
            matrix = np.array([
                [test_metrics["TN"][i], test_metrics["FP"][i]],
                [test_metrics["FN"][i], test_metrics["TP"][i]]
            ])
            final_log[f"test/{class_name}/Confusion_matrix"] = wandb.Image(matrix, caption=f"Confusion matrix for {class_name}")
            final_log[f"test/{class_name}/accuracy"] = test_metrics["accuracy"][i]
            final_log[f"test/{class_name}/precision"] = test_metrics["precision"][i]
            final_log[f"test/{class_name}/recall"] = test_metrics["recall"][i]
            final_log[f"test/{class_name}/f1"] = test_metrics["f1"][i]
        final_log["test/accuracy_macro"] = test_metrics["accuracy_macro"]
        final_log["test/precision_macro"] = test_metrics["precision_macro"]
        final_log["test/recall_macro"] = test_metrics["recall_macro"]
        final_log["test/f1_macro"] = test_metrics["f1_macro"]
        if test_loss is not None:
            final_log["test/loss"] = test_loss
        wandb_run.log(final_log)
        
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, max_grad_norm=1.0):
        """
        To be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement train_epoch().")

    def eval_epoch(self, dataloader: DataLoader, criterion: nn.Module):
        """
        To be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement eval_epoch().")
        
    def train_model(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        epochs: int = 5,
        optimizer: torch.optim.Optimizer = None,
        criterion: nn.Module = None,
        threshold: float = 0.5,
        scheduler: lr_scheduler._LRScheduler = None,
        patience: int = 3,
        show_progress: bool = True,
        prior_probability: Tensor = None,
        wandb_run=None,
        logger: logging.Logger = None
    ):
        """
        Train the model.
        """
        best_val_loss = float("inf")
        no_improve = 0
        logger.info(f"Training {self.model_name} model")
                
        bias = -(1 - prior_probability).log() + prior_probability.log()
        self.classifier.bias.data = bias.to(self.device)
        logger.info(f"Initial bias for the model: {bias}")
        
        epo_iter = tqdm(range(1, epochs + 1), desc="Epochs", unit="epoch") if show_progress else range(1, epochs + 1)
        
        for epoch in epo_iter:
            logger.info(f"Starting epoch {epoch}/{epochs}")
            train_loss, train_logits, train_labels = self.train_epoch(train_dataloader, optimizer, criterion)
            logger.info(f"Finished training epoch {epoch}")
            logger.info(f"Train loss: {train_loss:.4f}")
            
            log_msg = f"[{epoch:02d}/{epochs}] train-loss: {train_loss:.4f}"
            train_metrics = self.metric_computation(train_logits, train_labels, threshold)
            log_msg += f" | train-f1: {train_metrics['f1_macro']:.4f}"
            logger.info(f"Train metrics: {train_metrics}")

            if val_dataloader is not None:
                logger.info(f"Starting validation epoch {epoch}")
                val_loss, val_logits, val_labels = self.eval_epoch(val_dataloader, criterion)
                logger.info(f"Finished validation epoch {epoch}")
                logger.info(f"Validation loss: {val_loss:.4f}")
                
                log_msg += f" | val-loss: {val_loss:.4f}"
                val_metrics = self.metric_computation(val_logits, val_labels, threshold)
                log_msg += f" | val-f1: {val_metrics['f1_macro']:.4f}"
                logger.info(f"Validation metrics: {val_metrics}")
                
            if scheduler is not None:
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                logger.info(f"Scheduler step: {scheduler.get_last_lr()}")
            if show_progress:
                epo_iter.set_postfix_str(log_msg)
            if wandb_run is not None:
                logger.info(f"Logging to Weights & Biases")
                self.log_wandb(wandb_run = wandb_run, 
                               epoch = epoch, 
                               train_loss = train_loss, 
                               train_metrics = train_metrics, 
                               val_loss = val_loss if val_dataloader is not None else None,
                               val_metrics = val_metrics if val_dataloader is not None else None)
                logger.info(f"Finished logging to Weights & Biases")
            logger.info(f"Saving checkpoint for epoch {epoch}")
            self.save_checkpoint(epoch = epoch,
                                 optimizer = optimizer,
                                 scheduler = scheduler)
            logger.info(f"Finished saving checkpoint for epoch {epoch}")
            
            if val_dataloader is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                logger.info(f"Validation loss improved to {best_val_loss:.4f} at epoch {epoch}")
            else:
                no_improve += 1
                logger.info(f"No improvement in validation loss for {no_improve} epochs")
                if no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch} with patience {patience}.")
                    break
        
        logger.info(f"Training finished after {epoch} epochs")
        logger.info("Saving final model")
        self.save_model()
        logger.info("Final model saved")
        
        results = {"train_loss": train_loss,
                   "train_metrics": train_metrics,
                   "val_loss": val_loss if val_dataloader is not None else None,
                   "val_metrics": val_metrics if val_dataloader is not None else None}

        return results
    
    def test_model(self,
                   test_dataloader: DataLoader,
                   criterion: nn.Module = None,
                   threshold: float = 0.5,
                   wandb_run=None):
        """
        Test the model.
        """
        test_loss, test_logits, test_labels = self.eval_epoch(test_dataloader, criterion)
        
        test_metrics = self.metric_computation(test_logits, test_labels, threshold)
        
        if wandb_run is not None:
            self.log_test_wandb(wandb_run = wandb_run, 
                                test_loss = test_loss, 
                                test_metrics = test_metrics)
        
        return {"test_loss": test_loss,
                "test_metrics": test_metrics}