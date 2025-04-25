import torch.nn as nn
import logging
import os
from tqdm import tqdm
import cv2
import numpy as np
import torch
import glob
from torch.utils.data import Dataset, DataLoader


class BaseModel(nn.Module):
    """
    An abstract base class for video models.
    Provides a common interface for training, inference, and last-layer modifications.
    """
    def __init__(self, device = "cuda", model_name: str = "baseModel", 
                 system_message: str = "You are an helpful assistant.", 
                 prompt: str = "Describe the scene"):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.device = torch.device(device)
        self.to(self.device)
    
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement the forward method.")

    def modify_last_layer(self, new_layer_config):
        """
        Modify the last layer(s) of the model.
        :param new_layer_config: A new layer or sequential block to replace the final layer(s).
        """
        raise NotImplementedError("Subclasses must implement modify_last_layer().")
    
    def process_input(self, x):
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
                pixel_values = batch.get("pixel_values")
                input_ids = batch.get("input_ids")
                attention_mask = batch.get("attention_mask")
                
                if pixel_values is not None:
                    pixel_values = pixel_values
                if input_ids is not None:
                    input_ids = input_ids
                    attention_mask = attention_mask

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
        
        
    def define_optimizer(self, optimizer_name, learning_rate, momentum = None, weight_decay = None):
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
    
    def define_criterion(self, criterion_name, pos_weight=None, neg_weight=None):
        """
        Defines the criterion for the model.
        By now you can choose between BCE and CrossEntropy.
        """
        if criterion_name == "bce":
            criterion = torch.nn.BCEWithLogitsLoss()
        elif criterion_name == "crossentropy":
            criterion = torch.nn.CrossEntropyLoss()
        elif criterion_name == "wbce":
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            raise ValueError(f"Criterion {criterion_name} not available")
        return criterion

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
        
        folder_name = f'{clip_length}sec_{overlapping}overlap_{frame_per_second}fps'
        output_folder = os.path.join(output_folder, self.model_name, folder_name)
        logger.info(f"Output folder: {output_folder}")
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"Created output folder: {output_folder}")
        logger.info(f"Exporting tokens to {output_folder}")
        
        for i, _ in enumerate(video_files):
            
            logger.info("-" * 20)
            logger.info(f"Processing video {i + 1}/{len(video_files)}")
            logger.info("-" * 20)
            
            video_file = os.path.join(video_folder, video_files[i])
            annotation_file = os.path.join(annotation_folder, annotation_files[i])
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
                        file_name = "video_" + str(i) + "_clip_" + str(clip_index) + label_str + ".pt"
                        torch.save(os.path.join(output_folder, file_name), tokens)
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

