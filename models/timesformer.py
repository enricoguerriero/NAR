from models.basemodel import BaseModel
import torch
from transformers import AutoImageProcessor, TimesformerForVideoClassification, TimesformerConfig
from torch.amp import GradScaler, autocast
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from torch.optim import Optimizer, lr_scheduler
from torch.nn.utils import clip_grad_norm_

class TimeSformer(BaseModel):
    
    def __init__(self,  
                 base_model_id: str = "facebook/timesformer-base-finetuned-ssv2", 
                 device: str = "cuda", 
                 num_classes: int = 4):
        """
        Initialize the TimeSformer model.
        """
        super().__init__(device=device, model_name="TimeSformer")
        self.model_name = "TimeSformer"
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        config = TimesformerConfig.from_pretrained(base_model_id)
        self.processor = AutoImageProcessor.from_pretrained(base_model_id)
        self.backbone = TimesformerForVideoClassification.from_pretrained(base_model_id)
        self.backbone.gradient_checkpointing_enable() # Enable gradient checkpointing - save GPU memory
        self.backbone = torch.compile(self.backbone) # speed up training
        if torch.cuda.device_count() > 1:
            self.backbone = torch.nn.DataParallel(self.backbone) # Use DataParallel if multiple GPUs are available
                
        self.classifier = torch.nn.Linear(config.hidden_size, num_classes, bias=True)
        self.backbone.to(self.device)
        self.classifier.to(self.device)
        
    def forward(self, pixel_values, labels = None, loss_fct = None): 
        """
        Forward pass through the model.
        """
        pixel_values = pixel_values.to(self.device)
        labels = labels.to(self.device) if labels is not None else None
        outputs = self.backbone(pixel_values)
        hidden_states = outputs.hidden_states[-1]
        features = hidden_states.mean(dim=1)
        logits = self.classifier(features)
        loss = None
        if labels is not None and loss_fct is not None:
            loss = loss_fct(logits, labels.float())
        
        return {"loss": loss, "logits": logits}
    
    def process_input(self, frames_list, prompt = None, system_message = None):
        """
        Process the input frames and return the pixel values.
        """
        inputs = self.processor(frames_list,
                                return_tensors="pt",
                                padding=True)
        return inputs
    
    def train_epoch(self, dataloader, optimizer, loss_fct, max_grad_norm=1.0):
        """
        Train the model for one epoch.
        """
        self.backbone.train()
        total_loss = 0.0
        total_samples = 0
        labels_list = []
        logits_list = []
        # scaler  = GradScaler()
        
        for batch in tqdm(dataloader, desc="Training", unit="batch"):
            optimizer.zero_grad()
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # with autocast(device_type='cuda'):
                # outputs = self.forward(pixel_values, labels, loss_fct)
                # loss = outputs["loss"]
                # logits = outputs["logits"]
            outputs = self.forward(pixel_values, labels, loss_fct)
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            labels_list.append(labels.cpu())
            logits_list.append(logits.cpu())    
                       
            loss.backward()
            optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)
            # clip_grad_norm_(self.parameters(), max_grad_norm)
            # scaler.step(optimizer)
            # scaler.update()
            
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            
        logits_tensor = torch.cat(logits_list)
        labels_tensor = torch.cat(labels_list)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return avg_loss, logits_tensor, labels_tensor
    
    def eval_epoch(self, dataloader, loss_fct):
        """
        Evaluate the model for one epoch.
        """
        self.backbone.eval()
        total_loss = 0.0
        total_samples = 0
        labels_list = []
        logits_list = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.forward(pixel_values, labels, loss_fct)
                loss = outputs["loss"]
                logits = outputs["logits"]
                
                labels_list.append(labels.cpu())
                logits_list.append(logits.cpu())
                
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)
        
        logits_tensor = torch.cat(logits_list)
        labels_tensor = torch.cat(labels_list)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return avg_loss, logits_tensor, labels_tensor
