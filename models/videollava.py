import torch
import torch.nn as nn
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from models.basemodel import BaseModel
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from torch.optim import Optimizer, lr_scheduler
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

class VideoLlava(BaseModel):
    def __init__(self, checkpoint_path: str = None, base_model_id: str = "LanguageBind/Video-LLaVA-7B-hf", device=None, num_classes=4):
        super().__init__()
        self.model_name = "VideoLLaVA"
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the processor and model
        self.processor = VideoLlavaProcessor.from_pretrained(base_model_id)
        if checkpoint_path:
            self.backbone = VideoLlavaForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.backbone = VideoLlavaForConditionalGeneration.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16
            ).to(self.device)

        hidden_size = self.backbone.get_input_embeddings().embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        
        self.pos_weights = None

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None, loss_fct=None):
        
        outputs = self.backbone(
            pixel_values_videos=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )

        last_layer = outputs.hidden_states[-1]  # Use CLS or first token
        pooled = last_layer[:, 0, :]  # CLS token representation
        logits = self.classifier(pooled.float())
        
        
        if labels is not None and loss_fct is not None:
            loss = loss_fct(logits, labels.float())
        else:
            loss = None
        
        return {"loss": loss, "logits": logits}

    def prompt_definition(self, question: str, system_message: str = "You are a helpful assistant."):
        """
        Build the prompt text for a given question.
        Here, we follow the recommended prompt format for Video LLaVA.
        """
        prompt = f"USER: <video>\n{system_message}\n{question}\nASSISTANT:"
        
        return prompt
    
    def process_input(self, video: list, prompt: str, system_message: str):
        
        final_prompt = self.prompt_definition(prompt, system_message)
        inputs = self.processor(
            text = final_prompt,
            videos = video,
            padding = True,
            return_tensors = "pt")
        
        return inputs
    
    def forward_classifier(self, features, labels, loss_fct=None):
        """
        Forward pass through the classifier.
        """
        logits = self.classifier(features.float())
        
        if loss_fct is not None:
            loss = loss_fct(logits, labels.float())
        else:
            loss = None
        
        return {"loss": loss, "logits": logits}
    
    def feature_extraction(self, pixel_values=None, input_ids=None, attention_mask=None):
        """
        Extract features from the model.
        """
        outputs = self.backbone(
            pixel_values_videos=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )

        last_layer = outputs.hidden_states[-1]
        pooled = last_layer[:, 0, :]  # CLS token representation
        return pooled.float()
 
    def train_classifier_epoch(self, dataloader, optimizer, loss_fct, max_grad_norm=1.0):
        """
        Train the classifier for one epoch.
        """
        self.classifier.train()
        total_loss = 0.0
        total_samples = 0
        labels_list = []
        logits_list = []
        scaler  = GradScaler()
        
        for batch in tqdm(dataloader, desc="Training Classifier", unit="batch"):
            optimizer.zero_grad()
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            labels = inputs.pop("labels")
            with autocast(device_type='cuda'):
                outputs = self.forward_classifier(**inputs, loss_fct=loss_fct)
            loss = outputs["loss"]
            logits = outputs["logits"]
            labels_list.append(labels.cpu())
            logits_list.append(logits.cpu())
            
            if loss is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(self.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)
        
        logits_tensor = torch.cat(logits_list)
        labels_tensor = torch.cat(labels_list)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return avg_loss, logits_tensor, labels_tensor
    
    def eval_classifier_epoch(self, dataloader, loss_fct):
        """
        Evaluate the classifier for one epoch.
        """
        self.classifier.eval()
        total_loss = 0.0
        total_samples = 0
        labels_list = []
        logits_list = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating Classifier", unit="batch"):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                labels = inputs.pop("labels")
                
                outputs = self.forward_classifier(**inputs, loss_fct=loss_fct)
                loss = outputs["loss"]
                logits = outputs["logits"]
                labels_list.append(labels)
                logits_list.append(logits)
                
                if loss is not None:
                    total_loss += loss.item() * labels.size(0)
                    total_samples += labels.size(0)
        
        logits_tensor = torch.cat(logits_list)
        labels_tensor = torch.cat(labels_list)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return avg_loss, logits_tensor, labels_tensor
        
    def train_classifier(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        epochs: int = 5,
        optimizer: Optimizer = None,
        criterion: nn.Module = None,
        threshold: float = 0.5,
        scheduler: lr_scheduler = None,
        patience: int = 3,
        show_progress: bool = True,
        wandb_run=None
    ):
        self.classifier.to(self.device)
        best_val_loss = float('inf')
        no_improve = 0

        epo_iter = range(1, epochs + 1)
        if show_progress:
            epo_iter = tqdm(epo_iter, desc="Epochs", unit="epoch")

        for epoch in epo_iter:
            train_loss, logits, labels = self.train_classifier_epoch(train_dataloader, optimizer, criterion)

            log_msg = f"[{epoch:02d}/{epochs}] train-loss: {train_loss:.4f}"
            train_metrics = self.metric_computation(logits, labels, threshold)
            log_msg += f" | train-f1: {train_metrics['f1_macro']:.4f}"

            if val_dataloader is not None:
                val_loss, val_logits, val_labels = self.eval_classifier_epoch(val_dataloader, criterion)
                
                log_msg += f" | val-loss: {val_loss:.4f}"
                val_metrics = self.metric_computation(val_logits, val_labels, threshold)
                log_msg += f" | val-f1: {val_metrics["f1_macro"]:.4f}"

            if scheduler is not None:
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            if show_progress:
                epo_iter.set_postfix_str(log_msg)
            if wandb_run is not None:
                self.log_wandb(wandb = wandb_run, 
                               epoch = epoch, 
                               train_loss = train_loss, 
                               train_metrics = train_metrics, 
                               val_loss = val_loss if val_dataloader is not None else None,
                               val_metrics = val_metrics if val_dataloader is not None else None)
            self.save_checkpoint(epoch = epoch,
                                 optimizer = optimizer,
                                 scheduler = scheduler)
            
            if val_dataloader is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch} with patience {patience}.")
                    break

        results = {"train_loss": train_loss,
                   "train_metrics": train_metrics,
                   "val_loss": val_loss if val_dataloader is not None else None,
                   "val_metrics": val_metrics if val_dataloader is not None else None}

        return results
    
    def test_classifier(
        self,
        test_dataloader: DataLoader,
        threshold: float = 0.5,
        wandb_run=None
    ):
        self.classifier.to(self.device)
        test_loss, logits, labels = self.eval_classifier_epoch(test_dataloader, None)
        
        test_metrics = self.metric_computation(logits, labels, threshold)        
        
        if wandb_run is not None:
            self.log_test_wandb(wandb = wandb_run, 
                           test_loss = test_loss, 
                           test_metrics = test_metrics)

        return {"test_loss": test_loss, "test_metrics": test_metrics}

    @torch.no_grad()
    def generate_answer(
        self,
        inputs,
        max_new_tokens: int = 128,
        **generate_kwargs,
    ) -> str:
        """
        Traditional generative usage of Video-LLaVA (zero-shot).

        Args:
            video: list of frames or a video file path the processor can read.
            question: the user question about the clip.
            system_message: high-level system instruction to prepend.
            max_new_tokens: length of the answer to generate.
            **generate_kwargs: any other `generate` kwargs (e.g. temperature, top_p).

        Returns:
            A single decoded answer string.
        """

        # Move everything to the right device / dtype
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        self.backbone.eval()
        generated_ids = self.backbone.generate(
            pixel_values_videos=inputs["pixel_values_videos"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

        # Decode
        answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # The answer often comes after the original prompt – strip it if desired
        return answer.split("ASSISTANT:")[-1].strip()