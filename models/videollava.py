import torch
import torch.nn as nn
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor, TrainingArguments, Trainer
from models.basemodel import BaseModel
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import os

class VideoLlava(BaseModel):
    def __init__(self, checkpoint_path: str = None, base_model_id: str = "LanguageBind/Video-LLaVA-7B-hf", device=None, num_classes=4):
        super().__init__()
        self.name = "llava_video_classifier"
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the processor and model
        self.processor = VideoLlavaProcessor.from_pretrained(base_model_id)
        if checkpoint_path:
            self.model = VideoLlavaForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.model = VideoLlavaForConditionalGeneration.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16
            ).to(self.device)

        hidden_size = self.model.get_input_embeddings().embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        
        self.pos_weights = None

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None, loss_fct=None):
        outputs = self.model(
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
        outputs = self.model(
            pixel_values_videos=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )

        last_layer = outputs.hidden_states[-1]
        pooled = last_layer[:, 0, :]  # CLS token representation
        return pooled.float()
    
 