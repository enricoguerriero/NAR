import torch
import torch.nn as nn
from transformers import LlavaNextVideoForConditionalGeneration, AutoProcessor
from models.basemodel import BaseModel
from peft import LoraConfig, get_peft_model, TaskType

class LlavaNext(BaseModel):
    
    def __init__(self, 
                 checkpoint_path: str = None, 
                 base_model_id: str = "llava-hf/LLaVA-NeXT-Video-7B-hf", 
                 device=None, 
                 num_classes=4,
                 lora_modality = "language"):
        super().__init__()
        
        self.model_name = "LlavaNext"
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if checkpoint_path:
            self.backbone = LlavaNextVideoForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.backbone = LlavaNextVideoForConditionalGeneration.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16
            ).to(self.device)
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        if lora_modality == "language":
            self.backbone.language_model = get_peft_model(self.backbone.language_model, 
                                           lora_config)
        elif lora_modality == "total":
            self.backbone = get_peft_model(self.backbone, 
                                           lora_config)
        
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        ).to(self.device)
        
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
            
    
    def forward(self, 
                pixel_values = None, 
                input_ids = None, 
                attention_mask = None, 
                labels = None,
                loss_fct = None):
        
        outputs = self.backbone(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        
        last_layer = outputs.hidden_states[-1]
        pooled = last_layer.mean(dim=1)
        logits = self.classifier(pooled)
        
        if labels is not None and loss_fct is not None:
            loss = loss_fct(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"loss": None, "logits": logits}
        
    def prompt_definition(self,
                          question: str = None,
                          system_message: str = None):
        
        prompt = f"USER: {system_message}\n<video>\n{question}\nASSISTANT:"
        
        return prompt
    
    def process_input(self,
                      video: list, 
                      prompt: str, 
                      system_message: str):
        final_prompt = self.prompt_definition(
            question=prompt,
            system_message=system_message
        )
        inputs = self.processor(
            videos = video,
            text=final_prompt,
            return_tensors="pt",
            padding=True,
        )
        
        return inputs
    
    def set_freezing_condition(self,
                               mode: str):
        
        for param in self.parameters():
            param.requires_grad = False
        
        if mode == "none":
            for param in self.parameters():
                param.requires_grad = True
        
        elif mode == "lora":
            for name, param in self.backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
            for param in self.classifier.parameters():
                param.requires_grad = True
        
        elif mode == "all":
            for param in self.parameters():
                param.requires_grad = True
                
        else:
            raise ValueError(f"Unknown freezing condition: {mode}. Use 'none', 'lora', or 'all'.")
        
        return False
    
    def unfreeze_schedule(self, x):
        pass
    
    def collate_fn_tokens(self, batch):
        pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
        input_ids = torch.cat([item["input_ids"] for item in batch], dim=0)
        attention_mask = torch.cat([item["attention_mask"] for item in batch], dim=0)
        labels = torch.stack([item["labels"] for item in batch], dim=0)
        return {"pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels}
        


    # Training classifier from features:
    
    def forward_classifier(self,
                           features,
                           labels, 
                           loss_fct = None):
        
        logits = self.classifier(features.float())
        
        if loss_fct is not None:
            loss = loss_fct(logits, labels.float())
            return {"loss": loss, "logits": logits}
        else:
            return {"loss": None, "logits": logits}
        
    def get_input(self, batch):
        for key in batch:
            batch[key] = batch[key].squeeze(0).to(self.device)
        return batch
    
    def feature_extraction(self, pixel_values=None, input_ids=None, attention_mask=None):
        """
        Extract features from the model.
        """
        # print(f"pixel_values: {pixel_values}", flush=True)
        # print(f"input_ids: {input_ids}", flush=True)
        # print(f"attention_mask: {attention_mask}", flush=True)
        outputs = self.backbone(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )

        last_layer = outputs.hidden_states[-1] # (batch, seq_len, hidden_dim)
        pooled = last_layer.mean(dim=1)  # CLS token representation
        return pooled.float()
    
    