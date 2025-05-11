import torch
import torch.nn as nn
from transformers import (
    VideoLlavaForConditionalGeneration,
    VideoLlavaProcessor,
    VideoLlavaConfig,
    TimesformerConfig,
)
from models.basemodel import BaseModel
from models.timesformer import TimeSformer

class VideoLlava(BaseModel):
    """
    VideoLLaVA wrapper that substitutes the visual encoder with a pre-trained TimeSformer.
    """
    def __init__(
        self,
        timesformer_checkpoint: str,
        base_model_id: str = "LanguageBind/Video-LLaVA-7B-hf",
        text_model_id: str = "lmsys/vicuna-7b-v1.5",
        device: str = None,
        num_classes: int = 4,
        num_frames: int = 8,
    ):
        super().__init__(device=device, model_name="VideoLLaVA-Timesformer")
        # Device setup
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load processor for VideoLLaVA
        self.processor = VideoLlavaProcessor.from_pretrained(base_model_id)

        # Load TimeSformer config for vision side
        t_cfg = TimesformerConfig.from_pretrained(timesformer_checkpoint)

        # Build new VideoLLaVA config
        llava_cfg = VideoLlavaConfig(
            text_config=text_model_id,
            vision_config=t_cfg,
            mm_hidden_size=t_cfg.hidden_size,
            video_seq_length=num_frames,
        )

        # Instantiate VideoLLaVA model
        self.backbone = VideoLlavaForConditionalGeneration(llava_cfg).to(self.device)

        # Load the pre-trained TimeSformer as the vision tower
        self.timesformer = TimeSformer(
            base_model_id=timesformer_checkpoint,
            device=device,
            num_classes=num_classes,
            num_frames=num_frames,
        )
        # Substitute the vision tower and update config
        self.backbone.vision_tower = self.timesformer.backbone
        self.backbone.config.vision_config = self.timesformer.backbone.config
        self.backbone.config.mm_hidden_size = self.timesformer.backbone.config.hidden_size

        # Recreate the multimodal projector to match new hidden size
        self.backbone.mm_projector = nn.Linear(
            self.backbone.config.mm_hidden_size,
            self.backbone.config.hidden_size,
            bias=True
        ).to(self.device)
        self.backbone.mm_projector.apply(self.backbone._init_weights)

        # Freeze VideoLLaVA parameters
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False

        # Classification head on top of LLM hidden states
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        ).to(self.device)

    def forward(self, videos, text_prompts=None):
        """
        Forward pass that encodes videos (and optional text) through VideoLLaVA + classifier.
        """
        # Preprocess inputs
        inputs = self.processor(videos=videos, return_tensors="pt").to(self.device)
        if text_prompts is not None:
            txt = self.processor(text=text_prompts, return_tensors="pt")
            inputs["input_ids"] = txt.input_ids.to(self.device)
            inputs["attention_mask"] = txt.attention_mask.to(self.device)

        # Forward through VideoLLaVA, requesting hidden states
        outputs = self.backbone(
            **inputs,
            return_dict=True,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)
        cls_token = hidden_states[:, 0, :]
        logits = self.classifier(cls_token)
        return logits
