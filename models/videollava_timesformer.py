import torch
import torch.nn as nn
from transformers import (
    VideoLlavaForConditionalGeneration,
    VideoLlavaProcessor,
    AutoConfig,
)
from models.basemodel import BaseModel
from models.timesformer import TimeSformer


class VideoLlavaTimeSformer(BaseModel):
    """
    Video-LLaVA wrapper that swaps the vision tower for a pre-trained TimeSformer
    and exposes a video-aware classification head.
    """
    def __init__(
        self,
        timesformer_checkpoint: str,
        base_model_id: str = "LanguageBind/Video-LLaVA-7B-hf",
        device: str = None,
        num_classes: int = 4,
        num_frames: int = 8,
        freeze_llm: bool = True,
    ):
        super().__init__(device=device, model_name="VideoLLaVA-TimeSformer")
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ---------- Load processor & full pretrained Video-LLaVA ----------
        self.processor = VideoLlavaProcessor.from_pretrained(base_model_id)

        self.backbone = VideoLlavaForConditionalGeneration.from_pretrained(
            base_model_id, torch_dtype=torch.float16
        ).to(self.device)

        # ---------- Plug-in TimeSformer vision tower ----------
        self.timesformer = TimeSformer(
            base_model_id=timesformer_checkpoint,
            device=self.device,
            num_classes=num_classes,
            num_frames=num_frames,
        )
        self.backbone.vision_tower = self.timesformer.backbone
        self.backbone.config.vision_config = self.timesformer.backbone.config
        self.backbone.config.mm_hidden_size = self.timesformer.backbone.config.hidden_size
        self.backbone.cfg_num_frames = num_frames   # handy attr for later

        # ---------- Rebuild multimodal projector ----------
        self.backbone.mm_projector = nn.Linear(
            self.backbone.config.mm_hidden_size,
            self.backbone.config.text_config.hidden_size,
            bias=True,
            device=self.device,
        )
        self.backbone._init_weights(self.backbone.mm_projector)

        # ---------- Optional freezing ----------
        if freeze_llm:
            keep_trainable = {"vision_tower.", "mm_projector", "classifier"}
            for n, p in self.backbone.named_parameters():
                p.requires_grad = any(n.startswith(k) for k in keep_trainable)

        # ---------- Classification head (late-fusion style) ----------
        hidden_t = self.backbone.config.text_config.hidden_size
        self.fuse = nn.Sequential(
            nn.Linear(hidden_t * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        ).to(self.device)

    # ------------------------------------------------------------------ #
    #                               forward                              #
    # ------------------------------------------------------------------ #
    def forward(self, videos, text_prompts):
        """
        videos        : Tensor (B, T, C, H, W) *or* list of PIL frames
        text_prompts  : list[str] length B
        returns logits: (B, num_classes)
        """
        # 1) Pre-process both modalities
        vid_inputs = self.processor(
            videos=videos, return_tensors="pt", padding=True
        ).to(self.device)
        txt_inputs = self.processor(
            text=text_prompts, return_tensors="pt", padding=True
        )
        vid_inputs.update(
            {
                "input_ids": txt_inputs.input_ids.to(self.device),
                "attention_mask": txt_inputs.attention_mask.to(self.device),
            }
        )

        # 2) Forward pass; grab last hidden states
        out = self.backbone(
            **vid_inputs, return_dict=True, output_hidden_states=True
        )
        h = out.hidden_states[-1]  # (B, seq_len, hidden)

        # Token layout = [video_tokens] + [text_tokens]
        n_vid = self.backbone.cfg_num_frames  # may differ if processor pads
        vid_tok = h[:, : n_vid, :].mean(1)    # (B, hidden)
        txt_tok = h[:, n_vid:, :].mean(1)     # (B, hidden)

        fused = torch.cat([vid_tok, txt_tok], dim=-1)
        logits = self.fuse(fused)
        return logits
