import torch
import torch.nn as nn
from transformers import (
    VideoLlavaForConditionalGeneration,
    VideoLlavaProcessor,
)
from models.basemodel import BaseModel
from models.timesformer import TimeSformer


class VideoLlavaTimeSformer(BaseModel):
    """
    Video-LLaVA wrapper that swaps the vision tower for a pre-trained
    TimeSformer and exposes a video-aware classification head.
    """

    # ------------- constructor -------------------------------------------------
    def __init__(
        self,
        timesformer_checkpoint: str,
        base_model_id: str = "LanguageBind/Video-LLaVA-7B-hf",
        device: str | None = None,
        num_classes: int = 4,
        num_frames: int = 8,
        freeze_llm: bool = True,
        debug: bool = False,                 
    ):
        super().__init__(device=device, model_name="VideoLLaVA-TimeSformer")
        self.debug = debug
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ------------ processor & pretrained backbone -------------------------
        self.processor = VideoLlavaProcessor.from_pretrained(base_model_id)
        self.backbone = VideoLlavaForConditionalGeneration.from_pretrained(
            base_model_id, torch_dtype=torch.float16
        ).to(self.device)

        # ------------ plug in TimeSformer -------------------------------------
        self.timesformer = TimeSformer(
            base_model_id=timesformer_checkpoint,
            device=self.device,
            num_classes=num_classes,
            num_frames=num_frames,
        )
        self.backbone.vision_tower = self.timesformer.backbone
        self.backbone.config.vision_config = self.timesformer.backbone.config
        self.backbone.config.mm_hidden_size = self.timesformer.backbone.config.hidden_size
        self.backbone.cfg_num_frames = num_frames

        # ------------ rebuild mm_projector ------------------------------------
        self.backbone.mm_projector = nn.Linear(
            self.backbone.config.mm_hidden_size,
            self.backbone.config.text_config.hidden_size,
            bias=True,
            device=self.device,
        )
        self.backbone._init_weights(self.backbone.mm_projector)

        # ------------ optional freezing ---------------------------------------
        if freeze_llm:
            # keep_trainable = {"vision_tower.", "mm_projector", "fuse"}
            keep_trainable = {"mm_projector", "fuse"}
            for n, p in self.backbone.named_parameters():
                p.requires_grad = any(n.startswith(k) for k in keep_trainable)

        # ------------ classification head -------------------------------------
        hidden_t = self.backbone.config.text_config.hidden_size
        self.fuse = (
            nn.Sequential(
                nn.Linear(hidden_t * 2, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, num_classes),
            )
            .to(self.device)
        )

        # ------------ print summary -------------------------------------------
        if self.debug:
            self._print_param_summary()

    # ------------- private helpers -------------------------------------------
    def _print_param_summary(self):
        def count(module):
            return sum(p.numel() for p in module.parameters())

        groups = {
            "vision_tower (TimeSformer)": self.backbone.vision_tower,
            "mm_projector": self.backbone.mm_projector,
            "Vicuna LLM": self.backbone.language_model,  # attribute from transformers
            "Fuse head": self.fuse,
        }

        print("\n──────── Model parameter summary ────────")
        tot, train = 0, 0
        for name, module in groups.items():
            n_params = count(module)
            n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
            tot += n_params
            train += n_train
            print(f"{name:<30}: {n_train/1e6:7.1f}M / {n_params/1e6:7.1f}M trainable")
        print(f"{'TOTAL':<30}: {train/1e6:7.1f}M / {tot/1e6:7.1f}M\n")

    # ------------- forward ----------------------------------------------------
    def forward(self, pixel_values_videos, input_ids, attention_mask):
        vid_inputs = {
            "pixel_values_videos": pixel_values_videos.to(self.device),
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }

        out = self.backbone(
            **vid_inputs, return_dict=True, output_hidden_states=True
        )
        h = out.hidden_states[-1]               # (B, seq_len, hidden)

        # split tokens
        n_vid = self.backbone.cfg_num_frames
        vid_tok = h[:, :n_vid, :].mean(1)       # (B, hidden_t)
        txt_tok = h[:, n_vid:, :].mean(1)       # (B, hidden_t)

        if self.debug:
            print(f"[DEBUG] vid_tok shape : {tuple(vid_tok.shape)}")
            print(f"[DEBUG] txt_tok shape : {tuple(txt_tok.shape)}")
            print(f"[DEBUG] vid_tok sample: {vid_tok[0, :8].tolist()}")

        fused  = torch.cat([vid_tok, txt_tok], dim=-1)
        logits = self.fuse(fused)               # (B, num_classes)

        if self.debug:
            print(f"[DEBUG] logits shape  : {tuple(logits.shape)}")
            print(f"[DEBUG] logits sample : {logits[0].tolist()}")

        return logits

