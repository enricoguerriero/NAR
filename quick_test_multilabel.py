"""
quick_test_multilabel.py
Minimal sanity-check for VideoLlavaTimeSformer on a single video (multi-label).
"""

import torch
from PIL import Image
import torchvision.io as io
from models.videollava_timesformer import VideoLlavaTimeSformer  

# ------------------------------------------------------------------
# 1.  User settings
# ------------------------------------------------------------------
VIDEO_PATH         = "data/tokens/VideoLLaVA/test/2sec_4fps/video_1_clip_131_1_1_0_0.pt"
NUM_FRAMES         = 8
NUM_CLASSES        = 4                      # number of possible tags
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

BASE_VIDEOLLAVA_ID = "LanguageBind/Video-LLaVA-7B-hf"
TIMESFORMER_CKPT   = "facebook/timesformer-base-finetuned-ssv2"

LABELS = ["Baby visible", "Ventilation", "Stimulation", "Suction"]
THRESHOLD = 0.5          # ↑ probability cut-off for a tag to be “on”


# ------------------------------------------------------------------
# 2.  Load input
# ------------------------------------------------------------------
input = torch.load(VIDEO_PATH)


# ------------------------------------------------------------------
# 3.  Build model (classifier already uses a linear layer → logits)
# ------------------------------------------------------------------
model = VideoLlavaTimeSformer(
    timesformer_checkpoint=TIMESFORMER_CKPT,
    base_model_id=BASE_VIDEOLLAVA_ID,
    num_classes=NUM_CLASSES,
    num_frames=NUM_FRAMES,
    device=DEVICE,
    freeze_llm=True,
).eval()


# ------------------------------------------------------------------
# 4.  Run inference
# ------------------------------------------------------------------
with torch.inference_mode():
    logits = model(**input)  # (1, C)
    probs  = torch.sigmoid(logits)[0]                          # (C,)

# ------------------------------------------------------------------
# 5.  Pretty-print results
# ------------------------------------------------------------------
active = [LABELS[i] for i, p in enumerate(probs) if p > THRESHOLD]
print("Logits        :", logits.squeeze().tolist())
print("Probabilities :", probs.tolist())
print("Active labels :", active if active else "none above threshold")
