# dataset.py
"""
Dataset utilities for SHI-Labs SlowFast-Video-MLLM training *without*
an external annotation file - all labels are parsed from the video names.

Returned keys
-------------
'pixel_values' : float32 tensor (T, 3, H, W) in [0, 1] for the visual backbone
'input_ids'    : int64  tensor (seq,) containing exactly one image placeholder
'labels'       : bool   tensor (4,)  [baby_visible, ventilation, stimulation, suction]

Filename parsing rules
----------------------
* Baby visible    → token equals "Baby visible"
* Ventilation     → token contains "CPAP" **or** ends with "PPV"
* Stimulation     → token starts with "Stimulation backnates" or
                    "Stimulation trunk"  (optional 'P-' prefix allowed)
                    ("Stimulation extremities" is *ignored*)
* Suction         → token contains "Suction"
* A leading "P-"  → still counts as *present* (partial occurrence = 1)
"""
from __future__ import annotations

import itertools
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset


# ----------------------------------------------------------------------------- #
#  Helper ─────────────────────────────────────────────────────────────────────  #
# ----------------------------------------------------------------------------- #
def _all_video_paths(root: Path, exts=(".avi", ".mp4", ".mov", ".mkv")) -> List[Path]:
    root = Path(root).expanduser()
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


def parse_label_from_filename(path: str | Path) -> torch.BoolTensor:
    """Return BoolTensor [baby_visible, ventilation, stimulation, suction]."""
    parts: List[str] = Path(path).stem.split("_")

    # strip pure-digit tokens (timestamps) so they don't match anything
    parts = [p for p in parts if not p.isdigit()]

    baby_visible = any(p.strip().lower() == "baby visible" for p in parts)

    ventilation = any(p.lower().endswith("ppv") or "cpap" in p.lower() for p in parts)

    def _is_stim(p: str) -> bool:
        p = p.lower().lstrip("p-")  # drop optional P-
        return p.startswith("stimulation backnates") or p.startswith("stimulation trunk")

    stimulation = any(_is_stim(p) for p in parts)

    suction = any("suction" in p.lower() for p in parts)

    return torch.tensor(
        [baby_visible, ventilation, stimulation, suction], dtype=torch.bool
    )


# ----------------------------------------------------------------------------- #
#  Dataset ───────────────────────────────────────────────────────────────────── #
# ----------------------------------------------------------------------------- #
class SlowFastDataset(Dataset):
    """Video-prompt dataset that derives 4-bit labels from filenames."""

    def __init__(
        self,
        root_dir: str | Path,
        tokenizer,                       # HuggingFace tokenizer (LLaVA one)
        processor,                       # SlowFast visual processor
        n_frames: int = 64,
    ) -> None:
        super().__init__()
        self.video_paths = _all_video_paths(root_dir)
        if not self.video_paths:
            raise RuntimeError(f"No videos found under {root_dir}")

        self.tokenizer = tokenizer
        self.processor = processor
        self.n_frames = n_frames

    # ───────────────────────────────────────────────────────── frame sampler ── #
    @lru_cache(maxsize=128)
    def _load_decord(self, video_path: str | Path):
        """Cache VideoReader per worker-process."""
        return VideoReader(str(video_path), ctx=cpu(0))

    def _sample_uniform(self, video_path: str | Path) -> torch.FloatTensor:
        vr = self._load_decord(video_path)
        tot = len(vr)
        if tot == 0:
            raise RuntimeError(f"Empty video: {video_path}")

        # If the clip is shorter than n_frames, repeat last frame
        if tot < self.n_frames:
            idx = list(range(tot)) + [tot - 1] * (self.n_frames - tot)
        else:
            idx = np.linspace(0, tot - 1, self.n_frames, dtype=int).tolist()

        frames = vr.get_batch(idx).asnumpy()  # (T, H, W, 3) uint8

        # Give frames to the HF processor → returns dict with 'pixel_values'
        proc_out = self.processor(frames, return_tensors="pt")
        pixels: torch.Tensor = proc_out["pixel_values"].squeeze(0)  # (T, 3, H, W)
        return pixels  # float32 in [0, 1]

    # ───────────────────────────────────────────────────────── Dataset API ─── #
    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int):
        vid_path = self.video_paths[idx]

        # ---- video
        pixels = self._sample_uniform(vid_path)  # (T, 3, H, W)

        # ---- text  (single <image> placeholder - no extra prompt)
        text = "<image>"
        ids = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).input_ids.squeeze(0)

        # ---- labels
        label = parse_label_from_filename(vid_path)

        return {
            "pixel_values": pixels,  # (T, 3, H, W) float32
            "input_ids": ids,        # (seq_len,)
            "labels": label,         # (4,) bool
        }


# ----------------------------------------------------------------------------- #
#  Collate function ─────────────────────────────────────────────────────────── #
# ----------------------------------------------------------------------------- #
def build_collate_fn(tokenizer):
    """
    Returns a function that assembles a batch for SlowFast-LLaVA.

    * pads `input_ids`
    * swaps <image> ⇒ IMAGE_TOKEN_INDEX
    * stacks video tensors & labels
    """
    from llava.mm_utils import IMAGE_TOKEN_INDEX, tokenizer_image_token  # noqa: WPS433

    pad_id = tokenizer.pad_token_id or 0

    def collate(batch: Iterable[dict]):
        batch = list(batch)  # may arrive as generator from DataLoader
        # ---- videos --------------------------------------------------------- #
        vids = torch.stack([b["pixel_values"] for b in batch])  # (B, T, 3, H, W)

        # ---- text ----------------------------------------------------------- #
        seq_lens = [b["input_ids"].size(0) for b in batch]
        max_len = max(seq_lens)
        ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        for i, b in enumerate(batch):
            l = b["input_ids"].size(0)
            ids[i, :l] = b["input_ids"]

        # Replace <image> token with IMAGE_TOKEN_INDEX
        ids = tokenizer_image_token(ids, tokenizer, IMAGE_TOKEN_INDEX, is_input_ids=True)

        # ---- labels --------------------------------------------------------- #
        labels = torch.stack([b["labels"] for b in batch])  # (B, 4)

        return {
            "input_ids": ids,
            "pixel_values": vids,
            "labels": labels,
        }

    return collate
