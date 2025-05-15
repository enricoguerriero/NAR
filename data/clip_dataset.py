import os
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict

import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision.transforms.functional import to_pil_image
from PIL import Image


class ResuscitationVideoDataset(Dataset):
    """
    Multi-label video dataset that plugs into a Hugging Face *processor*.

    Returned dictionary:
        {"pixel_values": …,   # (or 'input_values', 'inputs', … - whatever
         "labels": tensor([4])}

    Label order (bool / float32):
        0 - baby visible
        1 - ventilation  (CPAP | PPV | P-CPAP | P-PPV)
        2 - stimulation  (Stimulation backnates | Stimulation trunk)
        3 - suction      (Suction)
    """

    # ------------ regexes to detect the four events in a file-name ------------ #
    _VENT_PAT    = re.compile(r"(?:^|[_\s])p?-?(?:cpap|ppv)(?:[_\s]|$)", flags=re.I)
    _STIM_PAT    = re.compile(r"(?:^|[_\s])stimulation[_\s](?:backnates|trunk)(?:[_\s]|$)",
                              flags=re.I)
    _SUCTION_PAT = re.compile(r"(?:^|[_\s])suction(?:[_\s]|$)", flags=re.I)
    _BABY_PAT    = re.compile(r"(?:^|[_\s])baby[_\s]visible(?:[_\s]|$)", flags=re.I)

    @staticmethod
    def _parse_labels(fname: str) -> torch.Tensor:
        """1-hot → float32 tensor[4]."""
        fname = fname.lower()
        labels = [
            bool(ResuscitationVideoDataset._BABY_PAT.search(fname)),
            bool(ResuscitationVideoDataset._VENT_PAT.search(fname)),
            bool(ResuscitationVideoDataset._STIM_PAT.search(fname)),
            bool(ResuscitationVideoDataset._SUCTION_PAT.search(fname)),
        ]
        return torch.tensor(labels, dtype=torch.float32)

    # ------------------------------------------------------------------------- #

    _VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}

    def __init__(
        self,
        root: str | Path,
        n_frames: int = 8,
        clip_len: Optional[int] = None,
        video_reader_kwargs: Optional[dict] = None,
        processor: Optional[Callable] = None,
        prompt: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        root : str | Path
            Directory containing the videos (recursively searched).
        processor : transformers.XXXProcessor / FeatureExtractor / ImageProcessor
            The object that converts a list/array of frames into model inputs.
        clip_len : int, optional
            Number of consecutive frames to sample. If None, the whole video is
            passed to the processor.
        video_reader_kwargs : dict, optional
            Extra kwargs forwarded to `torchvision.io.read_video`.
        """
        self.root = Path(root).expanduser()
        self.clip_len = clip_len
        self.vr_kwargs = video_reader_kwargs or {}
        self.n_frames = n_frames

        self.video_paths: List[Path] = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in self._VIDEO_EXTS
        )
        if not self.video_paths:
            raise RuntimeError(f"No videos with extensions {self._VIDEO_EXTS} found in {self.root}")

        self._labels = [self._parse_labels(p.name) for p in self.video_paths]
        
        self.processor = processor
        self.prompt = prompt

    # ------------------------------- Dataset API ------------------------------ #

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary ready for 🤗 models, plus `"labels"`.
        Typical keys: 'pixel_values', 'attention_mask', …
        """
        path = self.video_paths[idx]
        labels = self._labels[idx]

        # (T×H×W×C, audio, info)
        video, _, _ = read_video(str(path), **self.vr_kwargs)  # uint8, 0-255
        T = video.shape[0]
        
        if self.n_frames is not None and T != self.n_frames:
            indices = torch.linspace(0, T - 1, steps=self.n_frames).long()
        else:
            pad = self.n_frames - T
            indices = torch.cat([
                torch.arange(T),
                torch.full((pad,), T - 1, dtype=torch.long)
            ])
        video = video[indices]
        
        # -------------- convert to list[PIL.Image] for the processor ----------- #
        # read_video → T×H×W×C  ;  PIL expects H×W×C
        frames: List[Image.Image] = [to_pil_image(frame) for frame in video]
        
        if self.prompt is not None:
            out = self.processor(
                videos = frames,
                text = self.prompt,
                return_tensors="pt",
            )
            frames = out["pixel_values_videos"].squeeze(0)
        else:
            proc_out = self.processor(frames, return_tensors="pt")
            frames = proc_out["pixel_values"].squeeze(0)
        
        out["labels"] = labels  
        
        return out
