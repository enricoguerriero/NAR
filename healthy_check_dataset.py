from transformers import AutoTokenizer, AutoProcessor
from torch.utils.data import DataLoader

from data.clip_dataset import ResuscitationVideoDataset
from transformers import VideoLlavaProcessor

processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
prompt = "SYSTEM: check the video. USER:<video> WHat is happening in the video? Answer with a list of events. ASSISTANT:"

ds  = ResuscitationVideoDataset(
    root_dir="/data/clips/test",
    n_frames=8,
    processor=processor,
    prompt=prompt,
    )
loader = DataLoader(
    ds,
    batch_size=4,
    shuffle=True,
    num_workers=4
)

for batch in loader:
    print(batch["pixel_values"].shape)  # (4, 64, 3, H, W)
    print(batch["input_ids"].shape)     # (4, seq_len)
    print(batch["labels"].shape)        # (4, 4)
    break
