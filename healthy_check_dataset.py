from transformers import AutoTokenizer, AutoProcessor
from torch.utils.data import DataLoader

from data.slowfast_dataset import SlowFastDataset, build_collate_fn

tokenizer  = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")   # example
processor  = AutoProcessor.from_pretrained("ShiLab/slowfast-llava-visual")  # example

ds  = SlowFastDataset(
    root_dir="/data/clips/test",
    tokenizer=tokenizer,
    processor=processor,
    n_frames=64,
)
loader = DataLoader(
    ds,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=build_collate_fn(tokenizer),
)

for batch in loader:
    print(batch["pixel_values"].shape)  # (4, 64, 3, H, W)
    print(batch["input_ids"].shape)     # (4, seq_len)
    print(batch["labels"].shape)        # (4, 4)
    break
