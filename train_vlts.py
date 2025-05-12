# train_vlts.py
import os, math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from utils import setup_wandb
from tqdm import tqdm

from models.videollava_timesformer import VideoLlavaTimeSformer


# ──────────────────────────────────────────────────────────────────────────────
#  0.  Config
# ──────────────────────────────────────────────────────────────────────────────
class CFG:
    root         = "data/tokens/VideoLLaVA"  # path to train/val folders
    num_frames   = 8                         # number of frames per video
    num_classes  = 4                         # number of possible tags
    batch_size   = 8                        # batch size
    epochs       = 10                        # number of epochs
    lr_head      = 1e-5                      # learning rate for head
    warmup       = 0.1                       # warmup ratio
    max_grad_norm= 1.0                       # gradient clipping
    fp16         = True                      # use mixed precision
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    debug        = False                     # debug mode (no training)
    seed         = 42                        # random seed
    threshold    = 0.5                       # probability cut-off for a tag to be “on”

wandb_run = setup_wandb(
    model_name="videollava_timesformer",
    config=CFG
)

# ──────────────────────────────────────────────────────────────────────────────
#  1.  Dataset & collate
# ──────────────────────────────────────────────────────────────────────────────
from data.token_dataset import TokenDataset
train_ds = TokenDataset(os.path.join(CFG.root, "train", "2sec_4fps"))
val_ds   = TokenDataset(os.path.join(CFG.root, "validation",   "2sec_4fps"))

# ──────────────────────────────────────────────────────────────────────────────
#  2.  Build model
# ──────────────────────────────────────────────────────────────────────────────
model = VideoLlavaTimeSformer(
    timesformer_checkpoint="facebook/timesformer-base-finetuned-ssv2",
    base_model_id="LanguageBind/Video-LLaVA-7B-hf",
    num_frames=CFG.num_frames,
    num_classes=CFG.num_classes,
    freeze_llm=True,
    debug=False,
).to(CFG.device)

# --------- optional: un-freeze last 2 vision blocks -------------
for n, p in model.backbone.vision_tower.named_parameters():
    if n.startswith(("blocks.10.", "blocks.11.")):
        p.requires_grad = True

# ──────────────────────────────────────────────────────────────────────────────
#  3.  Optimizer & LR schedule
# ──────────────────────────────────────────────────────────────────────────────
decay, no_decay = [], []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    (no_decay if n.endswith(("bias", "norm.weight", "ln.weight"))
              else decay).append(p)

optimizer = torch.optim.AdamW(
    [
        {"params": decay,    "lr": CFG.lr_head, "weight_decay": 0.05},
        {"params": no_decay, "lr": CFG.lr_head, "weight_decay": 0.0},
    ]
)
total_steps = math.ceil(len(os.listdir(CFG.root)) / CFG.batch_size) * CFG.epochs
scheduler   = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=CFG.warmup * len(decay),
    num_training_steps=total_steps
)

scaler = torch.amp.GradScaler(device = "cuda", enabled=CFG.fp16)
pos_weight = torch.tensor([0.19311390817165375, 2.532083511352539, 7.530612468719482, 6.510387420654297])
criterion = nn.BCEWithLogitsLoss(weight = pos_weight.to(CFG.device))

def cast_trainable_to_fp32(model):
    n_fp16 = 0
    for n, p in model.named_parameters():
        if p.requires_grad and p.dtype == torch.float16:
            p.data = p.data.float()
            n_fp16 += 1
    print(f"✓ cast {n_fp16} trainable tensors to fp32")

# --------- optional: un-freeze last 2 vision blocks -------------
for n, p in model.backbone.vision_tower.named_parameters():
    if n.startswith(("blocks.10.", "blocks.11.")):
        p.requires_grad = True

cast_trainable_to_fp32(model)


# ──────────────────────────────────────────────────────────────────────────────
#  4.  DataLoaders
# ──────────────────────────────────────────────────────────────────────────────

train_dl = DataLoader(train_ds, batch_size=CFG.batch_size,
                      shuffle=True, collate_fn=model.collate_fn_tokens,
                      num_workers=4, pin_memory=True)
val_dl   = DataLoader(val_ds, batch_size=CFG.batch_size,
                      shuffle=False, collate_fn=model.collate_fn_tokens,
                      num_workers=2)


# ──────────────────────────────────────────────────────────────────────────────
#  5.  Evaluate function
# ──────────────────────────────────────────────────────────────────────────────
def evaluate():
    model.eval()
    logits_all, labels_all = [], []
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_dl, desc="Validation", unit="batch"):
            for k in batch:
                batch[k] = batch[k].to(CFG.device)

            with torch.amp.autocast(device_type="cuda", enabled=CFG.fp16):
                logits = model(**{k: batch[k] for k in batch if k != "labels"})
                logits = logits.float()
                loss = criterion(logits, batch["labels"].float().to(logits.device))


            logits_all.append(logits.float().cpu())
            labels_all.append(batch["labels"].float().cpu())
            val_loss += loss.item()

    logits_all = torch.cat(logits_all)
    labels_all = torch.cat(labels_all)
    val_loss  /= len(val_dl)

    val_metrics = model.metric_computation(
        logits=logits_all,
        labels=labels_all,
        threshold=None               # uses 0.5 or calibrated thresholds
    )
    return val_loss, val_metrics



# ──────────────────────────────────────────────────────────────────────────────
#  6. Training loop 
# ──────────────────────────────────────────────────────────────────────────────


best_map = 0.0
for epoch in range(CFG.epochs):
    model.train()
    running_loss = 0.0
    train_logits, train_labels = [], []

    for batch in tqdm(train_dl, desc=f"Training epoch {epoch}", unit="batch"):
        for k in batch:
            batch[k] = batch[k].to(CFG.device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=CFG.fp16):
            logits = model(**{k: batch[k] for k in batch if k != "labels"})
            loss = criterion(logits, batch["labels"].float().to(logits.device))
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()
        train_logits.append(logits.detach().float().cpu())
        train_labels.append(batch["labels"].float().cpu())

        # if step % 20 == 0:
        #     print(f"Epoch {epoch+1}/{CFG.epochs} | "
        #           f"Step {step}/{len(train_dl)} | "
        #           f"Loss {running_loss/step:.4f}")

    # ───────────────  aggregate train metrics  ─────────────────────────────
    train_logits = torch.cat(train_logits)
    train_labels = torch.cat(train_labels)
    train_loss   = running_loss / len(train_dl)
    train_metrics = model.metric_computation(
        logits=train_logits,
        labels=train_labels,
        threshold=CFG.threshold
    )
    print(f"Epoch {epoch+1}: "
          f"  |  train loss={train_loss:.4f}  |  "
          f"  |  train F1_macro={train_metrics['f1_macro']:.4f}  |  ")

    # ───────────────  validation  ──────────────────────────────────────────
    val_loss, val_metrics = evaluate()
    # print(f"Epoch {epoch+1}: "
    #       f"train loss={train_loss:.4f}  |  val loss={val_loss:.4f}  |  "
    #       f"val F1_macro={val_metrics['f1_macro']:.4f}")

    print(f"Epoch {epoch+1}: "
            f"  |  validation loss={val_loss:.4f}  |  "
            f"  |  validation F1_macro={val_metrics['f1_macro']:.4f}  |  ")
    # ───────────────  wandb logging  ───────────────────────────────────────
    model.log_wandb(
        wandb_run=wandb_run,
        epoch=epoch + 1,
        train_loss=train_loss,
        train_metrics=train_metrics,
        val_loss=val_loss,
        val_metrics=val_metrics
    )

    # ───────────────  model checkpoint  ────────────────────────────────────
    if val_metrics["f1_macro"] > best_map:
        best_map = val_metrics["f1_macro"]
        torch.save(model.state_dict(), "best_videollava_timesformer.pth")
        print("  ✔ Saved new best model\n")