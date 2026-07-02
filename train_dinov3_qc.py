import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm

from config import (
    DATA_ROOT,
    QC_LABELS,
    N_QC_CLASSES,
    CHECKPOINT_DIR,
    RANDOM_SEED,
    TEST_SIZE,
    DINOV3_MODEL_NAME,
    DINOV3_INPUT_SIZE,
    DINOV3_LORA_RANK,
    DINOV3_LORA_ALPHA,
    DINOV3_NUM_VISION_BLOCKS,
    DINOV3_USE_PATCH_CONCAT,
    DINOV3_LORA_LR,
    DINOV3_VISION_LR,
    DINOV3_HEAD_LR,
    DINOV3_FOCAL_GAMMA,
    DINOV3_FOCAL_ALPHA,
    DINOV3_MAX_EPOCHS,
    DINOV3_PATIENCE,
    DINOV3_WARMUP_EPOCHS,
    DINOV3_SLICE_STRIDE,
    DINOV3_BATCH_SIZE,
)
from data.task1a import QCImageDataset
from models.dinov3_qc import DINOv3QC


def collate_3d(batch):
    images, labels = zip(*batch)
    labels = torch.stack([torch.from_numpy(lb).long() for lb in labels])
    return list(images), labels


def collate_3d_pred(batch):
    return [b[0] for b in batch]


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()


def train(args):
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    num_epochs = args.num_epochs if args.num_epochs else DINOV3_MAX_EPOCHS
    patience = args.patience if args.patience else DINOV3_PATIENCE

    full_dataset = QCImageDataset(root=DATA_ROOT, split="train")
    if full_dataset.df is None:
        raise FileNotFoundError(f"Labels file not found at {DATA_ROOT}/train/labels.csv")

    n_val = int(len(full_dataset) * TEST_SIZE)
    n_train = len(full_dataset) - n_val

    if n_val > 0:
        train_ds, val_ds = random_split(full_dataset, [n_train, n_val])
        val_loader = DataLoader(
            val_ds, batch_size=DINOV3_BATCH_SIZE, shuffle=False,
            collate_fn=collate_3d, num_workers=0, pin_memory=True,
        )
    else:
        train_ds = full_dataset
        val_loader = None

    train_loader = DataLoader(
        train_ds, batch_size=DINOV3_BATCH_SIZE, shuffle=True,
        collate_fn=collate_3d, num_workers=0, pin_memory=True,
    )
    print(f"Train: {n_train}, Val: {n_val}")

    model = DINOv3QC(
        model_name=DINOV3_MODEL_NAME,
        num_labels=len(QC_LABELS),
        num_classes=N_QC_CLASSES,
        lora_rank=DINOV3_LORA_RANK,
        lora_alpha=DINOV3_LORA_ALPHA,
        num_vision_blocks=DINOV3_NUM_VISION_BLOCKS,
        use_patch_concat=DINOV3_USE_PATCH_CONCAT,
        input_size=DINOV3_INPUT_SIZE,
        slice_stride=DINOV3_SLICE_STRIDE,
        hf_token=args.hf_token,
    )
    model.to(device)

    lora_params = []
    vision_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "heads" in name or "slice_pool" in name:
            head_params.append(p)
        elif "vision_blocks" in name:
            vision_params.append(p)
        else:
            lora_params.append(p)

    print(f"  LoRA params: {sum(p.numel() for p in lora_params):,}")
    if vision_params:
        print(f"  VisionBlock params: {sum(p.numel() for p in vision_params):,}")
    print(f"  Head params: {sum(p.numel() for p in head_params):,}")

    param_groups = [
        {"params": lora_params, "lr": DINOV3_LORA_LR, "weight_decay": 1e-5},
    ]
    if vision_params:
        param_groups.append({"params": vision_params, "lr": DINOV3_VISION_LR, "weight_decay": 1e-5})
    param_groups.append({"params": head_params, "lr": DINOV3_HEAD_LR, "weight_decay": 1e-4})

    optimizer = optim.AdamW(param_groups)

    warmup_epochs = DINOV3_WARMUP_EPOCHS
    if warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
        )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, num_epochs - warmup_epochs),
    )

    criteria = FocalLoss(gamma=DINOV3_FOCAL_GAMMA, alpha=DINOV3_FOCAL_ALPHA)

    scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None

    save_dir = CHECKPOINT_DIR / "dinov3_qc"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_interval = max(1, num_epochs // 10)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{num_epochs} [train]", leave=False)
        for images, labels in pbar:
            labels = labels.to(device)

            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast(device_type="cuda"):
                    logits, _ = model(images)
                    loss = sum(criteria(logits[:, i], labels[:, i]) for i in range(len(QC_LABELS)))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(images)
                loss = sum(criteria(logits[:, i], labels[:, i]) for i in range(len(QC_LABELS)))
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)

        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        current_lrs = [f"{g['lr']:.2e}" for g in optimizer.param_groups]
        msg = f"Epoch {epoch:3d}/{num_epochs}  train={train_loss:.4f}  lr={current_lrs}"

        should_save = (epoch == num_epochs) or (epoch % save_interval == 0)
        if should_save:
            torch.save(model.state_dict(), save_dir / f"checkpoint_epoch{epoch:03d}.pt")
            print(f"{msg}  (saved epoch {epoch})")
        else:
            print(msg)

    torch.save(model.state_dict(), save_dir / "best_dinov3_qc.pt")
    print(f"Done. Final checkpoint saved.")


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DINOv3QC(
        model_name=DINOV3_MODEL_NAME,
        num_labels=len(QC_LABELS),
        num_classes=N_QC_CLASSES,
        lora_rank=DINOV3_LORA_RANK,
        lora_alpha=DINOV3_LORA_ALPHA,
        num_vision_blocks=DINOV3_NUM_VISION_BLOCKS,
        use_patch_concat=DINOV3_USE_PATCH_CONCAT,
        slice_stride=DINOV3_SLICE_STRIDE,
        hf_token=args.hf_token,
    )

    ckpt_path = CHECKPOINT_DIR / "dinov3_qc" / "best_dinov3_qc.pt"
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Loaded DINOv3 QC checkpoint from {ckpt_path}")

    dataset = QCImageDataset(root=DATA_ROOT, split="val")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_3d_pred)

    import pandas as pd

    rows = []
    with torch.no_grad():
        for i, images in enumerate(tqdm(loader, desc="Predicting")):
            logits, _ = model(images)
            preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy().tolist()
            rows.append({"filename": dataset.files[i].name, **{lbl: p for lbl, p in zip(QC_LABELS, preds)}})

    out_path = CHECKPOINT_DIR / "dinov3_qc" / "LISA_LF_QC_predictions.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "predict"])
    parser.add_argument("--num_epochs", type=int, default=None, help="Override DINOV3_MAX_EPOCHS")
    parser.add_argument("--patience", type=int, default=None, help="Override DINOV3_PATIENCE")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--hf_token", default=None, help="HuggingFace token for gated model access")
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    if args.mode == "train":
        train(args)
    else:
        predict(args)
