import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import numpy as np

from config import (
    DATA_ROOT,
    QC_LABELS,
    N_QC_CLASSES,
    QC_HEAD_LR,
    QC_HEAD_WEIGHT_DECAY,
    QC_HEAD_BATCH_SIZE,
    QC_HEAD_EPOCHS,
    QC_HEAD_PATIENCE,
    CHECKPOINT_DIR,
    RANDOM_SEED,
    TEST_SIZE,
    FINETUNE_N_BLOCKS,
    FINETUNE_LR,
)
from data.task1a import QCImageDataset
from models.encoder_qc import EncoderQC, Conv3DQC


def collate_3d(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack([torch.from_numpy(lb).long() for lb in labels])
    return images, labels


def collate_3d_pred(batch):
    images = torch.stack([b[0] for b in batch])
    return images


def train(args):
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    full_dataset = QCImageDataset(root=DATA_ROOT, split="train")
    n_val = int(len(full_dataset) * TEST_SIZE)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    # Per-label class weights (inverse frequency, clipped)
    label_arr = np.array([full_dataset[i][1] for i in range(len(full_dataset))])
    criteria = []
    for i in range(len(QC_LABELS)):
        counts = np.bincount(label_arr[:, i].astype(int), minlength=N_QC_CLASSES)
        weights = len(label_arr) / (N_QC_CLASSES * counts.astype(float))
        weights = np.clip(weights, None, 10.0)
        criteria.append(nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device)))

    train_loader = DataLoader(
        train_ds, batch_size=QC_HEAD_BATCH_SIZE, shuffle=True,
        collate_fn=collate_3d, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=QC_HEAD_BATCH_SIZE, shuffle=False,
        collate_fn=collate_3d, num_workers=2, pin_memory=True,
    )
    print(f"Train: {n_train}, Val: {n_val}")

    if args.backbone == "conv3d":
        model = Conv3DQC()
        model.to(device)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"Conv3D QC model — total params: {n_total:,} (all trainable)")
        param_groups = [
            {"params": model.parameters(), "lr": QC_HEAD_LR},
        ]
    else:
        ckpt_path = Path(args.checkpoint) if args.checkpoint else None
        model = EncoderQC(
            checkpoint_path=ckpt_path,
            freeze_encoder=True,
            n_unfreeze_blocks=FINETUNE_N_BLOCKS,
        )
        model.to(device)

        n_enc = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
        n_head = sum(p.numel() for p in model.qc_head.parameters())
        print(f"Trainable: encoder={n_enc:,}, head={n_head:,}")

        param_groups = [
            {"params": model.qc_head.parameters(), "lr": QC_HEAD_LR},
        ]
        if n_enc > 0:
            param_groups.append({
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": FINETUNE_LR,
            })

    optimizer = optim.AdamW(
        param_groups,
        lr=QC_HEAD_LR,
        weight_decay=QC_HEAD_WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=QC_HEAD_EPOCHS)

    best_val_loss = float("inf")
    patience_counter = 0
    save_dir = CHECKPOINT_DIR / "encoder_qc"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, QC_HEAD_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = sum(criteria[i](logits[:, i], labels[:, i]) for i in range(len(QC_LABELS)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = sum(criteria[i](logits[:, i], labels[:, i]) for i in range(len(QC_LABELS)))
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{QC_HEAD_EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}", end="")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            suffix = "_conv3d" if args.backbone == "conv3d" else ""
            torch.save(model.state_dict(), save_dir / f"best{suffix}.pt")
            patience_counter = 0
            print("  (saved)")
        else:
            patience_counter += 1
            print("")

        if patience_counter >= QC_HEAD_PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Done. Best val loss: {best_val_loss:.4f}")


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    suffix = "_conv3d" if args.backbone == "conv3d" else ""
    ckpt_path = CHECKPOINT_DIR / "encoder_qc" / f"best{suffix}.pt"

    if args.backbone == "conv3d":
        model = Conv3DQC()
    else:
        enc_ckpt = Path(args.checkpoint) if args.checkpoint else None
        model = EncoderQC(checkpoint_path=enc_ckpt, freeze_encoder=True)

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=args.backbone == "conv3d")
    model.to(device)
    model.eval()
    print(f"Loaded encoder QC model ({args.backbone})")

    dataset = QCImageDataset(root=DATA_ROOT, split="val")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_3d_pred)

    import pandas as pd

    rows = []
    with torch.no_grad():
        for i, images in enumerate(loader):
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy().tolist()
            rows.append({"filename": dataset.files[i].name, **{lbl: p for lbl, p in zip(QC_LABELS, preds)}})

    out_path = CHECKPOINT_DIR / "encoder_qc" / "LISA_LF_QC_predictions.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "predict"])
    parser.add_argument("--backbone", choices=["primus", "conv3d"], default="primus",
                        help="primus=frozen EVA+head (needs checkpoint), conv3d=fully trainable 3D ConvNet")
    parser.add_argument("--checkpoint", default=None, help="Path to Task 2 PrimusV3S checkpoint (primus only)")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    if args.mode == "train":
        train(args)
    else:
        predict(args)
