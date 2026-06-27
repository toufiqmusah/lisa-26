import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm

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
    RECON_FINETUNE_STAGES,
    RECON_FINETUNE_LR,
)
from data.task1a import QCImageDataset, get_train_transform
from models.encoder_qc import EncoderQC, Conv3DQC, ReconFeatureQC


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

    num_epochs = args.num_epochs if args.num_epochs else QC_HEAD_EPOCHS
    patience = args.patience if args.patience else QC_HEAD_PATIENCE

    full_dataset = QCImageDataset(root=DATA_ROOT, split="train", transform=get_train_transform())
    if full_dataset.df is None:
        raise FileNotFoundError(
            f"Labels file not found at {DATA_ROOT}/train/labels.csv. "
            "Set LISA_DATA_ROOT or place data at the expected location."
        )
    n_val = int(len(full_dataset) * TEST_SIZE)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    # Per-label class weights (inverse frequency, clipped)
    labels_list = [full_dataset[i][1] for i in range(len(full_dataset))]
    label_arr = np.array(labels_list)
    if label_arr.ndim != 2:
        raise ValueError(
            f"Expected 2D label array, got {label_arr.ndim}D with shape {label_arr.shape}. "
            "Check that labels.csv contains all QC_LABELS columns."
        )
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

    if args.backbone == "recon_feat":
        ckpt_path = Path(args.checkpoint) if args.checkpoint else None
        finetune_stages = args.finetune_stages if args.finetune_stages is not None else RECON_FINETUNE_STAGES
        model = ReconFeatureQC(checkpoint_path=ckpt_path, finetune_stages=finetune_stages)
        model.to(device)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"ReconFeature QC — trainable: {n_trainable:,} / {n_total:,} total")
        # Separate LR groups: head at QC_HEAD_LR, unfrozen stages at RECON_FINETUNE_LR
        head_params = []
        stage_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "net" in name or "heads" in name:
                head_params.append(p)
            else:
                stage_params.append(p)
        param_groups = [
            {"params": head_params, "lr": QC_HEAD_LR},
        ]
        if stage_params:
            param_groups.append({"params": stage_params, "lr": RECON_FINETUNE_LR})
    elif args.backbone == "conv3d":
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

    param_lrs = {f"group_{i}": g["lr"] for i, g in enumerate(param_groups)}
    optimizer = optim.AdamW(
        param_groups,
        lr=QC_HEAD_LR,
        weight_decay=QC_HEAD_WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    save_dir = CHECKPOINT_DIR / "encoder_qc"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{num_epochs} [train]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = sum(criteria[i](logits[:, i], labels[:, i]) for i in range(len(QC_LABELS)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0.0
        pbar = tqdm(val_loader, desc=f"Epoch {epoch:3d}/{num_epochs} [val]", leave=False)
        with torch.no_grad():
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = sum(criteria[i](logits[:, i], labels[:, i]) for i in range(len(QC_LABELS)))
                val_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        current_lrs = [f"{g['lr']:.2e}" for g in optimizer.param_groups]
        msg = f"Epoch {epoch:3d}/{num_epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={current_lrs}"

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            backbone_suffixes = {"conv3d": "_conv3d", "recon_feat": "_recon_feat"}
            suffix = backbone_suffixes.get(args.backbone, "")
            torch.save(model.state_dict(), save_dir / f"best{suffix}.pt")
            patience_counter = 0
            print(f"{msg}  (saved)")
        else:
            patience_counter += 1
            print(msg)

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Done. Best val loss: {best_val_loss:.4f}")


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone_suffixes = {"conv3d": "_conv3d", "recon_feat": "_recon_feat"}
    suffix = backbone_suffixes.get(args.backbone, "")
    ckpt_path = CHECKPOINT_DIR / "encoder_qc" / f"best{suffix}.pt"

    if args.backbone == "recon_feat":
        model = ReconFeatureQC()
    elif args.backbone == "conv3d":
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
        for i, images in enumerate(tqdm(loader, desc="Predicting")):
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
    parser.add_argument("--backbone", choices=["primus", "conv3d", "recon_feat"], default="primus",
                        help="primus=frozen EVA+head, conv3d=fully trainable 3D ConvNet, recon_feat=frozen conv stage+head")
    parser.add_argument("--checkpoint", default=None, help="Path to Task 2 PrimusV3S checkpoint")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Override QC_HEAD_EPOCHS (default: %(default)s)")
    parser.add_argument("--patience", type=int, default=None,
                        help="Override QC_HEAD_PATIENCE (default: %(default)s)")
    parser.add_argument("--finetune_stages", type=int, default=None,
                        help="Unfreeze last N conv stages for recon_feat (default: config RECON_FINETUNE_STAGES)")
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    if args.mode == "train":
        train(args)
    else:
        predict(args)
