from pathlib import Path
from typing import Optional

import torch
from torch import nn

from config import QC_LABELS, N_QC_CLASSES, QC_HEAD_HIDDEN, QC_HEAD_DROPOUT, CROP_SIZE, FINETUNE_N_BLOCKS, FINETUNE_LR, ensure_encoder_checkpoint


class QCHead(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        h = QC_HEAD_HIDDEN
        self.net = nn.Sequential(
            nn.Linear(in_features, h[0]),
            nn.LayerNorm(h[0]),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(QC_HEAD_DROPOUT),
            nn.Linear(h[0], h[1]),
            nn.LayerNorm(h[1]),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(QC_HEAD_DROPOUT),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(h[1], N_QC_CLASSES) for _ in QC_LABELS]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return torch.stack([h(x) for h in self.heads], dim=1)


class EncoderQC(nn.Module):
    """
    PrimusV3S encoder (down_projection + eva) pretrained on enhancement task,
    with a QC classification head substituted for the decoder.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        freeze_encoder: bool = False,
        n_unfreeze_blocks: int = 0,
    ):
        super().__init__()

        from dynamic_network_architectures.architectures.primus import PrimusV3S

        self.backbone = PrimusV3S(
            input_channels=1,
            output_channels=12,
            patch_embed_size=(8, 8, 8),
            input_shape=CROP_SIZE,
            drop_path_rate=0.2,
            scale_attn_inner=True,
            init_values=0.1,
        )

        if checkpoint_path is None:
            try:
                checkpoint_path = ensure_encoder_checkpoint()
            except Exception:
                print("Could not download encoder checkpoint, using random init")

        if checkpoint_path and Path(checkpoint_path).exists():
            state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
            nw = state["network_weights"]
            self.backbone.load_state_dict(nw, strict=True)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print("Checkpoint not found, using random init")

        embed_dim = self.backbone.eva.embed_dim
        self.qc_head = QCHead(in_features=embed_dim)

        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            if n_unfreeze_blocks > 0:
                for b in self.backbone.eva.blocks[-n_unfreeze_blocks:]:
                    for p in b.parameters():
                        p.requires_grad_(True)
                print(f"Unfrozen last {n_unfreeze_blocks} eva blocks")
            else:
                print("Encoder frozen")
        else:
            print("Encoder trainable (all params)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.down_projection(x)
        B, C, W, H, D = x.shape
        x = x.flatten(2).transpose(1, 2)
        x, _ = self.backbone.eva(x)
        x = x.mean(dim=1)
        logits = self.qc_head(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
        return probs
