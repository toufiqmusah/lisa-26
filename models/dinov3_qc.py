import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

from config import QC_LABELS, N_QC_CLASSES, DINOV3_MODEL_NAME, DINOV3_INPUT_SIZE, DINOV3_LORA_RANK, DINOV3_LORA_ALPHA, DINOV3_NUM_VISION_BLOCKS, DINOV3_USE_PATCH_CONCAT, DINOV3_SLICE_STRIDE

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


class VisionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VisionBlocks(nn.Module):
    _HEAD_MAP = {384: 6, 768: 12, 1024: 16}

    def __init__(self, embed_dim, num_blocks=2, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        num_heads = self._HEAD_MAP.get(embed_dim, max(1, embed_dim // 64))
        self.blocks = nn.Sequential(*[
            VisionBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, tokens):
        return self.blocks(tokens)


class AttentionPool(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Parameter(torch.empty(embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, H):
        scale = H.shape[-1] ** 0.5
        scores = torch.einsum('bsd,d->bs', H, self.query) / scale
        a = F.softmax(scores, dim=-1)
        v = torch.einsum('bs,bsd->bd', a, H)
        return v, a


class DINOv3QC(nn.Module):
    def __init__(
        self,
        model_name=DINOV3_MODEL_NAME,
        num_labels=7,
        num_classes=3,
        lora_rank=DINOV3_LORA_RANK,
        lora_alpha=DINOV3_LORA_ALPHA,
        num_vision_blocks=DINOV3_NUM_VISION_BLOCKS,
        use_patch_concat=DINOV3_USE_PATCH_CONCAT,
        input_size=DINOV3_INPUT_SIZE,
        slice_stride=DINOV3_SLICE_STRIDE,
        hf_token=None,
    ):
        super().__init__()

        self.input_size = input_size
        self.use_patch_concat = use_patch_concat
        self.num_labels = num_labels
        self.num_classes = num_classes
        self.slice_stride = slice_stride

        model_kwargs = {}
        if hf_token:
            model_kwargs["token"] = hf_token

        self.backbone = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.embed_dim = self.backbone.config.hidden_size
        self.num_register_tokens = getattr(self.backbone.config, "num_register_tokens", 0)
        print(f"Loaded {model_name}: embed_dim={self.embed_dim}, num_register_tokens={self.num_register_tokens}")

        for p in self.backbone.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_config)

        self.num_vision_blocks = num_vision_blocks
        if num_vision_blocks > 0:
            self.vision_blocks = VisionBlocks(
                embed_dim=self.embed_dim,
                num_blocks=num_vision_blocks,
            )

        self.slice_embed_dim = self.embed_dim * 2 if use_patch_concat else self.embed_dim
        self.slice_pool = AttentionPool(self.slice_embed_dim)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.slice_embed_dim, num_classes),
            )
            for _ in range(num_labels)
        ])

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"DINOv3QC — trainable: {n_trainable:,} / {n_total:,} total")
        print(f"  LoRA rank: {lora_rank}, VisionBlocks: {num_vision_blocks}, Patch concat: {use_patch_concat}")
        print(f"  Slice embed dim: {self.slice_embed_dim}")

    def forward(self, volumes):
        if isinstance(volumes, torch.Tensor) and volumes.dim() == 5:
            volumes = [volumes[i] for i in range(volumes.shape[0])]

        batch_logits = []
        batch_attns = []

        for vol in volumes:
            slices = self._extract_slices(vol)
            device = next(self.parameters()).device

            if slices is None or slices.shape[0] == 0:
                logits = torch.zeros(self.num_labels, self.num_classes, device=device)
                batch_logits.append(logits)
                batch_attns.append(None)
                continue

            slices = slices.to(device)
            slice_embeddings = self._encode_slices(slices)
            vol_embedding, attn = self.slice_pool(slice_embeddings.unsqueeze(0))
            vol_embedding = vol_embedding.squeeze(0)

            logits = torch.stack([h(vol_embedding) for h in self.heads], dim=0)
            batch_logits.append(logits)
            batch_attns.append(attn.squeeze(0))

        return torch.stack(batch_logits, dim=0), batch_attns

    def _extract_slices(self, volume):
        assert volume.dim() == 4 and volume.shape[0] == 1, f"Expected [1, H, W, D], got {volume.shape}"

        vol_sq = volume.squeeze(0)
        n_slices = vol_sq.shape[-1]

        slices_list = []
        for i in range(0, n_slices, self.slice_stride):
            s = vol_sq[..., i]
            s = torch.clamp(s, -3, 3)
            s = (s + 3) / 6
            s = s.unsqueeze(0).repeat(3, 1, 1)
            slices_list.append(s)

        if len(slices_list) == 0:
            return None

        slices = torch.stack(slices_list, dim=0)

        if slices.shape[-1] != self.input_size or slices.shape[-2] != self.input_size:
            slices = F.interpolate(
                slices,
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False,
            )

        mean = IMAGENET_MEAN.to(slices.device).view(1, 3, 1, 1)
        std = IMAGENET_STD.to(slices.device).view(1, 3, 1, 1)
        slices = (slices - mean) / std

        return slices

    def _encode_slices(self, slices):
        outputs = self.backbone(pixel_values=slices, output_hidden_states=False)
        tokens = outputs.last_hidden_state

        if hasattr(self, 'vision_blocks'):
            tokens = self.vision_blocks(tokens)

        cls_token = tokens[:, 0]

        if self.use_patch_concat:
            n_reg = self.num_register_tokens
            patch_tokens = tokens[:, 1 + n_reg:]
            patch_repr = patch_tokens.mean(dim=1)
            return torch.cat([cls_token, patch_repr], dim=-1)

        return cls_token

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
        return probs
