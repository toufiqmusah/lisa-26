import shutil
import json
from huggingface_hub import HfApi

# Save run metadata
meta = {
    "model": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "epochs": 150,
    "slice_stride": 2,
    "batch_size": 1,
    "lora_rank": 8,
    "vision_blocks": 2,
    "patch_concat": True,
    "focal_gamma": 2.0,
    "focal_alpha": 0.25,
    "best_val_loss": 0.XX,  # fill from training output
}
with open("/kaggle/working/lisa-26/checkpoints/dinov3_qc/run_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

# Zip checkpoint dir
shutil.make_archive(
    "/kaggle/working/lisa-26/checkpoints/dinov3_qc",
    "zip",
    "/kaggle/working/lisa-26/checkpoints/dinov3_qc",
)

# Upload to HF dataset repo
api = HfApi(token=" ... ")
api.upload_file(
    path_or_fileobj="/kaggle/working/lisa-26/checkpoints/dinov3_qc.zip",
    path_in_repo="Outputs/dinov3_qc.zip",
    repo_id="toufiqmusah/LISA-26",
    repo_type="dataset",
)
print("Uploaded!")
