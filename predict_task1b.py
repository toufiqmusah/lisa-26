import argparse
from pathlib import Path
import zipfile

from tqdm import tqdm

from config import OUTPUT_DIR, DATA_ROOT

TASK1B_ROOT = DATA_ROOT.parent / "task1b"


def enhance(args):
    input_dir = TASK1B_ROOT / "val" / "images"
    output_dir = OUTPUT_DIR / "task1b_enhanced"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        f for f in input_dir.glob("*.nii.gz")
        if ".mask." not in f.name
    )
    print(f"Enhancing {len(files)} images")

    from ulfsynth.enhance import enhance_file

    for src in tqdm(files, desc="Enhancing"):
        # LISA_VALIDATION_0001_LF_axi.nii.gz -> LISA_VALIDATION_0001_axi_enhanced.nii.gz
        base = src.name.replace(".nii.gz", "")
        # base = LISA_VALIDATION_0001_LF_axi
        subj = base.split("_")[2]
        view = base.split("_")[-1]
        out_name = f"LISA_VALIDATION_{subj}_{view}_enhanced.nii.gz"
        dst = output_dir / out_name

        if dst.exists() and not args.force:
            continue

        enhance_file(str(src), str(dst), device=args.device, verbose=False)

    print(f"Enhanced files saved to {output_dir}")


def zip_output(args):
    output_dir = OUTPUT_DIR / "task1b_enhanced"
    zip_path = OUTPUT_DIR / "LISA_enhanced_predictions.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(output_dir.glob("*_enhanced.nii.gz")):
            zf.write(f, arcname=f.name)

    print(f"Created {zip_path} ({len(list(output_dir.glob('*_enhanced.nii.gz')))} files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="Re-enhance existing files")
    args = parser.parse_args()

    enhance(args)
    zip_output(args)
