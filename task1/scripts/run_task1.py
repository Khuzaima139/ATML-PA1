"""Reproduce every Task 1 result in order.

Usage (from the repository root):
    python -m task1.scripts.run_task1

Hand-made inputs (style pool picks, rejection lists) are committed and must exist.
Cue-conflict generation is skipped when the final 200 images already exist, so a rerun
evaluates exactly the images that were judged. Every other step reuses its own caches.
"""

import subprocess
import sys

from task1.data.stl10 import ROOT, cfg

FINAL_DIR = ROOT / "task1/cache/cue_conflicts_final"

MANUAL_ARTIFACTS = [
    "task1/data/style_pool.json",
    "task1/data/cue_rejections.json",
    "task1/data/cue_rejections_extra.json",
]

# (module, description, skip this step if this path exists)
PIPELINE = [
    ("task1.data.make_subset", "Splits and 500-image evaluation subset", None),
    ("task1.scripts.extract_features", "Clean features for all backbones and CLIP prompts", None),
    ("task1.scripts.train_heads", "Linear heads with early stopping", None),
    ("task1.scripts.run_baseline", "Step 1: clean baseline", None),
    ("task1.scripts.run_colour", "Step 2: grayscale and hue rotation", None),
    ("task1.data.pick_style_pool", "Step 3: style candidate grids",
     ROOT / "task1/data/style_candidates.json"),
    ("task1.scripts.adain_pilot", "Step 3: alpha pilot figure",
     ROOT / "task1/results/adain_pilot.png"),
    ("task1.data.make_cue_conflicts", "Step 3: first 300 candidates", FINAL_DIR),
    ("task1.data.make_cue_conflicts_extra", "Step 3: extra 200 candidates", FINAL_DIR),
    ("task1.data.finalize_cue_conflicts", "Step 3: apply rejections, keep 20 per direction", FINAL_DIR),
    ("task1.scripts.extract_cue_features", "Step 3: features for the 200 conflicts", None),
    ("task1.scripts.run_cue_conflict", "Step 3: shape bias and coverage", None),
    ("task1.scripts.cue_failure_cases", "Step 3: failure case figure", None),
    ("task1.scripts.run_translation", "Step 4: translation curve", None),
    ("task1.scripts.run_patch_shuffle", "Step 5: patch shuffle", None),
    ("task1.scripts.run_representation", "Step 6: cosine stability and UMAP", None),
]


def exists(path):
    return path.exists() and (path.is_file() or any(path.iterdir()))


def main():
    missing = [p for p in MANUAL_ARTIFACTS if not (ROOT / p).exists()]
    if missing:
        sys.exit(f"missing hand-made files, cannot reproduce: {missing}")
    for key in ("adain_vgg", "adain_decoder"):
        if not (ROOT / cfg["cue_conflict"][key]).exists():
            sys.exit("AdaIN weights missing from checkpoints/adain, see README for download commands")

    for module, description, skip_if in PIPELINE:
        if skip_if is not None and exists(skip_if):
            print(f"=== skip {module} ({skip_if.relative_to(ROOT)} exists)")
            continue
        print(f"=== {module}: {description}")
        result = subprocess.run([sys.executable, "-m", module], cwd=ROOT)
        if result.returncode != 0:
            sys.exit(f"FAILED: {module} exited with code {result.returncode}")
    print("Task 1 pipeline complete. Results in task1/results/, figures in report/figures/.")


if __name__ == "__main__":
    main()