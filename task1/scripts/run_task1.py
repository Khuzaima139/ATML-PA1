"""Reproduce every Task 1 result in order.

Usage (from the repository root):
    python -m task1.scripts.run_task1

Steps that require manual input are not run here and are listed as prompts.
All other steps are cached: rerunning is cheap once features exist.
"""

import subprocess
import sys

from task1.data.stl10 import ROOT

# (module, description, needs_manual_input_before_it)
PIPELINE = [
    ("task1.data.make_subset",
     "Build the 80/20 train/val split and the 500-image evaluation subset (seed 6304)",
     None),
    ("task1.scripts.extract_features",
     "Cache clean features for ResNet-50, ViT-B/16, CLIP and the CLIP text prompts",
     None),
    ("task1.scripts.train_heads",
     "Train one linear head per backbone with early stopping",
     None),
    ("task1.scripts.run_baseline",
     "Step 1: clean baseline (accuracy, macro-F1, mean max confidence)",
     None),
    ("task1.scripts.run_colour",
     "Step 2: grayscale and 180-degree hue rotation",
     None),
    ("task1.data.pick_style_pool",
     "Step 3a: render style candidate grids for manual curation",
     None),
    ("task1.scripts.adain_pilot",
     "Step 3b: alpha pilot figure (alpha was fixed to 1.0 from this figure)",
     "task1/data/style_pool.json must contain 5 curated indices per class"),
    ("task1.data.make_cue_conflicts",
     "Step 3c: generate the first 300 cue-conflict candidates and review grids",
     None),
    ("task1.data.make_cue_conflicts_extra",
     "Step 3d: generate 200 further candidates from the unused evaluation images",
     None),
    ("task1.data.finalize_cue_conflicts",
     "Step 3e: apply the rejection lists and subsample to 20 per direction",
     "task1/data/cue_rejections.json and cue_rejections_extra.json must list rejected indices"),
    ("task1.scripts.extract_cue_features",
     "Step 3f: cache features for the 200 final cue conflicts",
     None),
    ("task1.scripts.run_cue_conflict",
     "Step 3g: shape bias, coverage and shape/texture/other counts",
     None),
    ("task1.scripts.cue_failure_cases",
     "Step 3h: informative agreement, disagreement and failure examples",
     None),
    ("task1.scripts.run_translation",
     "Step 4: translation by 0, 8, 16, 32 px averaged over four cardinal directions",
     None),
    ("task1.scripts.run_patch_shuffle",
     "Step 5: 4x4 patch shuffle with one non-identity permutation per image",
     None),
    ("task1.scripts.run_representation",
     "Step 6: cosine stability and the UMAP projection of clean vs transformed features",
     None),
]


def main():
    print(f"repository root: {ROOT}\n")
    for module, description, manual in PIPELINE:
        if manual:
            print(f"--- MANUAL STEP REQUIRED BEFORE {module}")
            print(f"    {manual}\n")
        print(f"=== {module}")
        print(f"    {description}")
        result = subprocess.run([sys.executable, "-m", module], cwd=ROOT)
        if result.returncode != 0:
            print(f"\nFAILED: {module} exited with code {result.returncode}")
            sys.exit(result.returncode)
        print()
    print("Task 1 pipeline complete. Results are in task1/results/ and report/figures/.")


if __name__ == "__main__":
    main()