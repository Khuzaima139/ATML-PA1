import json
import shutil

import numpy as np

from task1.data.stl10 import ROOT, cfg

CACHE = ROOT / "task1/cache/cue_conflicts"
FINAL = ROOT / "task1/cache/cue_conflicts_final"
MANIFEST = ROOT / "task1/data/cue_conflicts_generated.json"
FINAL_MANIFEST = ROOT / "task1/results/cue_conflicts_final.json"


def main():
    with open(MANIFEST) as f:
        all_images = json.load(f)
    with open(ROOT / "task1/data/cue_rejections.json") as f:
        rej_first = json.load(f)
    with open(ROOT / "task1/data/cue_rejections_extra.json") as f:
        rej_extra = json.load(f)

    by_direction = {}
    for m in all_images:
        by_direction.setdefault(m["direction"], []).append(m)

    rng = np.random.default_rng(cfg["seed"])
    target = cfg["cue_conflict"]["final_per_direction"]
    counts, kept = {}, []

    FINAL.mkdir(parents=True, exist_ok=True)
    for direction, images in by_direction.items():
        rejected = set(rej_first.get(direction, [])) | set(rej_extra.get(direction, []))
        accepted = [m for m in images if int(m["id"].rsplit("_", 1)[1]) not in rejected]

        n_keep = min(target, len(accepted))
        chosen = rng.choice(len(accepted), size=n_keep, replace=False)
        chosen_images = [accepted[i] for i in chosen]

        counts[direction] = {
            "generated": len(images),
            "rejected": len(rejected),
            "accepted": len(accepted),
            "kept": n_keep,
        }
        for m in chosen_images:
            shutil.copy(CACHE / f"{m['id']}.png", FINAL / f"{m['id']}.png")
            kept.append(m)

        print(f"{direction:<20} generated {len(images):>3}  accepted {len(accepted):>3}  kept {n_keep:>3}")

    with open(FINAL_MANIFEST, "w") as f:
        json.dump({"counts": counts, "images": kept}, f, indent=2)

    total_kept = sum(c["kept"] for c in counts.values())
    total_accepted = sum(c["accepted"] for c in counts.values())
    total_generated = sum(c["generated"] for c in counts.values())
    print(f"\ntotal generated {total_generated}, accepted {total_accepted}, kept {total_kept}")


if __name__ == "__main__":
    main()