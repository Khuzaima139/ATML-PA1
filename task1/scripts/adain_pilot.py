import matplotlib.pyplot as plt
from torchvision.datasets import STL10

from common.seed import get_device
from task1.data.stl10 import BASE_TRANSFORM, ROOT, cfg, get_sets, load_style_pool
from task1.models.adain import load_adain, style_transfer


def main():
    device = get_device()
    cc = cfg["cue_conflict"]
    sets, classes = get_sets()
    eval_set = sets["eval"]
    eval_labels = eval_set.dataset.labels[eval_set.indices].tolist()
    train_full = STL10(ROOT / cfg["data_root"], split="train", transform=BASE_TRANSFORM)
    pool = load_style_pool()
    encoder, decoder = load_adain(device)

    rows = []
    for a, b in cc["pairs"]:
        for content_cls, style_cls in [(a, b), (b, a)]:
            content = eval_set[eval_labels.index(classes.index(content_cls))][0]
            style = train_full[pool[style_cls][0]][0]
            outs = [
                style_transfer(encoder, decoder, content[None].to(device), style[None].to(device), alpha)[0].cpu()
                for alpha in cc["pilot_alphas"]
            ]
            rows.append((f"{content_cls}\n+ {style_cls}", [content, style, *outs]))

    titles = ["content", "style"] + [f"alpha={a}" for a in cc["pilot_alphas"]]
    fig, axes = plt.subplots(len(rows), len(titles), figsize=(2 * len(titles), 2 * len(rows)))
    for r, (name, images) in enumerate(rows):
        for c, img in enumerate(images):
            ax = axes[r, c]
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c])
            if c == 0:
                ax.set_ylabel(name)
    fig.tight_layout()
    fig.savefig(ROOT / "task1/results/adain_pilot.png", dpi=120)
    plt.close(fig)
    print("saved task1/results/adain_pilot.png")


if __name__ == "__main__":
    main()