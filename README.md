# ATML PA1: Learning Beyond the IID, Closed-Set Setting

Programming Assignment 1 of Advanced Topics in Machine Learning (EE-5102 / CS-6304), Fall 2026. The four tasks study what happens when the usual training assumptions break.

| Task | Question | Data | Methods |
|---|---|---|---|
| 1 | Which visual cues do different backbones rely on? | STL-10 | ResNet-50, ViT-B/16, CLIP ViT-B/32 |
| 2 | Does unlabeled target data help across a domain shift? | PACS, Sketch as target | Source-only, DAN, DANN, CDAN |
| 3 | Can a model generalize to a domain it never sees? | PACS, Sketch unseen | ERM, DAN-DG, SAM |
| 4 | Can a classifier reject inputs from unknown classes? | CIFAR-10 known, CIFAR-100 unknowns | Vanilla, GCSC, PROSER |

## Setup

```bash
python3.12 -m venv .atml_pa1
source .atml_pa1/bin/activate
pip install -r requirements.txt
```

`requirements-lock.txt` pins the exact versions used. Run every command from the repository root. Datasets are stored in `data/` and checkpoints in `checkpoints/`; neither is committed. Results are saved as JSON in `task*/results/` and figures in `report/figures/`.

## Task 1: Inductive biases and representations

Download STL-10, and download `vgg_normalised.pth` and `decoder.pth` from the releases of [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) into `checkpoints/adain/`. Then run the full pipeline:

```bash
python -c "from torchvision.datasets import STL10; STL10('data', split='train', download=True); STL10('data', split='test', download=True)"
python -m task1.scripts.run_task1
```

The style pool and the cue-conflict rejection lists were chosen by eye and are committed, so a rerun evaluates the same images.

## Task 2: Unsupervised domain adaptation

Download PACS once (Task 3 uses the same data and splits), then train and evaluate:

```bash
python -m shared.download_pacs
for c in source_only dan dan_lambda0.1 dan_lambda10 dann cdan dann_l2 cdan_l2; do
    python -m task2.train --config task2/configs/$c.yaml
done
python -m task2.evaluate_final
python -m task2.evaluation.feature_norms
python -m task2.evaluation.plot_results
```

## Task 3: Domain generalization

The ERM baseline is Task 2's Source-only checkpoint, so run Task 2 first.

```bash
python -m task3.methods.erm
for c in dan_dg sam dan_dg_lambda0.1 dan_dg_lambda10; do
    python -m task3.train --config task3/configs/$c.yaml
done
python -m task3.evaluation.source_domain_separability
python -m task3.evaluation.sharpness
python -m task3.evaluate_sketch
python -m task3.evaluation.plot_results
```

## Task 4: Open-set recognition

CIFAR-10 and CIFAR-100 download automatically. Training was run on a Kaggle T4 GPU; PROSER starts from the Vanilla checkpoint.

```bash
python -m task4.data.make_splits
python -m task4.train --config task4/configs/vanilla.yaml
python -m task4.train --config task4/configs/gcsc.yaml
python -m task4.train --config task4/configs/proser.yaml
python -m task4.data.cifar100_unknowns
for r in vanilla gcsc proser; do
    python -m task4.extract_outputs --run $r
done
python -m task4.evaluate_osr
```

## Attribution

- AdaIN architecture and pretrained weights: [naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) (Huang and Belongie, 2017).
- PROSER detection score: the authors' reference code, [LAMDA-CL/CVPR21-Proser](https://github.com/LAMDA-CL/CVPR21-Proser) (Zhou et al., 2021).