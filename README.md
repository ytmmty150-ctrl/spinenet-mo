# SpineNet-MO

A multi-objective neural architecture search framework based on NSGA-III for osteoporosis risk stratification from DEXA spine images.

## Main Features

- Multi-objective architecture search using NSGA-III.
- Automated CNN architecture generation.
- Clinical-data integration.
- Model training and evaluation.
- Grad-CAM-based visual interpretation.
- Computational efficiency analysis.
- Ablation and seed-sensitivity experiments.
- Baseline comparisons and cross-domain evaluation.
- Statistical analysis and figure generation.

## Data Structure

```text
data/
├── dexa/
│   ├── Normal/
│   ├── Osteopenia/
│   └── Osteoporosis/
├── clinical.csv
└── knee/
    ├── Normal/
    ├── Osteopenia/
    └── Osteoporosis/
```

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m spinenet_mo.cli prepare
python -m spinenet_mo.cli search
python -m spinenet_mo.cli train
python -m spinenet_mo.cli evaluate
python -m spinenet_mo.cli efficiency
python -m spinenet_mo.cli ablation
python -m spinenet_mo.cli seeds
python -m spinenet_mo.cli baselines
python -m spinenet_mo.cli crossdomain
python -m spinenet_mo.cli gradcam
python -m spinenet_mo.cli figures
```

To run the complete pipeline:

```bash
bash scripts/run_all.sh
```

Configuration settings can be adjusted in `configs/default.yaml`.

## Output

Results are saved in:

```text
runs/spinenet_mo/
```

The output directory contains architecture-search results, the selected model, evaluation metrics, experiment results, and generated figures.

## License

MIT
