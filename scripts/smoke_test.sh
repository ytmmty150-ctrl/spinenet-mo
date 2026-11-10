#!/usr/bin/env bash
# Tiny end-to-end run on synthetic data (a few minutes on CPU; requires PyTorch).
set -euo pipefail
python scripts/make_synthetic_data.py --out data --n-per-class 40
bash scripts/run_all.sh configs/default.yaml \
  data.test_size=0.2 data.val_size=0.2 data.image_size=64 data.num_workers=0 \
  data.clinical_csv=data/clinical.csv output_dir=runs/smoke \
  search.population_size=6 search.mating_pool_size=4 search.generations=1 search.candidate_epochs=1 search.ref_divisions=3 \
  train.epochs=2 evaluation.n_seeds=2 evaluation.latency_runs=5 evaluation.latency_warmup=1 \
  baselines.epochs=1 "baselines.models=[resnet50]" cross_domain.data_root=data/knee \
  cross_domain.phase1.epochs=1 cross_domain.phase2.epochs=1
