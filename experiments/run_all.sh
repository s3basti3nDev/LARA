#!/bin/bash
# Run all 5 Phase 1 experiments sequentially, then benchmark.
# Each experiment runs for 5000 iterations on Tiny Shakespeare.
# On a single RTX 3090/4090: ~20-30 min total.

set -e
cd "$(dirname "$0")/.."

echo "=============================="
echo " LARA Phase 1 — All Experiments"
echo "=============================="

python train.py --experiment baseline  --max_iters 5000
python train.py --experiment diff_attn --max_iters 5000
python train.py --experiment mor       --max_iters 5000
python train.py --experiment coconut   --max_iters 5000
python train.py --experiment lara_full --max_iters 5000

echo ""
echo "=============================="
echo " Benchmark"
echo "=============================="
python experiments/benchmark.py
