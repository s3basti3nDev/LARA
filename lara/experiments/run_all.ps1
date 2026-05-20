# Run all 5 Phase 1 experiments sequentially, then benchmark.
# PowerShell version for Windows.

Set-Location "$PSScriptRoot\.."

Write-Host "==============================" -ForegroundColor Cyan
Write-Host " LARA Phase 1 - All Experiments" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan

python train.py --experiment baseline  --max_iters 5000
python train.py --experiment diff_attn --max_iters 5000
python train.py --experiment mor       --max_iters 5000
python train.py --experiment coconut   --max_iters 5000
python train.py --experiment lara_full --max_iters 5000

Write-Host ""
Write-Host "==============================" -ForegroundColor Green
Write-Host " Benchmark" -ForegroundColor Green
Write-Host "==============================" -ForegroundColor Green
python experiments/benchmark.py
