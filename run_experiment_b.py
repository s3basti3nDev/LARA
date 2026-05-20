"""Re-run experiment B (LARA v1-3) with fixed seed=43."""
import sys
sys.path.insert(0, ".")
from train_cpu_v2 import run_experiment, get_dataloaders, CPU_TRAIN, LARA
from data.dataset import get_dataloaders

tc_obj = type("TC", (), CPU_TRAIN)()
tc_obj.block_size = CPU_TRAIN["block_size"]
tc_obj.batch_size = CPU_TRAIN["batch_size"]
tc_obj.dataset    = CPU_TRAIN["dataset"]
tc_obj.data_dir   = CPU_TRAIN["data_dir"]

train_loader, val_loader = get_dataloaders(tc_obj)

run_experiment(
    "B -- LARA v1-3      ",
    LARA,
    dict(use_diff_attention=True, use_mor=True, use_coconut=False, use_titans=True),
    train_loader,
    val_loader,
    seed=43,
)
print("\nDone. Checkpoint saved to results/B_-_LARA_v1-3_v2_ckpt.pt")
