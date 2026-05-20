"""
Benchmark LARA against Pythia and GPT-2 on standard tasks.

Usage (from lara/ directory):
    # LARA checkpoints
    python run_benchmarks.py --lara checkpoints/exp_a_baseline_best.pt
    python run_benchmarks.py --lara checkpoints/exp_a_baseline_best.pt checkpoints/exp_f_lara_v2_best.pt

    # All comparisons at once
    python run_benchmarks.py \
        --lara checkpoints/exp_a_baseline_best.pt checkpoints/exp_f_lara_v2_best.pt \
        --hf EleutherAI/pythia-160m:step163 EleutherAI/pythia-160m gpt2 gpt2-medium \
        --tasks hellaswag,arc_easy,lambada_openai \
        --device cuda
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(__file__))

# Import wrapper first — registers @register_model("lara") with lm_eval
import lm_eval_wrapper  # noqa: F401

import lm_eval
from lm_eval.models.huggingface import HFLM

TASK_DEFAULTS = "hellaswag,arc_easy,lambada_openai"


def _short_name(path_or_hf: str) -> str:
    if os.path.exists(path_or_hf):
        return os.path.basename(path_or_hf).replace("_best.pt", "").replace("checkpoints/", "")
    # HuggingFace model, possibly with revision
    name = path_or_hf.split("/")[-1]
    if ":" in name:
        model, rev = name.split(":", 1)
        return f"{model}@{rev}"
    return name


def eval_lara(checkpoint: str, tasks: list[str], device: str, num_fewshot: int):
    results = lm_eval.simple_evaluate(
        model="lara",
        model_args=f"checkpoint={checkpoint},device={device}",
        tasks=tasks,
        num_fewshot=num_fewshot,
        verbosity="WARNING",
    )
    return results["results"]


def eval_hf(pretrained: str, tasks: list[str], device: str, num_fewshot: int):
    if ":" in pretrained.split("/")[-1]:
        base, revision = pretrained.rsplit(":", 1)
        lm = HFLM(pretrained=base, revision=revision, dtype="float16", device=device)
    else:
        lm = HFLM(pretrained=pretrained, dtype="float16", device=device)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        verbosity="WARNING",
    )
    return results["results"]
    


def _acc(task_results: dict, task: str) -> str:
    r = task_results.get(task, {})
    # lm_eval uses "acc,none" or "acc_norm,none" depending on the task
    acc = r.get("acc_norm,none") or r.get("acc,none") or r.get("perplexity,none")
    if acc is None:
        return "  N/A  "
    return f"{acc*100:6.2f}%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lara", nargs="*", default=[],
                   help="LARA checkpoint paths")
    p.add_argument("--hf", nargs="*", default=[],
                   help="HuggingFace model IDs (use model:revision for specific step)")
    p.add_argument("--tasks", default=TASK_DEFAULTS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--output", default=None, help="Save JSON results to file")
    args = p.parse_args()

    tasks = args.tasks.split(",")
    all_results = {}

    print(f"\n{'='*72}")
    print(f"  LARA Benchmark Suite  —  tasks: {', '.join(tasks)}")
    print(f"{'='*72}")
    header = f"  {'Model':<35}" + "".join(f" {t[:10]:>10}" for t in tasks)
    print(header)
    print(f"  {'-'*68}")

    for ckpt in (args.lara or []):
        name = _short_name(ckpt)
        print(f"  Evaluating LARA: {name} ...", flush=True)
        try:
            res = eval_lara(ckpt, tasks, args.device, args.num_fewshot)
            all_results[name] = res
            row = f"  {name:<35}" + "".join(f" {_acc(res, t):>10}" for t in tasks)
            print(f"\r{row}")
        except Exception as e:
            print(f"\r  {name:<35}  ERROR: {e}")

    for hf_id in (args.hf or []):
        name = _short_name(hf_id)
        print(f"  Evaluating HF: {name} ...", flush=True)
        try:
            res = eval_hf(hf_id, tasks, args.device, args.num_fewshot)
            all_results[name] = res
            row = f"  {name:<35}" + "".join(f" {_acc(res, t):>10}" for t in tasks)
            print(f"\r{row}")
        except Exception as e:
            print(f"\r  {name:<35}  ERROR: {e}")

    print(f"{'='*72}\n")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
