"""
Compare training results across all LARA experiments.
Loads JSON result files from results/ and produces a convergence summary.

Usage:
    python compare_results.py              # compare all found results
    python compare_results.py --plot       # also save a convergence plot (requires matplotlib)
"""
import os, sys, json, argparse
from pathlib import Path

RESULTS_DIR = Path("results")


def load_results():
    results = {}
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            name = data.get("name", f.stem)
            results[name] = data
        except Exception as e:
            print(f"  [skip] {f.name}: {e}")
    return results


def ppl_at_iter(history, target_iter):
    """Interpolate PPL at a target iteration."""
    for h in history:
        if h["iter"] >= target_iter:
            return h["ppl"]
    return history[-1]["ppl"] if history else float("inf")


def summarise(results):
    print("\n" + "="*72)
    print("  LARA — Convergence Comparison")
    print("="*72)

    checkpoints = [0, 100, 200, 300, 500, 700, 1000]
    header = f"  {'Experiment':<28} {'Params':>10} " + "".join(f"{'PPL@'+str(c):>9}" for c in checkpoints)
    print(header)
    print(f"  {'-'*70}")

    all_final = []
    for name, data in results.items():
        hist = data.get("history", [])
        params = data.get("params", 0)
        row = f"  {name:<28} {params:>10,} "
        ppls = []
        for c in checkpoints:
            p = ppl_at_iter(hist, c)
            row += f"  {p:>7.1f}"
            ppls.append(p)
        print(row)
        best = min(h["ppl"] for h in hist) if hist else float("inf")
        all_final.append((name, params, best))

    print(f"\n  {'Experiment':<28} {'Params':>10} {'Best PPL':>10} {'Rank':>6}")
    print(f"  {'-'*60}")
    ranked = sorted(all_final, key=lambda x: x[2])
    for rank, (name, params, best) in enumerate(ranked, 1):
        print(f"  {name:<28} {params:>10,} {best:>10.2f} {rank:>6}")

    print()
    baselines = [x for x in ranked if "Baseline" in x[0] or "GPT" in x[0]]
    laras = [x for x in ranked if x not in baselines]
    if baselines and laras:
        baseline = baselines[0]   # best baseline
        best_lara = laras[0]      # best LARA variant
        if baseline[2] > 0:
            gain = (baseline[2] - best_lara[2]) / baseline[2] * 100
            direction = "improvement" if gain > 0 else "regression"
            print(f"  Best LARA vs Baseline: {abs(gain):.1f}% {direction}")
            print(f"  ({best_lara[0].strip()} PPL={best_lara[2]:.2f} vs {baseline[0].strip()} PPL={baseline[2]:.2f})")


def plot_convergence(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    linestyles = ["-", "--", "-.", ":", (0, (3,1,1,1))]
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]

    for i, (name, data) in enumerate(results.items()):
        hist = data.get("history", [])
        if not hist:
            continue
        iters = [h["iter"] for h in hist]
        ppls = [min(h["ppl"], 500) for h in hist]  # cap for readability
        ls = linestyles[i % len(linestyles)]
        c  = colors[i % len(colors)]
        label = name.strip()[:30]
        ax.semilogy(iters, ppls, linestyle=ls, color=c, linewidth=2, label=label, marker="o", markersize=4)

    ax.set_xlabel("Training iteration", fontsize=12)
    ax.set_ylabel("Validation perplexity (log scale)", fontsize=12)
    ax.set_title("LARA — Convergence Comparison", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    out = RESULTS_DIR / "convergence.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\n  Plot saved to {out}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Save convergence plot")
    args = parser.parse_args()

    if not RESULTS_DIR.exists():
        print("No results/ directory found. Run train_cpu.py first.")
        sys.exit(1)

    results = load_results()
    if not results:
        print("No result JSON files found in results/.")
        sys.exit(1)

    summarise(results)
    if args.plot:
        plot_convergence(results)
