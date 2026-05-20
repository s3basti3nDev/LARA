"""
lm_eval harness wrapper for LARA.

Allows running official benchmarks (WikiText-103, HellaSwag, ARC, MMLU, etc.)
using EleutherAI's lm_eval library.

Install: pip install lm_eval
Usage:
    # Run HellaSwag (0-shot):
    lm_eval --model hf --model_args "pretrained=." --tasks hellaswag --device cpu

    # Or use the Python API:
    python lm_eval_wrapper.py --checkpoint results/B_--_LARA_v1-3_ckpt.pt \
                               --tasks wikitext,hellaswag,arc_easy

Requirements:
    pip install lm_eval datasets tiktoken

Implements the lm_eval.api.model.LM interface.
See: https://github.com/EleutherAI/lm-evaluation-harness
"""
import sys, os, math, argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F

try:
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
    HAS_LMEVAL = True
except ImportError:
    HAS_LMEVAL = False
    LM = object   # fallback so the class can still be defined

import tiktoken
from config import ModelConfig
from model.baseline import GPT
from model.lara import LARA


_TOKENIZER = None

def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = tiktoken.get_encoding("gpt2")
    return _TOKENIZER


def _load_ckpt(path, device="cpu"):
    """Load a LARA or GPT checkpoint robustly (CPU and GPU scale)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    extra_cfg = ckpt.get("extra_cfg", {})

    # Support both key formats (GPU train.py → "model_state", legacy → "model")
    state = ckpt.get("model_state") or ckpt.get("model")
    # Strip torch.compile() prefix added when saving compiled models
    if state and any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}

    # Infer architecture dimensions from the state dict — works for any scale
    inferred = {}
    if state and "wte.weight" in state:
        inferred["n_embd"]      = state["wte.weight"].shape[1]
        inferred["block_size"]  = state["wpe.weight"].shape[0]
        inferred["n_layer"]     = sum(
            1 for k in state
            if k.startswith("transformer.layers.") and k.endswith(".ln1.weight")
        )
        # Infer n_head from MLA down_kv or DiffAttn q_proj shape
        if "transformer.layers.0.attn.down_kv.weight" in state:
            d_c = state["transformer.layers.0.attn.down_kv.weight"].shape[0]
            inferred["mla_kv_compress"] = d_c

    base_cfg = dict(
        n_embd=256, n_head=4, n_layer=6, block_size=128,
        memory_size=128, n_recursions=3, mor_top_k=0.5,
        memory_lr=0.01, n_thinking_steps=4,
        coconut_ramp_steps=3000, coconut_start_iter=9999,
    )
    # Priority: inferred (from weights) > extra_cfg (saved config) > base defaults
    mc = ModelConfig(**{**base_cfg, **extra_cfg, **inferred})

    is_baseline = not any(extra_cfg.get(k) for k in
                          ["use_diff_attention", "use_mla", "use_mor",
                           "use_recurrent_depth", "use_titans", "use_coconut"])
    cls = GPT if is_baseline else LARA
    model = cls(mc)

    # Migrate old single-projection Titans → MFE-PA multi-projection
    if state and "memory.q_proj.weight" in state and not any(
            k.startswith("memory.q_projs") for k in state):
        w = state.pop("memory.q_proj.weight")
        if hasattr(model, "memory") and model.memory is not None:
            for i in range(model.memory.n_proj):
                state[f"memory.q_projs.{i}.weight"] = w.clone()

    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


if HAS_LMEVAL:
    @register_model("lara")
    class LARAModel(LM):
        """
        lm_eval wrapper for LARA.

        Instantiate via:
            python -m lm_eval --model lara \
                --model_args checkpoint=results/B_ckpt.pt \
                --tasks hellaswag
        """

        def __init__(self, checkpoint, device="cpu", batch_size=8,
                     n_recursions_override=None, **kwargs):
            super().__init__()
            self._model = _load_ckpt(checkpoint, device)
            self._device = device
            self._batch_size = int(batch_size)
            self._n_rec = int(n_recursions_override) if n_recursions_override else None
            self._tokenizer = get_tokenizer()
            self._max_len = self._model.config.block_size

        @property
        def eot_token_id(self):
            return self._tokenizer.eot_token

        @property
        def max_length(self):
            return self._max_len

        @property
        def max_gen_toks(self):
            return 256

        @property
        def batch_size(self):
            return self._batch_size

        @property
        def device(self):
            return self._device

        def tok_encode(self, string):
            return self._tokenizer.encode(string)

        def tok_decode(self, tokens):
            return self._tokenizer.decode(tokens)

        def _model_kwargs(self):
            kwargs = {}
            if self._n_rec is not None and isinstance(self._model, LARA):
                kwargs["n_recursions_override"] = self._n_rec
            return kwargs

        def _forward_batch(self, batch_ids: list[list[int]]) -> list[torch.Tensor]:
            """
            Run model on a padded batch, return per-sample logits (unpadded).
            """
            max_len = max(len(s) for s in batch_ids)
            # Left-pad with 0 (padding doesn't affect causal positions before it)
            padded = torch.zeros(len(batch_ids), max_len, dtype=torch.long,
                                 device=self._device)
            for i, ids in enumerate(batch_ids):
                padded[i, max_len - len(ids):] = torch.tensor(ids, device=self._device)
            with torch.no_grad():
                out = self._model(padded, **self._model_kwargs())
                logits = out[0]
            # Return logits aligned to each sample (strip left-padding)
            return [logits[i, max_len - len(ids):] for i, ids in enumerate(batch_ids)]

        def loglikelihood(self, requests):
            results = []
            # Collect all (ctx_ids, cont_ids) then process in batches
            pairs = []
            for r in requests:
                ctx, cont = r.args
                ctx_ids  = self.tok_encode(ctx)
                cont_ids = self.tok_encode(cont)
                all_ids  = (ctx_ids + cont_ids)[-self._max_len:]
                pairs.append((all_ids, len(cont_ids)))

            for i in range(0, len(pairs), self._batch_size):
                batch = pairs[i: i + self._batch_size]
                batch_ids = [ids for ids, _ in batch]
                all_logits = self._forward_batch(batch_ids)

                for (all_ids, n_cont), logits in zip(batch, all_logits):
                    cont_start = len(all_ids) - n_cont
                    log_probs = F.log_softmax(logits[cont_start - 1:-1], dim=-1)
                    tgt = torch.tensor(all_ids[cont_start:], device=self._device)
                    ll = log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
                    is_greedy = (logits[cont_start - 1:-1].argmax(-1) == tgt).all().item()
                    results.append((ll, bool(is_greedy)))
            return results

        def loglikelihood_rolling(self, requests):
            results = []
            for r in requests:
                string = r.args[0]
                ids = self.tok_encode(string)
                total_ll = 0.0
                for start in range(0, len(ids) - 1, self._max_len):
                    chunk = ids[max(0, start - 1): start + self._max_len]
                    logits = self._forward_batch([chunk])[0]
                    log_probs = F.log_softmax(logits[:-1], dim=-1)
                    tgts = torch.tensor(chunk[1:], device=self._device)
                    total_ll += log_probs.gather(-1, tgts.unsqueeze(-1)).squeeze(-1).sum().item()
                results.append(total_ll)
            return results

        def generate_until(self, requests):
            results = []
            for r in requests:
                ctx  = r.args[0]
                until = r.args[1] if len(r.args) > 1 else []
                ids = self.tok_encode(ctx)
                x = torch.tensor([ids[-self._max_len:]], device=self._device)
                out = self._model.generate(x, self.max_gen_toks,
                                           **self._model_kwargs())
                new_ids = out[0][len(ids):].tolist()
                text = self.tok_decode(new_ids)
                for u in (until or []):
                    if u in text:
                        text = text[:text.index(u)]
                results.append(text)
            return results


# ── Standalone runner ─────────────────────────────────────────────
def run_perplexity(model, device, text, block_size=128):
    """Compute perplexity on raw text using sliding window."""
    enc = get_tokenizer()
    ids = enc.encode(text)
    if len(ids) < 2:
        return float("inf")

    total_ll = 0.0
    n_tokens = 0
    stride = block_size // 2

    for begin in range(0, len(ids) - 1, stride):
        chunk = ids[begin: begin + block_size + 1]
        x = torch.tensor([chunk[:-1]], device=device)
        y = torch.tensor([chunk[1:]], device=device)
        with torch.no_grad():
            out = model(x, y)
        total_ll -= out[1].item() * (len(chunk) - 1)
        n_tokens += len(chunk) - 1

    return math.exp(-total_ll / n_tokens) if n_tokens > 0 else float("inf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--tasks", default="wikitext",
                   help="Comma-separated lm_eval tasks (requires lm_eval installed)")
    p.add_argument("--depth-scaling", action="store_true",
                   help="Test RecurrentDepth at 1x,2x,3x before evaluating")
    args = p.parse_args()

    model = _load_ckpt(args.checkpoint, args.device)
    print(f"Loaded: {args.checkpoint}")
    print(f"Params: {model.num_params():,}")

    if not HAS_LMEVAL:
        print("\nlm_eval not installed. Install with:")
        print("  pip install lm_eval datasets")
        print("\nRunning local perplexity test on embedded corpus only...")

        # Load local val data for perplexity
        from data.dataset import get_dataloaders
        tc = type("TC", (), {"dataset": "local", "data_dir": "data",
                              "block_size": 128, "batch_size": 8})()
        _, val_loader = get_dataloaders(tc)
        losses = []
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                if i >= 100:
                    break
                out = model(x, y)
                losses.append(out[1].item())
        ppl = math.exp(sum(losses) / len(losses))
        print(f"Local val perplexity: {ppl:.3f}")
        return

    import lm_eval
    lm_eval.evaluate(
        lm=LARAModel(checkpoint=args.checkpoint, device=args.device),
        task_manager=lm_eval.tasks.TaskManager(),
        tasks=args.tasks.split(","),
        num_fewshot=0,
        verbosity="INFO",
    )


if __name__ == "__main__":
    main()
