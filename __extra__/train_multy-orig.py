"""
TTT-Discover for Circle Packing — runs locally.

No async, no Ray, no Tinker, no logging frameworks.
  - LFM2.5-350M via either Unsloth (fast) or plain HF+PEFT (fallback)
  - Subprocess sandbox for code execution
  - PUCT sampler for parent selection
  - Entropic adaptive β for advantage computation

Defaults are paper-scale: 50 steps × 8 groups × 64 rollouts.
On a single consumer GPU you'll probably want to scale down — every knob
is a CLI flag, see --help.

Quick examples:

    # Default (paper-scale)
    python train.py

    # Smaller for a 16 GB GPU
    python train.py --groups-per-step 2 --group-size 8

    # Force the plain HF backend (no Unsloth)
    python train.py --backend hf

    # Different problem size
    python train.py --num-circles 32 --target 2.940
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import os
import argparse
import time
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

# torch is imported lazily after we pick a backend (because Unsloth
# wants to monkey-patch transformers before they get imported).


# ======================================================================
# Config
# ======================================================================
@dataclass
class Config:
    # Model
    model_name: str = "Qwen/Qwen3-8B"
    backend: str = "auto"        # "auto" | "unsloth" | "hf"
    max_seq_length: int = 32000
    load_in_4bit: bool = False
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # Problem
    num_circles: int = 26
    target: float = 2.635983
    sandbox_timeout_s: float = 30.0

    # RL hyperparameters — paper scale
    num_steps: int = 50
    groups_per_step: int = 8
    group_size: int = 64
    num_seed_states: int = 8           # paper uses several seeds for diversity
    learning_rate: float = 4e-5
    kl_penalty_coef: float = 0.1
    max_new_tokens: int = 4200
    grad_clip: float = 1.0

    # Mini-batching for the training step (to control activation memory):
    # we accumulate grads over examples but each forward is one example.
    # If you want a true mini-batch, set this > 1 (no big benefit for now).
    train_examples_per_microbatch: int = 1

    # Sampling
    temperature: float = 1.0
    top_p: float = 1.0

    # PUCT
    puct_c: float = 1.0
    max_buffer_size: int = 1000
    topk_children_per_parent: int = 2

    # Misc
    seed: int = 42
    print_responses: int = 0           # how many rollouts to print per step (debug)

    # Multi-GPU generation
    num_gpus: int = 8
    gpu_ids: str = "0,1,2,3,4,5,6,7"


# ======================================================================
# CLI parsing
# ======================================================================
def parse_args_into_cfg() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="TTT-Discover for circle packing")
    # Backend
    p.add_argument("--backend", choices=["auto", "unsloth", "hf"], default=cfg.backend)
    p.add_argument("--model-name", default=cfg.model_name)
    p.add_argument("--load-in-4bit", action="store_true", default=cfg.load_in_4bit)
    p.add_argument("--max-seq-length", type=int, default=cfg.max_seq_length)
    # LoRA
    p.add_argument("--lora-rank", type=int, default=cfg.lora_rank)
    p.add_argument("--lora-alpha", type=int, default=cfg.lora_alpha)
    p.add_argument("--lora-dropout", type=float, default=cfg.lora_dropout)
    # Problem
    p.add_argument("--num-circles", type=int, default=cfg.num_circles)
    p.add_argument("--target", type=float, default=cfg.target)
    p.add_argument("--sandbox-timeout-s", type=float, default=cfg.sandbox_timeout_s)
    # Scale knobs — the ones you'll actually tweak
    p.add_argument("--num-steps", type=int, default=cfg.num_steps,
                   help="Number of TTT-Discover steps (paper: 50)")
    p.add_argument("--groups-per-step", type=int, default=cfg.groups_per_step,
                   help="Number of parent states sampled per step (paper: 8)")
    p.add_argument("--group-size", type=int, default=cfg.group_size,
                   help="Rollouts per parent per step (paper: 64)")
    p.add_argument("--num-seed-states", type=int, default=cfg.num_seed_states)
    p.add_argument("--max-new-tokens", type=int, default=cfg.max_new_tokens)
    # Optimization
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--kl-penalty-coef", type=float, default=cfg.kl_penalty_coef)
    p.add_argument("--grad-clip", type=float, default=cfg.grad_clip)
    # Sampling
    p.add_argument("--temperature", type=float, default=cfg.temperature)
    p.add_argument("--top-p", type=float, default=cfg.top_p)
    # Misc
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--print-responses", type=int, default=cfg.print_responses)
    # Multi-GPU generation
    p.add_argument("--num-gpus", type=int, default=cfg.num_gpus,
                   help="Number of GPUs for parallel generation. 1 = single-process "
                        "in-line generation (no worker pool). >1 spawns that many "
                        "plain-HF generation workers, one per GPU.")
    p.add_argument("--gpu-ids", type=str, default=cfg.gpu_ids,
                   help="Comma-separated physical GPU ids for the workers, e.g. "
                        "'0,1,2,3,4,5,6,7'. Defaults to 0..num_gpus-1.")
    args = p.parse_args()

    cfg.backend = args.backend
    cfg.model_name = args.model_name
    cfg.load_in_4bit = args.load_in_4bit
    cfg.max_seq_length = args.max_seq_length
    cfg.lora_rank = args.lora_rank
    cfg.lora_alpha = args.lora_alpha
    cfg.lora_dropout = args.lora_dropout
    cfg.num_circles = args.num_circles
    cfg.target = args.target
    cfg.sandbox_timeout_s = args.sandbox_timeout_s
    cfg.num_steps = args.num_steps
    cfg.groups_per_step = args.groups_per_step
    cfg.group_size = args.group_size
    cfg.num_seed_states = args.num_seed_states
    cfg.max_new_tokens = args.max_new_tokens
    cfg.learning_rate = args.lr
    cfg.kl_penalty_coef = args.kl_penalty_coef
    cfg.grad_clip = args.grad_clip
    cfg.temperature = args.temperature
    cfg.top_p = args.top_p
    cfg.seed = args.seed
    cfg.print_responses = args.print_responses
    cfg.num_gpus = args.num_gpus
    cfg.gpu_ids = args.gpu_ids
    return cfg


# ======================================================================
# Generation
# ======================================================================
def _generate_batch(model, tokenizer, inputs, input_len, n_samples, cfg):
    """
    Generate n_samples completions for a SINGLE prompt in ONE batched
    model.generate() call (via num_return_sequences). Returns a list of
    (text, gen_token_ids).
    """
    import torch
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            pad_token_id=pad_id,
            num_return_sequences=n_samples,
        )
    # out shape: (n_samples, input_len + gen_len). All share the same prompt,
    # so the prompt occupies the first input_len columns for every row.
    results = []
    for i in range(out.shape[0]):
        gen_ids = out[i, input_len:].tolist()
        if eos_id is not None and eos_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eos_id) + 1]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        results.append((text, gen_ids))
    return results


def generate_responses(model, tokenizer, prompt_text: str, group_size: int, cfg: Config):
    """
    Generate `group_size` responses from a single prompt, batched.

    We try to generate all `group_size` at once. If that OOMs, we halve the
    per-call batch size and retry, accumulating until we have group_size
    responses. This keeps the algorithm identical (still group_size IID
    samples from the same policy) while using the GPU in parallel.

    Returns (list of (text, gen_token_ids), prompt_len_in_tokens).
    """
    import torch
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    responses = []
    remaining = group_size
    # Start by trying the whole group in one call.
    batch = group_size

    while remaining > 0:
        n = min(batch, remaining)
        try:
            chunk = _generate_batch(model, tokenizer, inputs, input_len, n, cfg)
            responses.extend(chunk)
            remaining -= n
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch == 1:
                # Can't even do one — re-raise, nothing we can do
                raise
            batch = max(1, batch // 2)
            print(f"  [oom] halving generation batch size to {batch}")

    return responses, input_len


# ======================================================================
# Logprob computation
# ======================================================================
def compute_token_logprobs(model, prompt_ids, response_ids, with_grad: bool):
    """
    Returns the per-token log-probabilities of the response under the model.

    prompt_ids:   (1, P) tensor
    response_ids: (1, R) tensor
    Output:       (R,) tensor of token logprobs
    """
    import torch
    import torch.nn.functional as F

    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        out = model(full_ids)
        logits = out.logits  # (1, T, V)
        P = prompt_ids.shape[1]
        R = response_ids.shape[1]
        # Predict response token at position P+k from logits at position P+k-1
        pred_logits = logits[:, P - 1 : P - 1 + R, :]
        log_probs = F.log_softmax(pred_logits.float(), dim=-1)
        gathered = log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)  # (1, R)
    return gathered.squeeze(0)


# ======================================================================
# LoRA adapter sync (main process -> generation workers)
# ======================================================================
def _adapter_dir(exp_dir, step_idx):
    from pathlib import Path
    return str(Path(exp_dir) / f"adapter_step{step_idx:03d}")


def _adapter_exists(exp_dir):
    from pathlib import Path
    p = Path(exp_dir)
    return any(p.glob("adapter_step*"))


def _save_adapter(model, exp_dir, step_idx):
    """
    Save the current LoRA adapter to disk so generation workers can load it.
    Returns the directory path. Cleans up the previous step's adapter to
    avoid filling the disk (we only ever need the latest).
    """
    import shutil
    from pathlib import Path

    out_dir = _adapter_dir(exp_dir, step_idx)
    # PEFT/Unsloth models support save_pretrained, which writes just the adapter
    model.save_pretrained(out_dir)

    # Remove older adapter dirs (keep only the current one)
    for old in Path(exp_dir).glob("adapter_step*"):
        if str(old) != out_dir:
            try:
                shutil.rmtree(old)
            except Exception:
                pass

    return out_dir


# ======================================================================
# One training step
# ======================================================================
def train_step(backend, model, tokenizer, sampler, optimizer, step_idx: int,
               cfg: Config, exp_dir, gen_pool=None):
    import torch

    from advantage import entropic_adaptive_advantages
    from reward import compute_reward, extract_python_code
    from prompts import build_prompt
    from sampler import State
    from experiment_io import save_rollout

    step_t0 = time.time()
    parents = sampler.sample_states(cfg.groups_per_step)
    print(f"\n[step {step_idx}] parents picked: {len(parents)}")
    for i, info in enumerate(sampler.last_picks_info):
        tag = "seed" if info["is_seed"] else "expanded"
        print(f"  parent {i} [{tag}]  value={info['value']:.4f}  n={info['n']}  "
              f"Q={info['Q']:.4f}  P={info['P']:.4f}  bonus={info['bonus']:.4f}  "
              f"score={info['score']:.4f}")

    all_examples = []
    all_children = []

    # ----- BUILD PROMPTS (one per parent/group) -----
    prompts_by_group = []
    
    for g, parent in enumerate(parents):
        sampler.record_expansion(parent, count=cfg.group_size)
        messages = build_prompt(
            num_circles=cfg.num_circles,
            parent_code=parent.code,
            parent_value=parent.value or 0.0,
            target=cfg.target,
        )
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        prompts_by_group.append(prompt_text)

    # ----- ROLLOUTS -----
    rollout_t0 = time.time()

    if gen_pool is not None:
        # Multi-GPU path: save current LoRA adapter, then generate all groups
        # in parallel across all workers. We save every step (including step 0,
        # so workers use the same LoRA-wrapped policy the main process has).
        adapter_path = _save_adapter(model, exp_dir, step_idx)

        group_responses = gen_pool.generate_groups(
            prompts_by_group=prompts_by_group,
            group_size=cfg.group_size,
            adapter_path=adapter_path,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
    else:
        # Single-GPU fallback path: sequential generation in this process.
        backend.set_inference_mode()
        group_responses = {}
        for g, prompt_text in enumerate(prompts_by_group):
            responses, _ = generate_responses(
                model, tokenizer, prompt_text, cfg.group_size, cfg
            )
            group_responses[g] = responses

    # ----- SCORE + ADVANTAGE + SAVE + COLLECT TRAINING EXAMPLES -----
    for g, parent in enumerate(parents):
        prompt_text = prompts_by_group[g]
        responses = group_responses[g]

        rewards = []
        codes = []
        valids = []
        outs = []
        for r_idx, (text, _) in enumerate(responses):
            out = compute_reward(text, num_circles=cfg.num_circles,
                                 timeout_s=cfg.sandbox_timeout_s)
            rewards.append(out["reward"])
            codes.append(extract_python_code(text) or "")
            valids.append(out["valid"])
            outs.append(out)

        rewards_np = np.array(rewards, dtype=np.float64)
        advantages, beta = entropic_adaptive_advantages(rewards_np)
        print(f"  group {g}: rewards min={rewards_np.min():.4f} "
              f"mean={rewards_np.mean():.4f} max={rewards_np.max():.4f}  "
              f"valid={sum(valids)}/{len(valids)}  β={beta:.4f}")

        # Save every rollout (response + meta) to disk for debugging
        for r_idx, (text, gen_ids) in enumerate(responses):
            meta = {
                "step": step_idx,
                "group": g,
                "rollout": r_idx,
                "reward": float(rewards[r_idx]),
                "valid": bool(valids[r_idx]),
                "parsed": bool(outs[r_idx]["parsed"]),
                "ran": bool(outs[r_idx]["ran"]),
                "msg": outs[r_idx]["msg"],
                "advantage": float(advantages[r_idx]) if hasattr(advantages, "__len__") else 0.0,
                "beta": float(beta),
                "n_response_tokens": len(gen_ids),
                "sandbox_stdout": outs[r_idx].get("stdout", "")[:2000],
                "parent_value": float(parent.value) if parent.value is not None else None,
                "parent_is_seed": parent.id in sampler._seed_ids,
            }
            save_rollout(exp_dir, step_idx, g, r_idx, text, meta)

        # Children for the sampler
        for r_idx, (_, _) in enumerate(responses):
            if valids[r_idx] and codes[r_idx]:
                child = State.make(
                    timestep=step_idx,
                    value=rewards[r_idx],
                    code=codes[r_idx],
                )
                all_children.append((child, parent))

        # If reward is constant in this group, no training signal
        if float(rewards_np.max() - rewards_np.min()) < 1e-12:
            continue

        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
        for (text, gen_ids), adv in zip(responses, advantages):
            if len(gen_ids) == 0:
                continue
            response_ids = torch.tensor([gen_ids], device=model.device)
            all_examples.append({
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "advantage": float(adv),
            })

    rollout_time = time.time() - rollout_t0
    print(f"[step {step_idx}] rollout time: {rollout_time:.1f}s  "
          f"training examples: {len(all_examples)}  new children: {len(all_children)}")

    # Update archive
    sampler.update(all_children)

    if not all_examples:
        print(f"[step {step_idx}] no training signal (all groups had constant reward)")
        return

    # ----- TRAIN STEP -----
    backend.set_training_mode()
    optimizer.zero_grad()

    train_t0 = time.time()
    total_loss = 0.0
    total_logp_delta = 0.0
    n_examples = len(all_examples)

    for ex in all_examples:
        pid = ex["prompt_ids"]
        rid = ex["response_ids"]
        adv = ex["advantage"]

        # Current policy logprobs (with grad)
        cur_lp = compute_token_logprobs(model, pid, rid, with_grad=True)  # (R,)

        # Base policy logprobs (LoRA disabled)
        try:
            with backend.disable_adapter(), torch.no_grad():
                base_lp = compute_token_logprobs(model, pid, rid, with_grad=False)
        except Exception as e:
            # If disable_adapter isn't supported, skip KL (just warn once)
            if not hasattr(train_step, "_kl_warned"):
                print(f"[warn] disable_adapter failed ({e}); training without KL penalty")
                train_step._kl_warned = True
            base_lp = cur_lp.detach()

        # KL correction (Tang–Munos baselined form):
        # advantage_per_token = adv + kl_coef * (mean(logp_diff) - logp_diff_per_token)
        logp_diff = (cur_lp - base_lp).detach()
        avg_logp_diff = logp_diff.mean()
        kl_adv = cfg.kl_penalty_coef * (avg_logp_diff - (cur_lp - base_lp))
        eff_adv = adv + kl_adv  # broadcasts scalar adv over (R,) kl_adv

        # Loss: -E_token[advantage_token * logprob_token]
        loss = -(eff_adv.detach() * cur_lp).mean()
        # loss.backward()
        (loss / n_examples).backward()


        total_loss += float(loss.detach().item())
        total_logp_delta += float(logp_diff.mean().item())

    # Gradient clip + optimizer step
    import torch as _torch
    _torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        max_norm=cfg.grad_clip,
    )
    optimizer.step()

    train_time = time.time() - train_t0
    print(f"[step {step_idx}] train time: {train_time:.1f}s  "
          f"avg loss: {total_loss / n_examples:.4f}  "
          f"avg logπ_θ - logπ_base: {total_logp_delta / n_examples:.4f}")

    best = sampler.best_state()
    if best is not None:
        print(f"[step {step_idx}] best so far: {best.value:.6f}  "
              f"(step total {time.time() - step_t0:.1f}s, archive={sampler.archive_size()})")


# ======================================================================
# Main
# ======================================================================
def main():
    cfg = parse_args_into_cfg()

    print("=" * 70)
    print("TTT-Discover (Circle Packing) — local implementation")
    print("=" * 70)
    print(f"Model:              {cfg.model_name}")
    print(f"Backend:            {cfg.backend}")
    print(f"Num circles:        {cfg.num_circles}")
    print(f"Target:             {cfg.target}")
    print(f"Steps:              {cfg.num_steps}")
    print(f"Groups per step:    {cfg.groups_per_step}")
    print(f"Group size:         {cfg.group_size}")
    print(f"Total rollouts/step: {cfg.groups_per_step * cfg.group_size}")
    print(f"Seeds:              {cfg.num_seed_states}")
    print(f"LR:                 {cfg.learning_rate}")
    print(f"KL coef:            {cfg.kl_penalty_coef}")
    print(f"Max new tokens:     {cfg.max_new_tokens}")
    print("=" * 70)

    # ---- experiment dir ----
    from experiment_io import make_experiment_dir, save_final_summary
    exp_dir = make_experiment_dir(cfg)
    print(f"[init] writing all rollouts to: {exp_dir}")

    # ---- backend + model ----
    # Load backend FIRST so Unsloth can patch transformers if used.
    from model_backend import load_backend
    backend = load_backend(cfg.backend, cfg)
    model, tokenizer = backend.load()

    import torch  # safe to import now
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[init] trainable params: {trainable:,} / total {total:,} "
          f"({100 * trainable / total:.2f}%)")

    # ---- sampler ----
    from sampler import PUCTSampler
    sampler = PUCTSampler(
        num_seeds=cfg.num_seed_states,
        puct_c=cfg.puct_c,
        max_buffer_size=cfg.max_buffer_size,
        topk_children=cfg.topk_children_per_parent,
        seed_value=0.0,
    )
    print(f"[init] sampler archive size = {sampler.archive_size()}")

    # ---- generation pool (multi-GPU) ----
    gen_pool = None
    if cfg.num_gpus and cfg.num_gpus > 1:
        from gen_workers import GenerationPool
        if cfg.gpu_ids:
            gpu_ids = [int(x) for x in cfg.gpu_ids.split(",")]
        else:
            gpu_ids = list(range(cfg.num_gpus))
        print(f"[init] starting generation pool: {cfg.num_gpus} GPUs {gpu_ids}")
        gen_pool = GenerationPool(
            model_name=cfg.model_name,
            num_workers=cfg.num_gpus,
            gpu_ids=gpu_ids,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit,
        )
        print("[init] generation pool ready")
    else:
        print("[init] single-GPU generation (no worker pool)")

    # ---- main loop ----
    try:
        for step in range(cfg.num_steps):
            train_step(backend, model, tokenizer, sampler, optimizer, step,
                       cfg, exp_dir, gen_pool)
    finally:
        if gen_pool is not None:
            print("[shutdown] stopping generation pool ...")
            gen_pool.shutdown()

    # ---- summary ----
    print("\n" + "=" * 70)
    print("TRAINING DONE")
    print("=" * 70)
    best = sampler.best_state()
    if best is not None:
        print(f"Best sum of radii: {best.value:.6f}")
        print(f"Found at step:     {best.timestep}")
        print(f"\n--- best code ---\n{best.code}\n--- end ---")
        save_final_summary(exp_dir, best.value, best.code, best.timestep)
    else:
        print("No valid packing was ever produced.")
        save_final_summary(exp_dir, None, None, None)
    print(f"\nAll outputs saved under: {exp_dir}")


if __name__ == "__main__":
    main()