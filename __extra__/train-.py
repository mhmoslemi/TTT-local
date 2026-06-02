"""
TTT-Discover for Circle Packing — runs locally.


Defaults hyperparams as paper. 

"""


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import argparse
import time
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


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
    target_modules: Tuple[str, ...] = (  #  for LORA: which weight matrices inside the transformer to attach adapters
        "q_proj", "k_proj", "v_proj", "o_proj", # attention
        "gate_proj", "up_proj", "down_proj", # MLP / feed-forward
    )

    # Problem
    num_circles: int = 26
    target: float = 2.636
    sandbox_timeout_s: float = 30.0


    # max_seq_length (total context = prompt + response)
    
    # max_new_tokens only limits the response portion
    #   -->  higher = fewer truncation failures but slower generation and more memory.

    # RL hyperparameters — paper scale
    num_steps: int = 50
    groups_per_step: int = 8
    group_size: int = 64
    num_seed_states: int = 8           
    learning_rate: float = 1e-5
    kl_penalty_coef: float = 0.1
    max_new_tokens: int = 2048
    grad_clip: float = 1.0

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
    print_responses: int = 0           # how many rollouts to print per step


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
    # Scale knobs 
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
    return cfg


# ======================================================================
# Generation
# ======================================================================
def generate_responses(model, tokenizer, prompt_text: str, group_size: int, cfg: Config):
    """
    Generate `group_size` responses from a single prompt.
    Returns list of (text, gen_token_ids) and the prompt length in tokens.
    """
    import torch  
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    responses = []
    with torch.inference_mode():
        for _ in range(group_size):
            out = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                pad_token_id=pad_id,
            )
            gen_ids = out[0, input_len:].tolist()
            if eos_id is not None and eos_id in gen_ids:
                gen_ids = gen_ids[: gen_ids.index(eos_id) + 1]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            responses.append((text, gen_ids))

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
# One training step
# ======================================================================
def train_step(backend, model, tokenizer, sampler, optimizer, step_idx: int,
               cfg: Config, exp_dir):
    import torch

    from advantage import entropic_adaptive_advantages
    from reward import compute_reward, extract_python_code
    from prompts import build_prompt
    from sampler import State
    from experiment_io import save_rollout

    step_t0 = time.time()

    parents = sampler.sample_states(cfg.groups_per_step) # PUCT sample G parent, group per step is the size
    # For every state in the archive, compute the PUCT score
        #  Q: If never expanded, Q falls back to the state's own value. So a state is "worth" the best thing it has produced.
        # Scale:  max(value) − min(value) across the archive (_scale()), floored at 1.0 if there's no variance.
        # P: rank based: w / w.sum()
        # bonus: sqrt(1+T) / (1+n)
    # Sort all states by (score, value) descending.

    # IMPORTANT: _full_lineage collects the state's ancestors (from parents) plus all its descendants (BFS down a children map) plus itself. 
    #     --> Once you pick a state, its whole family tree is blocked from being picked again in this same batch. 


    ### KEEP IN MIND FOR LATER -->  The "search" is shallow and breadth-oriented, and its not fully like AlphaZero but similar 
        # maybe we can do island based later


    print(f"\n[step {step_idx}] parents picked: {len(parents)}")
    for i, info in enumerate(sampler.last_picks_info):
        tag = "seed" if info["is_seed"] else "expanded"
        print(f"  parent {i} [{tag}]  value={info['value']:.4f}  n={info['n']}  "
              f"Q={info['Q']:.4f}  P={info['P']:.4f}  bonus={info['bonus']:.4f}  "
              f"score={info['score']:.4f}")

    all_examples = []
    all_children = []

    # ----- ROLLOUTS -----
    backend.set_inference_mode()
    rollout_t0 = time.time()

    for g, parent in enumerate(parents):
        sampler.record_expansion(parent) # record that we expand this parent

        messages = build_prompt(
            num_circles=cfg.num_circles,
            parent_code=parent.code,
            parent_value=parent.value or 0.0,
            target=cfg.target,
        )
        # For Qwen3, enable_thinking=False so the
        # model doesn't waste the token budget on <think>...</think>.
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        responses, _ = generate_responses(              # for each parent, make K response (group size)
            model, tokenizer, prompt_text, cfg.group_size, cfg
        )

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

    for ex in all_examples:  # roll outs that we want to train on
        pid = ex["prompt_ids"]
        rid = ex["response_ids"]
        adv = ex["advantage"] # the entropic advantage of this rollout

        # Current policy logprobs (with grad)
        # ---> asks the current policy (base + LoRA): "what log-probability did you assign to the token that was actually generated here?" 
        cur_lp = compute_token_logprobs(model, pid, rid, with_grad=True)  # (R,) 
        # ---> cur_lp[k] = log \pi_\theta (token_k | everything before it).

        # Base policy logprobs (LoRA disabled)
        # Disabling LoRA is how we get the base model's logprobs, which is the reference distribution the KL penalty measures drift against.
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
        loss.backward()

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
    from model_backend import load_backend
    backend = load_backend(cfg.backend, cfg)
    model, tokenizer = backend.load()

    import torch  
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[init] trainable params: {trainable:,} / total {total:,} "
          f"({100 * trainable / total:.2f}%)")

    # ---- sampler ----
    from sampler import PUCTSampler
    # At init, all seeds are identical: value 0, empty code, n=0, no Q. They differ only by their random id.
    sampler = PUCTSampler(
        num_seeds=cfg.num_seed_states,
        puct_c=cfg.puct_c,
        max_buffer_size=cfg.max_buffer_size,
        topk_children=cfg.topk_children_per_parent,
        seed_value=0.0,
    )
    print(f"[init] sampler archive size = {sampler.archive_size()}")

    # ---- main loop ----
    for step in range(cfg.num_steps):
        train_step(backend, model, tokenizer, sampler, optimizer, step, cfg, exp_dir)

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