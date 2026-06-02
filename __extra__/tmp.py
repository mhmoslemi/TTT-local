# import numpy as np
# from scipy.optimize import minimize


# def run_packing():
#     n = 26

#     initial_centers = []
#     initial_radii = []

#     # Adjusted initial radius and spacing parameters
#     r_initial = 0.102  # Slightly smaller for better flexibility
#     buffer = 1e-6  # Small buffer to prevent boundary violations

#     # Generate staggered grid with 5 rows and varying number of circles per row
#     for row in range(5):  # 5 rows total
#         # Even rows start at r_initial, odd rows also start with buffer
#         if row % 2 == 0:
#             x_start = r_initial + buffer  # Even rows start slightly inside
#         else:
#             x_start = r_initial + buffer  # Odd rows also start with buffer

#         # Varying number of circles per row to fit better
#         if row == 0 or row == 2 or row == 4:
#             num_circles = 5  # Even rows (0, 2, 4) have 5 circles
#         elif row == 1:
#             num_circles = 6  # First odd row has 6 circles
#         else:  # row == 3
#             num_circles = 5  # Second odd row has 5 circles

#         if num_circles == 0:
#             continue

#         # Calculate horizontal spacing for this row
#         if num_circles == 1:
#             spacing_row = 0.0
#         else:
#             # Ensure horizontal spacing is at least 2*r_initial to prevent overlaps
#             max_horizontal = 1 - 2 * r_initial
#             spacing_row = max_horizontal / (num_circles - 1) if max_horizontal > 0 else 0.0

#         # Place circles in this row
#         for col in range(num_circles):
#             x = x_start + col * spacing_row
#             # Vertical positioning with refined vertical spacing
#             if row == 0:
#                 y = r_initial + buffer  # First row starts with buffer
#             else:
#                 # Vertical spacing with a refined factor for denser packing
#                 y = r_initial + buffer + row * 1.0 * np.sqrt(3) * r_initial

#             # Ensure y does not exceed 1 - r_initial
#             if y + r_initial > 1 + 1e-6:
#                 y = 1 - r_initial - 1e-6  # Clamp to prevent overflow

#             initial_centers.append([x, y])
#             # Assign initial radii based on row (middle row gets a slight boost)
#             if row == 2:
#                 initial_radii.append(r_initial + 0.003)  # Increased boost for central row
#             else:
#                 initial_radii.append(r_initial)

#     # Flatten the initial variables for optimization
#     variables_initial = []
#     for i in range(n):
#         variables_initial.extend(initial_centers[i])
#         variables_initial.append(initial_radii[i])

#     # Objective function to maximize sum of radii
#     def objective(vars):
#         total = 0.0
#         for i in range(n):
#             idx = i * 3
#             total += vars[idx + 2]
#         return -total  # Minimize negative sum to maximize

#     # Define constraints
#     constraints = []

#     # Constraints for center positions and radii
#     for i in range(n):
#         # x_i >= r_i
#         def constraint1(vars, i=i):
#             idx = i * 3
#             return vars[idx] - vars[idx + 2]
#         constraints.append({'type': 'ineq', 'fun': constraint1})

#         # x_i + r_i <= 1
#         def constraint2(vars, i=i):
#             idx = i * 3
#             return 1 - (vars[idx] + vars[idx + 2])
#         constraints.append({'type': 'ineq', 'fun': constraint2})

#         # y_i >= r_i
#         def constraint3(vars, i=i):
#             idx = i * 3
#             return vars[idx + 1] - vars[idx + 2]
#         constraints.append({'type': 'ineq', 'fun': constraint3})

#         # y_i + r_i <= 1
#         def constraint4(vars, i=i):
#             idx = i * 3
#             return 1 - (vars[idx + 1] + vars[idx + 2])
#         constraints.append({'type': 'ineq', 'fun': constraint4})

#     # Pairwise distance constraints
#     for i in range(n):
#         for j in range(i + 1, n):
#             def constraint_pair(vars, i=i, j=j):
#                 idx_i = i * 3
#                 idx_j = j * 3
#                 x_i, y_i, r_i = vars[idx_i], vars[idx_i + 1], vars[idx_i + 2]
#                 x_j, y_j, r_j = vars[idx_j], vars[idx_j + 1], vars[idx_j + 2]
#                 dist = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
#                 return dist - (r_i + r_j)
#             constraints.append({'type': 'ineq', 'fun': constraint_pair})

#     # Optimize using Sequential Least Squares Programming with refined parameters
#     result = minimize(
#         objective,
#         variables_initial,
#         method='SLSQP',
#         constraints=constraints,
#         options={
#             'ftol': 1e-14,
#             'maxiter': 1000000,
#             'disp': False,
#             'eps': 1e-12,
#             'iprint': 0,  # Suppress verbose output
#             'finite_diff_rel_step': np.sqrt(np.finfo(float).eps)
#         }
#     )

#     # Extract optimized centers and radii
#     optimized_vars = result.x
#     centers = []
#     radii = []
#     for i in range(n):
#         idx = i * 3
#         centers.append([optimized_vars[idx], optimized_vars[idx + 1]])
#         radii.append(optimized_vars[idx + 2])

#     sum_radii = sum(radii)
#     return np.array(centers), np.array(radii), sum_radii
    

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
    model_name: str =  "Qwen/Qwen3-8B"   #'Qwen/Qwen3-4B'    # "Qwen/Qwen3-8B"  #"Qwen/Qwen2.5-Coder-1.5B-Instruct" # "LiquidAI/LFM2.5-350M"
    backend: str = "auto"        # "auto" | "unsloth" | "hf"
    max_seq_length: int = 32768
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
    target: float = 2.636
    sandbox_timeout_s: float = 30.0

    # RL hyperparameters — paper scale
    num_steps: int = 50
    groups_per_step: int = 8
    group_size: int = 64
    num_seed_states: int = 8           # paper uses several seeds for diversity
    # learning_rate: float = 4e-5
    learning_rate: float = 1e-6
    kl_penalty_coef: float = 0.1
    max_new_tokens: int = 2048
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
    import torch  # local import; backend has been loaded by now
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
def train_step(backend, model, tokenizer, sampler, optimizer, step_idx: int, cfg: Config):
    import torch

    from advantage import entropic_adaptive_advantages
    from reward import compute_reward, extract_python_code
    from prompts import build_prompt
    from sampler import State

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

    # ----- ROLLOUTS -----
    backend.set_inference_mode()
    rollout_t0 = time.time()

    for g, parent in enumerate(parents):
        sampler.record_expansion(parent)

        messages = build_prompt(
            num_circles=cfg.num_circles,
            parent_code=parent.code,
            parent_value=parent.value or 0.0,
            target=cfg.target,
        )
        # For Qwen3 (a hybrid thinking model), pass enable_thinking=False so the
        # model doesn't waste the token budget on <think>...</think>. The kwarg
        # is silently ignored by templates that don't recognize it.
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        responses, _ = generate_responses(
            model, tokenizer, prompt_text, cfg.group_size, cfg
        )

        rewards = []
        codes = []
        valids = []
        for r_idx, (text, _) in enumerate(responses):
            out = compute_reward(text, num_circles=cfg.num_circles,
                                 timeout_s=cfg.sandbox_timeout_s)
            rewards.append(out["reward"])
            codes.append(extract_python_code(text) or "")
            valids.append(out["valid"])
            if r_idx < cfg.print_responses:
                print(f"  [g{g}/r{r_idx}] reward={out['reward']:.4f} "
                      f"valid={out['valid']} msg={out['msg']}")
                # Dump the actual response (truncated) so we can debug
                snippet = text if len(text) < 1500 else text[:750] + "\n... [truncated] ...\n" + text[-750:]
                print(f"  --- raw response (g{g}/r{r_idx}) ---")
                print(snippet)
                print(f"  --- end raw response ---")
                if out["stdout"]:
                    stdout_snippet = out["stdout"][:500]
                    print(f"  --- sandbox stdout ---\n{stdout_snippet}\n  --- end stdout ---")

        rewards_np = np.array(rewards, dtype=np.float64)
        advantages, beta = entropic_adaptive_advantages(rewards_np)
        print(f"  group {g}: rewards min={rewards_np.min():.4f} "
              f"mean={rewards_np.mean():.4f} max={rewards_np.max():.4f}  "
              f"valid={sum(valids)}/{len(valids)}  β={beta:.4f}")

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
        # loss = -(eff_adv.detach() * cur_lp).mean()
        # loss.backward()

        loss = -(eff_adv.detach() * cur_lp).mean()
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

    # ---- main loop ----
    for step in range(cfg.num_steps):
        train_step(backend, model, tokenizer, sampler, optimizer, step, cfg)

    # ---- summary ----
    print("\n" + "=" * 70)
    print("TRAINING DONE")
    print("=" * 70)
    best = sampler.best_state()
    if best is not None:
        print(f"Best sum of radii: {best.value:.6f}")
        print(f"Found at step:     {best.timestep}")
        print(f"\n--- best code ---\n{best.code}\n--- end ---")
    else:
        print("No valid packing was ever produced.")


if __name__ == "__main__":
    main()