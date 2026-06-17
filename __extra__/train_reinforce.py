"""
train_reinforce.py  -- drop-in replacement for the TTT-Discover train_step.

CONTROL / FLOOR for the benchmark.

This is your TTT train step with EXACTLY ONE axis changed: the entropic weight
w_beta(a) - 1 is replaced by a plain linear, z-scored, leave-nothing-else group
advantage  (R - mean) / (std + eps).  Everything else -- KL shaping, generation,
reward pool, rollout saving, child construction, archive update -- is identical
to your runner. So a TTT-vs-REINFORCE anytime-curve diff isolates the value of
the exponential reward upweighting and nothing else.

INTEGRATION
-----------
In your main runner, replace the inline train_step (and its import site) with:

    from train_reinforce import train_step

The signature is unchanged:
    train_step(backend, model, tokenizer, sampler, optimizer, step_idx,
               cfg, exp_dir, problem, gen_pool=None)

Optional cfg knob (defaults fine if absent):
    reward_workers: int = 0     # 0 => auto (cpu_count - num_gpus)
"""

import time
import numpy as np


# ----------------------------------------------------------------------
# Helpers (self-contained copies of the runner's primitives)
# ----------------------------------------------------------------------
def compute_token_logprobs(model, prompt_ids, response_ids, with_grad: bool):
    import torch
    import torch.nn.functional as F
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        logits = model(full_ids).logits             # (1, T, V)
        P = prompt_ids.shape[1]
        R = response_ids.shape[1]
        pred = logits[:, P - 1: P - 1 + R, :]
        lp = F.log_softmax(pred.float(), dim=-1)
        gathered = lp.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)
    return gathered.squeeze(0)


def _adapter_dir(exp_dir, step_idx):
    from pathlib import Path
    return str(Path(exp_dir) / f"adapter_step{step_idx:03d}")


def _save_adapter(model, exp_dir, step_idx):
    import shutil
    from pathlib import Path
    out_dir = _adapter_dir(exp_dir, step_idx)
    model.save_pretrained(out_dir)
    for old in Path(exp_dir).glob("adapter_step*"):
        if str(old) != out_dir:
            try:
                shutil.rmtree(old)
            except Exception:
                pass
    return out_dir


def _generate_batch(model, tokenizer, inputs, input_len, n_samples, cfg):
    import torch
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=cfg.max_new_tokens, do_sample=True,
            temperature=cfg.temperature, top_p=cfg.top_p,
            pad_token_id=pad_id, num_return_sequences=n_samples,
        )
    results = []
    for i in range(out.shape[0]):
        gen_ids = out[i, input_len:].tolist()
        if eos_id is not None and eos_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eos_id) + 1]
        results.append((tokenizer.decode(gen_ids, skip_special_tokens=True), gen_ids))
    return results


def generate_responses(model, tokenizer, prompt_text, group_size, cfg):
    import torch
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    responses, remaining, batch = [], group_size, group_size
    while remaining > 0:
        n = min(batch, remaining)
        try:
            responses.extend(_generate_batch(model, tokenizer, inputs, input_len, n, cfg))
            remaining -= n
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch == 1:
                raise
            batch = max(1, batch // 2)
            print(f"  [oom] halving generation batch size to {batch}")
    return responses, input_len


# ----------------------------------------------------------------------
# Shared rollout + scoring scaffolding (identical across all four algos)
# ----------------------------------------------------------------------
def _collect_groups(backend, model, tokenizer, sampler, step_idx, cfg,
                    exp_dir, problem, gen_pool):
    """Generate + score every group; return (parents, groups, rollout_time).
    `groups[g]` is a dict with parent / prompt_text / responses / outs / rewards.
    Search-level bookkeeping (record_expansion) happens here; advantage, rollout
    saving, child construction and the optimizer step are the caller's job."""
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from problems.base import ParentContext
    from gen_workers import make_progress_bar

    sampler.set_current_step(step_idx)
    parents = sampler.sample_states(cfg.groups_per_step)
    print(f"\n[step {step_idx}] parents picked: {len(parents)}")
    for i, info in enumerate(sampler.last_picks_info):
        tag = "seed" if info["is_seed"] else "expanded"
        print(f"  parent {i} [{tag}]  value={info['value']:.4f}  n={info['n']}  "
              f"Q={info['Q']:.4f}  P={info['P']:.4f}  bonus={info['bonus']:.4f}  "
              f"score={info['score']:.4f}")

    prompts_by_group, parent_ctxs = [], []
    for g, parent in enumerate(parents):
        sampler.record_expansion(parent, count=cfg.group_size)
        pc = ParentContext(
            code=parent.code,
            value=parent.value if parent.value is not None else 0.0,
            raw_score=parent.raw_score, construction=parent.construction,
        )
        parent_ctxs.append(pc)
        messages = problem.build_prompt(pc)
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        prompts_by_group.append(prompt_text)

    num_groups = len(parents)
    total_rollouts = num_groups * cfg.group_size

    n_reward_workers = getattr(cfg, "reward_workers", 0) or \
        max(1, (os.cpu_count() or 8) - max(0, cfg.num_gpus))
    reward_pool = ThreadPoolExecutor(max_workers=n_reward_workers)
    group_responses = {g: [] for g in range(num_groups)}
    reward_futures = {g: [] for g in range(num_groups)}

    def _submit(g, text, ids):
        group_responses[g].append((text, ids))
        reward_futures[g].append(reward_pool.submit(
            problem.compute_reward, text, parent_ctxs[g], cfg.sandbox_timeout_s))

    rollout_t0 = time.time()
    try:
        if gen_pool is not None:
            adapter_path = _save_adapter(model, exp_dir, step_idx)
            for gidx, job in gen_pool.iter_group_jobs(
                    prompts_by_group=prompts_by_group, group_size=cfg.group_size,
                    adapter_path=adapter_path, max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature, top_p=cfg.top_p):
                for (text, ids) in job:
                    _submit(gidx, text, ids)
        else:
            backend.set_inference_mode()
            bar = make_progress_bar(total_rollouts, desc="rollouts")
            try:
                for g, prompt_text in enumerate(prompts_by_group):
                    responses, _ = generate_responses(
                        model, tokenizer, prompt_text, cfg.group_size, cfg)
                    for (text, ids) in responses:
                        _submit(g, text, ids)
                    bar.update(len(responses))
            finally:
                bar.close()

        all_futs = [f for g in range(num_groups) for f in reward_futures[g]]
        ebar = make_progress_bar(len(all_futs), desc="evaluating")
        try:
            for _ in as_completed(all_futs):
                ebar.update(1)
        finally:
            ebar.close()
    finally:
        reward_pool.shutdown(wait=True)

    groups = []
    for g, parent in enumerate(parents):
        responses = group_responses[g]
        outs = [reward_futures[g][i].result() for i in range(len(responses))]
        rewards = np.array([o.reward for o in outs], dtype=np.float64)
        groups.append(dict(parent=parent, parent_ctx=parent_ctxs[g],
                           prompt_text=prompts_by_group[g],
                           responses=responses, outs=outs, rewards=rewards))
    return parents, groups, time.time() - rollout_t0


def _linear_group_advantages(rewards_np):
    """z-scored group advantage. The ONE axis that differs from TTT."""
    mean = rewards_np.mean()
    std = rewards_np.std()
    return (rewards_np - mean) / (std + 1e-8)


# ----------------------------------------------------------------------
# train_step
# ----------------------------------------------------------------------
def train_step(backend, model, tokenizer, sampler, optimizer, step_idx: int,
               cfg, exp_dir, problem, gen_pool=None):
    import torch
    from sampler import State
    from experiment_io import save_rollout

    step_t0 = time.time()
    parents, groups, rollout_time = _collect_groups(
        backend, model, tokenizer, sampler, step_idx, cfg, exp_dir, problem, gen_pool)

    all_examples, all_children = [], []

    for g, grp in enumerate(groups):
        parent = grp["parent"]
        responses = grp["responses"]
        outs = grp["outs"]
        rewards_np = grp["rewards"]
        valids = [o.valid for o in outs]
        codes = [o.code or "" for o in outs]

        advantages = _linear_group_advantages(rewards_np)   # <-- linear, not entropic
        print(f"  group {g}: rewards min={rewards_np.min():.4f} "
              f"mean={rewards_np.mean():.4f} max={rewards_np.max():.4f}  "
              f"valid={sum(valids)}/{len(valids)}  (REINFORCE linear adv)")

        # Save every rollout (same meta schema as your runner; beta dropped).
        for r_idx, (text, token_ids) in enumerate(responses):
            res = outs[r_idx]
            meta = {
                "step": step_idx, "group": g, "rollout": r_idx,
                "reward": float(rewards_np[r_idx]),
                "raw_score": (float(res.raw_score) if res.raw_score is not None else None),
                "valid": bool(valids[r_idx]), "parsed": bool(res.parsed),
                "ran": bool(res.ran), "msg": res.msg,
                "advantage": float(advantages[r_idx]),
                "algo": "reinforce",
                "n_response_tokens": len(token_ids),
                "sandbox_stdout": (res.stdout or "")[:2000],
                "parent_value": float(parent.value) if parent.value is not None else None,
                "parent_is_seed": parent.id in sampler._seed_ids,
            }
            save_rollout(exp_dir, step_idx, g, r_idx, text, meta)

        # Children for the shared search archive.
        for r_idx, (text, token_ids) in enumerate(responses):
            res = outs[r_idx]
            if valids[r_idx] and codes[r_idx]:
                all_children.append((State.make(
                    timestep=step_idx, value=float(rewards_np[r_idx]),
                    code=codes[r_idx], raw_score=res.raw_score,
                    construction=res.construction), parent))

        if float(rewards_np.max() - rewards_np.min()) < 1e-12:
            continue   # no training signal in this group

        prompt_ids = tokenizer(grp["prompt_text"], return_tensors="pt").input_ids.to(model.device)
        for (text, token_ids), adv in zip(responses, advantages):
            if len(token_ids) == 0:
                continue
            all_examples.append({
                "prompt_ids": prompt_ids,
                "response_ids": torch.tensor([token_ids], device=model.device),
                "advantage": float(adv),
            })

    print(f"[step {step_idx}] rollout+eval time: {rollout_time:.1f}s  "
          f"training examples: {len(all_examples)}  new children: {len(all_children)}")
    sampler.update(all_children)

    if not all_examples:
        print(f"[step {step_idx}] no training signal (all groups had constant reward)")
        return

    # ----- TRAIN STEP (KL shaping identical to your TTT runner) -----
    backend.set_training_mode()
    optimizer.zero_grad()
    train_t0 = time.time()
    total_loss = total_logp_delta = 0.0
    n_examples = len(all_examples)

    for ex in all_examples:
        pid, rid, adv = ex["prompt_ids"], ex["response_ids"], ex["advantage"]
        cur_lp = compute_token_logprobs(model, pid, rid, with_grad=True)     # (R,)
        try:
            with backend.disable_adapter(), torch.no_grad():
                base_lp = compute_token_logprobs(model, pid, rid, with_grad=False)
        except Exception as e:
            if not hasattr(train_step, "_kl_warned"):
                print(f"[warn] disable_adapter failed ({e}); training without KL penalty")
                train_step._kl_warned = True
            base_lp = cur_lp.detach()

        logp_diff = (cur_lp - base_lp).detach()
        kl_adv = cfg.kl_penalty_coef * (logp_diff.mean() - (cur_lp - base_lp))
        eff_adv = adv + kl_adv
        loss = -(eff_adv.detach() * cur_lp).mean()
        (loss / n_examples).backward()
        total_loss += float(loss.detach().item())
        total_logp_delta += float(logp_diff.mean().item())

    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=cfg.grad_clip)
    optimizer.step()

    print(f"[step {step_idx}] train time: {time.time() - train_t0:.1f}s  "
          f"avg loss: {total_loss / n_examples:.4f}  "
          f"avg logpi_theta - logpi_base: {total_logp_delta / n_examples:.4f}")

    best = sampler.best_state()
    if best is not None:
        raw = f" raw={best.raw_score:.6f}" if best.raw_score is not None else ""
        print(f"[step {step_idx}] best so far: value={best.value:.6f}{raw}  "
              f"(step total {time.time() - step_t0:.1f}s, archive={sampler.archive_size()})")
