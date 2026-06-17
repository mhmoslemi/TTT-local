"""
train_a2c.py  -- drop-in replacement for the TTT-Discover train_step.

THE VARIANCE / BASELINE AXIS.

Replaces the group-mean baseline with a LEARNED state value V(s). The advantage
for rollout i becomes  A_i = R_i - V(s_parent)  instead of  R_i - mean_group(R).
This is a one-step (terminal-reward) actor-critic: there is no bootstrapping, so
the critic target is simply the realized reward and V(s) regresses toward
E[R | s]. The KL shaping and everything search-side are identical to your runner.

HONEST CAVEAT
-------------
At group_size = 64 the group sample mean is already a low-variance estimate of
V(s), so the expected payoff over the REINFORCE control is small. A2C earns its
keep when groups are small, or when you want to share value signal across parents
you have only expanded once or twice. Run it precisely to confirm/deny that.

DESIGN CHOICES (each isolates the single axis cleanly)
------------------------------------------------------
* The value head reads the policy trunk's hidden state for the prompt, but the
  hidden state is DETACHED before the head. So the critic learns a baseline
  without perturbing the actor's representation -- the only thing that changes
  versus REINFORCE is where the baseline comes from.
* V(s) is computed once per parent (per group) from the prompt and shared across
  the group's rollouts, since the state is the parent context.
* The value head + its own AdamW are lazily created and stashed on `model`, so
  the train_step signature is unchanged and the head's optimizer state persists
  across steps. The `optimizer` argument (policy/LoRA params) is used as-is.

INTEGRATION
-----------
    from train_a2c import train_step      # signature unchanged

Optional cfg knobs:
    value_lr:       float = 1e-3
    value_coef:     float = 0.5     # (kept for parity; critic uses its own opt)
    reward_workers: int   = 0
"""

import time
import numpy as np


# ----------------------------------------------------------------------
# Helpers (self-contained)
# ----------------------------------------------------------------------
def compute_token_logprobs(model, prompt_ids, response_ids, with_grad: bool):
    import torch
    import torch.nn.functional as F
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        logits = model(full_ids).logits
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
            pad_token_id=pad_id, num_return_sequences=n_samples)
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


def _collect_groups(backend, model, tokenizer, sampler, step_idx, cfg,
                    exp_dir, problem, gen_pool):
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
            raw_score=parent.raw_score, construction=parent.construction)
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


# ----------------------------------------------------------------------
# Value head (critic)
# ----------------------------------------------------------------------
def _get_value_head(model, cfg):
    """Lazily build a small value head + its own AdamW, stash on the model."""
    import torch
    import torch.nn as nn
    if getattr(model, "_a2c_value_head", None) is not None:
        return model._a2c_value_head, model._a2c_value_opt

    # Resolve hidden size + dtype/device from the policy model.
    hidden = None
    for attr in ("config",):
        c = getattr(model, attr, None)
        if c is not None and getattr(c, "hidden_size", None):
            hidden = c.hidden_size
            break
    if hidden is None:  # Unsloth/PEFT wrappers expose the base config one level down
        base = getattr(model, "base_model", None)
        if base is not None and getattr(base, "config", None):
            hidden = base.config.hidden_size
    if hidden is None:
        raise RuntimeError("could not resolve hidden_size for the A2C value head")

    p = next(model.parameters())
    head = nn.Sequential(
        nn.Linear(hidden, hidden // 4),
        nn.Tanh(),
        nn.Linear(hidden // 4, 1),
    ).to(device=p.device, dtype=torch.float32)
    opt = torch.optim.AdamW(head.parameters(),
                            lr=float(getattr(cfg, "value_lr", 1e-3)))
    model._a2c_value_head = head
    model._a2c_value_opt = opt
    print(f"[init] A2C value head created (hidden={hidden}, "
          f"params={sum(x.numel() for x in head.parameters()):,})")
    return head, opt


def _state_value(model, head, prompt_ids):
    """V(s): last-layer hidden state of the prompt's last token -> scalar.
    The hidden state is DETACHED so the critic does not perturb the actor."""
    import torch
    with torch.no_grad():
        out = model(prompt_ids, output_hidden_states=True)
        h_last = out.hidden_states[-1][:, -1, :].detach().float()   # (1, hidden)
    return head(h_last).squeeze(-1).squeeze(0)                       # scalar tensor (grad in head)


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

    head, value_opt = _get_value_head(model, cfg)

    all_examples, all_children = [], []
    value_targets = []   # (prompt_ids, [rewards]) for the critic regression

    for g, grp in enumerate(groups):
        parent = grp["parent"]
        responses = grp["responses"]
        outs = grp["outs"]
        rewards_np = grp["rewards"]
        valids = [o.valid for o in outs]
        codes = [o.code or "" for o in outs]

        prompt_ids = tokenizer(grp["prompt_text"], return_tensors="pt").input_ids.to(model.device)

        # V(s) for this parent state (shared across the group).
        backend.set_training_mode()
        v_s = _state_value(model, head, prompt_ids)          # scalar tensor, grad in head
        v_scalar = float(v_s.detach().item())

        # Learned-baseline advantage: A_i = R_i - V(s).
        advantages = rewards_np - v_scalar
        value_targets.append((prompt_ids, rewards_np.copy()))

        print(f"  group {g}: rewards min={rewards_np.min():.4f} "
              f"mean={rewards_np.mean():.4f} max={rewards_np.max():.4f}  "
              f"valid={sum(valids)}/{len(valids)}  V(s)={v_scalar:.4f}  (A2C)")

        for r_idx, (text, token_ids) in enumerate(responses):
            res = outs[r_idx]
            meta = {
                "step": step_idx, "group": g, "rollout": r_idx,
                "reward": float(rewards_np[r_idx]),
                "raw_score": (float(res.raw_score) if res.raw_score is not None else None),
                "valid": bool(valids[r_idx]), "parsed": bool(res.parsed),
                "ran": bool(res.ran), "msg": res.msg,
                "advantage": float(advantages[r_idx]),
                "algo": "a2c", "value_baseline": v_scalar,
                "n_response_tokens": len(token_ids),
                "sandbox_stdout": (res.stdout or "")[:2000],
                "parent_value": float(parent.value) if parent.value is not None else None,
                "parent_is_seed": parent.id in sampler._seed_ids,
            }
            save_rollout(exp_dir, step_idx, g, r_idx, text, meta)

        for r_idx, (text, token_ids) in enumerate(responses):
            res = outs[r_idx]
            if valids[r_idx] and codes[r_idx]:
                all_children.append((State.make(
                    timestep=step_idx, value=float(rewards_np[r_idx]),
                    code=codes[r_idx], raw_score=res.raw_score,
                    construction=res.construction), parent))

        if float(rewards_np.max() - rewards_np.min()) < 1e-12:
            continue

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

    # ----- Critic update: regress V(s) -> realized rewards (MSE), own optimizer -----
    if value_targets:
        value_opt.zero_grad()
        critic_loss_total = 0.0
        for prompt_ids, rewards_np in value_targets:
            v_pred = _state_value(model, head, prompt_ids)       # scalar, grad in head
            target = torch.tensor(float(rewards_np.mean()),
                                  device=v_pred.device, dtype=v_pred.dtype)
            cl = (v_pred - target) ** 2
            (cl / len(value_targets)).backward()
            critic_loss_total += float(cl.detach().item())
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=cfg.grad_clip)
        value_opt.step()
        print(f"[step {step_idx}] critic MSE: {critic_loss_total / len(value_targets):.6f}")

    if not all_examples:
        print(f"[step {step_idx}] no policy signal (all groups had constant reward)")
        return

    # ----- ACTOR update (KL shaping identical to your TTT runner) -----
    backend.set_training_mode()
    optimizer.zero_grad()
    train_t0 = time.time()
    total_loss = total_logp_delta = 0.0
    n_examples = len(all_examples)

    for ex in all_examples:
        pid, rid, adv = ex["prompt_ids"], ex["response_ids"], ex["advantage"]
        cur_lp = compute_token_logprobs(model, pid, rid, with_grad=True)
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
