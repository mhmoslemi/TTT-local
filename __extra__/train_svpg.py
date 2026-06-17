"""
train_svpg.py  -- drop-in replacement for the TTT-Discover train_step.

THE DIVERSITY AXIS -- the only one that attacks the paper's stated mode-collapse
failure at the GRADIENT level rather than via search heuristics (entropic beta,
PUCT bonus, Elo reranker).

Stein Variational Policy Gradient (Liu et al., 2017). Maintain an ensemble of
`svpg_num_particles` policies as separate LoRA adapters over the SAME frozen
base. Each particle does its own on-policy rollouts and gets its own policy
gradient (driving it toward high reward). Then a single SVGD step couples them:

    phi(theta_i) = (1/n) * sum_j [ k(theta_j, theta_i) * grad_J(theta_j)
                                   + alpha * grad_{theta_j} k(theta_j, theta_i) ]

  - k is an RBF kernel over flattened LoRA params; bandwidth h via the median
    heuristic (h = med^2 / log n).
  - first term: kernel-smoothed reward gradient (attractive, shares signal).
  - second term: repulsion that pushes particles apart in parameter space.
    alpha = `svpg_temperature` trades exploitation (0) vs diversity (large).

We ASCEND J. The optimizer descends, so we write grad_i = -phi(theta_i) into
each particle's .grad and step. The RBF repulsion gradient is computed
analytically (no second-order autograd):  grad_{theta_j} k = k * (-2/h)(theta_j - theta_i).

COST: n x the per-step generation + scoring + backward. Default n = 3. This is
the speculative pick; budget for it. The OUTER SEARCH IS STILL SHARED: every
valid program from every particle feeds the one PUCT archive, so particles
diversify the *policies* exploring a single shared state space.

STATE MODEL (keeps the train_step signature unchanged)
------------------------------------------------------
The live `model` is a scratch compute engine. Each particle is a dict of
trainable (LoRA) param tensors held OUTSIDE the model, with its own AdamW so
Adam moments stay per-particle. Each step we copy a particle's params into the
model, generate/score/backward, read the grads back out, then run SVGD across
all particles and step their optimizers. Particle 0 is seeded from the model's
initial LoRA weights; particles 1..n-1 are small Gaussian perturbations. The
`optimizer` argument passed in is IGNORED (SVPG owns n internal optimizers); the
external code can keep constructing it harmlessly.

INTEGRATION
-----------
    from train_svpg import train_step      # signature unchanged

Optional cfg knobs:
    svpg_num_particles: int   = 3
    svpg_temperature:   float = 1.0     # alpha; repulsion strength
    svpg_init_std:      float = 0.01    # perturbation seeding particles 1..n-1
    svpg_lr:            float = 0.0     # 0 => reuse cfg.learning_rate
    reward_workers:     int   = 0
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


def _adapter_dir(exp_dir, step_idx, particle):
    from pathlib import Path
    return str(Path(exp_dir) / f"adapter_step{step_idx:03d}_p{particle}")


def _save_adapter(model, exp_dir, step_idx, particle):
    import shutil
    from pathlib import Path
    out_dir = _adapter_dir(exp_dir, step_idx, particle)
    model.save_pretrained(out_dir)
    # keep only this step's particle adapters
    keep_prefix = f"adapter_step{step_idx:03d}_"
    for old in Path(exp_dir).glob("adapter_step*"):
        if not old.name.startswith(keep_prefix):
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


# ----------------------------------------------------------------------
# Particle ensemble state (held outside the model)
# ----------------------------------------------------------------------
def _trainable_names(model):
    return [n for n, p in model.named_parameters() if p.requires_grad]


def _get_particles(model, cfg):
    """Lazily create n LoRA-param particles + n AdamW optimizers, stash on model."""
    import torch
    if getattr(model, "_svpg_particles", None) is not None:
        return (model._svpg_particles, model._svpg_opts,
                model._svpg_names, model._svpg_shapes)

    n = int(getattr(cfg, "svpg_num_particles", 3))
    init_std = float(getattr(cfg, "svpg_init_std", 0.01))
    lr = float(getattr(cfg, "svpg_lr", 0.0)) or cfg.learning_rate

    names = _trainable_names(model)
    name2param = dict(model.named_parameters())
    shapes = {nm: name2param[nm].shape for nm in names}

    particles = []   # each: dict name -> nn.Parameter (leaf, requires_grad)
    opts = []
    for i in range(n):
        pd = {}
        for nm in names:
            base = name2param[nm].detach().clone()
            if i > 0:
                base = base + init_std * torch.randn_like(base)
            param = torch.nn.Parameter(base, requires_grad=True)
            pd[nm] = param
        particles.append(pd)
        opts.append(torch.optim.AdamW(list(pd.values()), lr=lr,
                                      betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0))

    model._svpg_particles = particles
    model._svpg_opts = opts
    model._svpg_names = names
    model._svpg_shapes = shapes
    nparam = sum(int(np.prod(s)) for s in shapes.values())
    print(f"[init] SVPG ensemble: {n} particles x {nparam:,} LoRA params  "
          f"(temperature/alpha={float(getattr(cfg, 'svpg_temperature', 1.0))}, lr={lr})")
    return particles, opts, names, shapes


def _load_particle_into_model(model, particle, names):
    """Copy a particle's params into the live model (in-place, no grad tracking)."""
    import torch
    name2param = dict(model.named_parameters())
    with torch.no_grad():
        for nm in names:
            name2param[nm].data.copy_(particle[nm].data)


def _flatten(particle, names):
    import torch
    return torch.cat([particle[nm].data.reshape(-1).float() for nm in names])


def _flatten_grads(model, names):
    import torch
    name2param = dict(model.named_parameters())
    flats = []
    for nm in names:
        g = name2param[nm].grad
        flats.append((g.detach().reshape(-1).float()
                      if g is not None else torch.zeros(name2param[nm].numel(),
                                                        device=name2param[nm].device)))
    return torch.cat(flats)


def _unflatten_into_grad(particle, names, vec):
    """Write a flat vector into particle params' .grad (creating slices)."""
    import torch
    offset = 0
    for nm in names:
        p = particle[nm]
        numel = p.numel()
        chunk = vec[offset: offset + numel].reshape(p.shape).to(p.dtype)
        p.grad = chunk.clone()
        offset += numel


# ----------------------------------------------------------------------
# Per-particle rollout + scoring (single particle is loaded into `model`)
# ----------------------------------------------------------------------
def _rollout_and_score_particle(backend, model, tokenizer, parents, parent_ctxs,
                                prompts_by_group, cfg, exp_dir, step_idx, problem,
                                gen_pool, particle_idx):
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from gen_workers import make_progress_bar

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

    try:
        if gen_pool is not None:
            adapter_path = _save_adapter(model, exp_dir, step_idx, particle_idx)
            for gidx, job in gen_pool.iter_group_jobs(
                    prompts_by_group=prompts_by_group, group_size=cfg.group_size,
                    adapter_path=adapter_path, max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature, top_p=cfg.top_p):
                for (text, ids) in job:
                    _submit(gidx, text, ids)
        else:
            backend.set_inference_mode()
            bar = make_progress_bar(total_rollouts, desc=f"p{particle_idx} rollouts")
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
        for _ in as_completed(all_futs):
            pass
    finally:
        reward_pool.shutdown(wait=True)

    groups = []
    for g, parent in enumerate(parents):
        responses = group_responses[g]
        outs = [reward_futures[g][i].result() for i in range(len(responses))]
        rewards = np.array([o.reward for o in outs], dtype=np.float64)
        groups.append(dict(parent=parent, prompt_text=prompts_by_group[g],
                           responses=responses, outs=outs, rewards=rewards))
    return groups


def _linear_group_advantages(rewards_np):
    return (rewards_np - rewards_np.mean()) / (rewards_np.std() + 1e-8)


def _particle_policy_grad(backend, model, tokenizer, groups, cfg):
    """Accumulate this particle's policy-gradient (loss grad) into model.grad.
    Returns (n_examples, avg_loss). KL shaping identical to the TTT runner."""
    import torch
    model.zero_grad(set_to_none=True)
    backend.set_training_mode()

    examples = []
    for grp in groups:
        rewards_np = grp["rewards"]
        if float(rewards_np.max() - rewards_np.min()) < 1e-12:
            continue
        advantages = _linear_group_advantages(rewards_np)
        prompt_ids = tokenizer(grp["prompt_text"], return_tensors="pt").input_ids.to(model.device)
        for (text, token_ids), adv in zip(grp["responses"], advantages):
            if len(token_ids) == 0:
                continue
            examples.append((prompt_ids,
                             torch.tensor([token_ids], device=model.device),
                             float(adv)))

    if not examples:
        return 0, 0.0

    n = len(examples)
    total_loss = 0.0
    for pid, rid, adv in examples:
        cur_lp = compute_token_logprobs(model, pid, rid, with_grad=True)
        try:
            with backend.disable_adapter(), torch.no_grad():
                base_lp = compute_token_logprobs(model, pid, rid, with_grad=False)
        except Exception:
            base_lp = cur_lp.detach()
        logp_diff = (cur_lp - base_lp).detach()
        kl_adv = cfg.kl_penalty_coef * (logp_diff.mean() - (cur_lp - base_lp))
        eff_adv = adv + kl_adv
        loss = -(eff_adv.detach() * cur_lp).mean()
        (loss / n).backward()
        total_loss += float(loss.detach().item())
    return n, total_loss / n


# ----------------------------------------------------------------------
# SVGD combination
# ----------------------------------------------------------------------
def _svgd_descent_grads(thetas, grads, alpha):
    """thetas, grads: lists of flat tensors (one per particle).
    grads are LOSS gradients (descent). Returns the SVGD descent grad per
    particle, i.e. -phi where phi is the ascent direction on J."""
    import torch
    n = len(thetas)
    Theta = torch.stack(thetas)                      # (n, D)
    G = torch.stack(grads)                           # (n, D)  (= grad of loss = -grad J)

    # Pairwise squared distances + median-heuristic bandwidth.
    sq = torch.cdist(Theta, Theta) ** 2              # (n, n)
    med = torch.median(sq[sq > 0]) if (sq > 0).any() else torch.tensor(1.0, device=Theta.device)
    h = med / max(np.log(n), 1.0)
    h = torch.clamp(h, min=1e-8)
    K = torch.exp(-sq / h)                           # (n, n), K[j, i]

    out = []
    for i in range(n):
        # attractive: (1/n) sum_j K[j,i] * grad_loss_j  (= -(1/n) sum_j K[j,i] grad_J_j)
        kcol = K[:, i].unsqueeze(1)                  # (n, 1)
        attractive = (kcol * G).sum(dim=0) / n       # (D,)
        # repulsion ascent term: (1/n) sum_j grad_{theta_j} K[j,i]
        #   grad_{theta_j} K[j,i] = K[j,i] * (-2/h) * (theta_j - theta_i)
        diff = Theta - Theta[i].unsqueeze(0)         # (n, D)  theta_j - theta_i
        rep_ascent = (kcol * (-2.0 / h) * diff).sum(dim=0) / n
        # descent grad = -phi = attractive(loss) - alpha * rep_ascent
        out.append(attractive - alpha * rep_ascent)
    return out, float(K.mean().item()), float(h.item())


# ----------------------------------------------------------------------
# Build prompts once (shared across particles)
# ----------------------------------------------------------------------
def _build_prompts(sampler, tokenizer, cfg, problem, step_idx):
    from problems.base import ParentContext
    sampler.set_current_step(step_idx)
    parents = sampler.sample_states(cfg.groups_per_step)
    print(f"\n[step {step_idx}] parents picked: {len(parents)}")
    for i, info in enumerate(sampler.last_picks_info):
        tag = "seed" if info["is_seed"] else "expanded"
        print(f"  parent {i} [{tag}]  value={info['value']:.4f}  n={info['n']}  "
              f"Q={info['Q']:.4f}  P={info['P']:.4f}  bonus={info['bonus']:.4f}  "
              f"score={info['score']:.4f}")
    prompts_by_group, parent_ctxs = [], []
    for parent in parents:
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
    return parents, parent_ctxs, prompts_by_group


# ----------------------------------------------------------------------
# train_step
# ----------------------------------------------------------------------
def train_step(backend, model, tokenizer, sampler, optimizer, step_idx: int,
               cfg, exp_dir, problem, gen_pool=None):
    import torch
    from sampler import State
    from experiment_io import save_rollout

    alpha = float(getattr(cfg, "svpg_temperature", 1.0))
    step_t0 = time.time()

    particles, opts, names, shapes = _get_particles(model, cfg)
    n = len(particles)

    # Build the (shared) parents + prompts ONCE; record_expansion happens here.
    parents, parent_ctxs, prompts_by_group = _build_prompts(
        sampler, tokenizer, cfg, problem, step_idx)

    all_children = []
    thetas, grads = [], []
    particle_meanR, particle_nex, particle_loss = [], [], []

    rollout_t0 = time.time()
    for i in range(n):
        _load_particle_into_model(model, particles[i], names)

        groups = _rollout_and_score_particle(
            backend, model, tokenizer, parents, parent_ctxs, prompts_by_group,
            cfg, exp_dir, step_idx, problem, gen_pool, particle_idx=i)

        # Per-particle reward summary + SHARED-archive children + rollout saves.
        meanR = []
        for g, grp in enumerate(groups):
            rewards_np = grp["rewards"]
            outs = grp["outs"]
            valids = [o.valid for o in outs]
            codes = [o.code or "" for o in outs]
            advantages = _linear_group_advantages(rewards_np)
            meanR.append(float(rewards_np.mean()))

            for r_idx, (text, token_ids) in enumerate(grp["responses"]):
                res = outs[r_idx]
                meta = {
                    "step": step_idx, "group": g, "rollout": r_idx,
                    "reward": float(rewards_np[r_idx]),
                    "raw_score": (float(res.raw_score) if res.raw_score is not None else None),
                    "valid": bool(valids[r_idx]), "parsed": bool(res.parsed),
                    "ran": bool(res.ran), "msg": res.msg,
                    "advantage": float(advantages[r_idx]),
                    "algo": "svpg", "particle": i,
                    "n_response_tokens": len(token_ids),
                    "sandbox_stdout": (res.stdout or "")[:2000],
                    "parent_value": (float(grp["parent"].value)
                                     if grp["parent"].value is not None else None),
                    "parent_is_seed": grp["parent"].id in sampler._seed_ids,
                }
                save_rollout(exp_dir, step_idx, g * 100 + i, r_idx, text, meta)

            for r_idx in range(len(grp["responses"])):
                res = outs[r_idx]
                if valids[r_idx] and codes[r_idx]:
                    all_children.append((State.make(
                        timestep=step_idx, value=float(rewards_np[r_idx]),
                        code=codes[r_idx], raw_score=res.raw_score,
                        construction=res.construction), grp["parent"]))

        # This particle's policy gradient -> model.grad, then read it out.
        nex, avg_loss = _particle_policy_grad(backend, model, tokenizer, groups, cfg)
        thetas.append(_flatten(particles[i], names))
        grads.append(_flatten_grads(model, names))
        particle_meanR.append(float(np.mean(meanR)) if meanR else float("nan"))
        particle_nex.append(nex)
        particle_loss.append(avg_loss)
        print(f"  [particle {i}] meanR={particle_meanR[-1]:.4f}  "
              f"examples={nex}  policy_loss={avg_loss:.4f}")

    rollout_time = time.time() - rollout_t0
    print(f"[step {step_idx}] rollout+eval time (all particles): {rollout_time:.1f}s  "
          f"new children: {len(all_children)}")
    sampler.update(all_children)

    if all(nx == 0 for nx in particle_nex):
        print(f"[step {step_idx}] no training signal on any particle")
        return

    # ----- SVGD combination + per-particle optimizer step -----
    train_t0 = time.time()
    descent_grads, kmean, h = _svgd_descent_grads(thetas, grads, alpha)
    for i in range(n):
        _unflatten_into_grad(particles[i], names, descent_grads[i])
        torch.nn.utils.clip_grad_norm_(list(particles[i].values()), max_norm=cfg.grad_clip)
        opts[i].step()
        opts[i].zero_grad(set_to_none=True)

    # Load the best particle (highest meanR this step) back into the live model,
    # so the gen pool / best_state pipeline see a coherent default policy.
    best_p = int(np.nanargmax(particle_meanR)) if not all(
        np.isnan(particle_meanR)) else 0
    _load_particle_into_model(model, particles[best_p], names)

    print(f"[step {step_idx}] SVGD step: kernel_mean={kmean:.4f}  bandwidth_h={h:.4g}  "
          f"alpha={alpha}  best_particle={best_p}  (train {time.time() - train_t0:.1f}s)")

    best = sampler.best_state()
    if best is not None:
        raw = f" raw={best.raw_score:.6f}" if best.raw_score is not None else ""
        print(f"[step {step_idx}] best so far: value={best.value:.6f}{raw}  "
              f"(step total {time.time() - step_t0:.1f}s, archive={sampler.archive_size()})")
