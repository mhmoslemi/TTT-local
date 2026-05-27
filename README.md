# TTT-Discover (Circle Packing) — Local Implementation

A from-scratch reimplementation of *Learning to Discover at Test Time*
(Yuksekgonul et al., 2026) for the circle-packing problem from
section 4.1.3 of the paper.

Differences from the paper's release:

- Runs entirely **locally**. No Tinker, no Ray, no Anthropic/HF Inference API.
- Uses **LFM2.5-350M** (Liquid AI) as the base model — small enough to fit on
  a single consumer GPU. The paper uses gpt-oss-120b.
- Two interchangeable backends:
  - **Unsloth** (`--backend unsloth`): faster, lower memory. Tries this first by default.
  - **HF + PEFT** (`--backend hf`): plain transformers fallback if Unsloth
    can't load the LFM2 architecture.
- No async / await, no logging frameworks, no decorators. Everything prints
  to the terminal.

What's preserved from the paper:

- **Entropic objective with adaptive β** — bisection over the KL budget
  γ = ln 2 per group, then leave-one-out entropic advantages
  (`advantage.py`).
- **PUCT-style parent sampling** with rank-based prior, max-Q (not mean),
  lineage-blocked batches, and top-K children per parent (`sampler.py`).
- **KL penalty against the base policy** via the Tang–Munos
  zero-mean baselined form, with the LoRA adapter disabled for the base
  forward pass.
- **Sandboxed code execution** in a subprocess with a hard timeout,
  process-group kill on TLE, and BLAS thread caps (`sandbox.py`).
- **LoRA** with rank 32 (matching the paper).

## Installation

```bash
pip install -r requirements.txt
```

If `unsloth` fails to install (it's picky about CUDA versions), don't
worry — the script will fall back to plain transformers + PEFT.
See https://github.com/unslothai/unsloth for Unsloth-specific install help.

## Usage

```bash
# Paper-scale defaults (50 steps × 8 groups × 64 rollouts = 25,600 rollouts)
python train.py

# Conservative: fits a 16 GB GPU
python train.py --groups-per-step 2 --group-size 8

# Tiny smoke test
python train.py --num-steps 2 --groups-per-step 1 --group-size 2 \
                --max-new-tokens 256

# Force plain HF backend
python train.py --backend hf

# n=32 problem
python train.py --num-circles 32 --target 2.940

# All flags
python train.py --help
```

## Files

| File | Purpose |
|------|---------|
| `sandbox.py` | Subprocess code execution with timeout (no Ray) |
| `reward.py` | Validator + reward (sum of radii) |
| `sampler.py` | PUCT archive + state struct |
| `advantage.py` | Entropic adaptive-β advantage via KL-budget bisection |
| `prompts.py` | Chat-format prompt builder |
| `model_backend.py` | Unsloth ↔ HF+PEFT backends with auto-fallback |
| `train.py` | Main entry point — synchronous training loop |

## How it differs algorithmically from the paper

We faithfully implement Algorithm 1 (TTT-Discover) but at much smaller
scale and on a much smaller model. With LFM2.5-350M and modest compute,
don't expect to beat the human SOTA of 2.635983 on n=26. The point is to
verify the algorithm works and learns on the test problem.

Reasonable expectations for a single 24 GB GPU run with smaller scale:
- The model should start producing valid packings within a few steps
- Best sum of radii should climb noticeably over 50 steps
- Final value should land somewhere in the 2.0–2.5 range with good luck

The paper-scale settings (8 × 64) will OOM on most consumer GPUs. The
defaults are set to paper-scale so you can read the code knowing what
the authors intended; you'll want to scale down via the CLI flags on
your hardware.

## License

Code in this directory is original except for the `validate_packing`
function and circle-packing problem framing, which match the paper's
`examples/circle_packing/env.py`.
