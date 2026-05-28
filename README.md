# TTT-Discover (Circle Packing) — Local Implementation

A from-scratch reimplementation of *Learning to Discover at Test Time*
(Yuksekgonul et al., 2026) for the circle-packing problem from
section 4.1.3 of the paper.

## Model

**Works with any Hugging Face causal language model.** The default is
`Qwen/Qwen3-8B`; change it in `train.py` (`Config.model_name`) or pass it
on the command line:

```bash
python train.py --model-name "Qwen/Qwen3-8B"
python train.py --model-name "LiquidAI/LFM2.5-1.2B-Thinking"
python train.py --model-name "meta-llama/Llama-3.1-8B-Instruct"
```

The project was initially debugged and validated on **LiquidAI LFM2.5-350M**
(small enough for any GPU) and then scaled up to 8B models.

## Backends

Two interchangeable training/inference backends:

| Backend | How to select | Notes |
|---------|---------------|-------|
| **Unsloth** | `--backend unsloth` | Faster generation, lower VRAM. Recommended when available. |
| **HF + PEFT** | `--backend hf` | Plain `transformers` + `peft`. Works with every architecture. |
| **auto** (default) | `--backend auto` | Tries Unsloth first; silently falls back to HF+PEFT if Unsloth is unavailable or can't load the model. |

The auto-fallback means you never need to change code based on your environment — it just works.

See the [Unsloth install guide](https://github.com/unslothai/unsloth#-install)
for CUDA-version-specific wheels; if installation fails, the HF backend takes
over automatically.

For fine-tuning ideas and best practices (especially for thinking/reasoning
models), see:
- [LiquidAI LFM2.5-1.2B-Thinking fine-tuning guide](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking#%F0%9F%94%A7-fine-tuning)
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) — HuggingFace's library for GRPO, REINFORCE, PPO, and other RL fine-tuning recipes that informed this implementation

## What is preserved from the paper

- **Entropic objective with adaptive β** — bisection over the KL budget
  γ = ln 2 per group, then leave-one-out entropic advantages (`advantage.py`).
- **PUCT-style parent sampling** with rank-based prior, max-Q (not mean),
  lineage-blocked batches, and top-K children per parent (`sampler.py`).
- **KL penalty against the base policy** via the Tang–Munos
  zero-mean baselined form, with the LoRA adapter disabled for the base
  forward pass.
- **Sandboxed code execution** in a subprocess with a hard timeout,
  process-group kill on TLE, and BLAS thread caps (`sandbox.py`).
- **LoRA** with rank 32 (matching the paper).
- **AdamW** with β = (0.9, 0.95), gradient clipping.

## What differs from the paper

- Runs entirely **locally**. No Tinker, no Ray, no cloud inference APIs.
- **Any HF model** instead of the paper's gpt-oss-120b.
- Generation currently runs on a **single GPU** (one response at a time in a
  Python loop). Parallel generation across multiple GPUs or using vLLM/batched
  decoding is planned as the next improvement.
- No async/await, no logging frameworks. Everything prints to the terminal.

## Output logging

Every single rollout is saved to disk under `runs/`. The directory name
encodes the key hyperparameters and timestamp:

```
runs/n10_target1.6294_steps50_g8x64_lr1e-05_T1.1_klc0.1_model-Qwen_Qwen3-8B_20260528-010417/
  config.json                          ← full hyperparameter dump
  step03_group02_rollout017.txt        ← raw model response (the generated Python program)
  step03_group02_rollout017.meta.json  ← reward, valid, parsed, ran, error msg, advantage, β, ...
  ...
  final.summary.json                   ← best value, step, code
  best_code.py                         ← best packing found
```

The `.meta.json` files include:
- `reward` — sum of radii (0.0 if invalid)
- `valid` — whether the packing passed geometric validation
- `parsed` / `ran` — whether code extraction and execution succeeded
- `msg` — error message if anything failed
- `sandbox_stdout` — first 2000 chars of execution output
- `advantage`, `beta` — training signal for this rollout

This lets you inspect exactly what the model generated and why it received the
reward it did, at every step.

## Installation

```bash
pip install -r requirements.txt
```

For Unsloth (optional but recommended), follow the
[official install guide](https://github.com/unslothai/unsloth#-install)
matching your CUDA version. If it fails, the script falls back to plain
transformers + PEFT automatically.

## Usage

```bash
# Paper-scale defaults (50 steps × 8 groups × 64 rollouts)
python train.py

# Change model
python train.py --model-name "LiquidAI/LFM2.5-350M"

# Conservative: fits a 16 GB GPU
python train.py --groups-per-step 2 --group-size 8

# Tiny smoke test
python train.py --num-steps 2 --groups-per-step 1 --group-size 2 \
                --max-new-tokens 256

# n=10 problem (easier, good for validating the training loop)
python train.py --num-circles 10 --target 1.6294 --temperature 1.1

# n=26 paper problem
python train.py --num-circles 26 --target 2.636

# Force plain HF backend (useful if Unsloth is broken on your system)
python train.py --backend hf

# 4-bit quantization (reduces VRAM by ~50%, slight quality loss)
python train.py --load-in-4bit

# All flags
python train.py --help
```

Pre-written run scripts:

```bash
bash run.sh    # n=10 run, 50 steps, paper-scale groups
bash run2.sh   # n=26 run, 50 steps, paper-scale groups
```

## Files

| File | Purpose |
|------|---------|
| `train.py` | Main entry point — config, CLI, training loop |
| `model_backend.py` | Unsloth ↔ HF+PEFT backends with auto-fallback |
| `advantage.py` | Entropic adaptive-β advantage via KL-budget bisection |
| `sampler.py` | PUCT archive + state struct |
| `reward.py` | Validator + reward (sum of radii) |
| `prompts.py` | Chat-format prompt builder |
| `sandbox.py` | Subprocess code execution with timeout (no Ray) |
| `experiment_io.py` | Per-run directory creation and rollout logging |

## Expectations

With a single 24 GB GPU and smaller-scale settings:
- The model should start producing valid packings within a few steps.
- Best sum of radii should climb noticeably over 50 steps.
- For n=26, the human SOTA is 2.635983; a well-tuned run might reach 2.0–2.5.

The paper-scale settings (8 groups × 64 rollouts) require substantial VRAM for
an 8B model. Reduce via `--groups-per-step` and `--group-size` on smaller
hardware, or use `--load-in-4bit`.

## License

Code in this directory is original except for the `validate_packing`
function and circle-packing problem framing, which match the paper's
`examples/circle_packing/env.py`.
