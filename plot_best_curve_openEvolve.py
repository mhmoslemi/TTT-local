#!/usr/bin/env python3
"""
Compare OpenEvolve vs TTT vs TTT+Elo on circle_packing.

  - OpenEvolve : dense best-so-far curve parsed from the log, sampled every --every iterations.
  - TTT        : one point per step, placed at iteration (step+1)*STEP_ITERS.
  - TTT + Elo  : same.

Usage:
    python plot_best_curve.py                       # newest openevolve log + the TTT lists below
    python plot_best_curve.py path/to.log --every 32 --ylim 2.4 2.65 --target 2.635
    python plot_best_curve.py path/to.log --no-openevolve   # only the TTT series
    python plot_best_curve_openEvolve.py /work/mohammad/TTT-local/openevolve/examples/circle_packing/openevolve_output/logs/openevolve_20260611_184912.log --every 64 --target 2.635
"""

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")  # headless / SSH safe
import matplotlib.pyplot as plt
import numpy as np

# ============================ hyperparameters ============================
SAMPLE_EVERY = 32      # openevolve: plot one point every N iterations
STEP_ITERS   = 512     # iterations per TTT step (also the vertical-line spacing)
TTT_CUMMAX   = False   # True => plot cumulative best-so-far for the TTT series too
YLIM = None            # e.g. (2.4, 2.65); None = autoscale
XLIM = None            # e.g. (0, 7680); None = autoscale
TARGET = None          # horizontal reference line; ~2.635 for n=26 unit-square packing

# best sum_radii per step (step 0..14). Paste new runs here.
TTT = [
    2.5261384438, 2.6035808140, 2.6083336157, 2.6141354338, 2.6235348955,
    2.6282124182, 2.6282124182, 2.6302148014, 2.6310935891, 2.6310935891,
    2.6310935891, 2.6310935891, 2.6358957386, 2.6342924021, 2.6342924021,
]
TTT_ELO = [
    2.5518054052, 2.6221449045, 2.6221772829, 2.6270835278, 2.6297131377,
    2.6313341376, 2.6330195957, 2.6313343652, 2.6313496673, 2.6340630826,
    2.6323260073, 2.6323260265, 2.6353127766, 2.6359674489, 2.6353127769,
]
# =========================================================================

DEFAULT_LOG_DIR = "/work/mohammad/TTT-local/openevolve/examples/circle_packing/openevolve_output/logs"

_ITER_RE = re.compile(r"- Iteration (\d+):.*completed")
_SR_RE = re.compile(r"- Metrics:.*sum_radii=([0-9.]+)")


def parse_log(path):
    iters, srs, cur = [], [], None
    with open(path) as f:
        for line in f:
            m = _ITER_RE.search(line)
            if m:
                cur = int(m.group(1))
                continue
            if cur is not None:
                m = _SR_RE.search(line)
                if m:
                    iters.append(cur)
                    srs.append(float(m.group(1)))
                    cur = None
    return np.array(iters), np.array(srs)


def parse_json(path):
    with open(path) as f:
        data = json.load(f)
    return (np.array([d["iteration"] for d in data]),
            np.array([d["sum_radii"] for d in data]))


def best_so_far_curve(iters, srs, every):
    order = np.argsort(iters)
    iters, srs = iters[order], srs[order]
    cummax = np.maximum.accumulate(srs)
    grid = np.arange(0, int(iters.max()) + 1, every)
    idx = np.searchsorted(iters, grid, side="right") - 1
    mask = idx >= 0
    return grid[mask], cummax[idx[mask]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", nargs="?", help="openevolve .log or *_best_curve.json")
    p.add_argument("--every", type=int, default=SAMPLE_EVERY)
    p.add_argument("--step-iters", type=int, default=STEP_ITERS, help="iterations per TTT step")
    p.add_argument("--ttt-cummax", action="store_true", default=TTT_CUMMAX,
                   help="plot cumulative best-so-far for the TTT series")
    p.add_argument("--no-openevolve", action="store_true", help="skip the openevolve curve")
    p.add_argument("--ylim", type=float, nargs=2, default=YLIM, metavar=("LO", "HI"))
    p.add_argument("--xlim", type=float, nargs=2, default=XLIM, metavar=("LO", "HI"))
    p.add_argument("--target", type=float, default=TARGET)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    si = args.step_iters

    # resolve openevolve log
    log = args.log
    if log is None and not args.no_openevolve:
        cands = sorted(glob.glob(os.path.join(DEFAULT_LOG_DIR, "openevolve_*.log")),
                       key=os.path.getmtime)
        log = cands[-1] if cands else None
        if log is None:
            args.no_openevolve = True

    fig, ax = plt.subplots(figsize=(9, 5))

    # vertical step-boundary markers
    n_steps = max(len(TTT), len(TTT_ELO))
    xmax = args.xlim[1] if args.xlim else (n_steps + 1) * si
    for xb in range(si, int(xmax) + 1, si):
        ax.axvline(xb, color="0.8", ls="--", lw=0.8, zorder=0)

    # openevolve dense curve
    if not args.no_openevolve and log is not None:
        iters, srs = parse_json(log) if log.endswith(".json") else parse_log(log)
        if iters.size:
            gx, gy = best_so_far_curve(iters, srs, args.every)
            ax.plot(gx, gy, lw=1, color="#1f4e79", label="OpenEvolve", zorder=2)

    # TTT series: one point per step at x = (step+1)*step_iters
    def add_series(values, color, label):
        y = np.array(values, dtype=float)
        if args.ttt_cummax:
            y = np.maximum.accumulate(y)
        x = (np.arange(len(y)) + 1) * si
        ax.plot(x, y, marker="o", ms=5, lw=1.5, color=color, label=label, zorder=3)

    add_series(TTT, "#e07b39", "TTT")
    add_series(TTT_ELO, "#2a9d8f", "TTT + Elo")

    if args.target is not None:
        ax.axhline(args.target, ls="--", lw=1, color="#aa3333", zorder=1,
                   label=f"target = {args.target:g}")

    ax.set_xlabel("iteration")
    ax.set_ylabel("best seen sum_radii")
    ax.set_title("circle_packing: OpenEvolve vs TTT vs TTT+Elo")
    ax.grid(True, alpha=0.25)
    ax.legend()
    if args.ylim:
        ax.set_ylim(*args.ylim)
    if args.xlim:
        ax.set_xlim(*args.xlim)

    out = args.out or (os.path.join(os.path.dirname(log) if log else ".",
                                    "compare_best.png"))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"steps: TTT={len(TTT)}, TTT+Elo={len(TTT_ELO)}, step_iters={si}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()