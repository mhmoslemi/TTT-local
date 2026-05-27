"""
Entropic objective with adaptive β.

Paper section 3.2: instead of optimizing the expected reward, we optimize

    J_β(θ; s) = log E_{a~π_θ}[exp(β · R(s,a))]

The gradient gives a reweighted policy gradient with weights

    w_β(a) = exp(β·R(a)) / E[exp(β·R)]
    A(a)   = w_β(a) - 1                # baselined advantage

As β → ∞, this picks out the single best action. The trick is choosing β
adaptively for each parent: we enforce that the tilted distribution q_β
stays within KL budget γ = ln 2 of the uniform-over-batch distribution.

Implementation matches discover/ttt_discover/rl/train.py::compute_advantages
under the `entropic_adaptive_beta` branch.
"""

import math
import numpy as np


def _kl_to_uniform(beta: float, rewards: np.ndarray) -> float:
    """
    KL(q_β || uniform) where q_β(i) ∝ exp(β * r_i).
    Numerically stabilized by subtracting r.max().
    """
    K = len(rewards)
    if K <= 1:
        return 0.0
    logK = math.log(K)
    logits = beta * (rewards - rewards.max())
    # log q_β
    log_Z = np.log(np.exp(logits).sum())
    log_q = logits - log_Z
    q = np.exp(log_q)
    # KL(q || uniform) = sum q * (log q + log K)
    return float((q * (log_q + logK)).sum())


def entropic_adaptive_advantages(
    rewards: np.ndarray,
    gamma: float = math.log(2),
    beta_max: float = 1e6,
    n_bisect: int = 60,
    eps: float = 1e-12,
):
    """
    Compute leave-one-out entropic advantages for one group.

    rewards: shape (K,) — rewards of the K rollouts from the same parent.

    Returns:
      advantages: shape (K,) — what the policy gradient gets weighted by
      beta:       the temperature found by bisection

    All-equal-reward groups get a zero advantage vector (no gradient).
    """
    r = np.asarray(rewards, dtype=np.float64)
    K = r.shape[0]

    if K < 2 or float(r.max() - r.min()) < eps:
        return np.zeros_like(r), 0.0

    # Step 1: find β with KL(q_β || uniform) = gamma via bisection
    lo, hi = 0.0, 1.0

    # If even hi=1 has KL > gamma, then β is in (0, 1)
    if _kl_to_uniform(hi, r) < gamma:
        # Need to grow hi until KL exceeds gamma
        while hi < beta_max and _kl_to_uniform(hi, r) < gamma:
            hi *= 2.0
        if _kl_to_uniform(hi, r) < gamma:
            # Saturated; β = beta_max (effectively argmax)
            beta = hi
        else:
            beta = None
    else:
        beta = None

    if beta is None:
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            if _kl_to_uniform(mid, r) < gamma:
                lo = mid
            else:
                hi = mid
        beta = hi

    # Step 2: LOO entropic weights
    # w_n = e^{β(r_n - r_max)} / Z_{-n}
    # where Z_{-n} = (sum_m e^{β(r_m - r_max)} - e^{β(r_n - r_max)}) / (K-1)
    shift = r - r.max()
    e = np.exp(beta * shift)
    total = e.sum()
    Z_loo = (total - e) / (K - 1)
    w = e / (Z_loo + eps)
    advantages = w - 1.0

    return advantages, beta


if __name__ == "__main__":
    # Case 1: all rewards equal -> zero advantage
    r = np.array([0.5, 0.5, 0.5, 0.5])
    a, b = entropic_adaptive_advantages(r)
    print(f"all equal -> adv={a}, beta={b}")

    # Case 2: one big outlier -> outlier gets the positive signal
    r = np.array([0.1, 0.1, 0.1, 0.1, 2.5])
    a, b = entropic_adaptive_advantages(r)
    print(f"one outlier -> adv={a}, beta={b:.4f}")
    print(f"  sum adv (should ~= 0 in expectation but is LOO so not exactly): {a.sum():.4f}")

    # Case 3: small differences -> β grows large, still concentrates on best
    r = np.array([1.0, 1.001, 1.002, 1.003])
    a, b = entropic_adaptive_advantages(r)
    print(f"small diffs -> adv={a}, beta={b:.4f}")

    # Case 4: large spread -> β is small (KL budget hits fast)
    r = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    a, b = entropic_adaptive_advantages(r)
    print(f"big spread -> adv={a}, beta={b:.4f}")
