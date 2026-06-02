"""
Visualize a circle packing produced by a run_packing() function.

Usage:
    1. Paste your run_packing() below where indicated
    2. python plot_packing.py
       (or python plot_packing.py output.png   to save instead of display)

Validates the packing the same way reward.py does and prints any issues.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import minimize
import numpy.random as npr


# ====================================================================
# >>> PASTE YOUR run_packing() HERE <<<
# ====================================================================


import numpy as np
from scipy.optimize import minimize
import math

def run_packing():
    num_circles = 26

    # Generate a hexagonal close packing (HCP) layout within the unit square
    # Calculate the number of rows and columns needed
    rows = int(math.ceil(math.sqrt(num_circles)))
    cols = int(math.ceil(num_circles / rows))
    
    # Generate initial centers in a hexagonal close packing arrangement
    centers = np.zeros((num_circles, 2))
    for i in range(num_circles):
        row = i // cols
        col = i % cols
        x = col * (1.0 / (cols - 1)) * 0.9  # Add padding
        y = row * (1.0 / (rows - 1)) * 0.9  # Add padding
        # Add offset for staggered rows (hexagonal packing)
        if row % 2 == 1:
            x += 0.5 * (1.0 / (cols - 1)) * 0.9
        centers[i] = [x, y]
    
    # Trim to exactly 26 circles
    centers = centers[:26]
    radii = np.full(num_circles, 0.01)

    # Optimization step: maximize the sum of radii with constraints
    def objective(params):
        # params is a flat array: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
        params = params.reshape((num_circles, 3))
        x = params[:, 0]
        y = params[:, 1]
        r = params[:, 2]
        return -np.sum(r)  # Negative because we're minimizing

    def constraint_overlap(i, j, x, y, r):
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx*dx + dy*dy)
        return dist - r[i] - r[j]

    # Define boundary constraints
    def constraint_boundary(i, x, y, r):
        return x[i] - r[i]  # Ensure left boundary
    def constraint_boundary_right(i, x, y, r):
        return 1.0 - x[i] - r[i]  # Ensure right boundary
    def constraint_boundary_top(i, x, y, r):
        return 1.0 - y[i] - r[i]  # Ensure top boundary
    def constraint_boundary_bottom(i, x, y, r):
        return y[i] - r[i]  # Ensure bottom boundary

    # Prepare initial parameters: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
    initial_params = np.zeros(num_circles * 3)
    initial_params[::3] = centers[:, 0]  # x coordinates
    initial_params[1::3] = centers[:, 1]  # y coordinates
    initial_params[2::3] = radii  # radii

    # Define bounds for optimization: x, y ∈ [0, 1], r ≥ 0
    bounds = []
    for i in range(num_circles):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])  # x, y, r for each circle

    # Define constraints
    constraints = []
    for i in range(num_circles):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: constraint_boundary(i, p[::3], p[1::3], p[2::3])})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: constraint_boundary_right(i, p[::3], p[1::3], p[2::3])})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: constraint_boundary_top(i, p[::3], p[1::3], p[2::3])})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: constraint_boundary_bottom(i, p[::3], p[1::3], p[2::3])})

    for i in range(num_circles):
        for j in range(i + 1, num_circles):
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i, j=j: constraint_overlap(i, j, p[::3], p[1::3], p[2::3])})

    # Optimization
    result = minimize(objective, initial_params, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-10)
    optimized_params = result.x
    optimized_params = optimized_params.reshape((num_circles, 3))
    centers = optimized_params[:, :2]
    radii = optimized_params[:, 2]

    # Final validation
    valid, message = validate_packing(centers, radii)
    if not valid:
        print(f"Validation error: {message}")
        return np.zeros((num_circles, 2)), np.zeros(num_circles), 0.0

    # Return result
    return centers, radii, np.sum(radii)


    
# ====================================================================
# Validation (same as reward.py)
# ====================================================================
def validate_packing(centers, radii):
    """
    Paper-compatible signature: returns (bool, str).
    Matches reward.py so model code calling validate_packing(...) works.
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False, "NaN values present"
    for i in range(n):
        if radii[i] < 0:
            return False, f"Circle {i} has negative radius {radii[i]}"
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if (x - r < -1e-12 or x + r > 1 + 1e-12
                or y - r < -1e-12 or y + r > 1 + 1e-12):
            return False, f"Circle {i} at ({x},{y}) r={r} outside unit square"
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.sqrt(np.sum((centers[i] - centers[j]) ** 2)))
            if dist < radii[i] + radii[j] - 1e-12:
                return False, f"Circles {i} and {j} overlap"
    return True, "ok"


def collect_issues(centers, radii, tol=1e-9):
    """Plotter-only helper. Returns a list of strings, one per problem found."""
    issues = []
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        issues.append("NaN values present")
    for i in range(n):
        if radii[i] < 0:
            issues.append(f"Circle {i} has negative radius {radii[i]:.6f}")
        x, y = centers[i]
        r = radii[i]
        if (x - r < -tol or x + r > 1 + tol
                or y - r < -tol or y + r > 1 + tol):
            issues.append(f"Circle {i} at ({x:.4f},{y:.4f}) r={r:.4f} outside unit square")
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.sqrt(np.sum((centers[i] - centers[j]) ** 2)))
            gap = dist - (radii[i] + radii[j])
            if gap < -tol:
                issues.append(f"Circles {i},{j} overlap (gap={gap:.6f})")
    return issues


# ====================================================================
# Plotting
# ====================================================================
def plot_packing(centers, radii, sum_radii, save_to=None):
    n = len(radii)
    fig, ax = plt.subplots(figsize=(8, 8))

    # Unit square
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, linewidth=1.5, edgecolor="black"))

    # Color by radius so it's easy to see who's big and who's tiny
    cmap = plt.get_cmap("viridis")
    rmax = max(radii.max(), 1e-9)
    issues = collect_issues(centers, radii)
    invalid_ids = set()
    for msg in issues:
        # Best-effort: pull circle indices out of error messages so we can flag them
        for tok in msg.replace(",", " ").split():
            if tok.isdigit():
                invalid_ids.add(int(tok))

    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        color = cmap(r / rmax)
        edge = "red" if i in invalid_ids else "black"
        lw = 1.5 if i in invalid_ids else 0.7
        ax.add_patch(patches.Circle((x, y), r,
                                     facecolor=color, edgecolor=edge,
                                     linewidth=lw, alpha=0.65))
        ax.text(x, y, str(i), ha="center", va="center",
                fontsize=8, color="white", weight="bold")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(True, alpha=0.3)

    title = f"n={n}  sum of radii = {sum_radii:.6f}"
    if issues:
        title += f"  [INVALID: {len(issues)} issue(s)]"
    ax.set_title(title, fontsize=12)

    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_to}")
    else:
        plt.show()


def main():
    centers, radii, sum_radii = run_packing()
    centers = np.asarray(centers)
    radii = np.asarray(radii).ravel()

    print(f"n = {centers.shape[0]}")
    print(f"sum of radii (returned)  = {sum_radii:.6f}")
    print(f"sum of radii (recomputed)= {radii.sum():.6f}")
    print(f"radii: min={radii.min():.4f} max={radii.max():.4f} mean={radii.mean():.4f}")

    issues = collect_issues(centers, radii)
    if issues:
        print(f"\nVALIDATION FAILED: {len(issues)} issue(s)")
        for msg in issues[:10]:
            print(f"  - {msg}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("\nValidation OK.")

    save_to = sys.argv[1] if len(sys.argv) > 1 else None
    plot_packing(centers, radii, sum_radii, save_to='out.png')


if __name__ == "__main__":
    main()