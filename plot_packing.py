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
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Initial points based on known optimal configurations and a hexagonal grid
    # with boundary-aware shifts
    grid_points = []
    # Create a hexagonal grid with spacing that adjusts for boundary proximity
    for i in range(5):
        for j in range(5):
            # Shift even rows to allow better circle placement near boundaries
            x = i / 4.0 + (i % 2) * 0.025
            y = j / 4.0 + (i % 2) * 0.025
            grid_points.append((x, y))
    
    # Add points near boundaries and corners
    boundary_points = [
        (0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9),
        (0.2, 0.2), (0.2, 0.8), (0.8, 0.2), (0.8, 0.8),
        (0.15, 0.15), (0.15, 0.85), (0.85, 0.15), (0.85, 0.85),
        (0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75),
        (0.05, 0.5), (0.95, 0.5), (0.5, 0.05), (0.5, 0.95),
        (0.1, 0.5), (0.5, 0.1), (0.9, 0.5), (0.5, 0.9),
        (0.3, 0.3), (0.3, 0.7), (0.7, 0.3), (0.7, 0.7),
        (0.2, 0.5), (0.8, 0.5), (0.5, 0.2), (0.5, 0.8),
        (0.25, 0.5), (0.75, 0.5), (0.5, 0.25), (0.5, 0.75),
        (0.225, 0.5), (0.775, 0.5), (0.5, 0.225), (0.5, 0.775),
        (0.23, 0.5), (0.77, 0.5), (0.5, 0.23), (0.5, 0.77),
    ]
    grid_points += boundary_points

    # Limit the number of centers to 26
    adjusted_centers = []
    for x, y in grid_points:
        x = max(0.05, min(0.95, x))
        y = max(0.05, min(0.95, y))
        adjusted_centers.append((x, y))
    
    # Limit to 26 unique points
    adjusted_centers = adjusted_centers[:n]
    # Remove duplicates
    adjusted_centers = [tuple(p) for p in np.unique(adjusted_centers, axis=0)]
    adjusted_centers = adjusted_centers[:n]
    
    # Initialize centers
    centers = np.array(adjusted_centers)
    
    # We will use multiple starting layouts to enhance optimization
    start_centers = []
    # Generate multiple layout variations
    for variant in range(3):
        # Adjust the grid points slightly to allow for different layouts
        variant_centers = []
        for i in range(n):
            x, y = centers[i]
            if variant == 1:
                # Shift right and up
                x += 0.025
                y += 0.025
            elif variant == 2:
                # Shift left and down
                x -= 0.025
                y -= 0.025
            # Ensure within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            variant_centers.append((x, y))
        start_centers.append(np.array(variant_centers))
    
    # Optimization: maximize sum of radii with constraints
    def objective(vars):
        # Split the variables into centers and radii
        center_vars = vars[:2 * n].reshape(n, 2)
        radius_vars = vars[2 * n:]
        
        # Compute the sum of radii
        return -np.sum(radius_vars)  # minimize negative sum to maximize sum
    
    def constraints(vars):
        # Split the variables into centers and radii
        center_vars = vars[:2 * n].reshape(n, 2)
        radius_vars = vars[2 * n:]
        
        # Non-overlapping constraints: dist(c_i, c_j) >= r_i + r_j
        non_overlap_constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.hypot(center_vars[i][0] - center_vars[j][0],
                                center_vars[i][1] - center_vars[j][1])
                constraint = dist - radius_vars[i] - radius_vars[j]
                non_overlap_constraints.append(constraint)
        
        # Boundary constraints: x - r >= 0, x + r <= 1, same for y
        boundary_constraints = []
        for i in range(n):
            x, y = center_vars[i]
            r = radius_vars[i]
            boundary_constraints.append(x - r)
            boundary_constraints.append(1 - x - r)
            boundary_constraints.append(y - r)
            boundary_constraints.append(1 - y - r)
        
        return np.concatenate([non_overlap_constraints, boundary_constraints])

    best_sum_radii = 0.0
    best_centers = np.zeros((n, 2))
    best_radii = np.zeros(n)

    for start_idx in range(len(start_centers)):
        initial_guess = np.concatenate([
            start_centers[start_idx].reshape(n * 2),
            np.full(n, 0.15)  # Start with a moderate radius
        ])

        # Optimization bounds: center coordinates in [0,1], radii >= 0
        bounds = []
        for i in range(n):
            # Center coordinates
            bounds.append((0, 1))
            bounds.append((0, 1))
            # Radius
            bounds.append((0, 1))

        # Use SLSQP with tighter tolerances for better convergence
        result = opt.minimize(
            fun=objective,
            x0=initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraints},
            tol=1e-12,
            options={'maxiter': 1000, 'ftol': 1e-12, 'eps': 1e-12}
        )

        # Extract final centers and radii
        current_centers = result.x[:2 * n].reshape(n, 2)
        current_radii = result.x[2 * n:]
        current_sum_radii = np.sum(current_radii)

        # Validate the packing
        valid, msg = validate_packing(current_centers, current_radii)
        if valid and current_sum_radii > best_sum_radii:
            best_sum_radii = current_sum_radii
            best_centers = current_centers
            best_radii = current_radii

    return best_centers, best_radii, best_sum_radii

def validate_packing(centers, radii):
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
            dist = np.hypot(centers[i][0] - centers[j][0],
                            centers[i][1] - centers[j][1])
            if dist < radii[i] + radii[j] - 1e-12:
                return False, f"Circles {i} and {j} overlap"

    return True, "ok"

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
        lw = 1.5 if i in invalid_ids else 0.5
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