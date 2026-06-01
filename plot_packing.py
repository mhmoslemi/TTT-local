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
import math
import random

def run_packing():
    num_circles = 26

    def improved_initial_layout(num_circles, max_radius_initial=0.1):
        # Create centers using a hexagonal grid pattern
        row_count = int(np.ceil(np.sqrt(num_circles)))
        col_count = int(np.ceil(num_circles / row_count))
        centers = np.zeros((num_circles, 2))
        idx = 0

        for i in range(row_count):
            for j in range(col_count):
                if idx >= num_circles:
                    break
                # Hexagonal packing with dynamic spacing
                x = j * (2 * max_radius_initial) + max_radius_initial
                y = i * (2 * max_radius_initial) + max_radius_initial
                centers[idx] = [x, y]
                idx += 1

        # Adjust for boundary constraints
        for i in range(num_circles):
            x, y = centers[i]
            dist_to_boundary = min(x, 1 - x, y, 1 - y)
            if dist_to_boundary < max_radius_initial:
                # Push center towards the boundary
                dir_x = np.random.uniform(-0.5, 0.5)
                dir_y = np.random.uniform(-0.5, 0.5)
                new_x = x + (dist_to_boundary - max_radius_initial) * dir_x
                new_y = y + (dist_to_boundary - max_radius_initial) * dir_y
                new_x = max(0, min(1, new_x))
                new_y = max(0, min(1, new_y))
                centers[i] = [new_x, new_y]

        # Add circles near corners
        for i in range(num_circles):
            x, y = centers[i]
            dist_to_corner = min(x, 1 - x, y, 1 - y)
            if dist_to_corner < max_radius_initial:
                # Place new circle near corner
                dir_x = np.random.uniform(-0.5, 0.5)
                dir_y = np.random.uniform(-0.5, 0.5)
                new_x = max(0, min(1, x + (dist_to_corner - max_radius_initial) * dir_x))
                new_y = max(0, min(1, y + (dist_to_corner - max_radius_initial) * dir_y))
                centers[i] = [new_x, new_y]

        initial_radii = np.full(num_circles, max_radius_initial)
        for i in range(num_circles):
            x, y = centers[i]
            dist_to_boundary = min(x, 1 - x, y, 1 - y)
            if dist_to_boundary < max_radius_initial:
                initial_radii[i] = dist_to_boundary

        return centers, initial_radii

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
            if (x - r < -1e-12 or x + r > 1 + 1e-12 or
                y - r < -1e-12 or y + r > 1 + 1e-12):
                return False, f"Circle {i} at ({x},{y}) r={r} outside unit square"

        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < radii[i] + radii[j] - 1e-12:
                    return False, f"Circles {i} and {j} overlap"

        return True, "ok"

    def objective(params):
        flat_params = params.reshape(-1)
        centers_opt = flat_params[:num_circles * 2].reshape((num_circles, 2))
        radii_opt = flat_params[num_circles * 2:].reshape((num_circles,))
        return -np.sum(radii_opt)  # Minimization of negative sum

    def genetic_algorithm():
        population_size = 300
        mutation_rate = 0.05
        generations = 800
        elite_fraction = 0.1
        tolerance = 1e-6

        population = []
        for _ in range(population_size):
            centers, radii = improved_initial_layout(num_circles)
            params = np.concatenate([centers.flatten(), radii])
            population.append(params)

        for generation in range(generations):
            fitness = []
            for params in population:
                centers_opt = params[:num_circles * 2].reshape((num_circles, 2))
                radii_opt = params[num_circles * 2:].reshape((num_circles,))
                is_valid, message = validate_packing(centers_opt, radii_opt)
                if not is_valid:
                    fitness.append(-np.inf)
                else:
                    fitness.append(np.sum(radii_opt))

            # Select elite individuals
            indices = np.argsort(fitness)[-int(population_size * elite_fraction):]
            parents = [population[i] for i in indices]

            # Generate offspring
            offspring = []
            for _ in range(population_size - len(parents)):
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                child = np.copy(parent1)
                for i in range(len(child)):
                    if random.random() < mutation_rate:
                        child[i] += np.random.uniform(-0.01, 0.01)
                offspring.append(child)

            population = parents + offspring

            # Early stopping
            best_fitness = max(fitness)
            if best_fitness > 2.63:
                break

        best_params = None
        best_fitness = -np.inf
        for params in population:
            centers_opt = params[:num_circles * 2].reshape((num_circles, 2))
            radii_opt = params[num_circles * 2:].reshape((num_circles,))
            is_valid, message = validate_packing(centers_opt, radii_opt)
            if not is_valid:
                continue
            current_fitness = np.sum(radii_opt)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_params = params
        return best_params if best_params is not None else improved_initial_layout(num_circles)[1]

    def refine_optimization(centers, radii):
        bounds = []
        for i in range(num_circles):
            x, y = centers[i]
            r = radii[i]
            x_low = max(0, x - r)
            x_high = min(1, x + r)
            y_low = max(0, y - r)
            y_high = min(1, y + r)
            r_low = max(0, r - 0.01)
            r_high = min(1, r + 0.01)
            bounds.extend([(x_low, x_high), (y_low, y_high), (r_low, r_high)])

        def constraints(x):
            flat_x = x.reshape(-1)
            centers_opt = flat_x[:num_circles * 2].reshape((num_circles, 2))
            radii_opt = flat_x[num_circles * 2:].reshape((num_circles,))
            
            for i in range(num_circles):
                x, y = centers_opt[i]
                r = radii_opt[i]
                if (x - r < -1e-12 or x + r > 1 + 1e-12 or
                    y - r < -1e-12 or y + r > 1 + 1e-12):
                    return -1.0

            for i in range(num_circles):
                for j in range(i + 1, num_circles):
                    dx = centers_opt[i][0] - centers_opt[j][0]
                    dy = centers_opt[i][1] - centers_opt[j][1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < radii_opt[i] + radii_opt[j] - 1e-12:
                        return -1.0

            return 0.0

        initial_guess = np.concatenate([centers.flatten(), radii])
        result = opt.minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraints},
            tol=1e-8
        )

        if result.success:
            return result.x[:num_circles * 2].reshape((num_circles, 2)), result.x[num_circles * 2:].reshape((num_circles,))
        else:
            return centers, radii

    centers, radii = improved_initial_layout(num_circles)

    best_params = genetic_algorithm()
    centers_opt = best_params[:num_circles * 2].reshape((num_circles, 2))
    radii_opt = best_params[num_circles * 2:].reshape((num_circles,))

    is_valid, message = validate_packing(centers_opt, radii_opt)
    if not is_valid:
        print(f"Validation failed: {message}")
        centers_opt = centers
        radii_opt = radii
        is_valid, message = validate_packing(centers_opt, radii_opt)
        if not is_valid:
            print(f"Initial layout also failed: {message}")
            return np.zeros((26, 2)), np.zeros(26), 0.0

    centers_opt, radii_opt = refine_optimization(centers_opt, radii_opt)

    is_valid, message = validate_packing(centers_opt, radii_opt)
    if not is_valid:
        print(f"Refined layout failed: {message}")
        return np.zeros((26, 2)), np.zeros(26), 0.0

    sum_radii = np.sum(radii_opt)
    return centers_opt, radii_opt, sum_radii



    
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