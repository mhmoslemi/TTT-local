import numpy as np
import scipy.optimize as optimize
from typing import Tuple, List, Any

def run_packing() -> Tuple[np.ndarray, np.ndarray, float]:
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
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < radii[i] + radii[j] - 1e-12:
                    return False, f"Circles {i} and {j} overlap"

        return True, "ok"

    def objective_function(params, n_circles):
        centers = params[:n_circles * 2].reshape((n_circles, 2))
        radii = params[n_circles * 2:].reshape((n_circles,))
        
        # Penalize radii being too large or negative
        radius_penalty = np.sum(np.maximum(0, -radii))
        # Penalize circles being too close to the edges
        edge_penalty = np.sum(np.maximum(0, radii - (1 - np.abs(centers[:, 0])) - 1e-12))
        edge_penalty += np.sum(np.maximum(0, radii - (1 - np.abs(centers[:, 1])) - 1e-12))
        # Penalize circle overlaps
        overlap_penalty = 0
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < radii[i] + radii[j] - 1e-12:
                    overlap_penalty += (radii[i] + radii[j] - dist)
        return radius_penalty + edge_penalty + overlap_penalty

    def constraint_function(params, n_circles):
        centers = params[:n_circles * 2].reshape((n_circles, 2))
        radii = params[n_circles * 2:].reshape((n_circles,))
        
        # Constraint: circles cannot overlap
        constraints = []
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                # Penalty for overlapping: distance is less than r_i + r_j
                constraints.append(dist - (radii[i] + radii[j] - 1e-8))
        return constraints

    n_circles = 26
    # Initialize centers with a grid-like distribution
    centers = np.array([[i / 5 + 0.1, j / 5 + 0.1] for i in range(5) for j in range(6)])
    centers = centers[:n_circles]
    # Initialize radii with small values
    initial_radii = np.ones(n_circles) * 0.05

    # Flatten centers and radii for optimization
    initial_params = np.concatenate([centers.flatten(), initial_radii])

    # Optimization settings
    bounds = []
    for _ in range(n_circles * 2):
        bounds.append((0, 1))
    for _ in range(n_circles):
        bounds.append((0, 1))

    # Use Nelder-Mead for optimization
    result = optimize.minimize(
        fun=lambda x: objective_function(x, n_circles),
        x0=initial_params,
        bounds=bounds,
        constraints={"type": "ineq", "fun": lambda x: constraint_function(x, n_circles)},
        method="SLSQP"
    )

    # Extract the optimal solution
    optimal_params = result.x
    centers_opt = optimal_params[:n_circles * 2].reshape((n_circles, 2))
    radii_opt = optimal_params[n_circles * 2:].reshape((n_circles,))
    sum_radii = np.sum(radii_opt)

    # Validate the packing
    is_valid, message = validate_packing(centers_opt, radii_opt)
    if not is_valid:
        print(f"Validation failed: {message}")
        return np.array([]), np.array([]), 0.0

    return centers_opt, radii_opt, sum_radii