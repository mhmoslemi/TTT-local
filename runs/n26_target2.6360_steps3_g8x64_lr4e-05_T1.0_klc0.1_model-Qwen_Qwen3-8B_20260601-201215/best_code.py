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