import numpy as np
import scipy.optimize as opt

def run_packing():
    num_circles = 26

    # Initialize centers with a more efficient hexagonal close packing (HCP) pattern
    centers = np.zeros((num_circles, 2))
    idx = 0
    rows = int(np.ceil(np.sqrt(num_circles)))
    cols = int(np.ceil(num_circles / rows))
    spacing = 1.0 / (cols + 1)
    hex_offset = spacing * np.sqrt(3) / 2

    for row in range(rows):
        for col in range(cols):
            x = col * spacing + spacing / 2
            y = row * spacing + spacing / 2
            if row % 2 == 1:
                x += spacing / 2
            if idx < num_circles:
                centers[idx] = [x, y]
                idx += 1

    # Refine initial positions for better packing
    if idx < num_circles:
        # Fill remaining circles in a more efficient way
        for i in range(idx, num_circles):
            # Try placing them in the remaining space
            # For the sake of a better initial guess, spread remaining circles in free space
            remaining_circles = num_circles - idx
            # Distribute evenly with more spacing
            for j in range(remaining_circles):
                if j % 2 == 0:
                    centers[idx + j] = [0.1 + j * 0.1, 0.1]
                else:
                    centers[idx + j] = [0.9 - j * 0.1, 0.9]

    # Initial guess for radii
    # Estimate based on spacing between centers
    initial_radii = np.zeros(num_circles)
    for i in range(num_circles):
        min_dist = np.inf
        for j in range(num_circles):
            if i != j:
                dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if dist < min_dist:
                    min_dist = dist
        initial_radii[i] = min_dist / 3.0  # Ensure radius is safe to avoid overlap

    radii = initial_radii

    # Objective function: maximize sum of radii
    def objective_function(radii_params):
        return -np.sum(radii_params)  # Minimize negative sum

    # Constraint function for overlap between two circles
    def constraint_overlap(i, j, centers, radii_params):
        x1, y1 = centers[i]
        x2, y2 = centers[j]
        r1 = radii_params[i]
        r2 = radii_params[j]
        dist = np.hypot(x1 - x2, y1 - y2)
        return dist - (r1 + r2)

    # Constraint function for boundary (left)
    def constraint_boundary_left(i, centers, radii_params):
        x, y = centers[i]
        r = radii_params[i]
        return (x - r) + 1e-12

    # Constraint function for boundary (right)
    def constraint_boundary_right(i, centers, radii_params):
        x, y = centers[i]
        r = radii_params[i]
        return (1.0 - x - r) + 1e-12

    # Constraint function for boundary (top)
    def constraint_boundary_top(i, centers, radii_params):
        x, y = centers[i]
        r = radii_params[i]
        return (1.0 - y - r) + 1e-12

    # Constraint function for boundary (bottom)
    def constraint_boundary_bottom(i, centers, radii_params):
        x, y = centers[i]
        r = radii_params[i]
        return (y - r) + 1e-12

    # Bounds for radii
    bounds = [(0, None) for _ in range(num_circles)]

    # Constraints
    cons = []

    # Boundary constraints
    for i in range(num_circles):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: constraint_boundary_left(i, centers, x)})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: constraint_boundary_right(i, centers, x)})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: constraint_boundary_top(i, centers, x)})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: constraint_boundary_bottom(i, centers, x)})

    # Overlap constraints
    for i in range(num_circles):
        for j in range(i + 1, num_circles):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: constraint_overlap(i, j, centers, x)})

    # Initial guess
    initial_guess = radii.copy()

    # Run the optimization
    result = opt.minimize(
        objective_function,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        tol=1e-10,
        options={
            'maxiter': 5000,
            'ftol': 1e-10,
            'eps': 1e-8,
            'disp': False
        }
    )

    # Extract results
    radii_result = result.x
    total_sum = np.sum(radii_result)

    # Validate the packing
    is_valid, message = validate_packing(centers, radii_result)
    if not is_valid:
        print(f"Warning: Packing is invalid. Reason: {message}")
        # Fallback to initial guess if invalid
        radii_result = initial_guess
        total_sum = np.sum(radii_result)

    return centers, radii_result, total_sum