import numpy as np
import scipy.optimize as opt

def run_packing():
    # Number of circles
    num_circles = 26

    # Initial estimate of radius (based on hexagonal packing)
    initial_radius_estimate = 0.1  # Starting guess for radius

    # Initial positions using a hexagonal packing pattern
    centers = np.zeros((num_circles, 2))
    radii = np.full(num_circles, initial_radius_estimate)

    # Generate a hexagonal grid of points within the unit square
    # We'll pack a hexagonal grid of circles
    # For 26 circles, a rough layout could be 5 rows with 5, 5, 5, 5, 6 circles respectively

    # Hexagonal grid parameters
    hex_row_height = np.sqrt(3) * initial_radius_estimate  # vertical distance between rows
    hex_horizontal_offset = initial_radius_estimate  # horizontal offset between columns

    # Determine number of rows and columns based on hexagonal packing
    # Let's assume 5 rows with 5, 5, 5, 5, 6 circles for a total of 26
    num_rows = 5
    row_circle_counts = [5, 5, 5, 5, 6]
    row_positions = []

    # Generate positions for each row
    for i in range(num_rows):
        row_index = i
        y = row_index * hex_row_height + initial_radius_estimate
        if i % 2 == 0:
            x_positions = np.linspace(initial_radius_estimate, 1 - initial_radius_estimate, row_circle_counts[i])
        else:
            x_positions = np.linspace(initial_radius_estimate + hex_horizontal_offset, 1 - initial_radius_estimate - hex_horizontal_offset, row_circle_counts[i])
        for x in x_positions:
            row_positions.append((x, y))

    # Assign the calculated positions to centers
    for i, (x, y) in enumerate(row_positions):
        centers[i] = [x, y]

    # Objective function to maximize the sum of radii (we minimize the negative sum)
    def objective(radii_vec):
        return -np.sum(radii_vec)

    # Define constraints
    def constraint_overlap(i, j, centers, radii_vec):
        # Distance between centers minus sum of radii
        dist = np.linalg.norm(centers[i] - centers[j])
        return dist - (radii_vec[i] + radii_vec[j]) + 1e-12

    def constraint_boundary(i, centers, radii_vec):
        # Left boundary: x - r >= 0
        return centers[i][0] - radii_vec[i] + 1e-12

    def constraint_boundary2(i, centers, radii_vec):
        # Right boundary: x + r <= 1
        return 1.0 - centers[i][0] - radii_vec[i] + 1e-12

    def constraint_boundary3(i, centers, radii_vec):
        # Bottom boundary: y - r >= 0
        return centers[i][1] - radii_vec[i] + 1e-12

    def constraint_boundary4(i, centers, radii_vec):
        # Top boundary: y + r <= 1
        return 1.0 - centers[i][1] - radii_vec[i] + 1e-12

    # Define bounds for radii (non-negative)
    bounds = [(0.0, 1.0) for _ in range(num_circles)]

    # Define constraints
    constraints = []

    # Add boundary constraints
    for i in range(num_circles):
        constraints.append({'type': 'ineq', 'fun': lambda r, i=i: constraint_boundary(i, centers, r)})
        constraints.append({'type': 'ineq', 'fun': lambda r, i=i: constraint_boundary2(i, centers, r)})
        constraints.append({'type': 'ineq', 'fun': lambda r, i=i: constraint_boundary3(i, centers, r)})
        constraints.append({'type': 'ineq', 'fun': lambda r, i=i: constraint_boundary4(i, centers, r)})

    # Add overlap constraints
    for i in range(num_circles):
        for j in range(i + 1, num_circles):
            constraints.append({'type': 'ineq', 'fun': lambda r, i=i, j=j: constraint_overlap(i, j, centers, r)})

    # Flatten the radii vector for optimization
    flat_radii = radii.copy()

    # Run the optimization
    result = opt.minimize(
        objective,
        flat_radii,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    # Extract optimized radii
    optimized_radii = result.x
    optimized_centers = centers.copy()

    # Compute sum of radii
    sum_radii = np.sum(optimized_radii)

    return (optimized_centers, optimized_radii, sum_radii)