"""
Prompt construction for the circle-packing problem.

Mirrors the prompt template from the original TTT-Discover paper:
  test-time-training/discover/examples/circle_packing/env.py :: CirclePackingEnv.get_question

Differences from the paper (intentional, to fit our setup):
  - The paper renders parent state via `initial_state.to_prompt(target, metric_name=...)`.
    Our equivalent state context is the parent code block + previous best metric,
    formatted to match the paper's structure.
  - Validator source is shown verbatim (as the paper does). If the model tries to
    call validate_packing(...) inside its code, reward.py's prelude injection
    makes the function available so the call doesn't crash.
"""

import inspect
from reward import validate_packing


SYSTEM = "You are an expert mathematician specializing in circle packing problems and computational geometry."


def _state_context(num_circles: int, target: float, parent_code: str, parent_value: float) -> str:
    """
    Render the parent-state portion of the prompt, matching the paper's
    `initial_state.to_prompt(target, metric_name='sum of radii')` block.
    """
    if parent_code and parent_code.strip():
        return (
            f"Target sum of radii: {target}.\n"
            f"Your previous program achieved sum of radii = {parent_value:.6f}.\n"
            f"Here is the previous program:\n"
            f"```python\n{parent_code}\n```\n"
        )
    else:
        return (
            f"Target sum of radii: {target}.\n"
            f"No previous program. Write one from scratch.\n"
        )


def build_prompt(num_circles: int, parent_code: str = "",
                 parent_value: float = 0.0, target: float = 2.636):
    """
    Returns a list-of-messages chat prompt (system + user).
    Structure mirrors CirclePackingEnv.get_question from the paper repo.
    """
    validator_src = inspect.getsource(validate_packing)
    state_ctx = _state_context(num_circles, target, parent_code, parent_value)

    user = f"""Your task is to pack {num_circles} circles in a unit square [0,1]×[0,1] to maximize the sum of radii.

We will run the below validation function (read-only, do not modify this):
```python
{validator_src}
```

{state_ctx}
Reason about how you could further improve this packing. Consider:
- Are circles placed optimally near boundaries and corners?
- Could a different arrangement (hexagonal, nested, hybrid) yield better results?
- Are there gaps that could be filled with repositioned or resized circles?
- Could optimization parameters or methods be improved?

Rules:
- You must define the run_packing function: def run_packing() -> tuple[np.ndarray, np.ndarray, float]
- Returns (centers, radii, sum_radii) where centers has shape ({num_circles}, 2) and radii has shape ({num_circles},).
- You can use scientific libraries like scipy, numpy, cvxpy, math.
- Centers must lie within [0,1]^2 and radii must be nonnegative.
- The pair (centers, radii) must satisfy non-overlap and boundary constraints.
- Make all helper functions top level and have no closures from function nesting. Don't use any lambda functions.
- No filesystem or network IO.
- You need to get really creative and think from first principles.

Make sure to think step by step, first give your strategy between <strategy> and </strategy> tags, then finally return the final program between ```python and ```. /no_think

"""

# Make sure to /think step by step, first give your strategy between <strategy> and </strategy> tags, then finally return the final program between ```python and ```.


    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


if __name__ == "__main__":
    msgs = build_prompt(num_circles=26, target=2.635983)
    for m in msgs:
        print(f"\n=== {m['role']} ===\n{m['content']}")





# """
# Prompt construction for the circle-packing problem.

# """


# SYSTEM = (
#     "You are an expert Python programmer who writes correct, runnable code. "
#     "When asked for code, you respond with a single complete Python program "
#     "inside one ```python ... ``` block, with nothing after the closing fence."
# )


# def build_prompt(num_circles: int, parent_code: str = "",
#                  parent_value: float = 0.0, target: float = 2.636):
#     """
#     Returns a list-of-messages chat prompt (system + user).
#     """
#     if parent_code and parent_code.strip():
#         # Reuse the previous best as a starting point
#         history_block = f"""Your previous solution produced sum of radii = {parent_value:.6f}.
# Here it is. Improve it.

# ```python
# {parent_code}
# ```
# """
#     else:
#         history_block = ""

#     user = f"""Write a Python function `run_packing()` that packs {num_circles} non-overlapping circles inside the unit square [0,1] x [0,1] to maximize the sum of their radii. Target value: {target}.

# The function must return a tuple `(centers, radii, sum_radii)`:
#   - `centers`: numpy array of shape ({num_circles}, 2), each row is [x, y]
#   - `radii`:   numpy array of shape ({num_circles},)
#   - `sum_radii`: float, the sum of `radii`

# Constraints (your code must satisfy these):
#   - Every circle is fully inside the unit square: `r <= x <= 1-r` and `r <= y <= 1-r`
#   - No two circles overlap: distance between any two centers >= sum of their radii
#   - All radii are non-negative

# You have numpy available as `np`. You may use `scipy.optimize.minimize` if you import it (it may not always be installed, so wrap the import in try/except if you rely on it).

# Do NOT call any external validation function; just return your result.
# {history_block}
# Now write the complete program in ONE ```python ... ``` block. Include all imports inside the block. Define `run_packing()` and any helpers at module level (no closures, no lambdas)."""

#     return [
#         {"role": "system", "content": SYSTEM},
#         {"role": "user", "content": user},
#     ]


# if __name__ == "__main__":
#     msgs = build_prompt(num_circles=5, target=1.103553)
#     for m in msgs:
#         print(f"\n=== {m['role']} ===\n{m['content']}")