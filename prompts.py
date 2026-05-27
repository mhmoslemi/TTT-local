"""
Prompt construction for the circle-packing problem.

We keep the prompt SHORT and concrete. Small code models do much better
with focused instructions than long ones with lots of background.
"""


SYSTEM = (
    "You are an expert Python programmer who writes correct, runnable code. "
    "When asked for code, you respond with a single complete Python program "
    "inside one ```python ... ``` block, with nothing after the closing fence."
)


def build_prompt(num_circles: int, parent_code: str = "",
                 parent_value: float = 0.0, target: float = 2.636):
    """
    Returns a list-of-messages chat prompt (system + user).
    """
    if parent_code and parent_code.strip():
        # Reuse the previous best as a starting point
        history_block = f"""Your previous solution produced sum of radii = {parent_value:.6f}.
Here it is. Improve it.

```python
{parent_code}
```
"""
    else:
        history_block = ""

    user = f"""Write a Python function `run_packing()` that packs {num_circles} non-overlapping circles inside the unit square [0,1] x [0,1] to maximize the sum of their radii. Target value: {target}.

The function must return a tuple `(centers, radii, sum_radii)`:
  - `centers`: numpy array of shape ({num_circles}, 2), each row is [x, y]
  - `radii`:   numpy array of shape ({num_circles},)
  - `sum_radii`: float, the sum of `radii`

Constraints (your code must satisfy these):
  - Every circle is fully inside the unit square: `r <= x <= 1-r` and `r <= y <= 1-r`
  - No two circles overlap: distance between any two centers >= sum of their radii
  - All radii are non-negative

You have numpy available as `np`. You may use `scipy.optimize.minimize` if you import it (it may not always be installed, so wrap the import in try/except if you rely on it).

Do NOT call any external validation function; just return your result.
{history_block}
Now write the complete program in ONE ```python ... ``` block. Include all imports inside the block. Define `run_packing()` and any helpers at module level (no closures, no lambdas)."""

    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


if __name__ == "__main__":
    msgs = build_prompt(num_circles=5, target=1.103553)
    for m in msgs:
        print(f"\n=== {m['role']} ===\n{m['content']}")