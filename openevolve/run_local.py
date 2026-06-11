import asyncio

from openevolve.config import load_config
from openevolve.controller import OpenEvolve
from local_qwen import LocalQwen

config = load_config("config_local.yaml")
for m in set(config.llm.models + config.llm.evaluator_models):
    m.init_client = LocalQwen  # bypasses OpenAILLM entirely


async def main():
    evolve = OpenEvolve(
        initial_program_path="examples/circle_packing/initial_program.py",
        evaluation_file="examples/circle_packing/evaluator.py",
        config=config,
        output_dir="examples/circle_packing/openevolve_output_qwen",
    )
    best = await evolve.run(iterations=100)
    if best:
        print(best.metrics)


if __name__ == "__main__":
    asyncio.run(main())