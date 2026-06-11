import asyncio
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from openevolve.llm.base import LLMInterface

_model = None
_tokenizer = None
_lock = threading.Lock()

THINK_END_ID = 151668  # token id of </think> in Qwen3 tokenizer


def _load(name):
    # Lazy on purpose: the model must load inside the worker process,
    # not in the parent, because OpenEvolve forks workers and forking
    # an initialized CUDA context breaks things.
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(name)
        _model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        _model.eval()
    return _model, _tokenizer


class LocalQwen(LLMInterface):
    def __init__(self, model_cfg):
        self.name = model_cfg.name
        self.temperature = model_cfg.temperature if model_cfg.temperature is not None else 0.6
        self.top_p = model_cfg.top_p if model_cfg.top_p is not None else 0.95
        self.max_new_tokens = model_cfg.max_tokens or 24576
        self.system_message = model_cfg.system_message

    async def generate(self, prompt, **kwargs):
        return await self.generate_with_context(
            self.system_message, [{"role": "user", "content": prompt}], **kwargs
        )

    async def generate_with_context(self, system_message, messages, **kwargs):
        msgs = [{"role": "system", "content": system_message}] + messages
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_sync, msgs, kwargs)

    def _generate_sync(self, msgs, kwargs):
        model, tok = _load(self.name)
        with _lock:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,  # Qwen thinks
            )
            inputs = tok([text], return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_tokens", self.max_new_tokens),
                    do_sample=True,
                    temperature=kwargs.get("temperature", self.temperature),
                    top_p=kwargs.get("top_p", self.top_p),
                    top_k=20,
                    pad_token_id=tok.eos_token_id,
                )
            output_ids = out[0][inputs.input_ids.shape[1]:].tolist()

        # Official Qwen3 recipe: split at the last </think>, return only the answer
        try:
            idx = len(output_ids) - output_ids[::-1].index(THINK_END_ID)
        except ValueError:
            idx = 0
        return tok.decode(output_ids[idx:], skip_special_tokens=True).strip()