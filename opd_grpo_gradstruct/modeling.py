from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import dtype_from_name


def load_tokenizer(model_name: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_causal_lm(
    model_name: str,
    device: str,
    dtype_name: str,
    four_bit: bool = False,
):
    dtype = dtype_from_name(dtype_name)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": device}
    else:
        kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.inference_mode()
def generate_group(
    model,
    tokenizer,
    prompt: str,
    group_size: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
):
    enc = tokenizer(
        [prompt],
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
        padding=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    input_ids = input_ids.repeat(group_size, 1)
    attention_mask = attention_mask.repeat(group_size, 1)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=False,
    )
    prompt_len = input_ids.shape[1]
    generated = outputs[:, prompt_len:]
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return outputs.detach().cpu(), prompt_len, generated.detach().cpu(), texts
