"""
Motor de inferência partilhado: base Hugging Face + adapter LoRA (ou pasta modelo fundido).
Usado por inferir.py e server/serve_lora.py.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from threading import Event, Thread

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)


class _CancelStoppingCriteria(StoppingCriteria):
    """Interrompe ``model.generate`` quando ``cancel_event`` está definido."""

    def __init__(self, cancel_event: Event) -> None:
        super().__init__()
        self.cancel_event = cancel_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.cancel_event.is_set()


def load_lora_pipeline(
    model_name: str,
    adapter_dir: Path | None,
    merged_model_dir: Path | None,
    *,
    trust_remote_code: bool = False,
    fix_generation_max_length: bool = True,
) -> tuple[AutoTokenizer, torch.nn.Module]:
    """
    Carrega tokenizer + modelo.
    - Se ``merged_model_dir`` existir e for pasta válida, carrega só o modelo fundido.
    - Senão: base ``model_name`` + ``PeftModel`` em ``adapter_dir``.
    """
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    else:
        compute_dtype = torch.float32

    common_kw = dict(
        trust_remote_code=trust_remote_code,
        device_map="auto" if use_cuda else None,
        low_cpu_mem_usage=True,
    )

    merged: Path | None = None
    if merged_model_dir is not None and merged_model_dir.is_dir():
        if (merged_model_dir / "config.json").is_file():
            merged = merged_model_dir

    if merged is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            str(merged), trust_remote_code=trust_remote_code
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(merged), dtype=compute_dtype, **common_kw
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                str(merged), torch_dtype=compute_dtype, **common_kw
            )
    else:
        if adapter_dir is None or not adapter_dir.is_dir():
            raise FileNotFoundError(
                f"Adapter não encontrado: {adapter_dir}. "
                "Treine com train_lora.py ou passe --merged_model_dir."
            )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=compute_dtype, **common_kw
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=compute_dtype, **common_kw
            )
        model = PeftModel.from_pretrained(model, str(adapter_dir))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not use_cuda:
        model = model.to("cpu")
    model.eval()

    if fix_generation_max_length and hasattr(model, "generation_config"):
        model.generation_config.max_length = None

    return tokenizer, model


def generate_chat_reply(
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    messages: list[dict],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """``messages``: lista de ``{\"role\": \"user\"|\"assistant\", \"content\": str}``."""
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    if isinstance(model_inputs, torch.Tensor):
        model_inputs = {"input_ids": model_inputs.to(device)}
    else:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    input_ids = model_inputs["input_ids"]
    with torch.no_grad():
        out = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(0.01, temperature),
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out[0, input_ids.shape[1] :]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return reply


def generate_chat_reply_stream(
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    messages: list[dict],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    cancel_event: Event | None = None,
) -> Iterator[str]:
    """
    Igual a ``generate_chat_reply``, mas produz **pedaços de texto** à medida que o modelo gera
    (para SSE no servidor). O Hugging Face pode devolver texto acumulado ou por token; normalizamos
    para **deltas** (só o que é novo em cada passo).
    """
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    if isinstance(model_inputs, torch.Tensor):
        model_inputs = {"input_ids": model_inputs.to(device)}
    else:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    gen_kwargs = {
        **model_inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": max(0.01, temperature),
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if cancel_event is not None:
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [_CancelStoppingCriteria(cancel_event)]
        )

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    accumulated = ""
    try:
        for text in streamer:
            if cancel_event is not None and cancel_event.is_set():
                break
            if not text:
                continue
            if text.startswith(accumulated):
                delta = text[len(accumulated) :]
                accumulated = text
            else:
                delta = text
                accumulated += delta
            if delta:
                yield delta
    finally:
        thread.join(timeout=600)
