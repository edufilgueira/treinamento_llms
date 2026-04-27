"""
Motor Hugging Face: **só modelo fundido** (pasta merge_lora), sem base+LoRA em runtime.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from threading import Event, Thread
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)


class _CancelStoppingCriteria(StoppingCriteria):
    def __init__(self, cancel_event: Event) -> None:
        super().__init__()
        self.cancel_event = cancel_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.cancel_event.is_set()


def _env_flag(name: str, default: bool) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def apply_chat_template(
    tokenizer: AutoTokenizer,
    conversation: list,
    *,
    tokenize: bool,
    add_generation_prompt: bool,
    return_tensors: str | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
    }
    if return_tensors is not None:
        kwargs["return_tensors"] = return_tensors
    try:
        return tokenizer.apply_chat_template(
            conversation,
            **kwargs,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(conversation, **kwargs)


def _attn_implementation() -> str | None:
    raw = (os.environ.get("ORACULO_ATTN_IMPLEMENTATION") or "").strip()
    if raw:
        return raw
    if torch.cuda.is_available():
        return "sdpa"
    return "eager"


def _apply_cuda_runtime_prefs() -> None:
    if not torch.cuda.is_available():
        return
    prec = (os.environ.get("ORACULO_MATMUL_PRECISION") or "high").strip().lower()
    if prec in ("high", "highest", "medium"):
        try:
            torch.set_float32_matmul_precision(prec)  # type: ignore[arg-type]
        except Exception:
            pass
    if _env_flag("ORACULO_CUDNN_BENCHMARK", False):
        torch.backends.cudnn.benchmark = True  # type: ignore[union-attr]


def _maybe_compile_model(model: torch.nn.Module) -> torch.nn.Module:
    if not _env_flag("ORACULO_TORCH_COMPILE", False):
        return model
    mode = (os.environ.get("ORACULO_TORCH_COMPILE_MODE") or "reduce-overhead").strip()
    try:
        out = torch.compile(model, mode=mode)  # type: ignore[assignment, misc]
    except Exception as e:
        print(
            f"Aviso: ORACULO_TORCH_COMPILE=1 mas torch.compile falhou ({e!s}); a usar modelo normal.",
            file=sys.stderr,
            flush=True,
        )
        return model
    print(
        "torch.compile activo (1.ª geração pode ser mais lenta). ORACULO_TORCH_COMPILE=0 para desligar.",
        file=sys.stderr,
        flush=True,
    )
    return out


def _forward_warmup(
    tokenizer: AutoTokenizer, model: torch.nn.Module, skip_if_no_cuda: bool = False
) -> None:
    if skip_if_no_cuda and not torch.cuda.is_available():
        return
    if not _env_flag("ORACULO_TORCH_WARMUP", True):
        return
    try:
        device = next(model.parameters()).device
        with torch.inference_mode():
            t = tokenizer(" ", return_tensors="pt", add_special_tokens=True)
            ids = t["input_ids"].to(device)
            model(ids)
    except Exception as e:
        print(f"Aviso: warmup (forward) ignorado: {e!s}", file=sys.stderr, flush=True)


def hf_local_dir_has_model_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file():
        return True
    if (path / "model.safetensors.index.json").is_file():
        return True
    for child in path.iterdir():
        if not child.is_file():
            continue
        name = child.name
        if name.startswith("model-") and name.endswith(".safetensors"):
            return True
        if name.startswith("pytorch_model-") and name.endswith(".bin"):
            return True
    return False


def load_merged_pipeline(
    merged_model_dir: Path,
    *,
    trust_remote_code: bool = False,
    fix_generation_max_length: bool = True,
) -> tuple[AutoTokenizer, torch.nn.Module]:
    p = merged_model_dir.resolve()
    if not p.is_dir() or not (p / "config.json").is_file():
        raise FileNotFoundError(
            f"Pasta do modelo fundido inválida (falta config.json): {merged_model_dir}"
        )
    if not hf_local_dir_has_model_weights(p):
        raise FileNotFoundError(
            f"Modelo fundido sem pesos carregáveis (.safetensors / .bin): {merged_model_dir}. "
            "Corre merge_lora.py e confirma que o merge terminou."
        )

    _apply_cuda_runtime_prefs()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    else:
        compute_dtype = torch.float32

    attn_impl = _attn_implementation()
    common_kw = dict(
        trust_remote_code=trust_remote_code,
        device_map="auto" if use_cuda else None,
        low_cpu_mem_usage=True,
    )
    if attn_impl:
        common_kw["attn_implementation"] = attn_impl

    def _load_causal_pretrained(weights_ref: str) -> torch.nn.Module:
        kw_tries: list[dict] = [common_kw]
        if "attn_implementation" in common_kw:
            kw_tries.append(
                {k: v for k, v in common_kw.items() if k != "attn_implementation"}
            )
        for kw in kw_tries:
            try:
                return AutoModelForCausalLM.from_pretrained(
                    weights_ref, dtype=compute_dtype, **kw
                )
            except TypeError:
                try:
                    return AutoModelForCausalLM.from_pretrained(
                        weights_ref, torch_dtype=compute_dtype, **kw
                    )
                except TypeError:
                    continue
        raise RuntimeError(f"Falha ao carregar o modelo: {weights_ref!r}")

    tokenizer = AutoTokenizer.from_pretrained(str(p), trust_remote_code=trust_remote_code)
    model = _load_causal_pretrained(str(p))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not use_cuda:
        model = model.to("cpu")
    model.eval()

    if fix_generation_max_length and hasattr(model, "generation_config"):
        model.generation_config.max_length = None

    model = _maybe_compile_model(model)
    _forward_warmup(tokenizer, model)

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
    model_inputs = apply_chat_template(
        tokenizer,
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
    with torch.inference_mode():
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
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


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
    model_inputs = apply_chat_template(
        tokenizer,
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
