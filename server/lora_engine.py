"""
Motor de inferência partilhado: base Hugging Face + adapter LoRA (ou pasta modelo fundido).
Usado por `server/serve_lora.py` (e pode ser importado por scripts de treino que partilhem a mesma stack).
"""

from __future__ import annotations

import os
import re
import sys
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


def _env_flag(name: str, default: bool) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _attn_implementation() -> str | None:
    """sdpa = PyTorch SDPA (rápido, sem deps extra); eager = compatível. flash_attention_2 requer o pacote."""
    raw = (os.environ.get("ORACULO_ATTN_IMPLEMENTATION") or "").strip()
    if raw:
        return raw
    if torch.cuda.is_available():
        return "sdpa"
    return "eager"


def _apply_cuda_runtime_prefs() -> None:
    if not torch.cuda.is_available():
        return
    # TF32 em matmuls float32 (alguns módulos internos) — ligeiro ganho em A4500/Ampera+
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
    """Um forward curto para aquecer CUDA/cuDNN (e o grafo de compile, se houver)."""
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
    """
    True se a pasta tiver pesos carregáveis pelo Transformers (merge completo ou Hub local).
    Evita tratar como fundido uma pasta só com config.json (merge interrompido).
    """
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


def _reject_gguf_only_hub_id(model_name: str) -> None:
    """
    Repositórios no Hub com sufixo *-GGUF* (ex. TheBloke/...) publicam ficheiros `.gguf` para
    llama.cpp, não um modelo carregável por `transformers` + PyTorch.
    """
    s = (model_name or "").strip()
    if not s:
        return
    p = Path(s)
    if p.is_dir() or p.is_file():
        return
    if not re.search(r"[-_]GGUF\b", s, re.I):
        return
    raise ValueError(
        f"O ID {s!r} parece ser um repositório **só GGUF** (ficheiros .gguf para llama.cpp), "
        "não um modelo Hugging Face completo (config + safetensors).\n"
        "• Modo **PyTorch** (actual): use o ID **sem** sufixo GGUF, ex. "
        "`mistralai/Mistral-7B-Instruct-v0.3`.\n"
        "• Para **.gguf**: defina `ORACULO_INFERENCE_BACKEND=gguf` e `ORACULO_GGUF_PATH` "
        "com o caminho local de um ficheiro `.gguf` (pode descarregar desse repositório TheBloke). "
        "Ver `server/README_INFERENCE_HF_GGUF.md`."
    )


def load_lora_pipeline(
    model_name: str,
    adapter_dir: Path | None,
    merged_model_dir: Path | None,
    *,
    base_only: bool = False,
    trust_remote_code: bool = False,
    fix_generation_max_length: bool = True,
) -> tuple[AutoTokenizer, torch.nn.Module, Path | None]:
    """
    Carrega tokenizer + modelo.
    - Se ``base_only`` é True: carrega **só** o modelo base a partir de ``model_name`` (Hugging Face ou pasta local);
      ignora merge e LoRA (útil para testar Qwen, etc. antes de treinar adapter).
    - Se ``merged_model_dir`` existir e for pasta válida **com pesos** (e **não** ``base_only``), carrega só o modelo fundido.
    - Senão: base ``model_name`` + ``PeftModel`` em ``adapter_dir``.

    O terceiro valor devolvido é o caminho do merge **efetivamente** usado, ou ``None`` se carregou base+adapter ou só base.

    **Performance (GPU):** `ORACULO_ATTN_IMPLEMENTATION` (padrão ``sdpa``), matmul `ORACULO_MATMUL_PRECISION`,
    opcional `ORACULO_TORCH_COMPILE=1` e `ORACULO_CUDNN_BENCHMARK=1` — ver comentário em `server/.env.example`.
    """
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

    merged: Path | None = None
    if not base_only and merged_model_dir is not None and merged_model_dir.is_dir():
        if (merged_model_dir / "config.json").is_file():
            if hf_local_dir_has_model_weights(merged_model_dir):
                merged = merged_model_dir
            else:
                print(
                    f"Aviso: pasta fundida incompleta (sem pesos .safetensors/.bin): {merged_model_dir}. "
                    "A usar modelo base + adapter (ou defina --base-only / ORACULO_BASE_ONLY=1 se não tiver LoRA).",
                    file=sys.stderr,
                    flush=True,
                )

    if merged is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            str(merged), trust_remote_code=trust_remote_code
        )
        model = _load_causal_pretrained(str(merged))
    else:
        _reject_gguf_only_hub_id(model_name)
        if base_only:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )
            model = _load_causal_pretrained(model_name)
        else:
            if adapter_dir is None or not adapter_dir.is_dir():
                raise FileNotFoundError(
                    f"Adapter não encontrado: {adapter_dir}. "
                    "Treine com train_lora.py, passe --merged_model_dir, "
                    "ou use --base-only (ORACULO_BASE_ONLY=1) para carregar só o modelo base do Hub."
                )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )
            model = _load_causal_pretrained(model_name)
            model = PeftModel.from_pretrained(model, str(adapter_dir))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not use_cuda:
        model = model.to("cpu")
    model.eval()

    if fix_generation_max_length and hasattr(model, "generation_config"):
        model.generation_config.max_length = None

    model = _maybe_compile_model(model)
    _forward_warmup(tokenizer, model)

    return tokenizer, model, merged


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
