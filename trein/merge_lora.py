#!/usr/bin/env python3
"""
Funde o adapter LoRA no modelo base e grava um único modelo Hugging Face
(sem carregar adapter separado na inferência). Usa GPU (CUDA) se disponível.

Use depois do train_lora.py. Exige VRAM/RAM suficiente para o modelo base inteiro.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TREIN = Path(__file__).resolve().parent
if str(_TREIN) not in sys.path:
    sys.path.insert(0, str(_TREIN))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_config import DEFAULT_ADAPTER_DIR, DEFAULT_MERGED_MODEL_DIR, DEFAULT_MODEL_NAME


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge LoRA: incorpora adapter ao modelo base e salva em --output_dir (GPU se houver CUDA)."
    )
    p.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_NAME,
        help="Mesmo modelo base usado no train_lora.py (padrão: data_config.DEFAULT_MODEL_NAME).",
    )
    p.add_argument(
        "--adapter_dir",
        type=Path,
        default=DEFAULT_ADAPTER_DIR,
        help="Pasta com adapter (padrão: data_config.DEFAULT_ADAPTER_DIR).",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_MERGED_MODEL_DIR,
        help="Pasta de saída do modelo fundido (padrão: data_config.DEFAULT_MERGED_MODEL_DIR).",
    )
    p.add_argument("--trust_remote_code", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.adapter_dir.is_dir():
        print(f"Pasta não encontrada: {args.adapter_dir}", file=sys.stderr)
        sys.exit(1)
    if not (args.adapter_dir / "adapter_config.json").is_file():
        print(
            f"adapter_config.json não encontrado em {args.adapter_dir}. "
            "Use a pasta salva pelo train_lora.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_cuda = torch.cuda.is_available()

    tokenizer_src = args.adapter_dir
    if not (args.adapter_dir / "tokenizer_config.json").is_file():
        tokenizer_src = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_src),
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_cuda:
        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    else:
        compute_dtype = torch.float32
    common_kw = dict(
        trust_remote_code=args.trust_remote_code,
        device_map="auto" if use_cuda else None,
        low_cpu_mem_usage=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model_name),
            dtype=compute_dtype,
            **common_kw,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model_name),
            torch_dtype=compute_dtype,
            **common_kw,
        )

    model = PeftModel.from_pretrained(model, str(args.adapter_dir))
    print("Fundindo LoRA no modelo base (pode usar bastante VRAM/RAM)...")
    merged = model.merge_and_unload()
    merged.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Concluído. Modelo fundido salvo em: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
