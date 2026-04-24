#!/usr/bin/env python3
"""
Carrega o modelo base + adapter LoRA treinado e gera uma resposta de teste.
Usa GPU (CUDA) se disponível, senão CPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO / "trein", _REPO / "server"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data_config import (
    DEFAULT_ADAPTER_DIR,
    DEFAULT_MODEL_NAME,
    FIX_GENERATION_LENGTH_CONFLICT,
    apply_loading_progress_env,
)

apply_loading_progress_env()

from lora_engine import generate_chat_reply, load_lora_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Teste rápido do adapter após o treino (GPU se houver CUDA).")
    p.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_NAME,
        help="Mesmo modelo base usado no train_lora.py (padrão: data_config.DEFAULT_MODEL_NAME).",
    )
    p.add_argument(
        "--adapter_dir",
        type=Path,
        default=DEFAULT_ADAPTER_DIR,
        help="Pasta onde o train_lora.py salvou o adapter (padrão: data_config.DEFAULT_ADAPTER_DIR).",
    )
    p.add_argument(
        "--prompt",
        default="O que é um oráculo digital?",
        help="Pergunta para o modelo.",
    )
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--trust_remote_code", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer, model = load_lora_pipeline(
        args.model_name,
        args.adapter_dir,
        None,
        trust_remote_code=args.trust_remote_code,
        fix_generation_max_length=FIX_GENERATION_LENGTH_CONFLICT,
    )
    messages = [{"role": "user", "content": args.prompt}]
    text = generate_chat_reply(
        tokenizer,
        model,
        messages,
        max_new_tokens=args.max_new_tokens,
    )
    print(text)


if __name__ == "__main__":
    main()
