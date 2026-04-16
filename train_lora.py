#!/usr/bin/env python3
"""
Pipeline de treino LoRA (fine-tuning leve).

- **GPU (CUDA):** usa FP16/BF16 e `device_map="auto"` — é o cenário típico para Qwen 3B.
- **Só CPU:** FP32; modelos 3B+ podem esgotar RAM («Morto») em máquinas com pouca memória.
  O script apenas **avisa**; use `--allow_large_model_on_cpu` para omitir o aviso.

Memória (CPU): o tamanho em disco do modelo não é o uso em RAM — em FP32, ~3B parâmetros
≈ 12–13 GiB só nos pesos, mais o pico do 1.º backward (activações ~ proporcionais ao
``max_seq_length``). ``free`` mostra memória *antes* do pico; «Morto» costuma ser OOM nesse pico.

CPU + modelo ~3B: sem ``TRAIN_LORA_NO_CPU_SEQ_CAP=1``, ``max_seq_length`` é limitado a
``TRAIN_LORA_CPU_MAX_SEQ`` (padrão 128) para caber melhor em 16 GiB.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# O pacote opcional `gptqmodel` (GPT-QModel), se instalado, ao importar ajusta
# PYTORCH_ALLOC_CONF e mostra o banner com versões. Este script não usa GPTQ.
# Definir aqui evita que o gptqmodel sobrescreva o allocator (comportamento mais estável).
if not os.environ.get("PYTORCH_ALLOC_CONF"):
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from data_config import DEFAULT_ADAPTER_DIR, DEFAULT_MODEL_NAME, resolve_train_file


def _likely_too_large_for_cpu_only(model_id: str) -> bool:
    """Heurística: modelos vários-B em FP32 na CPU costumam morrer por OOM no 1.º passo."""
    s = model_id.lower().replace("_", "-")
    safe = ("tinyllama", "1.1b", "0.5b", "380m", "500m", "smollm")
    if any(x in s for x in safe):
        return False
    hints = (
        "2.5-3b",
        "-3b-",
        "/3b-",
        "3b-instruct",
        "-7b-",
        "/7b-",
        "-8b-",
        "-13b-",
        "-32b-",
        "-70b-",
        "llama-3.2-3b",
        "llama-3.1-8b",
    )
    return any(h in s for h in hints)


def _effective_max_seq_length_for_cpu(args: argparse.Namespace) -> int:
    """
    Em CPU, activações do backward com seq longa empurram o pico muito acima dos ~12 GiB dos pesos.
    """
    max_seq = args.max_seq_length
    if torch.cuda.is_available() or not _likely_too_large_for_cpu_only(args.model_name):
        return max_seq
    if os.environ.get("TRAIN_LORA_NO_CPU_SEQ_CAP", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return max_seq
    cap = int(os.environ.get("TRAIN_LORA_CPU_MAX_SEQ", "128"))
    if max_seq > cap:
        print(
            f"Aviso (CPU + modelo ~3B): max_seq_length {max_seq} → {cap} para reduzir pico de RAM "
            f"(pesos FP32 já são ~12 GiB; ficheiro no disco «~5 GiB» não inclui FP32 nem activações). "
            f"Aumente com cuidado: TRAIN_LORA_CPU_MAX_SEQ=160, ou desligue o teto: "
            f"TRAIN_LORA_NO_CPU_SEQ_CAP=1. Se ainda «Morto»: experimente cap=96 ou 64.",
            file=sys.stderr,
        )
        return cap
    return max_seq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Treino LoRA com TRL (SFT) + PEFT — usa GPU (CUDA) se disponível, senão CPU."
    )
    p.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_NAME,
        help="Nome do modelo no Hugging Face Hub (padrão: data_config.DEFAULT_MODEL_NAME).",
    )
    p.add_argument(
        "--train_file",
        type=Path,
        default=None,
        help="JSONL com 'messages'. Se omitido: maior versão em data/snapshots/ (train_*_v*.jsonl), senão data/exemplo_treino.jsonl. Edite data_config.py (versão/prefixo) e rode build_snapshot.py.",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_ADAPTER_DIR,
        help="Pasta de saída: adapter LoRA + tokenizer (padrão: data_config.DEFAULT_ADAPTER_DIR).",
    )
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Se > 0, interrompe após N passos (útil para testar a instalação; ignora --epochs).",
    )
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_seq_length",
        type=int,
        default=256,
        help="Comprimento máximo tokenizado. Em CPU com modelo ~3B, o script pode aplicar um teto "
        "(variável TRAIN_LORA_CPU_MAX_SEQ, padrão 128) para evitar OOM — ver docstring do módulo.",
    )
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--no_gradient_checkpointing",
        action="store_true",
        help="Desativa gradient checkpointing (mais RAM, pode ser um pouco mais rápido).",
    )
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--allow_large_model_on_cpu",
        action="store_true",
        help="Omite o aviso ao treinar modelos 3B+ só em CPU (o treino corre na mesma; "
        "isto só suprime a mensagem no stderr).",
    )
    return p.parse_args()


def messages_to_text(tokenizer: AutoTokenizer, example: dict) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main() -> None:
    args = parse_args()
    try:
        train_path = resolve_train_file(args.train_file)
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        sys.exit(1)
    if not train_path.is_file():
        print(f"Arquivo não encontrado: {train_path}", file=sys.stderr)
        sys.exit(1)

    if (
        not torch.cuda.is_available()
        and _likely_too_large_for_cpu_only(args.model_name)
        and not args.allow_large_model_on_cpu
    ):
        print(
            "Aviso (CPU, modelo ~3B+ em FP32): o treino prossegue, mas em máquinas com pouca RAM "
            "o kernel pode matar o processo (OOM / «Morto»), sobretudo no 1.º passo.\n"
            "Se já treinaste assim com sucesso, pode ignorar — em algum momento este script "
            "passou a **bloquear** antes; isso foi revertido: só resta este aviso. "
            "Para não o ver: --allow_large_model_on_cpu\n"
            "Alternativas se voltar a falhar: TinyLlama, TRAIN_LORA_CPU_MAX_SEQ=96, mais swap, "
            "ou GPU com CUDA. (Disco ~5 GiB ≠ RAM: 3B×FP32 ≈ 12 GiB só nos pesos.)",
            file=sys.stderr,
        )

    max_seq = _effective_max_seq_length_for_cpu(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = load_dataset("json", data_files=str(train_path), split="train")
    if "messages" not in dataset.column_names:
        print(
            "O dataset precisa ter a coluna 'messages' (lista de objetos com 'role' e 'content').",
            file=sys.stderr,
        )
        sys.exit(1)

    n_before = len(dataset)

    def _has_messages(ex: dict) -> bool:
        m = ex.get("messages")
        return isinstance(m, list) and len(m) > 0

    dataset = dataset.filter(_has_messages)
    n_after = len(dataset)
    if n_after < n_before:
        print(
            f"Aviso: ignorados {n_before - n_after} exemplos com 'messages' vazio ou inválido.",
            file=sys.stderr,
        )
    if n_after == 0:
        print("Nenhum exemplo válido após filtrar messages.", file=sys.stderr)
        sys.exit(1)

    dataset = dataset.map(
        lambda ex: messages_to_text(tokenizer, ex),
        remove_columns=dataset.column_names,
    )

    use_cuda = torch.cuda.is_available()
    use_cpu = not use_cuda

    if (
        use_cpu
        and _likely_too_large_for_cpu_only(args.model_name)
        and os.environ.get("TRAIN_LORA_CPU_THREADS", "").strip()
    ):
        torch.set_num_threads(int(os.environ["TRAIN_LORA_CPU_THREADS"]))
    elif use_cpu and _likely_too_large_for_cpu_only(args.model_name):
        # Menos threads → buffers temporários de BLAS menores em alguns sistemas (picos mais baixos).
        n = min(4, max(1, torch.get_num_threads()))
        torch.set_num_threads(n)

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
            args.model_name, dtype=compute_dtype, **common_kw
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=compute_dtype, **common_kw
        )

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if not args.no_gradient_checkpointing:
        model.enable_input_require_grads()

    optim = "adamw_torch" if use_cpu else "adamw_torch_fused"

    sft_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5,
        save_steps=200,
        save_total_limit=2,
        warmup_steps=5,
        lr_scheduler_type="cosine",
        fp16=bool(use_cuda and compute_dtype == torch.float16),
        bf16=bool(use_cuda and compute_dtype == torch.bfloat16),
        use_cpu=use_cpu,
        report_to="none",
        max_length=max_seq,
        dataset_text_field="text",
        packing=False,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        optim=optim,
        dataloader_pin_memory=use_cuda,
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"\nConcluído. Adapter e tokenizer salvos em: {args.output_dir.resolve()}")
    print(f"Dataset usado: {train_path.resolve()}")


if __name__ == "__main__":
    main()
