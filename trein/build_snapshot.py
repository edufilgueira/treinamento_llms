#!/usr/bin/env python3
"""
Unifica todos os JSONL em data/raw/ (recursivo) num único snapshot em data/snapshots/.

O caminho de saída vem de data_config.py (train_<SNAPSHOT_DATE_PREFIX>_<DATASET_VERSION>.jsonl).
Enquanto VERSION e PREFIX não mudarem, o mesmo ficheiro é sobrescrito.

Cada linha deve ser JSON com pelo menos a coluna 'messages' (formato do train_lora.py).
Colunas extra (ex.: meta) são removidas na saída — só fica 'messages'.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_TREIN = Path(__file__).resolve().parent
if str(_TREIN) not in sys.path:
    sys.path.insert(0, str(_TREIN))

from data_config import RAW_DIR, SNAPSHOTS_DIR, snapshot_output_path


def collect_jsonl_files(raw_dir: Path) -> list[Path]:
    return sorted(p for p in raw_dir.rglob("*.jsonl") if p.is_file())


def load_valid_rows(paths: list[Path]) -> tuple[list[dict], int]:
    """Retorna linhas válidas e contagem de linhas ignoradas."""
    rows: list[dict] = []
    skipped = 0
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    print(f"Aviso: JSON inválido em {path}:{line_no}", file=sys.stderr)
                    continue
                if "messages" not in obj:
                    skipped += 1
                    print(
                        f"Aviso: sem 'messages' em {path}:{line_no}",
                        file=sys.stderr,
                    )
                    continue
                rows.append({"messages": obj["messages"]})
    return rows, skipped


def main() -> None:
    p = argparse.ArgumentParser(description="Unifica datasets de data/raw/ num snapshot.")
    p.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Mantém a ordem (concatenação por caminho de ficheiro); por defeito embaralha.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente para embaralhamento (só com shuffle).",
    )
    args = p.parse_args()

    if not RAW_DIR.is_dir():
        print(f"Pasta não encontrada: {RAW_DIR}", file=sys.stderr)
        sys.exit(1)

    files = collect_jsonl_files(RAW_DIR)
    if not files:
        print(
            f"Nenhum ficheiro .jsonl em {RAW_DIR}. Adicione dados em subpastas (ex.: raw/exemplo/).",
            file=sys.stderr,
        )
        sys.exit(1)

    rows, skipped = load_valid_rows(files)
    if not rows:
        print("Nenhuma linha válida com 'messages'.", file=sys.stderr)
        sys.exit(1)

    if not args.no_shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)

    out_path = snapshot_output_path()
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Snapshot: {out_path.resolve()}")
    print(f"Linhas: {len(rows)} (ignoradas: {skipped})")
    print(f"Ficheiros de origem: {len(files)}")


if __name__ == "__main__":
    main()
