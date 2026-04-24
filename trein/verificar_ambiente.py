#!/usr/bin/env python3
"""Imprime versões das dependências (ambiente pensado para treino em CPU)."""

from __future__ import annotations

import sys


def main() -> None:
    try:
        import torch
    except ImportError:
        print("PyTorch não está instalado. Ative o venv e rode: pip install -r trein/requirements.txt")
        sys.exit(1)

    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)
    print("Dispositivo de treino/inferência deste projeto: CPU (fp32).")

    for name in ("transformers", "datasets", "peft", "trl", "accelerate"):
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "?")
            print(f"{name}: {ver}")
        except ImportError:
            print(f"{name}: não instalado")


if __name__ == "__main__":
    main()
