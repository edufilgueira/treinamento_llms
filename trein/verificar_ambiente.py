#!/usr/bin/env python3
"""Diagnóstico: driver NVIDIA, PyTorch, CUDA; útil para perceber lentidão do pip ou incompatibilidades."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from importlib import metadata
from datetime import datetime, timezone
from typing import TextIO


def _find_nvidia_smi() -> str | None:
    p = shutil.which("nvidia-smi")
    if p:
        return p
    for candidate in ("/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi"):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _run_smi(
    nvidia_smi: str, args: list[str], timeout: float = 20.0
) -> tuple[int, str, str]:
    try:
        r = subprocess.run(  # noqa: S603
            [nvidia_smi, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return r.returncode, r.stdout, r.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return 1, "", str(e)


def nvidia_smi_block() -> list[str]:
    out: list[str] = ["--- nvidia-smi (sistema) ---"]
    nvidia_smi = _find_nvidia_smi()
    if not nvidia_smi:
        out.append("nvidia-smi: não encontrado (sem driver no PATH, ou máquina sem GPU / container sem --gpus all).")
        return out
    out.append(f"comando: {nvidia_smi}")

    code, so, se = _run_smi(
        nvidia_smi, ["-q", "-d", "DRIVER,MEMORY", "-L"]
    )
    if code == 0 and (so or "").strip():
        out.extend([ln for ln in so.rstrip().splitlines() if ln.strip()][:40])
    else:
        code2, s2, se2 = _run_smi(
            nvidia_smi,
            ["--query-gpu=name,driver_version,memory.total,compute_mode", "--format=csv"],
        )
        if code2 == 0 and s2.strip():
            out.extend(s2.rstrip().splitlines())
        else:
            for err in (se, se2):
                if err and err.strip():
                    out.append("stderr: " + err.strip()[:500])
                    break

    # "CUDA Version: X.X" na primeira linha do nvidia-smi
    c0, smi_head, _ = _run_smi(nvidia_smi, [])
    for line in (smi_head or "").splitlines()[:4]:
        if "CUDA" in line.upper() or "Driver" in line:
            out.append("resumo: " + line.strip())
            break
    m = re.search(
        r"CUDA Version:\s*([0-9.]+)", (smi_head or ""), re.IGNORECASE
    )
    if m:
        out.append(
            f"nota: versão de CUDA do driver (máx. runtime) ≈ {m.group(1)} — o wheel do PyTorch deve ser compatível com o driver; "
            "se o torch tiver CUDA mais nova, pode haver 'CUDA capability' ou avisos ao importar."
        )
    return out


def torch_block() -> list[str]:
    out: list[str] = ["--- PyTorch (wheel em uso) ---"]
    try:
        import torch
    except ImportError:
        out.append("torch: não importável (ainda a instalar ou venv vazio).")
        return out

    out.append(f"torch.__version__: {torch.__version__}")
    out.append(f"torch.version.cuda (CUDA do build do wheel): {getattr(torch.version, 'cuda', None)}")
    out.append(
        f"torch.cuda.is_available(): {torch.cuda.is_available()}"
    )
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        for i in range(torch.cuda.device_count()):
            out.append(
                f"  device[{i}]: {torch.cuda.get_device_name(i)} "
                f"cap {torch.cuda.get_device_capability(i)}"
            )
    try:
        v = torch.backends.cudnn.version()
        out.append(f"cuDNN (biblioteca carregada): {v}")
    except Exception:  # noqa: BLE001
        out.append("cuDNN: n/d (normal em CPU)")

    # pip metadata do pacote torch
    try:
        dist = metadata.distribution("torch")
        m = re.search(r"\+cu(\d+)", (dist.version or ""))
        if m:
            out.append(
                f"metadado do wheel (hint): {dist.version!s}  →  sufixo +cu{m.group(1)} = stack CUDA aprox. do índice PyTorch"
            )
        else:
            out.append(f"metadado pip torch: {dist.version!s}")
    except Exception:  # noqa: BLE001
        pass
    return out


def env_block() -> list[str]:
    out = ["--- ambiente ---" ]
    out.append(f"quando: {datetime.now(timezone.utc).astimezone().isoformat()}")
    out.append(f"python: {sys.executable} — {sys.version.split()[0]}")
    if shutil.which("pip"):
        try:
            r = subprocess.run(  # noqa: S603
                [shutil.which("pip") or "pip", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.stdout:
                out.append("pip: " + r.stdout.strip()[:200])
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
    return out


def pip_nvidia_sanity() -> list[str]:
    out = ["--- pacotes nvidia* em pip (se instalados) ---"]
    try:
        r = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "pip", "list", "--format=freeze"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        out.append("(pip list falhou.)")
        return out
    lines = [l for l in (r.stdout or "").splitlines() if "nvidia" in l.lower() or l.startswith("torch")]
    if not lines:
        out.append("(nenhum pacote cujo nome contenha nvidia, ou ainda vazio)")
    else:
        for l in lines[:40]:
            out.append(l)
        if len(lines) > 40:
            out.append(f"... (+{len(lines) - 40} linhas)")
    return out


def hf_block() -> list[str]:
    out = ["--- Hugging Face / treino ---" ]
    for name in ("transformers", "datasets", "peft", "trl", "accelerate"):
        try:
            mod = __import__(name)
            out.append(f"{name}: {getattr(mod, '__version__', '?')}")
        except ImportError:
            out.append(f"{name}: não instalado")
    return out


def run(out_f: TextIO | None) -> int:
    lines: list[str] = []
    lines.extend(env_block())
    lines.append("")
    lines.extend(nvidia_smi_block())
    lines.append("")
    lines.extend(torch_block())
    lines.append("")
    lines.extend(pip_nvidia_sanity())
    lines.append("")
    lines.extend(hf_block())

    text = "\n".join(lines) + "\n"
    if out_f is not None:
        out_f.write(text)
        if out_f is not sys.stdout:
            print(text, end="")
    else:
        print(text, end="")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Imprime (e opcionalmente grava) driver NVIDIA, CUDA do PyTorch e dependências de treino."
    )
    ap.add_argument(
        "-o",
        "--output",
        help="Ficheiro para gravar o relatório (além de stdout).",
    )
    args = ap.parse_args()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:  # noqa: PTH123
            return run(f)
    return run(None)


if __name__ == "__main__":
    raise SystemExit(main())
