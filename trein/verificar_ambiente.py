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


def _pip_freeze() -> str:
    try:
        r = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "pip", "list", "--format=freeze"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return (r.stdout or "") if r.returncode == 0 else ""
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""


def _torch_import_ok() -> bool:
    try:
        import torch  # noqa: F401
    except (ImportError, OSError):
        # OSError: ex. libtorch_global_deps.so em falta (instalação interrompida a meio do wheel).
        return False
    return True


def torch_block() -> list[str]:
    out: list[str] = ["--- PyTorch (wheel em uso) ---"]
    try:
        import torch
    except (ImportError, OSError) as e:
        out.append("torch: não importa correctamente (instalação incompleta, interrompida, ou ficheiros .so em falta).")
        out.append(f"  detalhe: {e!s}")
        if "libtorch" in str(e) or "shared object" in str(e) or "No such file" in str(e):
            out.append("  (típico: pip a meio de 'Installing torch' foi parado com Ctrl+Z, SIGKILL, ou quota/disco; o pacote fica partido no site-packages).")
        out.append("  repara:  pip uninstall -y torch")
        out.append("          export TMPDIR=/workspace/_pip_tmp; mkdir -p $TMPDIR")
        out.append("          pip install --no-cache-dir 'torch>=2.1'")
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
    out.append(f"diretório corrente: {os.getcwd()}")
    ve = os.environ.get("VIRTUAL_ENV")
    if ve:
        out.append(f"VIRTUAL_ENV: {ve}  (venv na raiz do clone: …/treinamento_llms/.venv-trein — não fica em trein/)")
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


def pip_nvidia_sanity(freeze: str | None = None) -> list[str]:
    out = ["--- pacotes nvidia* / torch (pip) ---"]
    if freeze is None:
        rtxt = _pip_freeze()
    else:
        rtxt = freeze
    if not rtxt.strip():
        out.append("(pip list falhou.)")
        return out
    lines = [l for l in rtxt.splitlines() if "nvidia" in l.lower() or l.startswith("torch")]
    if not lines:
        out.append("(nenhum pacote cujo nome contenha nvidia, ou ainda vazio)")
    else:
        for l in lines[:40]:
            out.append(l)
        if len(lines) > 40:
            out.append(f"... (+{len(lines) - 40} linhas)")
    return out


def partial_install_block(freeze: str) -> list[str]:
    """Estado comum: pip instalou muitas dependências nvidia-* e a seguir a instalação parou (SIGKILL, OOM, Ctrl+C) antes de torch/."""
    if not _torch_import_ok() and not any(
        l.strip().startswith("torch==") for l in freeze.splitlines()
    ):
        nvidia_guess = [l for l in freeze.splitlines() if l.strip().startswith("nvidia-")]
        if nvidia_guess:
            return [
                "!!! AVISO: instalação a meio (dependências CUDA em pip, mas sem pacote 'torch' / import).",
                "    Isto costuma acontecer se o processo morreu na fase 'Installing collected packages' (memória, tempo, interrupção).",
                "    O teu driver (CUDA 13.0) combina com a stack nvidia-*-cu13 que já está no venv; faltam o wheel do PyTorch e o resto (transformers, …).",
                "    Recuperação neste venv (mantém a stack cu13 que já baixou):",
                "      pip install 'torch>=2.1' && pip install -r trein/requirements.txt  # 2.º comando pula/actualiza o resto",
                "    Ou, em comandos mínimos:  pip install 'torch>=2.1'  e depois o que faltar (transformers, datasets, trl, …) — ver trein/requirements.txt",
                "    Cuidado: trein/treina.sh (passo 1) por defeito usa o índice PyTorch *cu124*; num venv que já tem nvidia-*-cu13, podes trocar muita coisa. Para treinar já neste venv, usa o pip com PyPI (como acima) ou apaga o venv e recomeça:  rm -rf .venv-trein  &&  ./trein/treina.sh",
                "",
            ]
    return []


def slow_torch_wheel_tips(freeze: str) -> list[str]:
    """Ajudar quem fica muito tempo em 'Installing torch' (ficheiro .whl ~0.5GB)."""
    if _torch_import_ok():
        return []
    if any(l.strip().startswith("torch==") for l in freeze.splitlines()):
        return []  # torch já consta no pip: falha = pacote a meio, ver bloco PyTorch
    return [
        "--- se travas em 'Installing collected packages: torch' (download já a 100+ MB/s) ---",
        "O download do .whl é rapido; a fase 'Installing' = escrever dezenas de milhares de ficheiros em site-packages. Em /workspace tipo MooseFS/Rede (ex. RunPod) o *metadata* fica muito lento: 30-90+ min e parece parado, mas o pip está a extrair (não é a rede a falhar).",
        "  noutro shell:  watch -n3 du -s .venv-trein  (se a pasta cresce, ainda a trabalhar)  |  iotop / dstat  (I/O?)",
        "  df -h /workspace  (espaço). TMPDIR e PIP no workspace:  trein/treina.sh  já define _pip_tmp e .pip_cache no repo, ou: export TMPDIR=.../ _pip_tmp",
        "  se o plano tiver NVMe/SSD local, repõe o venv:  TREIN_VENV_DIR=/caminho/rapido/.venv-trein  ./trein/treina.sh",
        "Bash: 'Stopped' = Ctrl+Z —  fg  ou  kill %N; não suspendas o pip a meio.",
        "",
    ]


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
    freeze = _pip_freeze()
    lines: list[str] = []
    lines.extend(env_block())
    lines.append("")
    lines.extend(nvidia_smi_block())
    lines.append("")
    lines.extend(torch_block())
    lines.append("")
    extra = partial_install_block(freeze)
    if extra:
        lines.extend(extra)
    lines.extend(slow_torch_wheel_tips(freeze))
    lines.extend(pip_nvidia_sanity(freeze))
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
