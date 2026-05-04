"""
Runpod Serverless: um llama-server (CUDA) por worker; o handler só faz HTTP local.

Modelo: carrega uma vez no arranque do llama-server (não por pedido).

---
Relação com o admin do Oráculo (PostgreSQL / app_global)

Este worker NÃO lê a base de dados do Oráculo. Definições como llama_max_new_tokens,
llama_temperature e llama_top_p (ver server/db/auth_db.get_llama_server_settings) só
se aplicam aqui se:

1) Configurares no painel Runpod as ENV abaixo (DEFAULT_*) com os mesmos valores
   que vês no admin (cópia manual ao criar/atualizar o endpoint), ou

2) O cliente (ex.: backend Oráculo a chamar api.runpod.ai) passar em cada input:
   max_tokens, temperature, top_p, system, messages, etc.

Por omissão usamos ENV alinhadas aos defaults do admin (2048 / 0.8 / 0.9).
"""

from __future__ import annotations

import atexit
import os
import subprocess
import time
from typing import Any

import httpx
import runpod

LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8080"))
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/model.gguf")
LLAMA_CTX = os.environ.get("LLAMA_CTX", "8192")
N_GPU_LAYERS = os.environ.get("N_GPU_LAYERS", "99")

# Espelho opcional do admin (app_global); ajusta no Runpod «Environment».
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("DEFAULT_MAX_NEW_TOKENS", "2048"))
DEFAULT_TEMPERATURE = float(os.environ.get("DEFAULT_TEMPERATURE", "0.8"))
DEFAULT_TOP_P = float(os.environ.get("DEFAULT_TOP_P", "0.9"))

_server_proc: subprocess.Popen[bytes] | None = None


def _health_url() -> str:
    return f"http://127.0.0.1:{LLAMA_PORT}/health"


def _server_responds() -> bool:
    try:
        r = httpx.get(_health_url(), timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def _start_llama_server() -> None:
    global _server_proc

    if _server_responds():
        return
    if _server_proc is not None and _server_proc.poll() is None:
        return

    cmd: list[str] = [
        "/app/llama-server",
        "-m",
        MODEL_PATH,
        "--host",
        "127.0.0.1",
        "--port",
        str(LLAMA_PORT),
        "-c",
        LLAMA_CTX,
        "--n-gpu-layers",
        N_GPU_LAYERS,
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
    ]
    _server_proc = subprocess.Popen(cmd, cwd="/app")
    atexit.register(_shutdown_server)


def _shutdown_server() -> None:
    global _server_proc
    if _server_proc is None:
        return
    if _server_proc.poll() is not None:
        return
    _server_proc.terminate()
    try:
        _server_proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        _server_proc.kill()


def _wait_ready(timeout_s: float = 900.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _server_responds():
            return
        if _server_proc is not None and _server_proc.poll() is not None:
            raise RuntimeError("llama-server terminou durante o arranque (ver logs do worker)")
        time.sleep(0.25)
    raise RuntimeError("timeout à espera de /health do llama-server")


_start_llama_server()
_wait_ready()


def _num(v: Any, default: float) -> float:
    if v is None:
        return float(default)
    return float(v)


def _num_i(v: Any, default: int) -> int:
    if v is None:
        return int(default)
    return int(v)


def _build_messages(inp: dict[str, Any], user_text: str) -> list[dict[str, Any]]:
    raw = inp.get("messages")
    if isinstance(raw, list) and len(raw) > 0:
        return raw  # confiar no emissor (ex. Oráculo com histórico)
    out: list[dict[str, Any]] = []
    sys_t = inp.get("system")
    if isinstance(sys_t, str) and sys_t.strip():
        out.append({"role": "system", "content": sys_t.strip()})
    out.append({"role": "user", "content": user_text.strip()})
    return out


def handler(event: dict[str, Any]) -> dict[str, Any]:
    inp = event.get("input") or {}
    raw_msgs = inp.get("messages")
    prompt = inp.get("prompt")

    if isinstance(raw_msgs, list) and len(raw_msgs) > 0:
        messages = raw_msgs
    elif isinstance(prompt, str) and prompt.strip():
        messages = _build_messages(inp, prompt)
    else:
        return {"error": "missing_input_prompt_or_messages", "output": None}

    payload: dict[str, Any] = {
        "model": "local",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    url = f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions"
    try:
        r = httpx.post(
            url,
            json=payload,
            timeout=httpx.Timeout(600.0, connect=30.0),
        )
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
    except Exception as e:
        return {"error": str(e), "output": None}

    return {"output": text}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
