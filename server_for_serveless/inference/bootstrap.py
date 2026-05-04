"""
Carrega o backend de inferência (Runpod ou llama-server) no runtime.
Usado no arranque da app e quando o administrador desactiva «só UI» em tempo real.

A escolha vem da configuração global do admin (``app_global.runpod_serverless_enabled``):

- **Ligada:** Runpod Serverless (endpoint, chave e opções na BD; fall-back para ``ORACULO_RUNPOD_*``
  no .env se campos na BD vazios).
- **Desligada (omissão):** llama.cpp em HTTP (host/porta no admin + ``ORACULO_LLAMA_CPP_BASE_URL`` / omissão).
"""

from __future__ import annotations

import os
import sys

from server_for_serveless.db.auth_db import (
    get_runpod_server_settings,
    llama_upstream_base_url_from_db,
)
from server_for_serveless.inference.runtime import ModelRuntime

_DEFAULT_LLAMA_UPSTREAM = "http://127.0.0.1:8080"


def _llama_cpp_upstream_from_env() -> tuple[str | None, str]:
    v = (os.environ.get("ORACULO_LLAMA_CPP_BASE_URL") or "").strip().rstrip("/")
    if v:
        return v, ".env ORACULO_LLAMA_CPP_BASE_URL"
    strict = (os.environ.get("ORACULO_LLAMA_CPP_REQUIRE_EXPLICIT_URL") or "").strip().lower()
    if strict in ("1", "true", "yes", "on"):
        return None, ""
    d = _DEFAULT_LLAMA_UPSTREAM.rstrip("/")
    return d, f"omissão {d!r} (defina ORACULO_LLAMA_CPP_BASE_URL se o llama-server for noutro host/porta)"


def _missing_llama_server_msg() -> str:
    return (
        "Modo llama-server local (admin «Runpod Serverless» desligado):\n"
        "  Define host e porta nas configurações globais, ou ORACULO_LLAMA_CPP_BASE_URL no .env.\n"
        "  Se usas ORACULO_LLAMA_CPP_REQUIRE_EXPLICIT_URL=1, a URL explícita é obrigatória."
    )


def load_inference_backend(rt: ModelRuntime) -> None:
    """
    Configura ``rt`` conforme admin + .env.
    Levanta RuntimeError se faltar destino ou credenciais.
    """
    rp = get_runpod_server_settings()
    if bool(rp["serverless_enabled"]):
        eid = (rp["endpoint_id"] or os.environ.get("ORACULO_RUNPOD_ENDPOINT_ID") or "").strip()
        key = (rp["api_key"] or os.environ.get("ORACULO_RUNPOD_API_KEY") or "").strip()
        if not (eid and key):
            raise RuntimeError(
                "Runpod Serverless: preenche endpoint e API key nas configurações globais (admin) "
                "ou define ORACULO_RUNPOD_ENDPOINT_ID e ORACULO_RUNPOD_API_KEY no .env."
            )
        mid = (rp["model_id"] or os.environ.get("ORACULO_RUNPOD_MODEL_ID") or "").strip() or None
        rt.load_runpod(
            eid,
            api_key=key,
            model_id=mid,
            poll_timeout_s=float(rp["poll_timeout_s"]),
            poll_interval_s=float(rp["poll_interval_s"]),
            startup_health=bool(rp["startup_health"]),
        )
        return

    db_upstream = llama_upstream_base_url_from_db()
    env_upstream, _env_src = _llama_cpp_upstream_from_env()
    upstream = (db_upstream or env_upstream or "").strip().rstrip("/")
    if not upstream:
        print(_missing_llama_server_msg(), file=sys.stderr, flush=True)
        raise RuntimeError(
            "Sem URL do llama-server: host/porta no admin, ou ORACULO_LLAMA_CPP_BASE_URL no .env, "
            "ou retire ORACULO_LLAMA_CPP_REQUIRE_EXPLICIT_URL para usar a omissão 127.0.0.1:8080."
        )

    api_key = (os.environ.get("ORACULO_LLAMA_CPP_API_KEY") or "").strip() or None
    model_ov = (os.environ.get("ORACULO_LLAMA_CPP_MODEL") or "").strip() or None
    rt.load_llama_server(upstream, api_key=api_key, model=model_ov)
