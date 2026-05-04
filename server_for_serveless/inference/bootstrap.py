"""
Carrega o backend de inferência (Runpod ou llama-server) no runtime.
Usado no arranque da app e quando o administrador desactiva «só UI» em tempo real.
"""

from __future__ import annotations

import os
import sys

from server_for_serveless.db.auth_db import llama_upstream_base_url_from_db
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
        "Inferência: o Oráculo só delega ao llama-server.\n"
        "  Tens ORACULO_LLAMA_CPP_REQUIRE_EXPLICIT_URL=1 mas não definiste ORACULO_LLAMA_CPP_BASE_URL.\n"
        "  Ou no admin activa «Usar llama-server» com host/porta, ou no .env define por exemplo:\n"
        "    ORACULO_LLAMA_CPP_BASE_URL=http://127.0.0.1:8080\n"
        "  Modo só UI: activar «Só interface» nas configurações globais (admin)."
    )


def load_inference_backend(rt: ModelRuntime) -> None:
    """
    Configura ``rt`` a partir de ORACULO_RUNPOD_* ou llama-server (DB admin + .env).
    Levanta RuntimeError se não houver destino válido.
    """
    runpod_eid = (os.environ.get("ORACULO_RUNPOD_ENDPOINT_ID") or "").strip()
    runpod_key = (os.environ.get("ORACULO_RUNPOD_API_KEY") or "").strip()

    if runpod_eid or runpod_key:
        if not (runpod_eid and runpod_key):
            raise RuntimeError(
                "Runpod Serverless: ORACULO_RUNPOD_ENDPOINT_ID e ORACULO_RUNPOD_API_KEY "
                "têm de estar ambos definidos."
            )
        model_r = (os.environ.get("ORACULO_RUNPOD_MODEL_ID") or "").strip() or None
        rt.load_runpod(runpod_eid, api_key=runpod_key, model_id=model_r)
        return

    db_upstream = llama_upstream_base_url_from_db()
    env_upstream, _env_src = _llama_cpp_upstream_from_env()
    upstream = db_upstream or env_upstream
    if not upstream:
        print(_missing_llama_server_msg(), file=sys.stderr, flush=True)
        raise RuntimeError(
            "Sem URL do llama-server: ORACULO_LLAMA_CPP_BASE_URL, ou admin, ou retire "
            "ORACULO_LLAMA_CPP_REQUIRE_EXPLICIT_URL para usar a omissão 127.0.0.1:8080. "
            "Ou define ORACULO_RUNPOD_ENDPOINT_ID + ORACULO_RUNPOD_API_KEY para Runpod."
        )

    api_key = (os.environ.get("ORACULO_LLAMA_CPP_API_KEY") or "").strip() or None
    model_ov = (os.environ.get("ORACULO_LLAMA_CPP_MODEL") or "").strip() or None
    rt.load_llama_server(upstream, api_key=api_key, model=model_ov)
