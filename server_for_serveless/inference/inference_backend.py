"""Origem da inferência por mensagem (persistida em ``chat_messages.inference_backend``)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class InferenceBackend(StrEnum):
    """Valores estáveis na BD; alargar quando existirem novos backends."""

    LLAMA_SERVER = "llama_server"
    RUNPOD_SERVERLESS = "runpod_serverless"


_ALL_VALUES = frozenset(e.value for e in InferenceBackend)


def normalize_inference_backend(value: str | None) -> str | None:
    if value is None:
        return None
    t = str(value).strip()[:32]
    return t if t in _ALL_VALUES else None


def inference_backend_for_runtime(rt: Any) -> str | None:
    if not rt.is_loaded:
        return None
    b = rt.backend
    if b == "llama_server":
        return InferenceBackend.LLAMA_SERVER.value
    if b == "runpod":
        return InferenceBackend.RUNPOD_SERVERLESS.value
    return None
