"""Preferências de chat e mensagens com system prompt."""

from __future__ import annotations

from server_for_serveless.db.auth_db import (
    get_global_system_prompt,
    get_llama_server_settings,
    get_user_model_settings,
    is_user_admin,
)
from server_for_serveless.inference.runtime import get_runtime

DEFAULT_MAX_NEW = 2048
DEFAULT_TEMP = 0.7
DEFAULT_TOP_P = 0.9


def merge_system_blocks(global_text: str, user_text: str) -> str | None:
    g = (global_text or "").strip()
    u = (user_text or "").strip()
    if g and u:
        return f"{g}\n\n{u}"
    if g:
        return g
    if u:
        return u
    return None


def chat_messages_for_user(user_id: int, base: list[dict]) -> list[dict]:
    prefs = get_user_model_settings(int(user_id))
    gsp = get_global_system_prompt()
    no_client_system = [m for m in base if m.get("role") != "system"]
    usp = (prefs.get("system_prompt") or "").strip()
    merged = merge_system_blocks(gsp, usp)
    if merged:
        return [{"role": "system", "content": merged}, *no_client_system]
    return no_client_system


def infer_params_for_user(user_id: int) -> tuple[int, float, float]:
    rt = get_runtime()
    if rt.backend in ("llama_server", "runpod") or is_user_admin(int(user_id)):
        ls = get_llama_server_settings()
        return (
            int(ls["max_new_tokens"]),
            float(ls["temperature"]),
            float(ls["top_p"]),
        )
    return DEFAULT_MAX_NEW, DEFAULT_TEMP, DEFAULT_TOP_P


def model_unavailable_detail() -> str:
    rt = get_runtime()
    if rt.ui_only:
        return "Modo --ui-only (sem modelo). Suba o servidor sem --ui-only para o chat."
    return "Modelo ainda não carregado."
