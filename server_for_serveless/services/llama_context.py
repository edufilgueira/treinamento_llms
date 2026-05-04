"""
Orçamento de contexto alinhado ao n_ctx do admin: trunca histórico antes do llama-server.

Regra: tokens estimados do prompt ≤ (n_ctx − max_tokens). Heurística ~2 chars/token no texto.
"""

from __future__ import annotations

from copy import deepcopy

from server_for_serveless.db.auth_db import get_llama_server_settings

_ROLE_OVERHEAD_TOKENS = 12
_CHARS_PER_TOKEN_EST = 2


def max_allowed_max_new_tokens_for_n_ctx(n_ctx: int) -> int:
    """Teto de geração: max_tokens < n_ctx (sobra pelo menos 1 token lógico para o prompt)."""
    n_ctx = max(256, int(n_ctx))
    return max(16, n_ctx - 1)


def cap_max_new_tokens_for_n_ctx(n_ctx: int, max_new_tokens: int) -> int:
    mnt = max(16, int(max_new_tokens))
    return min(mnt, max_allowed_max_new_tokens_for_n_ctx(int(n_ctx)))


def estimate_message_tokens(msg: dict) -> int:
    role = str(msg.get("role") or "user").lower()
    content = str(msg.get("content") or "")
    base = max(
        1,
        (len(content) + _CHARS_PER_TOKEN_EST - 1) // _CHARS_PER_TOKEN_EST
        + _ROLE_OVERHEAD_TOKENS,
    )
    if role == "system":
        base += 4
    return base


def total_messages_estimate(messages: list[dict]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


def _total_content_chars(messages: list[dict]) -> int:
    return sum(len(str(m.get("content") or "")) for m in messages)


def _prompt_token_budget(
    *, n_ctx: int, max_new_tokens: int, reserve_prompt_tokens: int = 0
) -> int:
    """Sobra para o prompt; ``reserve_prompt_tokens`` reserva espaço (ex.: bloco dinâmico pós-truncagem)."""
    n_ctx = max(256, int(n_ctx))
    mnt = max(16, int(max_new_tokens))
    rp = max(0, int(reserve_prompt_tokens))
    return max(1, n_ctx - mnt - rp)


def truncate_messages_to_ctx_budget(
    messages: list[dict],
    *,
    n_ctx: int,
    max_new_tokens: int,
    reserve_prompt_tokens: int = 0,
) -> list[dict]:
    """Remove turnos antigos até a estimativa do prompt caber em (n_ctx − max_tokens).

    ``reserve_prompt_tokens`` reduz o orçamento útil antes de cortar mensagens —
    usar quando, após truncar, se acrescenta texto ao prompt (ex.: contexto sessão).
    """
    if not messages:
        return messages
    prompt_budget = _prompt_token_budget(
        n_ctx=n_ctx,
        max_new_tokens=max_new_tokens,
        reserve_prompt_tokens=reserve_prompt_tokens,
    )

    msgs = deepcopy(messages)
    system_msgs = [m for m in msgs if str(m.get("role") or "").lower() == "system"]
    rest = [m for m in msgs if str(m.get("role") or "").lower() != "system"]

    def fits_pre_combined() -> bool:
        return total_messages_estimate(system_msgs + rest) <= prompt_budget

    while system_msgs and not fits_pre_combined():
        if len(system_msgs) == 1:
            sm = system_msgs[0]
            c = str(sm.get("content") or "")
            need = total_messages_estimate(system_msgs) - prompt_budget
            drop_chars = min(len(c), max(0, need * _CHARS_PER_TOKEN_EST + 50))
            if drop_chars >= len(c):
                system_msgs = []
            else:
                system_msgs[0] = {
                    **sm,
                    "content": "…\n" + c[drop_chars:],
                }
            break
        system_msgs = system_msgs[1:]

    while rest and not fits_pre_combined():
        if len(rest) <= 1:
            break
        rest = rest[1:]

    combined: list[dict] = system_msgs + rest

    def fits_combined() -> bool:
        return total_messages_estimate(combined) <= prompt_budget

    char_cap = max(256, prompt_budget * _CHARS_PER_TOKEN_EST)

    def shrink_until_fits() -> None:
        nonlocal combined
        for _ in range(4096):
            if not combined:
                return
            if fits_combined() and _total_content_chars(combined) <= char_cap:
                return
            non_sys_idx = None
            for i in range(len(combined) - 1, -1, -1):
                if str(combined[i].get("role") or "").lower() != "system":
                    non_sys_idx = i
                    break
            if non_sys_idx is None:
                sm = combined[0]
                c = str(sm.get("content") or "")
                if len(c) <= 80:
                    return
                cut = max(1, len(c) // 5)
                combined[0] = {**sm, "content": "…\n" + c[cut:]}
                continue
            msg = combined[non_sys_idx]
            c = str(msg.get("content") or "")
            if len(c) > 120:
                cut = max(1, len(c) // 4)
                combined[non_sys_idx] = {**msg, "content": "…\n" + c[cut:]}
                continue
            if len(combined) > 1:
                combined.pop(non_sys_idx)
                continue
            return

    shrink_until_fits()

    if not combined:
        return combined

    last = combined[-1]
    if str(last.get("role") or "").lower() != "system":
        c = str(last.get("content") or "")
        for _ in range(256):
            if fits_combined() and _total_content_chars(combined) <= char_cap:
                break
            if len(c) <= 1:
                break
            over = max(1, total_messages_estimate(combined) - prompt_budget)
            drop = min(len(c), over * _CHARS_PER_TOKEN_EST + 48)
            c = ("…\n" + c[drop:]) if drop < len(c) else "…"
            combined[-1] = {**last, "content": c}
    return combined


def prepare_messages_for_llama_upstream(
    messages: list[dict],
    *,
    max_new_tokens: int,
) -> list[dict]:
    ls = get_llama_server_settings()
    return truncate_messages_to_ctx_budget(
        messages,
        n_ctx=int(ls["n_ctx"]),
        max_new_tokens=max_new_tokens,
    )
