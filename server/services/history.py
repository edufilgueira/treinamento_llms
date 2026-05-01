"""
Montagem final do histórico enviado ao modelo (upstream).

- Nome do utilizador: ``display_name`` da BD se preenchido, senão ``username``.
- Data atual em português, timezone ``Europe/Lisbon`` (data do servidor quando corre).

A meta de sessão (nome + data) é acrescentada **ao último mensagem ``system``** *depois* de
``truncate_messages_to_ctx_budget``, com orçamento de tokens reservado antes da truncagem.
Assim não cria mensagem ``user`` falsa — o último pedido ``user`` continua a ser a pergunta
real para ``append_turn`` / persistência — e não é cortada primeiro como um turno velho.

Para pedidos sem utilizador web (ex. API OpenAI com ``user_id is None``), não se injeta nada.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo

from server.db.auth_db import get_user_names

_PT_WEEKDAY = (
    "segunda-feira",
    "terça-feira",
    "quarta-feira",
    "quinta-feira",
    "sexta-feira",
    "sábado",
    "domingo",
)

_PT_MONTH = (
    "janeiro",
    "fevereiro",
    "março",
    "abril",
    "maio",
    "junho",
    "julho",
    "agosto",
    "setembro",
    "outubro",
    "novembro",
    "dezembro",
)

_TZ = ZoneInfo("Europe/Lisbon")

# Pequeno colchão sobre a heurística de tokens do bloco (llama_context).
_RESERVE_TOKEN_BUFFER = 24


def data_atual_para_modelo(now: datetime | None = None) -> str:
    """Ex.: ``sexta-feira, 01 de maio de 2026`` (calendário Lisboa)."""
    dt = now or datetime.now(_TZ)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_TZ)
    else:
        dt = dt.astimezone(_TZ)
    w = _PT_WEEKDAY[dt.weekday()]
    m = _PT_MONTH[dt.month - 1]
    day = f"{dt.day:02d}"
    return f"{w}, {day} de {m} de {dt.year}"


def nome_utilizador_para_modelo(user_id: int) -> str:
    row = get_user_names(int(user_id))
    if not row:
        return "Utilizador"
    d = (row.get("display_name") or "").strip()
    if d:
        return d
    u = str(row.get("username") or "").strip()
    return u or "Utilizador"


def session_context_block_text(user_id: int) -> str:
    """Texto curto só para orientação interna ao modelo."""
    nome = nome_utilizador_para_modelo(int(user_id))
    data_s = data_atual_para_modelo()
    return (
        "[Contexto da sessão (não é uma pergunta do utilizador). "
        f"Nome do interlocutor: «{nome}». Data atual: {data_s}.]"
    )


def append_session_snippet_to_system_messages(
    messages: list[dict], snippet: str
) -> list[dict]:
    """Acrescenta ``snippet`` ao último ``system``. Se não houver ``system``, cria um novo."""
    out = deepcopy(messages)
    sep = "\n\n"
    last_sys_i: int | None = None
    for i, m in enumerate(out):
        if str(m.get("role") or "").lower() == "system":
            last_sys_i = int(i)
    if last_sys_i is not None:
        m = out[last_sys_i]
        base = str(m.get("content") or "")
        out[last_sys_i] = {
            **m,
            "content": ((base + sep + snippet) if base.strip() else snippet).strip(),
        }
        return out
    out.insert(0, {"role": "system", "content": snippet})
    return out


def finalize_messages_for_upstream(
    messages: list[dict],
    *,
    user_id: int | None,
    max_new_tokens: int,
) -> list[dict]:
    """
    Trunca ao orçamento de contexto e injeta nome + data atuais (``user_id`` conhecido).

    Usar no caminho único antes de ``chat_completions_*`` quando há utilizador na sessão web.
    """
    from server.db.auth_db import get_llama_server_settings
    from server.services.llama_context import (
        cap_max_new_tokens_for_n_ctx,
        estimate_message_tokens,
        truncate_messages_to_ctx_budget,
    )

    ls = get_llama_server_settings()
    n_ctx = int(ls["n_ctx"])
    mnt_eff = cap_max_new_tokens_for_n_ctx(int(n_ctx), int(max_new_tokens))

    snippet = ""
    reserve = 0
    if user_id is not None:
        snippet = session_context_block_text(int(user_id))
        reserve = estimate_message_tokens({"role": "system", "content": snippet})
        reserve = min(max(1, reserve + _RESERVE_TOKEN_BUFFER), max(256, n_ctx // 8))

    truncated = truncate_messages_to_ctx_budget(
        messages,
        n_ctx=n_ctx,
        max_new_tokens=mnt_eff,
        reserve_prompt_tokens=reserve,
    )
    if user_id is not None and snippet:
        truncated = append_session_snippet_to_system_messages(truncated, snippet)
    return truncated
