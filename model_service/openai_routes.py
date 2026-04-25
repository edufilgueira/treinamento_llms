"""
API compatível com OpenAI Chat Completions (subconjunto usado por clientes comuns).

POST /v1/chat/completions — corpo estilo OpenAI; resposta e SSE alinhados ao esperado
por bibliotecas que usam base_url .../v1 e Authorization: Bearer.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .runtime import get_runtime

router = APIRouter(prefix="/v1", tags=["openai"])


def _openai_api_key() -> str:
    return (os.environ.get("ORACULO_OPENAI_API_KEY") or "").strip()


def _verify_bearer(authorization: str | None) -> None:
    expected = _openai_api_key()
    if not expected:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid or missing Authorization header.")
    got = authorization[7:].strip()
    if got != expected:
        raise HTTPException(status_code=401, detail="Invalid API key.")


class OAIChatMessage(BaseModel):
    role: str
    content: str | None = None

    @field_validator("role")
    @classmethod
    def _role_ok(cls, v: str) -> str:
        r = (v or "").strip().lower()
        if r not in ("system", "user", "assistant"):
            raise ValueError("role must be system, user, or assistant.")
        return r


class OAIChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = Field(default="default", max_length=256)
    messages: list[OAIChatMessage] = Field(..., min_length=1)
    max_tokens: int | None = Field(default=2048, ge=1, le=4096)
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float | None = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False


def _messages_to_dicts(body: OAIChatCompletionRequest) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in body.messages:
        if m.content is None:
            raise HTTPException(
                status_code=400,
                detail="Null message content is not supported (tool calls not implemented).",
            )
        out.append({"role": m.role, "content": m.content})
    return out


def _prompt_token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(ids, list):
        return len(ids)
    if isinstance(ids, dict):
        t = ids.get("input_ids")
        if t is not None and hasattr(t, "shape"):
            return int(t.shape[-1])
        return 0
    if hasattr(ids, "shape"):
        return int(ids.shape[-1])
    return 0


def _completion_tokens(tokenizer: Any, text: str) -> int:
    if not (text or "").strip():
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def _echo_model_id(req_model: str, loaded_id: str) -> str:
    if req_model and req_model != "default":
        return req_model
    return loaded_id or "local"


@router.get("/models")
async def list_models(authorization: Annotated[str | None, Header()] = None) -> dict[str, Any]:
    _verify_bearer(authorization)
    rt = get_runtime()
    if not rt.is_loaded:
        return {"object": "list", "data": []}
    mid = rt.model_id or "local"
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": mid,
                "object": "model",
                "created": now,
                "owned_by": "local",
            }
        ],
    }


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: Request,
    body: OAIChatCompletionRequest,
    authorization: Annotated[str | None, Header()] = None,
):
    _verify_bearer(authorization)
    rt = get_runtime()
    if rt.ui_only or not rt.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded (ui-only mode or load error).",
        )
    messages = _messages_to_dicts(body)
    tok = rt.tokenizer
    model = rt.model
    assert tok is not None and model is not None
    max_new = int(body.max_tokens or 2048)
    temp = float(body.temperature if body.temperature is not None else 0.7)
    top_p = float(body.top_p if body.top_p is not None else 0.9)
    resp_model = _echo_model_id(body.model, rt.model_id)
    prompt_toks = _prompt_token_count(tok, messages)

    if body.stream:

        def ndjson_stream():
            cid = f"chatcmpl-{secrets.token_hex(12)}"
            created = int(time.time())
            first = True
            cancel_event = threading.Event()
            acc = ""
            try:
                for delta in rt.stream(
                    messages,
                    max_new_tokens=max_new,
                    temperature=temp,
                    top_p=top_p,
                    cancel_event=cancel_event,
                    user_id=None,
                ):
                    acc += delta
                    if first:
                        chunk = {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": resp_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": delta},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        first = False
                    else:
                        chunk = {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": resp_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": delta},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                final = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": resp_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
            except Exception as err:  # noqa: BLE001
                err_chunk = {"error": {"message": str(err), "type": "server_error"}}
                yield f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n".encode("utf-8")

        return StreamingResponse(
            ndjson_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        content = rt.generate(
            messages,
            max_new_tokens=max_new,
            temperature=temp,
            top_p=top_p,
            user_id=None,
        )
    except Exception as err:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(err)) from err

    comp_toks = _completion_tokens(tok, content)
    cid = f"chatcmpl-{secrets.token_hex(12)}"
    created = int(time.time())
    payload = {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": resp_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_toks,
            "completion_tokens": comp_toks,
            "total_tokens": prompt_toks + comp_toks,
        },
    }
    return JSONResponse(content=payload)
