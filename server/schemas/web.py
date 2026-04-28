"""Esquemas Pydantic da API web (chat, auth, definições)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    messages: list[Message] = Field(..., min_length=1)
    max_new_tokens: int = Field(2048, ge=16, le=32768)
    temperature: float = Field(0.7, ge=0.01, le=2.0)
    top_p: float = Field(0.9, ge=0.05, le=1.0)


class ChatJobIn(BaseModel):
    messages: list[Message] = Field(..., min_length=1)
    max_new_tokens: int = Field(2048, ge=16, le=32768)
    temperature: float = Field(0.7, ge=0.01, le=2.0)
    top_p: float = Field(0.9, ge=0.05, le=1.0)
    session_id: int | None = None


class ChatOut(BaseModel):
    reply: str


class JobCreateOut(BaseModel):
    job_id: str


class JobStateOut(BaseModel):
    status: str
    text: str
    error: str | None = None
    output_tokens: int | None = None
    gen_seconds: float | None = None
    tokens_per_sec: float | None = None


class SessionTitleUpdate(BaseModel):
    title: str = Field(..., max_length=200)


class UserProfileUpdate(BaseModel):
    display_name: str = Field("", max_length=64)
    email: str = Field("", max_length=254)


class LlamaServerSettingsIn(BaseModel):
    upstream_enabled: bool | None = None
    api_host: str | None = Field(None, max_length=256)
    api_port: int | None = Field(None, ge=1, le=65535)
    n_ctx: int | None = Field(None, ge=256, le=1_000_000)
    max_new_tokens: int | None = Field(None, ge=16, le=32768)
    temperature: float | None = Field(None, ge=0.01, le=2.0)
    top_p: float | None = Field(None, ge=0.05, le=1.0)
    repeat_penalty: float | None = Field(None, ge=0.5, le=2.0)
    repeat_last_n: int | None = Field(None, ge=0, le=131072)
    reasoning: str | None = Field(None, max_length=16)
    reasoning_budget: int | None = Field(None, ge=-1, le=1_000_000)

    @field_validator("reasoning")
    @classmethod
    def _reasoning_ok(cls, v: str | None) -> str | None:
        if v is None:
            return None
        t = (v or "").strip().lower()
        if t not in ("off", "on", "auto"):
            raise ValueError("reasoning: use off, on ou auto.")
        return t


class LlamaServerSettingsOut(BaseModel):
    upstream_enabled: bool
    api_host: str
    api_port: int
    n_ctx: int
    max_new_tokens: int
    temperature: float
    top_p: float
    repeat_penalty: float
    repeat_last_n: int
    reasoning: str
    reasoning_budget: int


class UserModelSettingsIn(BaseModel):
    system_prompt: str | None = None
    global_system_prompt: str | None = None
    # max_new_tokens / temperature / top_p: ignorados no PATCH (admin); usar body.llama → app_global.
    max_new_tokens: int | None = Field(None, ge=16, le=32768)
    temperature: float | None = Field(None, ge=0.01, le=2.0)
    top_p: float | None = Field(None, ge=0.05, le=1.0)
    llama: LlamaServerSettingsIn | None = None


class UserModelSettingsOut(BaseModel):
    is_admin: bool
    system_prompt: str
    global_system_prompt: str
    max_new_tokens: int
    temperature: float
    top_p: float
    llama_server: LlamaServerSettingsOut | None = None


class AuthIn(BaseModel):
    username: str = Field(..., max_length=32)
    password: str = Field(..., max_length=128)

    @field_validator("username")
    @classmethod
    def _username_len(cls, v: str) -> str:
        t = (v or "").strip()
        if len(t) < 3 or len(t) > 32:
            raise ValueError("Utilizador: entre 3 e 32 caracteres.")
        return t

    @field_validator("password")
    @classmethod
    def _password_len(cls, v: str) -> str:
        if (v is None) or (len(v) < 8):
            raise ValueError("Palavra-passe: pelo menos 8 caracteres.")
        if len(v) > 128:
            raise ValueError("Palavra-passe: no máximo 128 caracteres.")
        return v
