"""Dependências FastAPI (sessão, admin)."""

from __future__ import annotations

import threading
import time
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from server_for_serveless.db.auth_db import is_user_admin

_user_presence_lock = threading.Lock()
_user_presence: dict[int, float] = {}
PRESENCE_ONLINE_S = 300.0


def touch_user_presence(user_id: int) -> None:
    with _user_presence_lock:
        _user_presence[int(user_id)] = time.time()


def presence_snapshot() -> dict[int, float]:
    with _user_presence_lock:
        return dict(_user_presence)


def require_user_id(request: Request) -> int:
    uid = request.session.get("user_id")
    if uid is None:
        raise HTTPException(status_code=401, detail="Não autenticado.")
    u = int(uid)
    touch_user_presence(u)
    return u


def require_admin(request: Request) -> int:
    uid = request.session.get("user_id")
    if uid is None:
        raise HTTPException(status_code=401, detail="Não autenticado.")
    u = int(uid)
    touch_user_presence(u)
    if not is_user_admin(u):
        raise HTTPException(status_code=403, detail="Apenas administrador.")
    return u


UserIdDep = Annotated[int, Depends(require_user_id)]
AdminIdDep = Annotated[int, Depends(require_admin)]
