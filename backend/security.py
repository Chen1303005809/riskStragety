"""Bearer API key authentication for the shared service caller."""

from __future__ import annotations

from dataclasses import dataclass
from secrets import compare_digest

from fastapi import Header, HTTPException, Request


@dataclass(frozen=True)
class Principal:
    """A caller authenticated with the shared service API key."""


async def require_principal(
    request: Request, authorization: str | None = Header(default=None)
) -> Principal:
    scheme, _, credential = (authorization or "").partition(" ")
    expected = request.app.state.settings.api_key
    if (
        scheme.lower() != "bearer"
        or not credential
        or not compare_digest(credential.encode("utf-8"), expected.encode("utf-8"))
    ):
        raise HTTPException(
            status_code=401,
            detail="API Key 无效",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Principal()
