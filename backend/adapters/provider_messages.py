"""Shared parent type for concrete providers' request-message structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ProviderWireMessage(ABC):
    """A provider-specific message that can be converted to its wire payload."""

    @abstractmethod
    def to_payload(self) -> dict[str, Any]:
        raise NotImplementedError
