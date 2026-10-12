"""Run the combined backend and built frontend."""

from __future__ import annotations

import uvicorn

from backend.app import create_app
from backend.config import Settings


def main() -> None:
    settings = Settings.from_env()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
