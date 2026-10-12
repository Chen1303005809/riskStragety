#!/usr/bin/env python3
"""Download an artifact and decode its streamed UTF-8 bytes as Python source."""

from __future__ import annotations

import codecs
import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


CHUNK_SIZE = 64 * 1024


def download_artifact_source(base_url: str, api_key: str, artifact_id: str) -> str:
    """Read the binary response in chunks and return its UTF-8 source text."""
    request = Request(
        f"{base_url.rstrip('/')}/api/artifacts/download",
        data=json.dumps({"artifact_id": artifact_id}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
            text_chunks: list[str] = []

            while chunk := response.read(CHUNK_SIZE):
                # The incremental decoder handles a UTF-8 character split across
                # two HTTP chunks, which a per-chunk bytes.decode() cannot do.
                text_chunks.append(decoder.decode(chunk))

            text_chunks.append(decoder.decode(b"", final=True))
            return "".join(text_chunks)
    except HTTPError as error:
        body = error.read()
        try:
            payload = json.loads(body.decode("utf-8"))
            detail = payload.get("detail", payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            detail = body.decode("utf-8", errors="replace")
        raise RuntimeError(f"下载失败：HTTP {error.code}: {detail}") from error
    except UnicodeDecodeError as error:
        raise RuntimeError("下载内容不是有效的 UTF-8 文本") from error


if __name__ == "__main__":
    base_url = "http://127.0.0.1:8765"
    api_key = ""
    artifact_id = ""
    output_path = Path("strategy.py")

    try:
        source = download_artifact_source(base_url, api_key, artifact_id)
    except (OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error

    # newline="" avoids changing the source's existing line endings.
    try:
        output_path.write_text(source, encoding="utf-8", newline="")
    except OSError as error:
        print(f"保存文件失败：{error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"已保存到 {output_path}", file=sys.stderr)
