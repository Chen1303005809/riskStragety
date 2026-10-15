#!/usr/bin/env python3
"""Download an artifact and decode its streamed UTF-8 bytes as Python source."""

from __future__ import annotations

import ast
import codecs
import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


CHUNK_SIZE = 64 * 1024


def strategy_filename(source: str) -> str:
    """Derive the .py filename from the strategy class's name attribute."""
    source_tree = ast.parse(source)
    strategy_classes = [
        node
        for node in source_tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(base, ast.Name) and base.id == "RHTemplate"
            for base in node.bases
        )
    ]
    if len(strategy_classes) != 1:
        raise ValueError("源码必须包含且只包含一个直接继承 RHTemplate 的顶层策略类")
    name_values = [
        statement.value
        for statement in strategy_classes[0].body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name) and target.id == "name"
    ]
    name_values.extend(
        statement.value
        for statement in strategy_classes[0].body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.target.id == "name"
        and statement.value is not None
    )
    if len(name_values) != 1:
        raise ValueError("策略类必须且只能定义一次 name 字符串属性")
    name = ast.literal_eval(name_values[0])
    if (
        not isinstance(name, str)
        or not name
        or name != name.strip()
        or name.startswith(".")
        or name.lower().endswith(".py")
        or "/" in name
        or "\\" in name
        or any(ord(char) < 32 or ord(char) == 127 for char in name)
    ):
        raise ValueError("策略类的 name 属性必须是有效的纯文件名，且不含 .py 后缀")
    filename = f"{name}.py"
    if len(filename) > 120:
        raise ValueError("策略文件名不能超过 120 个字符")
    return filename


def download_artifact_source(
    base_url: str, api_key: str, account_id: str, artifact_id: str
) -> str:
    """Read the binary response in chunks and return its UTF-8 source text."""
    request = Request(
        f"{base_url.rstrip('/')}/api/artifacts/download",
        data=json.dumps({"artifact_id": artifact_id, "account_id": account_id}).encode("utf-8"),
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
    account_id = ""
    artifact_id = ""

    try:
        source = download_artifact_source(base_url, api_key, account_id, artifact_id)
        output_path = Path(strategy_filename(source))
    except (OSError, RuntimeError, SyntaxError, ValueError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error

    # newline="" avoids changing the source's existing line endings.
    try:
        output_path.write_text(source, encoding="utf-8", newline="")
    except OSError as error:
        print(f"保存文件失败：{error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"已保存到 {output_path}", file=sys.stderr)
