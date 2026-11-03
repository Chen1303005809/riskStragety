"""Uploaded XLSX storage, parsing, ownership, and privacy boundaries."""

from __future__ import annotations

import asyncio
import json
import zipfile
from collections.abc import AsyncIterator
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree

import pytest
from fastapi.testclient import TestClient
from openpyxl import Workbook

from agent.contracts import AgentMessage, ModelChunk, ModelGateway, ToolCall, ToolResultMessage
from backend.adapters.excel_files import (
    MAX_EXCEL_UPLOAD_BYTES,
    ExcelFileError,
    ExcelFileStore,
)
from backend.adapters.monitoring import (
    ExcelAttachmentNotFound,
    MonitoringStore,
)
from backend.config import Settings
from backend.app import create_app


KEY = "excel-test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}
CONVERSATION_ID = "550e8400-e29b-41d4-a716-446655440000"
SENSITIVE_VALUE = "XLSX-SENSITIVE-CELL-8f3e2d"


def workbook_bytes(*, formula: bool = False, second_sheet: bool = True) -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "数据"
    sheet["A1"] = "名称"
    sheet["B1"] = "数值"
    sheet["A2"] = SENSITIVE_VALUE
    sheet["B2"] = "=SUM(B3:B3)" if formula else 12
    sheet["B3"] = 15
    if second_sheet:
        workbook.create_sheet("空表")
    output = BytesIO()
    workbook.save(output)
    return output.getvalue()


def workbook_bytes_with_cached_formula() -> bytes:
    source = workbook_bytes(formula=True, second_sheet=True)
    output = BytesIO()
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    with zipfile.ZipFile(BytesIO(source)) as original, zipfile.ZipFile(output, "w") as updated:
        for item in original.infolist():
            content = original.read(item.filename)
            if item.filename == "xl/worksheets/sheet1.xml":
                root = ElementTree.fromstring(content)
                cell = root.find(f".//{{{namespace}}}c[@r='B2']")
                assert cell is not None
                cached_value = cell.find(f"{{{namespace}}}v")
                assert cached_value is not None
                cached_value.text = "27"
                content = ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)
            updated.writestr(item, content)
    return output.getvalue()


def make_stores(tmp_path: Path) -> tuple[MonitoringStore, ExcelFileStore]:
    artifact_directory = tmp_path / "artifacts"
    artifact_directory.mkdir()
    monitoring = MonitoringStore(tmp_path / "monitoring.sqlite3", artifact_directory)
    store = ExcelFileStore(tmp_path / "excel_uploads", monitoring)
    return monitoring, store


def test_xlsx_upload_is_private_scoped_and_reads_sheets_ranges_and_cached_formulas(
    tmp_path: Path,
) -> None:
    monitoring, store = make_stores(tmp_path)
    asyncio.run(monitoring.prepare_conversation("alice", CONVERSATION_ID, "读取表格"))
    upload = asyncio.run(store.save_upload(
        "alice", "风险数据.xlsx", workbook_bytes_with_cached_formula()
    ))
    attachment_id = upload["attachment_id"]
    file_path = tmp_path / "excel_uploads" / f"{attachment_id}.xlsx"

    assert file_path.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "excel_uploads").stat().st_mode & 0o777 == 0o700
    assert asyncio.run(store.list_files("alice", CONVERSATION_ID)) == []

    asyncio.run(store.bind_uploads("alice", CONVERSATION_ID, [attachment_id]))
    overview = asyncio.run(store.read_workbook("alice", CONVERSATION_ID, attachment_id))
    assert [sheet["name"] for sheet in overview["sheets"]] == ["数据", "空表"]

    rows = asyncio.run(store.read_workbook(
        "alice", CONVERSATION_ID, attachment_id, "数据", "A1:B3"
    ))
    assert rows["range"] == "A1:B3"
    assert rows["row_count"] == 3
    assert rows["values"][1] == [SENSITIVE_VALUE, 27]
    assert rows["formulas"][1][1] == "=SUM(B3:B3)"
    assert asyncio.run(store.list_files("alice", CONVERSATION_ID))[0]["filename"] == "风险数据.xlsx"


def test_formula_without_cached_value_falls_back_to_formula_text(tmp_path: Path) -> None:
    monitoring, store = make_stores(tmp_path)
    asyncio.run(monitoring.prepare_conversation("alice", CONVERSATION_ID, "读取公式"))
    upload = asyncio.run(store.save_upload("alice", "formula.xlsx", workbook_bytes(formula=True)))
    asyncio.run(store.bind_uploads("alice", CONVERSATION_ID, [upload["attachment_id"]]))

    rows = asyncio.run(store.read_workbook(
        "alice", CONVERSATION_ID, upload["attachment_id"], "数据", "B2:B2"
    ))
    assert rows["values"] == [["=SUM(B3:B3)"]]
    assert rows["formulas"] == [["=SUM(B3:B3)"]]


def test_rejects_unsupported_corrupt_oversized_ranges_and_unknown_files(tmp_path: Path) -> None:
    monitoring, store = make_stores(tmp_path)
    asyncio.run(monitoring.prepare_conversation("alice", CONVERSATION_ID, "读取表格"))

    with pytest.raises(ExcelFileError, match="仅支持"):
        asyncio.run(store.save_upload("alice", "legacy.xls", b"not xlsx"))
    with pytest.raises(ExcelFileError, match="超过 20 MiB"):
        asyncio.run(store.save_upload("alice", "large.xlsx", b"x" * (MAX_EXCEL_UPLOAD_BYTES + 1)))
    with pytest.raises(ExcelFileError, match="无法解析"):
        asyncio.run(store.save_upload("alice", "corrupt.xlsx", b"not a zip workbook"))

    upload = asyncio.run(store.save_upload("alice", "data.xlsx", workbook_bytes()))
    asyncio.run(store.bind_uploads("alice", CONVERSATION_ID, [upload["attachment_id"]]))
    with pytest.raises(ExcelFileError, match="最多读取 1,000"):
        asyncio.run(store.read_workbook(
            "alice", CONVERSATION_ID, upload["attachment_id"], "数据", "A1:AK30"
        ))
    with pytest.raises(ExcelFileError, match="单个 A1 范围"):
        asyncio.run(store.read_workbook(
            "alice", CONVERSATION_ID, upload["attachment_id"], "数据", "A1,B2"
        ))
    with pytest.raises(ExcelFileError, match="不存在或不属于"):
        asyncio.run(store.read_workbook("alice", CONVERSATION_ID, "0" * 32))


def test_pending_uploads_expire_and_cross_account_conversation_binding_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import backend.adapters.excel_files as excel_module

    monitoring, store = make_stores(tmp_path)
    asyncio.run(monitoring.prepare_conversation("alice", CONVERSATION_ID, "读取表格"))
    other_conversation = "650e8400-e29b-41d4-a716-446655440000"
    asyncio.run(monitoring.prepare_conversation("bob", other_conversation, "另一个会话"))

    upload = asyncio.run(store.save_upload("alice", "data.xlsx", workbook_bytes()))
    with pytest.raises(ExcelAttachmentNotFound):
        asyncio.run(store.bind_uploads("bob", other_conversation, [upload["attachment_id"]]))
    asyncio.run(store.bind_uploads("alice", CONVERSATION_ID, [upload["attachment_id"]]))
    assert asyncio.run(store.list_files("bob", other_conversation)) == []

    monkeypatch.setattr(excel_module, "PENDING_UPLOAD_TTL", excel_module.timedelta(seconds=-1))
    expired = asyncio.run(store.save_upload("alice", "expired.xlsx", workbook_bytes()))
    expired_path = tmp_path / "excel_uploads" / f"{expired['attachment_id']}.xlsx"
    assert expired_path.is_file()
    assert asyncio.run(store.purge_expired()) == 1
    assert not expired_path.exists()
    with pytest.raises(ExcelAttachmentNotFound):
        asyncio.run(store.bind_uploads("alice", CONVERSATION_ID, [expired["attachment_id"]]))


class ExcelReadingModel(ModelGateway):
    def __init__(self) -> None:
        self.main_step = 0
        self.read_result: dict[str, object] | None = None

    async def list_models(self) -> list[str]:
        return ["model-a"]

    async def stream_chat(
        self,
        model: str,
        messages: list[AgentMessage],
        tools: list[dict[str, object]],
        *,
        options: dict[str, int | float],
        think: bool,
    ) -> AsyncIterator[ModelChunk]:
        if len(tools) == 1 and tools[0]["function"]["name"] == "lookup_exchange_by_contract_code":
            yield ModelChunk()
            return
        self.main_step += 1
        if self.main_step == 1:
            yield ModelChunk(tool_calls=(ToolCall("list_excel_files", {}),))
        elif self.main_step == 2:
            result_message = next(
                item for item in reversed(messages)
                if isinstance(item, ToolResultMessage) and item.tool_name == "list_excel_files"
            )
            file_id = json.loads(result_message.content)["files"][0]["attachment_id"]
            yield ModelChunk(tool_calls=(ToolCall("read_excel", {
                "file_id": file_id,
                "sheet_name": "数据",
                "cell_range": "A1:A2",
            }),))
        else:
            result_message = next(
                item for item in reversed(messages)
                if isinstance(item, ToolResultMessage) and item.tool_name == "read_excel"
            )
            self.read_result = json.loads(result_message.content)
            # Deliberately echo a workbook value to ensure model trace logs redact it too.
            yield ModelChunk(content=self.read_result["values"][1][0])


class NoopFunctionalTester:
    async def run(self, source, *, requirements, model, options):
        return {"passed": True, "case_count": 0, "cases": []}


def test_upload_chat_read_and_resume_exposes_files_without_logging_cell_values(
    tmp_path: Path,
) -> None:
    model = ExcelReadingModel()
    settings = Settings(
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="model-a",
        api_key=KEY,
        project_root=tmp_path,
        app_data_dir=tmp_path / "data",
    )
    client = TestClient(create_app(settings, model_gateway=model, functional_tester=NoopFunctionalTester()))
    with client:
        unauthorized = client.post(
            "/api/agent/uploads",
            data={"account_id": "alice"},
            files={"file": ("data.xlsx", workbook_bytes(), "application/octet-stream")},
        )
        assert unauthorized.status_code == 401

        uploaded = client.post(
            "/api/agent/uploads",
            headers=AUTH,
            data={"account_id": "alice"},
            files={"file": ("data.xlsx", workbook_bytes(), "application/octet-stream")},
        )
        assert uploaded.status_code == 200
        attachment = uploaded.json()
        response = client.post("/api/agent/chat/stream", headers=AUTH, json={
            "account_id": "alice",
            "conversation_id": CONVERSATION_ID,
            "message": "读取 Excel 的数据",
            "model": "model-a",
            "attachment_ids": [attachment["attachment_id"]],
        })
        assert response.status_code == 200
        assert model.read_result is not None
        assert model.read_result["values"][1][0] == SENSITIVE_VALUE

        detail = client.get(
            f"/api/monitor/accounts/alice/conversations/{CONVERSATION_ID}", headers=AUTH
        )
        assert detail.status_code == 200
        assert detail.json()["excel_attachments"][0]["filename"] == "data.xlsx"

        trace_path = tmp_path / "data" / "conversation_logs" / "alice" / f"{CONVERSATION_ID}.jsonl"
        trace = trace_path.read_text(encoding="utf-8")
        assert SENSITIVE_VALUE not in trace
        assert attachment["attachment_id"] in trace
        assert "A1:A2" in trace


def test_upload_rejects_xls_and_chat_rejects_another_accounts_attachment(tmp_path: Path) -> None:
    settings = Settings(
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="model-a",
        api_key=KEY,
        project_root=tmp_path,
        app_data_dir=tmp_path / "data",
    )
    client = TestClient(create_app(
        settings,
        model_gateway=ExcelReadingModel(),
        functional_tester=NoopFunctionalTester(),
    ))
    with client:
        old_format = client.post(
            "/api/agent/uploads",
            headers=AUTH,
            data={"account_id": "alice"},
            files={"file": ("legacy.xls", b"legacy", "application/octet-stream")},
        )
        assert old_format.status_code == 400
        assert old_format.json()["ErrCode"] == "excel_file_invalid"

        uploaded = client.post(
            "/api/agent/uploads",
            headers=AUTH,
            data={"account_id": "alice"},
            files={"file": ("data.xlsx", workbook_bytes(), "application/octet-stream")},
        )
        other_conversation = "750e8400-e29b-41d4-a716-446655440000"
        response = client.post("/api/agent/chat/stream", headers=AUTH, json={
            "account_id": "bob",
            "conversation_id": other_conversation,
            "message": "读取 Excel",
            "attachment_ids": [uploaded.json()["attachment_id"]],
        })
        assert response.status_code == 404
        assert response.json()["ErrCode"] == "excel_attachment_not_found"
