"""Private, conversation-scoped storage and read-only parsing for XLSX uploads."""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
import zipfile
from datetime import date, datetime, time, timedelta, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from openpyxl import load_workbook
from openpyxl.utils.cell import range_boundaries
from openpyxl.utils.exceptions import InvalidFileException

from backend.adapters.monitoring import (
    ConversationNotFound,
    ExcelAttachmentNotFound,
    MonitoringStore,
)


MAX_EXCEL_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_EXCEL_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
MAX_EXCEL_ZIP_ENTRIES = 10_000
MAX_EXCEL_READ_CELLS = 1_000
MAX_EXCEL_SHEETS_IN_OVERVIEW = 200
DEFAULT_READ_ROWS = 50
DEFAULT_READ_COLUMNS = 20
PENDING_UPLOAD_TTL = timedelta(hours=24)
ATTACHMENT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
CELL_RANGE_PATTERN = re.compile(
    r"^\$?[A-Z]{1,3}\$?[1-9][0-9]{0,6}(?::\$?[A-Z]{1,3}\$?[1-9][0-9]{0,6})?$",
    re.IGNORECASE,
)


class ExcelFileError(ValueError):
    """A workbook cannot be accepted or read with the supported XLSX interface."""


class ExcelFileStore:
    def __init__(self, directory: Path, monitoring: MonitoringStore) -> None:
        self._directory = directory
        self._monitoring = monitoring
        self._ensure_private_directory()

    def _ensure_private_directory(self) -> None:
        if self._directory.is_symlink():
            raise OSError("Excel upload directory cannot be a symbolic link")
        self._directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self._directory.is_dir():
            raise OSError("Excel upload path is not a directory")
        os.chmod(self._directory, 0o700)

    async def purge_expired(self) -> int:
        expired_ids = await self._monitoring.purge_expired_excel_uploads()
        for attachment_id in expired_ids:
            if ATTACHMENT_ID_PATTERN.fullmatch(attachment_id):
                self._path_for_id(attachment_id).unlink(missing_ok=True)
        return len(expired_ids)

    async def save_upload(
        self, account_id: str, filename: str, content: bytes
    ) -> dict[str, Any]:
        if len(content) > MAX_EXCEL_UPLOAD_BYTES:
            raise ExcelFileError("Excel 文件超过 20 MiB 限制")
        safe_filename = _display_filename(filename)
        if not safe_filename.lower().endswith(".xlsx"):
            raise ExcelFileError("仅支持 .xlsx 文件")
        if not content:
            raise ExcelFileError("Excel 文件为空")

        await asyncio.to_thread(_validate_xlsx, content)
        await self.purge_expired()
        attachment_id = uuid4().hex
        path = self._path_for_id(attachment_id)
        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".upload-", suffix=".tmp", dir=self._directory
            )
            temporary_path = Path(temporary_name)
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as target:
                target.write(content)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary_path, path)
            os.chmod(path, 0o600)
            expires_at = (
                datetime.now(timezone.utc) + PENDING_UPLOAD_TTL
            ).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            record = await self._monitoring.register_excel_upload(
                attachment_id,
                account_id,
                safe_filename,
                len(content),
                expires_at,
            )
            return record
        except Exception:
            path.unlink(missing_ok=True)
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

    async def bind_uploads(
        self, account_id: str, conversation_id: str, attachment_ids: list[str]
    ) -> list[dict[str, Any]]:
        return await self._monitoring.bind_excel_uploads(
            account_id, conversation_id, attachment_ids
        )

    async def list_files(
        self, account_id: str, conversation_id: str | None
    ) -> list[dict[str, Any]]:
        if conversation_id is None:
            return []
        files = await self._monitoring.list_excel_uploads(account_id, conversation_id)
        return [
            {
                "attachment_id": item["attachment_id"],
                "filename": item["filename"],
                "size_bytes": int(item["size_bytes"]),
                "created_at": item["created_at"],
            }
            for item in files
        ]

    async def read_workbook(
        self,
        account_id: str,
        conversation_id: str | None,
        attachment_id: str,
        sheet_name: str | None = None,
        cell_range: str | None = None,
    ) -> dict[str, Any]:
        if conversation_id is None:
            raise ExcelFileError("当前会话没有可读取的 Excel 文件")
        if not ATTACHMENT_ID_PATTERN.fullmatch(attachment_id):
            raise ExcelFileError("Excel 文件不存在或不属于当前会话")
        try:
            record = await self._monitoring.excel_upload_for_conversation(
                account_id, conversation_id, attachment_id
            )
        except (ExcelAttachmentNotFound, ConversationNotFound) as exc:
            raise ExcelFileError("Excel 文件不存在或不属于当前会话") from exc
        path = self._path_for_id(attachment_id)
        if path.is_symlink() or not path.is_file():
            raise ExcelFileError("Excel 文件不存在或无法读取")
        return await asyncio.to_thread(
            _read_workbook,
            path,
            attachment_id,
            str(record["filename"]),
            sheet_name,
            cell_range,
        )

    def _path_for_id(self, attachment_id: str) -> Path:
        if not ATTACHMENT_ID_PATTERN.fullmatch(attachment_id):
            raise ExcelAttachmentNotFound(attachment_id)
        return self._directory / f"{attachment_id}.xlsx"


def _display_filename(filename: str) -> str:
    if not isinstance(filename, str) or not filename.strip():
        raise ExcelFileError("请上传带有文件名的 .xlsx 文件")
    normalized = filename.replace("\\", "/")
    basename = PurePosixPath(normalized).name
    safe = "".join(char for char in basename if ord(char) >= 32 and ord(char) != 127)
    safe = safe.strip()[:180]
    if not safe or safe in {".", ".."}:
        raise ExcelFileError("Excel 文件名无效")
    return safe


def _validate_xlsx(content: bytes) -> None:
    try:
        with zipfile.ZipFile(BytesIO(content)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_EXCEL_ZIP_ENTRIES:
                raise ExcelFileError("Excel 文件包含过多压缩条目")
            expanded_size = sum(entry.file_size for entry in entries)
            if expanded_size > MAX_EXCEL_UNCOMPRESSED_BYTES:
                raise ExcelFileError("Excel 文件解压后超过安全限制")
            names = {entry.filename for entry in entries}
            if "[Content_Types].xml" not in names or "xl/workbook.xml" not in names:
                raise ExcelFileError("文件内容不是有效的 .xlsx 工作簿")
            if archive.testzip() is not None:
                raise ExcelFileError("Excel 文件压缩数据已损坏")
        workbook = load_workbook(BytesIO(content), read_only=True, data_only=False)
        try:
            if not workbook.sheetnames:
                raise ExcelFileError("Excel 工作簿没有工作表")
        finally:
            workbook.close()
    except ExcelFileError:
        raise
    except (InvalidFileException, OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise ExcelFileError("无法解析该 Excel 文件，请确认它是未加密的 .xlsx 工作簿") from exc


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sheet_dimensions(worksheet: Any) -> dict[str, Any]:
    try:
        dimension = worksheet.calculate_dimension()
    except (ValueError, TypeError):
        dimension = None
    return {
        "name": worksheet.title,
        "state": worksheet.sheet_state,
        "dimension": dimension,
        "max_row": worksheet.max_row,
        "max_column": worksheet.max_column,
    }


def _read_workbook(
    path: Path,
    attachment_id: str,
    filename: str,
    sheet_name: str | None,
    cell_range: str | None,
) -> dict[str, Any]:
    try:
        formula_book = load_workbook(path, read_only=True, data_only=False)
        values_book = load_workbook(path, read_only=True, data_only=True)
        try:
            if sheet_name is None:
                if cell_range is not None:
                    raise ExcelFileError("指定 cell_range 时必须同时提供 sheet_name")
                sheet_names = formula_book.sheetnames
                overview = {
                    "file_id": attachment_id,
                    "filename": filename,
                    "sheet_count": len(sheet_names),
                    "sheets": [
                        _sheet_dimensions(formula_book[name])
                        for name in sheet_names[:MAX_EXCEL_SHEETS_IN_OVERVIEW]
                    ],
                    "sheets_truncated": len(sheet_names) > MAX_EXCEL_SHEETS_IN_OVERVIEW,
                }
                if len(str(overview).encode("utf-8")) > 64 * 1024:
                    raise ExcelFileError("工作簿工作表列表超过 64 KiB")
                return overview
            if sheet_name not in formula_book.sheetnames:
                raise ExcelFileError(f"工作表不存在：{sheet_name}")
            formula_sheet = formula_book[sheet_name]
            values_sheet = values_book[sheet_name]
            if cell_range is None:
                min_column, min_row = 1, 1
                max_column, max_row = DEFAULT_READ_COLUMNS, DEFAULT_READ_ROWS
                resolved_range = f"A1:T{DEFAULT_READ_ROWS}"
            else:
                if not isinstance(cell_range, str) or not CELL_RANGE_PATTERN.fullmatch(cell_range):
                    raise ExcelFileError("cell_range 必须是单个 A1 范围，例如 A1:D20")
                boundaries = range_boundaries(cell_range.replace("$", "").upper())
                min_column, min_row, max_column, max_row = boundaries
                if None in boundaries or min_column > max_column or min_row > max_row:
                    raise ExcelFileError("cell_range 无效")
                if max_row > 1_048_576 or max_column > 16_384:
                    raise ExcelFileError("cell_range 超出 Excel 支持范围")
                resolved_range = cell_range.replace("$", "").upper()
            cell_count = (max_row - min_row + 1) * (max_column - min_column + 1)
            if cell_count > MAX_EXCEL_READ_CELLS:
                raise ExcelFileError("单次最多读取 1,000 个单元格，请缩小范围")

            result_rows: list[list[Any]] = []
            formula_rows: list[list[str | None]] = []
            formula_iterator = formula_sheet.iter_rows(
                min_row=min_row,
                max_row=max_row,
                min_col=min_column,
                max_col=max_column,
            )
            values_iterator = values_sheet.iter_rows(
                min_row=min_row,
                max_row=max_row,
                min_col=min_column,
                max_col=max_column,
            )
            for formula_row, value_row in zip(formula_iterator, values_iterator):
                row_values: list[Any] = []
                row_formulas: list[str | None] = []
                for formula_cell, value_cell in zip(formula_row, value_row):
                    formula = (
                        str(formula_cell.value)
                        if getattr(formula_cell, "data_type", None) == "f"
                        else None
                    )
                    value = value_cell.value if formula is not None else formula_cell.value
                    if formula is not None and value is None:
                        value = formula
                    row_values.append(_safe_json_value(value))
                    row_formulas.append(formula)
                result_rows.append(row_values)
                formula_rows.append(row_formulas)

            output = {
                "file_id": attachment_id,
                "filename": filename,
                "sheet_name": sheet_name,
                "range": resolved_range,
                "row_count": len(result_rows),
                "column_count": max_column - min_column + 1,
                "values": result_rows,
                "formulas": formula_rows,
                "sheet": _sheet_dimensions(formula_sheet),
            }
            if len(str(output).encode("utf-8")) > 64 * 1024:
                raise ExcelFileError("读取结果超过 64 KiB，请缩小单元格范围")
            return output
        finally:
            formula_book.close()
            values_book.close()
    except ExcelFileError:
        raise
    except (InvalidFileException, OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise ExcelFileError("无法读取 Excel 工作簿或工作表数据") from exc
