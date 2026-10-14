"""Contract-code exchange lookup behavior."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from agent.exchange_lookup import lookup_exchange_by_contract_code


SHANGHAI = ZoneInfo("Asia/Shanghai")
FIXED_NOW = datetime(2026, 10, 14, 12, 0, tzinfo=SHANGHAI)


@pytest.mark.parametrize(
    ("contract_code", "exchange", "product_code", "product_name", "delivery_month"),
    [
        ("rb2710", "SHFE", "rb", "螺纹钢", "2027-10"),
        ("sc2710", "INE", "sc", "原油", "2027-10"),
        ("i2710", "DCE", "i", "铁矿石", "2027-10"),
        ("sr710", "CZCE", "SR", "白糖", "2027-10"),
        ("IF2710", "CFFEX", "if", "沪深300股指", "2027-10"),
        ("LC2710", "GFEX", "lc", "碳酸锂", "2027-10"),
        ("PT2710", "GFEX", "pt", "铂", "2027-10"),
    ],
)
def test_identifies_representative_contracts_from_all_six_exchanges(
    contract_code: str,
    exchange: str,
    product_code: str,
    product_name: str,
    delivery_month: str,
) -> None:
    result = lookup_exchange_by_contract_code(contract_code, now=FIXED_NOW)

    assert result["status"] == "matched"
    assert result["is_valid"] is True
    assert result["exchange_code"] == exchange
    assert result["exchange_name"]
    assert result["product_code"] == product_code
    assert result["product_name"] == product_name
    assert result["delivery_month"] == delivery_month


def test_normalizes_case_and_outer_whitespace() -> None:
    result = lookup_exchange_by_contract_code("  rB2710  ", now=FIXED_NOW)

    assert result["contract_code"] == "RB2710"
    assert result["product_code"] == "rb"
    assert result["exchange_code"] == "SHFE"
    assert result["status"] == "matched"


@pytest.mark.parametrize(
    ("contract_code", "expected_month", "expected_status"),
    [
        ("IF2610", "2026-10", "matched"),
        ("IF2611", "2026-11", "matched"),
        ("IF2609", "2026-09", "expired"),
        ("SR610", "2026-10", "matched"),
        ("SR609", "2026-09", "expired"),
    ],
)
def test_validity_is_month_based_and_czce_uses_yymm_suffix(
    contract_code: str,
    expected_month: str,
    expected_status: str,
) -> None:
    result = lookup_exchange_by_contract_code(contract_code, now=FIXED_NOW)

    assert result["delivery_month"] == expected_month
    assert result["status"] == expected_status
    assert result["is_valid"] is (expected_status == "matched")


@pytest.mark.parametrize("contract_code", ["rb2700", "rb27x0", "rb27", "SR1000"])
def test_known_product_with_invalid_month_format_is_not_valid(contract_code: str) -> None:
    result = lookup_exchange_by_contract_code(contract_code, now=FIXED_NOW)

    assert result["status"] == "invalid_format"
    assert result["is_valid"] is False
    assert result["exchange_code"] is not None
    assert result["delivery_month"] is None


def test_unknown_product_does_not_guess_an_exchange() -> None:
    result = lookup_exchange_by_contract_code("XYZ2710", now=FIXED_NOW)

    assert result["status"] == "unrecognized"
    assert result["is_valid"] is False
    assert result["exchange_code"] is None
    assert result["product_code"] is None


def test_czce_single_digit_year_ambiguity_is_reported_as_invalid() -> None:
    result = lookup_exchange_by_contract_code("CF105", now=FIXED_NOW)

    assert result["status"] == "invalid_format"
    assert result["exchange_code"] == "CZCE"
    assert result["delivery_month"] is None
