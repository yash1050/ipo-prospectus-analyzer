"""
Financial metrics extraction for IPO prospectus companies (XBRL via SEC, PDF fallback).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pdfplumber
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

COMPANIES = [
    {"name": "instacart", "pdf": "data/instacart s1.pdf", "ticker": "CART", "cik": "1579091"},
    {"name": "reddit", "pdf": "data/reddit s1.pdf", "ticker": "RDDT", "cik": "1713445"},
    {"name": "klaviyo", "pdf": "data/klaviyo s1.pdf", "ticker": "KVYO", "cik": "1899888"},
    {"name": "arm", "pdf": "data/arm s1.pdf", "ticker": "ARM", "cik": "0001898098"},
    {"name": "coinbase", "pdf": "data/coinbase s1.pdf", "ticker": "COIN", "cik": "1679788"},
    {"name": "doordash", "pdf": "data/doordash s1.pdf", "ticker": "DASH", "cik": "1792789"},
    {"name": "snowflake", "pdf": "data/snowflake s1.pdf", "ticker": "SNOW", "cik": "1640147"},
]

SEC_USER_AGENT = "IPOAnalyzer contact@ipoanalyzer.com"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

REVENUE_TAGS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
)

GROSS_PROFIT_TAGS = (
    "GrossProfit",
    "GrossProfitLoss",
)

OPERATING_EXPENSE_TAGS = (
    "OperatingExpenses",
    "CostsAndExpenses",
    "OperatingCostsAndExpenses",
    "NoninterestExpense",
)

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
PDF_EXTRACTOR_MODEL = "gpt-4.1-mini"


def _cik_10(cik: str) -> str:
    digits = re.sub(r"\D", "", str(cik))
    return digits.zfill(10)


def _format_usd_amount(val: float | int) -> str:
    neg = float(val) < 0
    v = abs(float(val))
    if v >= 1e9:
        num = f"{v / 1e9:.1f}B"
    elif v >= 1e6:
        num = f"{v / 1e6:.0f}M"
    elif v >= 1e3:
        num = f"{v / 1e3:.0f}K"
    else:
        num = f"{v:.0f}"
    if neg:
        return f"-${num}"
    return f"${num}"


def _form_allowed(form: str) -> bool:
    u = str(form).upper()
    return "S-1" in u or "10-K" in u


def _latest_annual_entry(concept_block: dict[str, Any] | None) -> dict[str, Any] | None:
    if not concept_block or not isinstance(concept_block, dict):
        return None
    units = concept_block.get("units")
    if not isinstance(units, dict):
        return None
    candidates: list[dict[str, Any]] = []
    for _unit_key, rows in units.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            if not _form_allowed(str(row.get("form", ""))):
                continue
            end = row.get("end")
            if end:
                candidates.append(row)
    if not candidates:
        return None
    return max(candidates, key=lambda r: str(r.get("end", "")))


def _value_from_entry(entry: dict[str, Any] | None) -> str | None:
    if not entry:
        return None
    raw = entry.get("val")
    if raw is None:
        return None
    try:
        return _format_usd_amount(float(raw))
    except (TypeError, ValueError):
        return None


def _value_from_first_tag(us_gaap: dict[str, Any], tags: tuple[str, ...]) -> str | None:
    for tag in tags:
        val = _value_from_entry(_latest_annual_entry(us_gaap.get(tag)))
        if val is not None:
            return val
    return None


def extract_financials_xbrl(cik: str, company_name: str) -> dict[str, Any]:
    _ = company_name
    try:
        url = SEC_FACTS_URL.format(cik=_cik_10(cik))
        resp = requests.get(
            url,
            headers={"User-Agent": SEC_USER_AGENT},
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError, json.JSONDecodeError):
        return {}

    facts_root = payload.get("facts") or {}
    us_gaap = facts_root.get("us-gaap")
    if not isinstance(us_gaap, dict):
        return {}

    revenue_val: str | None = None
    for tag in REVENUE_TAGS:
        entry = _latest_annual_entry(us_gaap.get(tag))
        revenue_val = _value_from_entry(entry)
        if revenue_val is not None:
            break

    gp = _value_from_first_tag(us_gaap, GROSS_PROFIT_TAGS)
    ni = _value_from_entry(_latest_annual_entry(us_gaap.get("NetIncomeLoss")))
    opex = _value_from_first_tag(us_gaap, OPERATING_EXPENSE_TAGS)

    out = {
        "revenue": revenue_val,
        "gross_profit": gp,
        "net_income": ni,
        "operating_expenses": opex,
        "source": "xbrl",
    }
    if all(out[k] is None for k in ("revenue", "gross_profit", "net_income", "operating_expenses")):
        return {}
    return out


def _tables_to_text(tables: list[list[list[str | None]]]) -> str:
    lines: list[str] = []
    for t_idx, table in enumerate(tables):
        lines.append(f"--- table {t_idx + 1} ---")
        for row in table:
            cells = [str(c).strip() if c is not None else "" for c in row]
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _openai_extract_from_tables(table_blob: str, company_name: str) -> dict[str, Any] | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    system = "You are a financial data extractor. Return only JSON matching the requested schema."
    user = (
        f"Company: {company_name}\n\n"
        "You are a financial data extractor. Given these raw table rows from an S-1 filing, "
        "extract the most recent annual figures for: revenue, gross_profit, net_income, operating_expenses. "
        "Return only valid JSON with these exact keys and string values like '$2.9B' or '-$482M'. "
        "If a value cannot be found return null.\n\n"
        f"Raw tables:\n{table_blob}"
    )

    body = {
        "model": PDF_EXTRACTOR_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    try:
        r = requests.post(
            OPENAI_CHAT_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(body).encode(),
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (requests.RequestException, KeyError, IndexError, ValueError, json.JSONDecodeError):
        return None

    if not isinstance(parsed, dict):
        return None
    return parsed


def extract_financials_pdfplumber(pdf_path: str, company_name: str) -> dict[str, Any]:
    path = Path(pdf_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return {}

    keywords = (
        "consolidated statements of operations",
        "total revenue",
        "gross profit",
    )
    try:
        with pdfplumber.open(path) as pdf:
            target_pages: list[Any] = []
            for page in pdf.pages:
                text = (page.extract_text() or "").lower()
                if any(k in text for k in keywords):
                    target_pages.append(page)

            all_tables: list[list[list[str | None]]] = []
            for page in target_pages:
                for tbl in page.extract_tables() or []:
                    all_tables.append(tbl)

        if not all_tables:
            return {}

        blob = _tables_to_text(all_tables)
        parsed = _openai_extract_from_tables(blob, company_name)
        if not parsed:
            return {}

        def _norm(v: Any) -> str | None:
            if v is None:
                return None
            if isinstance(v, str) and v.strip().lower() in ("null", "none", ""):
                return None
            return str(v).strip()

        out = {
            "revenue": _norm(parsed.get("revenue")),
            "gross_profit": _norm(parsed.get("gross_profit")),
            "net_income": _norm(parsed.get("net_income")),
            "operating_expenses": _norm(parsed.get("operating_expenses")),
            "source": "pdfplumber",
        }
        if all(out[k] is None for k in ("revenue", "gross_profit", "net_income", "operating_expenses")):
            return {}
        return out
    except Exception:
        return {}


def _filled_financial_count(d: dict[str, Any]) -> int:
    keys = ("revenue", "gross_profit", "net_income", "operating_expenses")
    n = 0
    for k in keys:
        v = d.get(k)
        if v is not None and str(v).strip() != "":
            n += 1
    return n


def _na_result() -> dict[str, Any]:
    return {
        "revenue": "N/A",
        "gross_profit": "N/A",
        "net_income": "N/A",
        "operating_expenses": "N/A",
        "source": "unavailable",
    }


def get_financials(company: dict[str, Any]) -> dict[str, Any]:
    label = company.get("name", str(company))
    cik = str(company.get("cik", ""))
    pdf_rel = str(company.get("pdf", ""))

    xbrl = extract_financials_xbrl(cik, label)
    if xbrl and _filled_financial_count(xbrl) >= 2:
        print(f"✅ XBRL success: {label}")
        return {
            "revenue": xbrl.get("revenue") or "N/A",
            "gross_profit": xbrl.get("gross_profit") or "N/A",
            "net_income": xbrl.get("net_income") or "N/A",
            "operating_expenses": xbrl.get("operating_expenses") or "N/A",
            "source": "xbrl",
        }

    print(f"⚠️ pdfplumber fallback: {label}")
    pdf_result = extract_financials_pdfplumber(pdf_rel, label)
    if pdf_result and _filled_financial_count(pdf_result) >= 1:
        return {
            "revenue": pdf_result.get("revenue") or "N/A",
            "gross_profit": pdf_result.get("gross_profit") or "N/A",
            "net_income": pdf_result.get("net_income") or "N/A",
            "operating_expenses": pdf_result.get("operating_expenses") or "N/A",
            "source": "pdfplumber",
        }

    print(f"❌ unavailable: {label}")
    return _na_result()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    results: dict[str, Any] = {}
    counts = {"xbrl": 0, "pdfplumber": 0, "unavailable": 0}

    for co in COMPANIES:
        name = co["name"]
        row = get_financials(co)
        results[name] = row

        src = row.get("source", "")
        if src == "xbrl":
            counts["xbrl"] += 1
        elif src == "pdfplumber":
            counts["pdfplumber"] += 1
        else:
            counts["unavailable"] += 1

        print(f"\n--- {name.upper()} ({co.get('ticker', '')}) ---")
        for k in ("revenue", "gross_profit", "net_income", "operating_expenses", "source"):
            print(f"  {k}: {row.get(k)}")

    out_path = PROJECT_ROOT / "data" / "financials.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")

    print("\nSummary:")
    print(f"  XBRL: {counts['xbrl']}")
    print(f"  pdfplumber: {counts['pdfplumber']}")
    print(f"  unavailable: {counts['unavailable']}")


if __name__ == "__main__":
    main()
