"""Export all evaluation ground truth (GT) datasets to a single Excel file.

Each query file under eval/data/queries/ becomes a sheet in the workbook,
with one row per query and columns for query text, GT doc IDs, type, etc.

Usage::

    uv run python eval/scripts/export_gt_to_excel.py
    → writes eval/data/gt_datasets.xlsx
"""

from __future__ import annotations

import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

REPO_ROOT = Path(__file__).resolve().parents[2]
QUERIES_DIR = REPO_ROOT / "eval" / "data" / "queries"
OUTPUT_PATH = REPO_ROOT / "eval" / "data" / "gt_datasets.xlsx"


HEADER_FILL = PatternFill("solid", fgColor="2F5496")
HEADER_FONT = Font(bold=True, color="FFFFFF", size=11)


def _flatten_query(q: dict) -> dict:
    """Normalize a query dict to a flat set of columns."""
    relevant = q.get("relevant_docs") or q.get("answer_ids") or []
    if isinstance(relevant, dict):
        relevant = list(relevant.keys())
    return {
        "qid": q.get("qid", q.get("query_id", "")),
        "query": q.get("query", q.get("question", "")),
        "type": q.get("type", q.get("query_type", "")),
        "level": q.get("level", q.get("difficulty", "")),
        "category": q.get("category", ""),
        "description": q.get("description", ""),
        "relevant_count": len(relevant),
        "relevant_docs": "\n".join(str(x) for x in relevant),
    }


def _write_sheet(wb: Workbook, name: str, meta: dict, queries: list[dict]) -> None:
    ws = wb.create_sheet(title=name[:31])  # Excel sheet name limit

    # Metadata header (first rows)
    ws["A1"] = "Dataset"
    ws["B1"] = name
    ws["A2"] = "Description"
    ws["B2"] = meta.get("description", "")
    ws["A3"] = "id_field"
    ws["B3"] = meta.get("id_field", "doc_id")
    ws["A4"] = "Total queries"
    ws["B4"] = len(queries)

    for row in range(1, 5):
        ws[f"A{row}"].font = Font(bold=True)

    # Column headers
    columns = ["qid", "query", "type", "level", "category", "description",
               "relevant_count", "relevant_docs"]
    header_row = 6
    for i, col in enumerate(columns, start=1):
        cell = ws.cell(row=header_row, column=i, value=col)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")

    # Data rows
    for r, q in enumerate(queries, start=header_row + 1):
        flat = _flatten_query(q)
        for c, col in enumerate(columns, start=1):
            cell = ws.cell(row=r, column=c, value=flat.get(col, ""))
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    # Column widths
    widths = {"A": 8, "B": 45, "C": 18, "D": 8, "E": 22, "F": 42, "G": 14, "H": 60}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w

    # Freeze header
    ws.freeze_panes = f"A{header_row + 1}"


def _write_summary(wb: Workbook, stats: list[dict]) -> None:
    ws = wb.create_sheet(title="Summary", index=0)

    ws["A1"] = "Synaptic Memory — Evaluation GT Datasets"
    ws["A1"].font = Font(bold=True, size=14)
    ws.merge_cells("A1:E1")

    ws["A3"] = "Generated from"
    ws["B3"] = "eval/data/queries/*.json"
    ws["A3"].font = Font(bold=True)

    header_row = 5
    headers = ["Dataset", "Description", "Queries", "id_field", "Language"]
    for i, h in enumerate(headers, start=1):
        cell = ws.cell(row=header_row, column=i, value=h)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center")

    for r, s in enumerate(stats, start=header_row + 1):
        for c, key in enumerate(["dataset", "description", "queries", "id_field", "language"], start=1):
            cell = ws.cell(row=r, column=c, value=s.get(key, ""))
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    widths = {"A": 22, "B": 60, "C": 10, "D": 14, "E": 12}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w

    ws.freeze_panes = f"A{header_row + 1}"


def _guess_language(name: str, queries: list[dict]) -> str:
    # Sample a few queries
    text = " ".join(str(q.get("query", "")) for q in queries[:5])
    if any(ord(c) > 0x3000 and ord(c) < 0xD7AF for c in text):
        return "KO"
    return "EN"


def main() -> None:
    wb = Workbook()
    # Remove default sheet
    default = wb.active
    if default is not None:
        wb.remove(default)

    files = sorted(QUERIES_DIR.glob("*.json"))
    stats: list[dict] = []

    for path in files:
        with open(path, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                print(f"  ⚠ skip {path.name}: {exc}")
                continue

        queries = data.get("queries", [])
        if not isinstance(queries, list) or not queries:
            print(f"  ⚠ skip {path.name}: no queries list")
            continue

        name = path.stem
        meta = {
            "description": data.get("description", ""),
            "id_field": data.get("id_field", "doc_id"),
        }

        _write_sheet(wb, name, meta, queries)
        stats.append({
            "dataset": name,
            "description": meta["description"][:100],
            "queries": len(queries),
            "id_field": meta["id_field"],
            "language": _guess_language(name, queries),
        })
        print(f"  ✓ {name}: {len(queries)} queries")

    _write_summary(wb, stats)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUTPUT_PATH)
    print(f"\nSaved → {OUTPUT_PATH}")
    print(f"Total datasets: {len(stats)}")


if __name__ == "__main__":
    main()
