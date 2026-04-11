#!/usr/bin/env python3
"""
Extract recognized pseudo-words and their attributed meanings from gibberish probe result files.

Input:
  results_dir/{model}_{lang}.json

Output:
  out_dir/pseudoword_meanings_long.csv
  out_dir/pseudoword_meanings_long.xlsx
  out_dir/pseudoword_meanings_by_model_lang.csv
  out_dir/pseudoword_meanings_by_model_lang.xlsx

Only keeps pseudo-words where the model assigned a meaning:
- explicit unknown with likely_meaning
- implicit normalization with predicted_meaning
Optionally keeps other cases with non-empty predicted_meaning.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _json_load(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_model_lang_from_filename(path: Path) -> Optional[Tuple[str, str]]:
    m = re.match(r"^(?P<model>.+)_(?P<lang>de|es|fr|it|en)\.json$", path.name)
    if not m:
        return None
    return m.group("model"), m.group("lang")


def deduplicate_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep one record per file. Prefer latest successful response, else latest.
    """
    by_file: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = row.get("file")
        if not key:
            continue
        prev = by_file.get(key)
        if prev is None:
            by_file[key] = row
            continue

        prev_ok = prev.get("response") is not None
        curr_ok = row.get("response") is not None

        if curr_ok and not prev_ok:
            by_file[key] = row
        elif curr_ok == prev_ok:
            by_file[key] = row
    return list(by_file.values())


def meaning_is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    text = normalize_text(str(value))
    if not text:
        return False
    nullish = {
        "null", "none", "unknown", "unk", "n/a", "na",
        "no meaning", "cannot infer", "can't infer", "unclear", "noise", "gibberish"
    }
    return text.lower() not in nullish


def extract_rows_from_result_file(path: Path) -> List[Dict[str, Any]]:
    parsed = extract_model_lang_from_filename(path)
    if not parsed:
        return []
    model, lang = parsed

    raw = _json_load(path)
    if not isinstance(raw, list):
        return []

    rows = deduplicate_results(raw)
    out: List[Dict[str, Any]] = []

    for row in rows:
        file_name = row.get("file")
        row_idx = row.get("row_idx")
        original_text = row.get("original_text")
        modified_text = row.get("modified_text")
        parsed_response = row.get("parsed_response") or {}
        transcript = parsed_response.get("transcript")
        pseudo_evals = row.get("pseudo_evaluation") or []

        for item in pseudo_evals:
            pseudo_text = normalize_text(item.get("pseudo_text"))
            predicted_meaning = item.get("predicted_meaning")
            meaning_assigned = bool(item.get("meaning_assigned"))
            handling_type = item.get("handling_type")
            meaning_source = item.get("meaning_source")
            detected = item.get("detected")
            detected_in_transcript = item.get("detected_in_transcript")
            detected_in_unknowns = item.get("detected_in_unknowns")
            implicit_normalization = item.get("implicit_normalization")
            context_window_modified = item.get("context_window_modified")
            context_window_transcript = item.get("context_window_transcript")

            keep = meaning_assigned and meaning_is_nonempty(predicted_meaning)
            if not keep:
                continue

            out.append({
                "model": model,
                "lang": lang,
                "file": file_name,
                "row_idx": row_idx,
                "pseudo_text": pseudo_text,
                "predicted_meaning": normalize_text(str(predicted_meaning)) if predicted_meaning is not None else None,
                "handling_type": handling_type,
                "meaning_source": meaning_source,
                "detected": detected,
                "detected_in_transcript": detected_in_transcript,
                "detected_in_unknowns": detected_in_unknowns,
                "implicit_normalization": implicit_normalization,
                "original_text": original_text,
                "modified_text": modified_text,
                "transcript": transcript,
                "context_window_modified": context_window_modified,
                "context_window_transcript": context_window_transcript,
            })

    return out


def build_outputs(results_dir: Path):
    long_rows: List[Dict[str, Any]] = []

    for path in sorted(results_dir.glob("*.json")):
        if path.name.endswith("_gibberish_summary.json"):
            continue
        long_rows.extend(extract_rows_from_result_file(path))

    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        return long_df, pd.DataFrame()

    long_df = long_df.sort_values(["model", "lang", "row_idx", "file", "pseudo_text"]).reset_index(drop=True)

    grouped = (
        long_df.groupby(
            ["model", "lang", "pseudo_text", "predicted_meaning", "handling_type", "meaning_source"],
            dropna=False
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            ["model", "lang", "count", "pseudo_text", "predicted_meaning"],
            ascending=[True, True, False, True, True]
        )
        .reset_index(drop=True)
    )

    return long_df, grouped


def save_outputs(long_df: pd.DataFrame, grouped_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    long_csv = out_dir / "pseudoword_meanings_long.csv"
    long_xlsx = out_dir / "pseudoword_meanings_long.xlsx"
    grouped_csv = out_dir / "pseudoword_meanings_by_model_lang.csv"
    grouped_xlsx = out_dir / "pseudoword_meanings_by_model_lang.xlsx"

    long_df.to_csv(long_csv, index=False)
    grouped_df.to_csv(grouped_csv, index=False)

    with pd.ExcelWriter(long_xlsx, engine="openpyxl") as w:
        long_df.to_excel(w, index=False, sheet_name="pseudowords_long")

    with pd.ExcelWriter(grouped_xlsx, engine="openpyxl") as w:
        grouped_df.to_excel(w, index=False, sheet_name="by_model_lang")

    return long_csv, long_xlsx, grouped_csv, grouped_xlsx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=Path, required=True, help="Directory containing {model}_{lang}.json files")
    ap.add_argument("--out_dir", type=Path, default=Path("pseudoword_extract_out"), help="Directory for CSV/XLSX outputs")
    args = ap.parse_args()

    long_df, grouped_df = build_outputs(args.results_dir)
    if long_df.empty:
        print(f"[WARN] No recognized pseudo-words with attributed meanings found in {args.results_dir}")
        return

    paths = save_outputs(long_df, grouped_df, args.out_dir)
    for p in paths:
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
