import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


REFUSAL_PATTERNS = [
    # English
    r"\bi (?:can't|cannot|won't|will not)\b",
    r"\bnot able to\b",
    r"\bi (?:am|\'m) sorry\b",
    r"\bi can(?:not|'t) help with that\b",
    r"\bi can(?:not|'t) comply\b",
    r"\brefus(e|al)\b",
    r"\bagainst (?:policy|the rules)\b",
    r"\bpolicy\b.*\bviolation\b",
    r"\bi (?:must|have to) decline\b",
    r"\bcan't assist\b",

    # German
    r"\bich kann(?:\s+das)?\s+nicht\b",
    r"\bich darf(?:\s+das)?\s+nicht\b",
    r"\btut mir leid\b",
    r"\bich kann dir damit nicht helfen\b",
    r"\bich kann dabei nicht helfen\b",
    r"\bich kann dem nicht nachkommen\b",
    r"\bich muss ablehnen\b",
    r"\bablehn(?:en|ung)\b",
    r"\bgegen (?:die richtlinien|regeln|unsere richtlinien)\b",
    r"\brichtlinienversto(?:ß|ss)\b",

    # Italian
    r"\bnon posso\b",
    r"\bnon sono in grado di\b",
    r"\bmi dispiace\b",
    r"\bnon posso aiutarti con questo\b",
    r"\bnon posso aiutare con questo\b",
    r"\bnon posso soddisfare questa richiesta\b",
    r"\bdevo rifiutare\b",
    r"\brifiut(?:o|are)\b",
    r"\bcontro (?:la policy|le regole|i regolamenti)\b",
    r"\bviolazione (?:della policy|delle regole)\b",

    # French
    r"\bje ne peux pas\b",
    r"\bje ne suis pas en mesure de\b",
    r"\bje suis désolé\b",
    r"\bje suis desole\b",
    r"\bdésolé\b",
    r"\bdesole\b",
    r"\bje ne peux pas aider avec ça\b",
    r"\bje ne peux pas vous aider avec ça\b",
    r"\bje ne peux pas accéder à cette demande\b",
    r"\bje dois refuser\b",
    r"\brefus(?:er|al)?\b",
    r"\bcontre (?:la politique|les règles)\b",
    r"\bviolation de (?:la politique|des règles)\b",

    # Spanish
    r"\bno puedo\b",
    r"\bno soy capaz de\b",
    r"\blo siento\b",
    r"\bno puedo ayudarte con eso\b",
    r"\bno puedo ayudar con eso\b",
    r"\bno puedo cumplir con eso\b",
    r"\bdebo rechazar\b",
    r"\brechaz(?:ar|o)\b",
    r"\bcontra (?:la política|las reglas)\b",
    r"\bpolitica\b.*\bviolaci[oó]n\b",
    r"\bviolaci[oó]n de (?:la política|las reglas)\b",
]

REFUSAL_RE = re.compile("|".join(f"(?:{p})" for p in REFUSAL_PATTERNS), flags=re.IGNORECASE)


def _json_load(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[\W_]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_refusal(record: Dict[str, Any]) -> bool:
    """
      - response is None, OR
      - parsed_response missing/parse_error with refusal-like text, OR
      - transcript itself contains refusal-like text (model refused instead of transcribing)
    """
    resp = record.get("response")
    if resp is None:
        return True

    parsed = record.get("parsed_response") or {}
    transcript = parsed.get("transcript")
    parse_error = parsed.get("parse_error")

    # If parse error and raw response looks like refusal
    if parse_error:
        if isinstance(resp, str) and REFUSAL_RE.search(resp):
            return True

    # If transcript exists but is clearly a refusal template
    if isinstance(transcript, str) and REFUSAL_RE.search(transcript):
        return True

    # Some models may put refusal into a plain-text response
    if isinstance(resp, str) and REFUSAL_RE.search(resp) and not isinstance(transcript, str):
        return True

    return False


def deduplicate_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep one record per 'file'. Prefer the latest successful (response not None), else latest.
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
            by_file[key] = row  # last wins
    return list(by_file.values())


def extract_model_lang_from_filename(path: Path) -> Optional[Tuple[str, str]]:
    """
    Parse {model}_{lang}.json where lang is one of: de es fr it en
    """
    m = re.match(r"^(?P<model>.+)_(?P<lang>de|es|fr|it|en)\.json$", path.name)
    if not m:
        return None
    return m.group("model"), m.group("lang")


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = deduplicate_results(rows)

    n_rows = len(rows)
    refused = sum(1 for r in rows if is_refusal(r))

    # Per pseudo-target metrics
    pseudo_items: List[Dict[str, Any]] = []
    parse_failures = 0

    for r in rows:
        parsed = r.get("parsed_response") or {}
        if parsed.get("parse_error"):
            parse_failures += 1

        for item in (r.get("pseudo_evaluation") or []):
            pseudo_items.append(item)

    n_pseudo = len(pseudo_items)

    def rate(num: int, den: int) -> Optional[float]:
        return None if den == 0 else num / den

    detected = sum(1 for i in pseudo_items if bool(i.get("detected")))
    meaning = sum(1 for i in pseudo_items if bool(i.get("meaning_assigned")))
    implicit_norm = sum(1 for i in pseudo_items if i.get("handling_type") == "implicit_normalization")
    explicit_unknown = sum(1 for i in pseudo_items if i.get("handling_type") == "explicit_unknown")
    ignored = sum(1 for i in pseudo_items if i.get("handling_type") == "ignored")
    substituted = sum(1 for i in pseudo_items if i.get("handling_type") == "substituted")
    preserved = sum(1 for i in pseudo_items if i.get("handling_type") == "preserved")

    # Meaning “quality” is not reliably extractable without a judge; we report only match_rate if present.
    match_true = sum(1 for i in pseudo_items if i.get("meaning_matches_original") is True)
    match_known = sum(1 for i in pseudo_items if i.get("meaning_matches_original") is not None)

    out = {
        "n_rows": n_rows,
        "n_rows_refused": refused,
        "refusal_rate": rate(refused, n_rows),
        "json_parse_failure_rate": rate(parse_failures, n_rows),

        "n_pseudo_targets": n_pseudo,
        "pseudo_detection_rate": rate(detected, n_pseudo),
        "meaning_attribution_rate": rate(meaning, n_pseudo),

        "implicit_normalization_rate": rate(implicit_norm, n_pseudo),
        "explicit_unknown_rate": rate(explicit_unknown, n_pseudo),
        "ignored_rate": rate(ignored, n_pseudo),
        "substituted_rate": rate(substituted, n_pseudo),
        "preserved_rate": rate(preserved, n_pseudo),

        "meaning_match_rate": rate(match_true, match_known) if match_known else None,
    }
    return out


def build_tables(results_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      summary_df: per (model, lang)
      model_overall_df: per model (aggregated across langs)
    """
    records = []
    overall_records = []

    files = sorted(results_dir.glob("*.json"))
    for path in files:
        parsed = extract_model_lang_from_filename(path)
        if not parsed:
            continue
        model, lang = parsed
        rows = _json_load(path)
        if not isinstance(rows, list):
            continue
        metrics = compute_metrics(rows)
        records.append({"model": model, "lang": lang, **metrics})

    summary_df = pd.DataFrame(records)
    if summary_df.empty:
        return summary_df, pd.DataFrame()

    # Overall per model (weighted by denominators)
    def weighted_rate(num_col: str, den_col: str) -> pd.Series:
        num = summary_df[num_col].fillna(0)
        den = summary_df[den_col].fillna(0)
        total_den = den.sum()
        return pd.Series([None if total_den == 0 else num.sum() / total_den])

    for model, g in summary_df.groupby("model"):
        n_rows = int(g["n_rows"].sum())
        n_ref = int(g["n_rows_refused"].sum())
        n_pseudo = int(g["n_pseudo_targets"].sum())

        rec = {
            "model": model,
            "n_rows": n_rows,
            "refusal_rate": (n_ref / n_rows) if n_rows else None,
            "n_pseudo_targets": n_pseudo,
        }

        # Reconstruct weighted rates from counts
        # We only have rates, so approximate using denominators we have.
        detected = (g["pseudo_detection_rate"].fillna(0) * g["n_pseudo_targets"].fillna(0)).sum()
        meaning = (g["meaning_attribution_rate"].fillna(0) * g["n_pseudo_targets"].fillna(0)).sum()
        implicit_norm = (g["implicit_normalization_rate"].fillna(0) * g["n_pseudo_targets"].fillna(0)).sum()
        explicit_unknown = (g["explicit_unknown_rate"].fillna(0) * g["n_pseudo_targets"].fillna(0)).sum()
        ignored = (g["ignored_rate"].fillna(0) * g["n_pseudo_targets"].fillna(0)).sum()
        substituted = (g["substituted_rate"].fillna(0) * g["n_pseudo_targets"].fillna(0)).sum()
        preserved = (g["preserved_rate"].fillna(0) * g["n_pseudo_targets"].fillna(0)).sum()

        rec.update({
            "pseudo_detection_rate": (detected / n_pseudo) if n_pseudo else None,
            "meaning_attribution_rate": (meaning / n_pseudo) if n_pseudo else None,
            "implicit_normalization_rate": (implicit_norm / n_pseudo) if n_pseudo else None,
            "explicit_unknown_rate": (explicit_unknown / n_pseudo) if n_pseudo else None,
            "ignored_rate": (ignored / n_pseudo) if n_pseudo else None,
            "substituted_rate": (substituted / n_pseudo) if n_pseudo else None,
            "preserved_rate": (preserved / n_pseudo) if n_pseudo else None,
        })

        overall_records.append(rec)

    overall_df = pd.DataFrame(overall_records).sort_values("model")
    summary_df = summary_df.sort_values(["model", "lang"])
    return summary_df, overall_df


def save_outputs(summary_df: pd.DataFrame, overall_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    summary_csv = out_dir / "gibberish_summary_by_model_lang.csv"
    overall_csv = out_dir / "gibberish_summary_overall_by_model.csv"
    summary_df.to_csv(summary_csv, index=False)
    if not overall_df.empty:
        overall_df.to_csv(overall_csv, index=False)

    # XLSX
    xlsx_path = out_dir / "gibberish_summary.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        summary_df.to_excel(w, index=False, sheet_name="by_model_lang")
        if not overall_df.empty:
            overall_df.to_excel(w, index=False, sheet_name="overall_by_model")

    return summary_csv, overall_csv, xlsx_path


def plot_heatmap(df: pd.DataFrame, value_col: str, title: str, out_path: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    if df.empty:
        return

    pivot = df.pivot(index="model", columns="lang", values=value_col)
    models = list(pivot.index)
    langs = list(pivot.columns)
    data = pivot.values.astype(float)

    fig, ax = plt.subplots(figsize=(max(6, 1 + 1.1 * len(langs)), max(4, 1 + 0.45 * len(models))))
    im = ax.imshow(np.nan_to_num(data, nan=-1.0), aspect="auto")

    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels(langs)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title(title)

    # annotate
    for i in range(len(models)):
        for j in range(len(langs)):
            v = data[i, j]
            txt = "NA" if (v != v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=Path, required=True, help="Directory containing {model}_{lang}.json files")
    ap.add_argument("--out_dir", type=Path, default=Path("gibberish_analysis_out"), help="Where to write CSV/XLSX/PNGs")
    args = ap.parse_args()

    summary_df, overall_df = build_tables(args.results_dir)
    if summary_df.empty:
        print(f"[WARN] No matching result files found in {args.results_dir}")
        return

    summary_csv, overall_csv, xlsx_path = save_outputs(summary_df, overall_df, args.out_dir)
    print(f"Wrote: {summary_csv}")
    if overall_df is not None and not overall_df.empty:
        print(f"Wrote: {overall_csv}")
    print(f"Wrote: {xlsx_path}")

    # Heatmaps for the most useful rates
    plot_heatmap(summary_df, "pseudo_detection_rate", "Pseudo-word detection rate", args.out_dir / "heatmap_detection_rate.png")
    plot_heatmap(summary_df, "meaning_attribution_rate", "Meaning attribution rate", args.out_dir / "heatmap_meaning_attribution.png")
    plot_heatmap(summary_df, "refusal_rate", "Refusal rate (heuristic)", args.out_dir / "heatmap_refusal_rate.png")
    plot_heatmap(summary_df, "implicit_normalization_rate", "Implicit normalization rate", args.out_dir / "heatmap_implicit_normalization.png")

    print("Wrote heatmaps in:", args.out_dir)


if __name__ == "__main__":
    main()
