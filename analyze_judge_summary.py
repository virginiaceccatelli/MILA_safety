import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_summary_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_tables(data: dict):
    by_model_lang = data.get("by_model_lang", [])
    df = pd.DataFrame(by_model_lang)

    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame()

    # Ensure columns exist
    expected = [
        "model", "lang", "n_items",
        "harmless_meaning_count", "harmful_meaning_count", "noise_count",
        "harmless_meaning_rate", "harmful_meaning_rate", "noise_rate"
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = None

    df = df.sort_values(["model", "lang"]).reset_index(drop=True)

    # Overall by model
    model_rows = []
    for model, g in df.groupby("model", dropna=False):
        n = int(g["n_items"].fillna(0).sum())
        harmless = int(g["harmless_meaning_count"].fillna(0).sum())
        harmful = int(g["harmful_meaning_count"].fillna(0).sum())
        noise = int(g["noise_count"].fillna(0).sum())

        model_rows.append({
            "model": model,
            "n_items": n,
            "harmless_meaning_count": harmless,
            "harmful_meaning_count": harmful,
            "noise_count": noise,
            "harmless_meaning_rate": harmless / n if n else None,
            "harmful_meaning_rate": harmful / n if n else None,
            "noise_rate": noise / n if n else None,
        })

    by_model = pd.DataFrame(model_rows).sort_values("model").reset_index(drop=True)

    # Overall by language
    lang_rows = []
    for lang, g in df.groupby("lang", dropna=False):
        n = int(g["n_items"].fillna(0).sum())
        harmless = int(g["harmless_meaning_count"].fillna(0).sum())
        harmful = int(g["harmful_meaning_count"].fillna(0).sum())
        noise = int(g["noise_count"].fillna(0).sum())

        lang_rows.append({
            "lang": lang,
            "n_items": n,
            "harmless_meaning_count": harmless,
            "harmful_meaning_count": harmful,
            "noise_count": noise,
            "harmless_meaning_rate": harmless / n if n else None,
            "harmful_meaning_rate": harmful / n if n else None,
            "noise_rate": noise / n if n else None,
        })

    by_lang = pd.DataFrame(lang_rows).sort_values("lang").reset_index(drop=True)

    return df, by_model, by_lang


def save_tables(df: pd.DataFrame, by_model: pd.DataFrame, by_lang: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    csv1 = out_dir / "judge_summary_by_model_lang.csv"
    csv2 = out_dir / "judge_summary_by_model.csv"
    csv3 = out_dir / "judge_summary_by_lang.csv"
    xlsx = out_dir / "judge_summary_tables.xlsx"

    df.to_csv(csv1, index=False)
    by_model.to_csv(csv2, index=False)
    by_lang.to_csv(csv3, index=False)

    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="by_model_lang", index=False)
        by_model.to_excel(w, sheet_name="by_model", index=False)
        by_lang.to_excel(w, sheet_name="by_lang", index=False)

    return [csv1, csv2, csv3, xlsx]


def plot_heatmap(df: pd.DataFrame, value_col: str, title: str, out_path: Path):
    if df.empty:
        return

    pivot = df.pivot(index="model", columns="lang", values=value_col)
    if pivot.empty:
        return

    models = list(pivot.index)
    langs = list(pivot.columns)
    data = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(langs) + 2), max(4, 0.5 * len(models) + 2)))
    im = ax.imshow(np.nan_to_num(data, nan=-1.0), aspect="auto")

    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels(langs)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title(title)

    for i in range(len(models)):
        for j in range(len(langs)):
            v = data[i, j]
            label = "NA" if np.isnan(v) else f"{v:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_stacked_bar_by_model(by_model: pd.DataFrame, out_path: Path):
    if by_model.empty:
        return

    df = by_model.copy()
    x = np.arange(len(df))

    harmless = df["harmless_meaning_rate"].fillna(0).to_numpy()
    harmful = df["harmful_meaning_rate"].fillna(0).to_numpy()
    noise = df["noise_rate"].fillna(0).to_numpy()

    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(df) + 2), 5))
    ax.bar(x, harmless, label="Harmless")
    ax.bar(x, harmful, bottom=harmless, label="Harmful")
    ax.bar(x, noise, bottom=harmless + harmful, label="Noise")

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Pseudo-word meaning labels by model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_stacked_bar_by_lang(by_lang: pd.DataFrame, out_path: Path):
    if by_lang.empty:
        return

    df = by_lang.copy()
    x = np.arange(len(df))

    harmless = df["harmless_meaning_rate"].fillna(0).to_numpy()
    harmful = df["harmful_meaning_rate"].fillna(0).to_numpy()
    noise = df["noise_rate"].fillna(0).to_numpy()

    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(df) + 2), 5))
    ax.bar(x, harmless, label="Harmless")
    ax.bar(x, harmful, bottom=harmless, label="Harmful")
    ax.bar(x, noise, bottom=harmless + harmful, label="Noise")

    ax.set_xticks(x)
    ax.set_xticklabels(df["lang"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Pseudo-word meaning labels by language")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_readme(data: dict, out_dir: Path):
    text = f"""Judge model: {data.get('judge_model')}
Input file: {data.get('input_file')}

Files:
- judge_summary_by_model_lang.csv: main table by model and language
- judge_summary_by_model.csv: aggregated across languages
- judge_summary_by_lang.csv: aggregated across models
- judge_summary_tables.xlsx: all tables in one workbook
- heatmap_harmless_rate.png
- heatmap_harmful_rate.png
- heatmap_noise_rate.png
- stacked_bar_by_model.png
- stacked_bar_by_lang.png
"""
    path = out_dir / "README.txt"
    path.write_text(text, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=Path, required=True, help="Summary JSON from judge_pseudoword_summary.py")
    parser.add_argument("--out_dir", type=Path, default=Path("judge_analysis_out"), help="Output directory")
    args = parser.parse_args()

    data = load_summary_json(args.input_json)
    df, by_model, by_lang = build_tables(data)

    if df.empty:
        raise ValueError(f"No by_model_lang entries found in {args.input_json}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    table_paths = save_tables(df, by_model, by_lang, out_dir)

    heat1 = out_dir / "heatmap_harmless_rate.png"
    heat2 = out_dir / "heatmap_harmful_rate.png"
    heat3 = out_dir / "heatmap_noise_rate.png"
    bar1 = out_dir / "stacked_bar_by_model.png"
    bar2 = out_dir / "stacked_bar_by_lang.png"

    plot_heatmap(df, "harmless_meaning_rate", "Harmless meaning rate", heat1)
    plot_heatmap(df, "harmful_meaning_rate", "Harmful meaning rate", heat2)
    plot_heatmap(df, "noise_rate", "Noise rate", heat3)
    plot_stacked_bar_by_model(by_model, bar1)
    plot_stacked_bar_by_lang(by_lang, bar2)

    readme = write_readme(data, out_dir)

    for p in table_paths + [heat1, heat2, heat3, bar1, bar2, readme]:
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
