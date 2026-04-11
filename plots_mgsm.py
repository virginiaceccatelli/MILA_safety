import json
import math as _math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SUMMARY_PATH = "mgsm/judge_results_mgsm/summary.json"
OUT_DIR      = Path("plots_mgsm")
OUT_DIR.mkdir(exist_ok=True)

with open(SUMMARY_PATH) as f:
    data = json.load(f)

rows = []
for model, langs in data.items():
    for lang, metrics in langs.items():
        n_judged    = metrics["n_judged"]
        correct     = metrics["correct"]
        incorrect   = metrics["incorrect"]
        no_answer   = max(0, n_judged - (correct + incorrect))
        n_total     = metrics["n_total"]
        no_response = metrics.get("n_no_response", 0)

        rows.append({
            "model":          model,
            "language":       lang,
            "n_total":        n_total,
            "n_judged":       n_judged,
            "n_no_response":  no_response,
            "correct":        correct,
            "incorrect":      incorrect,
            "no_answer":      no_answer,
            "correct_rate":   metrics["correct_rate"],
            "incorrect_rate": metrics["incorrect_rate"],
            "no_answer_rate": max(0.0, 1.0 - (metrics["correct_rate"] + metrics["incorrect_rate"])),
        })

df = pd.DataFrame(rows).drop_duplicates(subset=["model", "language"])

single_lang  = sorted([c for c in df["language"].unique() if "-" not in c and c != "en"])
mixed_lang   = sorted([c for c in df["language"].unique() if "-" in c])
ordered_cols = ["en"] + single_lang + mixed_lang
languages    = ordered_cols
models       = sorted(df["model"].unique())

summary = (
    df
    .assign(language=pd.Categorical(df["language"], categories=ordered_cols, ordered=True))
    .sort_values(["model", "language"])
    [[
        "model", "language",
        "n_total", "n_judged", "n_no_response",
        "correct", "incorrect", "no_answer",
        "correct_rate", "incorrect_rate", "no_answer_rate",
    ]]
    .reset_index(drop=True)
)

summary.to_csv("summary_table.csv", index=False)
summary.to_excel("summary_table.xlsx", index=False)
print("Saved summary_table.csv / .xlsx")


heatmap_specs = [
    ("correct_rate",   "Greens",  "Correct Rate"),
    ("incorrect_rate", "Reds",    "Incorrect Rate"),
    ("no_answer_rate", "Oranges", "No-Answer Rate"),
]

fig, axes = plt.subplots(1, 3, figsize=(18, max(4, len(models) * 0.8 + 2)))

for ax, (metric, cmap, title) in zip(axes, heatmap_specs):
    pivot = (
        df.pivot(index="model", columns="language", values=metric)
          .reindex(columns=languages)
    )
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, vmin=0, vmax=1, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title(title)
    ax.set_xlabel("Language")
    ax.set_ylabel("Model")

fig.suptitle("Accuracy by Model and Language (MGSM)", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "heatmap_all_outcomes.png", bbox_inches="tight")
plt.close()
print("Saved heatmap_all_outcomes.png")


# ── 2. Line charts — all three outcomes on one chart per model ─────────────────
n_models = len(models)
n_cols   = min(3, n_models)
n_rows   = _math.ceil(n_models / n_cols)

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(7 * n_cols, 4 * n_rows),
                         sharey=True, sharex=False)
axes_flat = np.array(axes).flatten().tolist()

line_styles = {
    "correct_rate":   ("o", "-",  "#4caf50", "Correct"),
    "incorrect_rate": ("s", "--", "#f44336", "Incorrect"),
    "no_answer_rate": ("^", ":",  "#ff9800", "No Answer"),
}

for ax, model in zip(axes_flat, models):
    model_df = df[df["model"] == model].set_index("language").reindex(languages)
    for metric, (marker, ls, color, label) in line_styles.items():
        ax.plot(
            languages,
            model_df[metric] * 100,
            marker=marker, linestyle=ls, color=color, label=label,
        )
    ax.set_title(model)
    ax.set_xticks(range(len(languages)))
    ax.set_xticklabels(languages, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Rate (%)")

for ax in axes_flat[n_models:]:
    ax.set_visible(False)

handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Outcome", loc="upper right",
           bbox_to_anchor=(1.0, 1.0))
fig.suptitle("Correct / Incorrect / No-Answer by Language and Model (MGSM)", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "linechart_all_outcomes.png", bbox_inches="tight")
plt.close()
print("Saved linechart_all_outcomes.png")


# ── 3. Stacked bar — correct / incorrect / no-answer / no-response ────────────
x        = np.arange(len(languages))
bar_width = 0.6

colors = {
    "Correct":     "#4caf50",
    "Incorrect":   "#f44336",
    "No Answer":   "#ff9800",
    "No Response": "#9e9e9e",
}

fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(6 * n_cols, 5 * n_rows),
                          sharey=True)
axes_flat = np.array(axes).flatten().tolist()

for ax, model in zip(axes_flat, models):
    mdf         = df[df["model"] == model].set_index("language").reindex(languages)
    total       = mdf["n_total"].fillna(1)
    correct_pct     = mdf["correct"].fillna(0)      / total * 100
    incorrect_pct   = mdf["incorrect"].fillna(0)    / total * 100
    no_answer_pct   = mdf["no_answer"].fillna(0)    / total * 100
    no_response_pct = mdf["n_no_response"].fillna(0) / total * 100

    ax.bar(x, correct_pct,     width=bar_width, label="Correct",
           color=colors["Correct"])
    ax.bar(x, incorrect_pct,   width=bar_width, label="Incorrect",
           color=colors["Incorrect"],   bottom=correct_pct)
    ax.bar(x, no_answer_pct,   width=bar_width, label="No Answer",
           color=colors["No Answer"],   bottom=correct_pct + incorrect_pct)
    ax.bar(x, no_response_pct, width=bar_width, label="No Response",
           color=colors["No Response"], bottom=correct_pct + incorrect_pct + no_answer_pct)

    ax.set_title(model)
    ax.set_xticks(x)
    ax.set_xticklabels(languages, rotation=45, ha="right")
    ax.set_ylim(0, 100)

for ax in axes_flat[n_models:]:
    ax.set_visible(False)

for row in range(n_rows):
    axes_flat[row * n_cols].set_ylabel("Share of Questions (%)")

handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Outcome", loc="upper right",
           bbox_to_anchor=(1.0, 1.0))
fig.suptitle("Outcomes by Language and Model (MGSM)", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "stacked_outcomes.png", bbox_inches="tight")
plt.close()
print("Saved stacked_outcomes.png")