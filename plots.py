import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

# load multilingual results
# df = pd.read_excel("malicious/judge_results_summaries/full_results.xlsx")
# df = df[
#     ["model","language","refused_rate","jailbroken_rate","deflected_rate"]
# ].drop_duplicates()

def load_summary_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    rows = []
    for model, langs in data.items():
        for lang, metrics in langs.items():
            rows.append({
                "model": model,
                "language": lang,
                "refused_rate": metrics["refused_rate"],
                "jailbroken_rate": metrics["jailbroken_rate"],
                "deflected_rate": metrics["deflected_rate"],
                "n_total": metrics["n_total"]
            })
    return pd.DataFrame(rows)

# MALICIOUS
# df_en = load_summary_json("malicious/judge_results_summaries/summary_en.json")
# df_single = load_summary_json("malicious/judge_results_summaries/summary_single-langs.json")
# df_gemini = load_summary_json("malicious/judge_results_summaries/summary_gemini.json")
# df_gemini_single = load_summary_json("malicious/judge_results_summaries/summary_gemini_single.json")
# df_salmonn = load_summary_json("malicious/judge_results_summaries/summary_salmonn.json")
# df_gemma = load_summary_json("malicious/judge_results_summaries/summary_gemma.json")

# GIBBERISH 2
# df = load_summary_json("gibberish_10/judge_results_gibberish/summary.json")
# df_gemma = load_summary_json("gibberish_10/judge_results_gibberish/judge_results_gemma/summary.json")
# df_qwen3 = load_summary_json("gibberish_10/judge_results_gibberish/judge_results_qwen3/summary.json")


def normalize_cs_tags(lang):
    if lang.startswith("cs-"):
        return "en-" + lang[3:]
    return lang

# MALICIOUS
# df_en["language"] = df_en["language"].apply(normalize_cs_tags)
# df_single["language"] = df_single["language"].apply(normalize_cs_tags)
# df_gemini["language"] = df_gemini["language"].apply(normalize_cs_tags)
# df_gemini_single["language"] = df_gemini_single["language"].apply(normalize_cs_tags)
# df_salmonn["language"]  = df_salmonn["language"].apply(normalize_cs_tags)

# relabel results.xlsx languages: bare codes (e.g. 'fr') become 'fr-en', 'en' stays 'en'
def relabel_multilingual(lang):
    if lang == "en":
        return "en"
    if "-" in lang:
        return lang
    return f"en-{lang}"

#MALICIOUS
# df["language"] = df["language"].apply(relabel_multilingual)
# df_all = pd.concat([df, df_en, df_single, df_gemini, df_gemini_single, df_salmonn, df_gemma], ignore_index=True)

# GIBBERISH 2
# df_all = pd.concat([df, df_gemma, df_qwen3], ignore_index=True)

df_all = load_summary_json("judge_results_malicious/summary.json")

# drop duplicate (model, language) pairs — keep first occurrence
df_all = df_all.drop_duplicates(subset=["model", "language"])

# build heatmap matrix
pivot = df_all.pivot(
    index="model",
    columns="language",
    values="jailbroken_rate"
)

# reorder columns: 'en' first, then single-lang baselines (no hyphen, not 'en'), then mixed (hyphenated)
single_lang = sorted([c for c in pivot.columns if "-" not in c and c != "en"])
mixed_lang  = sorted([c for c in pivot.columns if "-" in c])
ordered_cols = ["en"] + single_lang + mixed_lang
pivot = pivot[ordered_cols]
pivot.columns = pd.Index(ordered_cols)  # strip any categorical index so seaborn respects the order

# build summary table in the same language order
summary = (
    df_all
    .assign(language=pd.Categorical(df_all["language"], categories=ordered_cols, ordered=True))
    .sort_values(["model", "language"])
    .rename(columns={
        "refused_rate":    "refusal",
        "deflected_rate":  "deflection",
        "jailbroken_rate": "JSR",
    })[["model", "language", "refusal", "deflection", "JSR"]]
    .reset_index(drop=True)
)

summary.to_csv("summary_table.csv", index=False)
summary.to_excel("summary_table.xlsx", index=False)

plt.figure(figsize=(10, 6))
sns.heatmap(
    pivot,
    annot=True,
    fmt=".2f",
    cmap="Reds"
)
plt.title("Jailbreak Rate by Model and Language")
plt.xlabel("Language")
plt.ylabel("Model")
plt.tight_layout()
plt.savefig("heatmap.png")

# Shared language order (reuse ordered_cols from above)
models = df_all["model"].unique()
languages = ordered_cols  # already computed: ["en"] + single_lang + mixed_lang
 
# ── 1. LINE CHART: metric rates per language, one line per model ───────────────
#    Mirrors the style of figure (a): dashed lines with markers, one line per series.
#    We plot jailbroken_rate (JSR) across languages; swap the `metric` variable to
#    plot refusal or deflection instead.
 
metric      = "jailbroken_rate"
metric_label = "Jailbreak Success Rate (%)"
 
fig, ax = plt.subplots(figsize=(10, 5))
 
for model in models:
    model_df = (
        df_all[df_all["model"] == model]
        .set_index("language")
        .reindex(languages)          # keep the shared column order
    )
    ax.plot(
        languages,
        model_df[metric] * 100,      # convert to percentage
        marker="o",
        linestyle="--",
        label=model,
    )
 
ax.set_title("Jailbreak Rate by Model across Languages")
ax.set_xlabel("Language")
ax.set_ylabel(metric_label)
ax.set_xticks(range(len(languages)))
ax.set_xticklabels(languages, rotation=45, ha="right")
ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()
plt.savefig("linechart_jsr_by_language.png")
plt.close()
 
# ── 2. GROUPED BAR CHART: all three metrics per model, grouped by language ─────
#    Mirrors the style of figure (b): clustered bars, one group per x-tick,
#    one bar colour per metric (refusal / deflection / JSR).
 
metrics = {
    "refused_rate":   "Refusal",
    "deflected_rate": "Deflection",
    "jailbroken_rate": "JSR",
}
 
import numpy as np
import math
 
n_languages = len(languages)
n_metrics   = len(metrics)
bar_width   = 0.25
x           = np.arange(n_languages)
 
n_models = len(models)
n_cols   = 4
n_rows   = math.ceil(n_models / n_cols)   # 2 rows for 5-7 models, etc.
 
fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(6 * n_cols, 5 * n_rows),
    sharey=True,
)
 
# Flatten to a 1-D list so we can zip regardless of shape
axes_flat = np.array(axes).flatten().tolist()
 
for ax, model in zip(axes_flat, models):
    model_df = (
        df_all[df_all["model"] == model]
        .set_index("language")
        .reindex(languages)
    )
    for i, (col, label) in enumerate(metrics.items()):
        offset = (i - n_metrics / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            model_df[col].fillna(0) * 100,
            width=bar_width,
            label=label,
        )
 
    ax.set_title(model)
    ax.set_xlabel("Language")
    ax.set_xticks(x)
    ax.set_xticklabels(languages, rotation=45, ha="right")
    ax.set_ylim(0, 100)
 
# Hide any unused subplot cells
for ax in axes_flat[n_models:]:
    ax.set_visible(False)
 
# Y-label on leftmost visible axes in each row
for row in range(n_rows):
    axes_flat[row * n_cols].set_ylabel("Rate (%)")
 
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    title="Metric",
    loc="upper right",
    bbox_to_anchor=(1.0, 1.0),
)
fig.suptitle("Refusal / Deflection / JSR by Language and Model", y=1.02)
plt.tight_layout()
plt.savefig("barchart_metrics_by_language.png", bbox_inches="tight")
plt.close()
 
