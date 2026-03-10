import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

# load multilingual results
df = pd.read_excel("judge_results_summaries/full_results.xlsx")
df = df[
    ["model","language","refused_rate","jailbroken_rate","deflected_rate"]
].drop_duplicates()

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

# load english baseline
df_en = load_summary_json("judge_results_summaries/summary_en.json")

# load single-language baselines (non-english)
df_single = load_summary_json("judge_results_summaries/summary_single-langs.json")

df_gemini = load_summary_json("results_one/summary_gemini.json")
df_gemini_single = load_summary_json("judge_results_summaries/summary_gemini_single.json")
df_salmonn = load_summary_json("judge_results_summaries/summary_salmonn.json")
def normalize_cs_tags(lang):
    if lang.startswith("cs-"):
        return lang[3:] + "-en"
    return lang

df_gemini["language"]  = df_gemini["language"].apply(normalize_cs_tags)
df_gemini_single["language"]  = df_gemini_single["language"].apply(normalize_cs_tags)
df_salmonn["language"] = df_salmonn["language"].apply(normalize_cs_tags)

# relabel results.xlsx languages: bare codes (e.g. 'fr') become 'fr-en', 'en' stays 'en'
def relabel_multilingual(lang):
    if lang == "en":
        return "en"
    if "-" in lang:
        return lang
    return f"{lang}-en"

df["language"] = df["language"].apply(relabel_multilingual)

# single-lang and english baseline labels are kept as-is ('fr', 'de', 'es', 'it', 'en')

# combine all datasets
df_all = pd.concat([df, df_en, df_single, df_gemini, df_gemini_single, df_salmonn], ignore_index=True)

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

# delta relative to english baseline
baseline = pivot["en"]
delta = pivot.subtract(baseline, axis=0)

plt.figure(figsize=(10, 6))
sns.heatmap(
    delta,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Change in Jailbreak Rate Relative to English")
plt.xlabel("Language")
plt.ylabel("Model")
plt.tight_layout()
plt.savefig("heatmap_delta_en.png")