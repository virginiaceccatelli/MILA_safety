import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

# load multilingual results
df = pd.read_excel("results.xlsx")
df = df[
    ["model","language","refused_rate","jailbroken_rate","deflected_rate"]
].drop_duplicates()

# load english baseline
with open("summary_en.json", "r") as f:
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

df_en = pd.DataFrame(rows)

# combine datasets
df_all = pd.concat([df, df_en], ignore_index=True)

def relabel(lang):
    if lang == "en":
        return "en"
    if "-" in lang:
        return lang
    return f"{lang}-en"

df_all["language"] = df_all["language"].apply(relabel)

# build heatmap matrix
pivot = df_all.pivot(
    index="model",
    columns="language",
    values="jailbroken_rate"
)

plt.figure(figsize=(10,6))

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

baseline = pivot["en"]

delta = pivot.subtract(baseline, axis=0)

plt.figure(figsize=(10,6))

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
