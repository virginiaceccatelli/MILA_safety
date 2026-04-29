import json
import os
import re
import pandas as pd
from collections import Counter
from jiwer import wer, cer

MODELS = [
    "flamingo", "gemini", "gemma3", "gemma4", "gpt",
    "qwen3omni", "qwen25omni", "salmonn", "voxtral"
]
LANGUAGES = ["de", "es", "fr", "it", "en"]

PATH_TEMPLATE = "fleurs_sib/results_sib_fleurs_{model}/{model}_{lang}.json"

ALIASES = {
    "science": "science technology",
    "technology": "science technology",
    "science and technology": "science technology",
    "sci tech": "science technology",
    "stem": "science technology",
    "sport": "sports",
    "arts and entertainment": "entertainment",
    "media": "entertainment",
    "culture": "entertainment",
    "government": "politics",
    "political science": "politics",
    "international relations": "politics",
    "earth science": "science technology",
}

def strip_json_wrapper(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
            for v in parsed.values():
                if isinstance(v, str):
                    return v.strip()
        except json.JSONDecodeError:
            pass
    pattern = r'^\{\\?"?\w+\\?"?\s*:\s*\\?"?(.*?)\\?"?\}?$'
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def normalize(text: str) -> str:
    text = strip_json_wrapper(text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return ALIASES.get(text, text)


def token_f1(hyp: str, ref: str) -> float:
    hyp_tokens = hyp.split()
    ref_tokens = ref.split()
    if not ref_tokens and not hyp_tokens:
        return 1.0
    if not ref_tokens or not hyp_tokens:
        return 0.0
    hyp_counts = Counter(hyp_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((hyp_counts & ref_counts).values())
    precision = overlap / len(hyp_tokens)
    recall    = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


records = []

for model in MODELS:
    for lang in LANGUAGES:
        path = PATH_TEMPLATE.format(model=model, lang=lang)

        if not os.path.exists(path):
            print(f"[MISSING] {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            for key in ("results", "data", "items", model, lang):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                for v in data.values():
                    if isinstance(v, list):
                        data = v
                        break

        hypotheses, references = [], []
        f1_scores = []
        correct = 0
        no_response = 0

        for item in data:
            hyp = item.get("response", "")
            ref = item.get("ground_truth", "")

            if not isinstance(hyp, str) or not isinstance(ref, str):
                no_response += 1
                continue
            if hyp.strip() in ('""', "''", ""):
                no_response += 1
                continue
            if ref.strip() == "":
                no_response += 1
                continue

            hyp_n = normalize(hyp)
            ref_n = normalize(ref)

            hypotheses.append(hyp_n)
            references.append(ref_n)
            f1_scores.append(token_f1(hyp_n, ref_n))
            if hyp_n == ref_n:
                correct += 1

        if not hypotheses:
            print(f"[EMPTY]   {path}  (no_response={no_response})")
            continue

        n = len(hypotheses)
        accuracy        = correct / n
        word_error_rate = wer(references, hypotheses)
        char_error_rate = cer(references, hypotheses)
        mean_f1         = sum(f1_scores) / n

        records.append({
            "model":       model,
            "language":    lang,
            "n_samples":   n,
            "no_response": no_response,
            "correct":     correct,
            "accuracy":    round(accuracy,        4),
            "WER":         round(word_error_rate, 4),
            "CER":         round(char_error_rate, 4),
            "token_F1":    round(mean_f1,         4),
        })

        print(
            f"[OK] {model:12s} {lang}  n={n:4d}  "
            f"acc={accuracy:.3f}  WER={word_error_rate:.3f}  "
            f"CER={char_error_rate:.3f}  token_F1={mean_f1:.3f}"
        )


if not records:
    print("\nNo results found.")
else:
    df = pd.DataFrame(records)

    print("\nSUMMARY PIVOT TABLES  (models × languages)")
    for metric in ("accuracy", "WER", "CER", "token_F1"):
        pivot = df.pivot_table(index="model", columns="language", values=metric)
        pivot["avg"] = pivot.mean(axis=1)
        pivot.loc["avg"] = pivot.mean()
        print(f"\n── {metric} ──")
        print(pivot.round(4).to_string())

    out_csv = "sib_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nFull results saved to: {out_csv}")

    try:
        out_xlsx = "sib_results.xlsx"
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="detail", index=False)
            for metric in ("accuracy", "WER", "CER", "token_F1"):
                pivot = df.pivot_table(index="model", columns="language", values=metric)
                pivot["avg"] = pivot.mean(axis=1)
                pivot.loc["avg"] = pivot.mean()
                pivot.round(4).to_excel(writer, sheet_name=metric)
        print(f"Excel summary saved to: {out_xlsx}")
    except ImportError:
        print("(openpyxl not installed — skipping Excel export)")