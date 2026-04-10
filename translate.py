import os
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from datasets import load_dataset
from transformers import pipeline


MODEL_ID = "google/translategemma-12b-it"
DATASET_ID = "CohereLabs/global-mgsm"
SPLIT = "test"

LIMIT = 0

BATCH_SIZE = 8
MAX_NEW_TOKENS = 256
SOURCE_LANG = "en"
TARGET_LANG = "it-IT"

OUT_DIR = "global_mgsm"

# Languages to extract directly from dataset
LANG_SUBSETS = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
}

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def chunked(xs: List[str], n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def build_conversations(
    texts: List[str], target_lang: str
) -> List[List[Dict[str, Any]]]:
    """
    Build Translategemma-style chat payloads.
    """
    return [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": SOURCE_LANG,
                        "target_lang_code": target_lang,
                        "text": t,
                    }
                ],
            }
        ]
        for t in texts
    ]


def extract_translation(item: Any) -> str:
    """
    Robust extraction from pipeline output.
    """
    if isinstance(item, list) and item:
        item = item[0]

    generated = item.get("generated_text") if isinstance(item, dict) else item

    if isinstance(generated, str):
        return generated.strip()

    if isinstance(generated, list) and generated:
        last = generated[-1]

        if isinstance(last, dict):
            content = last.get("content", "")
        else:
            content = last

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            chunks = []
            for c in content:
                if isinstance(c, dict) and "text" in c:
                    chunks.append(str(c["text"]))
                else:
                    chunks.append(str(c))
            return " ".join(chunks).strip()

        return str(content).strip()

    return str(generated).strip()


def translate_column(
    texts: List[str],
    textgen,
    target_lang: str,
    batch_size: int,
) -> List[str]:
    results: List[str] = []
    total = len(texts)

    for i, batch in enumerate(chunked(texts, batch_size)):
        start = i * batch_size + 1
        end = min((i + 1) * batch_size, total)
        print(f"  Translating batch {start}-{end} / {total}", flush=True)

        convs = build_conversations(batch, target_lang)
        outs = textgen(
            convs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            truncation=True,
        )

        results.extend(extract_translation(o) for o in outs)

    return results


def load_language_subset(
    dataset_id: str,
    subset: str,
    split: str,
    limit: int = 0,
) -> pd.DataFrame:
    """
    Load one language subset such as 'en', 'de', 'fr', 'es'.
    Expected columns from HF viewer: question, answer, instruction, answer_prefix.
    """
    print(f"Loading subset '{subset}'...", flush=True)
    ds = load_dataset(dataset_id, subset, split=split)

    if limit and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    required_cols = ["question", "answer"]
    for col in required_cols:
        if col not in ds.column_names:
            raise ValueError(
                f"Subset '{subset}' is missing required column '{col}'. "
                f"Found columns: {ds.column_names}"
            )

    optional_cols = [c for c in ["instruction", "answer_prefix"] if c in ds.column_names]

    data = {
        "question": list(ds["question"]),
        "answer": list(ds["answer"]),
    }

    for c in optional_cols:
        data[c] = list(ds[c])

    return pd.DataFrame(data)


def save_per_language_csv(
    df: pd.DataFrame,
    lang_code: str,
    out_dir: str,
):
    out_path = os.path.join(out_dir, f"global_mgsm_{lang_code}.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load existing dataset subsets
    language_frames: Dict[str, pd.DataFrame] = {}
    for lang_code in LANG_SUBSETS:
        language_frames[lang_code] = load_language_subset(
            DATASET_ID, lang_code, SPLIT, LIMIT
        )

    # Basic consistency check
    lengths = {lang: len(df) for lang, df in language_frames.items()}
    print(f"Loaded row counts: {lengths}", flush=True)

    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(
            f"Language subsets do not have matching lengths: {lengths}"
        )

    n = next(iter(unique_lengths))
    print(f"Using {n} aligned examples.", flush=True)

    # Load translation model
    print(f"Loading model {MODEL_ID}...", flush=True)
    textgen = pipeline(
        "text-generation",
        model=MODEL_ID,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    dm = getattr(textgen.model, "hf_device_map", None)
    if dm:
        cpu_entries = sum(1 for v in dm.values() if v == "cpu")
        print(f"Device map: cpu-mapped entries = {cpu_entries}", flush=True)

    # Translate EN question -> IT
    print(f"Translating English questions to Italian ({TARGET_LANG})...", flush=True)
    en_questions = language_frames["en"]["question"].tolist()

    translated_it_questions = translate_column(
        en_questions,
        textgen,
        target_lang=TARGET_LANG,
        batch_size=BATCH_SIZE,
    )

    if len(translated_it_questions) != n:
        raise RuntimeError(
            f"Translation count mismatch: got {len(translated_it_questions)}, expected {n}"
        )

    # Build combined dataframe
    # We keep answers from each language subset separately, since wording/localization
    # may differ, though the numeric answer should usually match.
    combined_df = pd.DataFrame(
        {
            "question_en": language_frames["en"]["question"],
            "answer_en": language_frames["en"]["answer"],
            "question_de": language_frames["de"]["question"],
            "answer_de": language_frames["de"]["answer"],
            "question_fr": language_frames["fr"]["question"],
            "answer_fr": language_frames["fr"]["answer"],
            "question_es": language_frames["es"]["question"],
            "answer_es": language_frames["es"]["answer"],
            "question_it": translated_it_questions,
            # Keep English answer as the answer for Italian too, since you are only
            # translating the question and MGSM answers are numeric strings.
            "answer_it": language_frames["en"]["answer"],
        }
    )

    # Add instructions/prefixes where available
    for lang_code in ["en", "de", "fr", "es"]:
        df_lang = language_frames[lang_code]
        if "instruction" in df_lang.columns:
            combined_df[f"instruction_{lang_code}"] = df_lang["instruction"]
        if "answer_prefix" in df_lang.columns:
            combined_df[f"answer_prefix_{lang_code}"] = df_lang["answer_prefix"]

    # For Italian, reuse English instruction/prefix unless you want to translate those too
    if "instruction" in language_frames["en"].columns:
        combined_df["instruction_it_source_en"] = language_frames["en"]["instruction"]
    if "answer_prefix" in language_frames["en"].columns:
        combined_df["answer_prefix_it_source_en"] = language_frames["en"]["answer_prefix"]

    # Save combined CSV
    combined_out = os.path.join(OUT_DIR, "global_mgsm_en_de_fr_es_it.csv")
    combined_df.to_csv(combined_out, index=False)
    print(f"Wrote {combined_out} with {len(combined_df)} rows.", flush=True)

    # Save per-language CSVs
    save_per_language_csv(language_frames["en"], "en", OUT_DIR)
    save_per_language_csv(language_frames["de"], "de", OUT_DIR)
    save_per_language_csv(language_frames["fr"], "fr", OUT_DIR)
    save_per_language_csv(language_frames["es"], "es", OUT_DIR)

    it_df = pd.DataFrame(
        {
            "question": translated_it_questions,
            "answer": language_frames["en"]["answer"],
        }
    )
    if "instruction" in language_frames["en"].columns:
        it_df["instruction_source_en"] = language_frames["en"]["instruction"]
    if "answer_prefix" in language_frames["en"].columns:
        it_df["answer_prefix_source_en"] = language_frames["en"]["answer_prefix"]

    save_per_language_csv(it_df, "it", OUT_DIR)

    print("\nAll done.", flush=True)
    print(f"Files are in: {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()