import os
from typing import Any, Dict, List

import pandas as pd
import torch
from datasets import load_dataset
from transformers import pipeline

MODEL_ID = "google/translategemma-12b-it"  
SPLIT = "benign"                         
LIMIT = 0         # set 0 for all
BATCH_SIZE = 4
MAX_NEW_TOKENS = 128
SOURCE_LANG = "en"
TARGET_LANG = "it-IT"
OUT_CSV = "jbb_behaviors_benign_it_12b.csv"

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def chunked(xs: List[str], n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]


def build_conversations(texts: List[str]) -> List[List[Dict[str, Any]]]:
    # Batch must be: list of conversations; each conversation is a list of messages
    return [[{
        "role": "user",
        "content": [{
            "type": "text",
            "source_lang_code": SOURCE_LANG,
            "target_lang_code": TARGET_LANG,
            "text": t,
        }],
    }] for t in texts]


def extract_translation(item: Any) -> str:
    if isinstance(item, list) and item:
        item = item[0]

    generated = item.get("generated_text") if isinstance(item, dict) else item

    # Sometimes it's a plain string
    if isinstance(generated, str):
        return generated.strip()

    # Sometimes it's a list of chat messages
    if isinstance(generated, list) and generated:
        last = generated[-1]
        content = last.get("content", "") if isinstance(last, dict) else last

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict) and "text" in chunk:
                    return str(chunk["text"]).strip()
            return " ".join(str(x) for x in content).strip()

        return str(content).strip()

    return str(generated).strip()


def main():
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split=SPLIT)

    if "Goal" not in ds.column_names:
        raise RuntimeError(f"'Goal' not found; columns are: {ds.column_names}")

    if LIMIT and LIMIT > 0:
        ds = ds.select(range(min(LIMIT, len(ds))))

    goals = list(ds["Goal"])

    textgen = pipeline(
        "text-generation",
        model=MODEL_ID,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        )

    # Quick check for CPU offload 
    dm = getattr(textgen.model, "hf_device_map", None)
    if dm:
        cpu_entries = sum(1 for v in dm.values() if v == "cpu")
        print(f"Device map: cpu-mapped entries = {cpu_entries}", flush=True)

    translations: List[str] = []
    for batch in chunked(goals, BATCH_SIZE):
        convs = build_conversations(batch)
        outs = textgen(convs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, truncation=True)
        translations.extend(extract_translation(o) for o in outs)

    df = ds.to_pandas()
    df["Goal_de"] = translations
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} with {len(df)} rows", flush=True)


if __name__ == "__main__":
    main()