import torch
import re
import unicodedata
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import soundfile as sf

_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)
torch.load = _torch_load_compat

from TTS.api import TTS


# Folder containing:
#   global_mgsm_de.csv
#   global_mgsm_en.csv
#   global_mgsm_es.csv
#   global_mgsm_fr.csv
#   global_mgsm_it.csv
CSV_DIR = "mgsm/global_mgsm"

# Output folder for wav files + manifests
BASE_OUT_DIR = "audio_xtts_mgsm"

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# Use ONE of these:
SPEAKER_WAV: Optional[str] = None
SPEAKER_IDX: Optional[str] = "Ana Florence"

LIMIT_ROWS = 0               # 0 = all rows
MAX_CHARS_PER_CHUNK = 200
PAUSE_SILENCE_MS = 100

USE_GPU = torch.cuda.is_available()

# Only process these five single-language CSVs
TARGET_FILES = [
    "global_mgsm_de.csv",
    "global_mgsm_en.csv",
    "global_mgsm_es.csv",
    "global_mgsm_fr.csv",
    "global_mgsm_it.csv",
]

# XTTS language codes
LANG_MAP = {
    "de": "de",
    "en": "en",
    "es": "es",
    "fr": "fr",
    "it": "it",
}


def infer_lang_from_csv_name(csv_name: str) -> str:
    m = re.match(r"global_mgsm_([a-z]{2})\.csv$", Path(csv_name).name)
    if not m:
        raise ValueError(f"Could not infer language from filename: {csv_name}")
    code = m.group(1)
    if code not in LANG_MAP:
        raise ValueError(f"Unsupported language code '{code}' in filename: {csv_name}")
    return LANG_MAP[code]

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def flatten_text_for_xtts(text: str) -> str:
    """
    Convert multi-sentence text into one smooth sentence.
    Keeps meaning, removes sentence-boundary artifacts.
    """
    import unicodedata
    import re

    text = str(text)
    text = unicodedata.normalize("NFKC", text)

    # Standardize quotes
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")

    # Replace sentence boundaries with commas
    text = re.sub(r"[.!?]+", ",", text)

    # Keep commas but clean duplicates
    text = re.sub(r",+", ",", text)

    # Remove brackets only
    text = re.sub(r"[\[\]\(\)\{\}<>]", " ", text)

    # Fix spacing
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Remove trailing comma
    text = text.rstrip(",")

    return text

def clean_text_for_xtts(text: str) -> str:
    """
    - normalize unicode
    - standardize quotes/dashes
    - remove brackets
    - collapse repeated punctuation
    - keep normal punctuation and wording intact
    """
    text = str(text)
    text = unicodedata.normalize("NFKC", text)

    # Standardize quotes/apostrophes
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")

    # Standardize dashes/ellipsis, but minimally
    text = text.replace("—", " - ")
    text = text.replace("–", " - ")
    text = text.replace("…", "...")

    # Remove brackets only
    text = re.sub(r"[\[\]\(\)\{\}<>]", " ", text)

    # Collapse repeated punctuation
    text = re.sub(r"([!?.,])\1+", r"\1", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.?!])", r"\1", text)

    # Ensure a space after sentence punctuation if missing
    text = re.sub(r"([.?!,])([^\s])", r"\1 \2", text)

    return normalize_whitespace(text)

def tts_synthesize_text(
    tts: TTS,
    text: str,
    language: str,
    sr: int,
) -> np.ndarray:
    text = flatten_text_for_xtts(text)

    if not text:
        return np.zeros(0, dtype=np.float32)

    try:
        if SPEAKER_WAV is not None:
            wav = tts.tts(
                text=text,
                speaker_wav=SPEAKER_WAV,
                language=language,
            )
        else:
            wav = tts.tts(
                text=text,
                speaker=SPEAKER_IDX,
                language=language,
            )
    except Exception as e:
        print(f"[WARN] Failed text: {text[:120]!r} -> {e}")
        return np.zeros(0, dtype=np.float32)

    return np.asarray(wav, dtype=np.float32)

def build_jobs(csv_dir: str) -> list[dict]:
    jobs = []
    for filename in TARGET_FILES:
        csv_path = Path(csv_dir) / filename
        if not csv_path.exists():
            print(f"[WARN] Missing file, skipping: {csv_path}")
            continue

        lang = infer_lang_from_csv_name(filename)
        jobs.append({
            "csv_path": csv_path,
            "tag": Path(filename).stem,
            "language": lang,
        })
    return jobs


def run_job(job: dict, tts: TTS, sr: int):
    csv_path = job["csv_path"]
    tag = job["tag"]
    language = job["language"]

    out_dir = Path(BASE_OUT_DIR) / tag
    manifest_csv = Path(BASE_OUT_DIR) / f"{tag}_manifest.csv"

    df = pd.read_csv(csv_path)

    if "question" not in df.columns:
        raise ValueError(
            f"'question' column not found in {csv_path}. "
            f"Columns are: {list(df.columns)}"
        )

    if LIMIT_ROWS and LIMIT_ROWS > 0:
        df = df.iloc[:LIMIT_ROWS].copy()

    done_indices: set[int] = set()
    rows_out: list[dict] = []

    if manifest_csv.exists():
        try:
            existing = pd.read_csv(manifest_csv)
            done_indices = set(existing["row_index"].tolist())
            rows_out = existing.to_dict("records")
            print(f"[{tag}] Resuming — {len(done_indices)} rows already done")
        except Exception:
            pass

    remaining_indices = [i for i in df.index.tolist() if i not in done_indices]

    if not remaining_indices:
        print(f"[{tag}] Already complete — skipping")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"[{tag}]")
    print(f"  csv      : {csv_path.name}")
    print(f"  language : {language}")
    print(f"  rows todo: {len(remaining_indices)}")
    print(f"{'=' * 60}")

    for n, i in enumerate(remaining_indices, start=1):
        raw_text = str(df.at[i, "question"])
        clean_text = clean_text_for_xtts(raw_text)

        if not clean_text:
            continue

        audio = tts_synthesize_text(
            tts=tts,
            text=clean_text,
            language=language,
            sr=sr,
        )

        out_path = out_dir / f"row_{i:04d}.wav"
        sf.write(str(out_path), audio, sr)

        rows_out.append({
            "row_index": i,
            "wav_path": str(out_path),
            "language": language,
            "text_raw": raw_text,
            "text_clean": clean_text,
            "source_csv": str(csv_path),
        })

        if n % 10 == 0:
            pd.DataFrame(rows_out).to_csv(manifest_csv, index=False)
            print(f"  [{tag}] {n}/{len(remaining_indices)} newly done", flush=True)

    pd.DataFrame(rows_out).to_csv(manifest_csv, index=False)
    print(f"  [{tag}] ✓ {len(rows_out)} total wav files listed in manifest")
    print(f"  [{tag}] ✓ audio saved to {out_dir}")


def main():
    Path(BASE_OUT_DIR).mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(CSV_DIR)
    if not jobs:
        print(f"No target CSV files found in '{CSV_DIR}'.")
        return

    print(f"Found {len(jobs)} jobs:")
    for j in jobs:
        print(f"  {j['tag']:20s} language={j['language']}")

    print(f"\nLoading TTS model: {MODEL_NAME}")
    tts = TTS(MODEL_NAME)
    tts.to("cuda" if USE_GPU else "cpu")

    sr = 24000
    try:
        if hasattr(tts, "synthesizer") and hasattr(tts.synthesizer, "output_sample_rate"):
            sr = int(tts.synthesizer.output_sample_rate)
    except Exception:
        pass

    for job in jobs:
        run_job(job, tts, sr)

    print("\n" + "=" * 60)
    print("ALL JOBS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()