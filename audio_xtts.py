import torch
import os
import re
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import soundfile as sf
import json

_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)
torch.load = _torch_load_compat

from TTS.api import TTS

FOREIGN_JSON_DIR  = "malicious/output_json"        
BASE_OUT_DIR     = "audio_xtts_foreign"

MODEL_NAME   = "tts_models/multilingual/multi-dataset/xtts_v2"
SPEAKER_WAV: Optional[str] = None
SPEAKER_IDX: Optional[str] = "Ana Florence"

LIMIT_ROWS          = 0    # 0 = all
MAX_CHARS_PER_CHUNK = 250
PAUSE_SILENCE_MS    = 120
USE_GPU             = torch.cuda.is_available()

# Map nordic_lang field -> XTTS language code
# NORDIC_LANG_CODE = {
#     "Danish":    "da",
#     "Norwegian": "no",
#     "Finnish":   "fi",
#     "Icelandic": "is",
#     "Swedish":   "sv",
# }
FOREIGN_LANG_CODE = {
    "Hungarian": "hu",
}

# Map language codes used in filenames -> XTTS language code
FILENAME_LANG_CODE = {
    "de": "de",
    "es": "es",
    "fr": "fr",
    "it": "it",
    "en": "en",
}

def infer_base_language(json_filename: str) -> str:
    """
    Always returns "fi" (Finnish).
    Finnish is used as the single TTS rendering language across all sentences
    because its typological distance from all base languages (German, French,
    Spanish, Italian, English) maximises acoustic disruption and model confusion.
    """
    return "hu"


def build_jobs(json_dir: str) -> list[dict]:
    """
    One job per JSON file in json_dir.
    Each job: {
        "json_path": Path,
        "tag":       str,   # used for output subfolder + manifest name
        "base_lang": str,   # XTTS language code for the non-Nordic content
    }
    """
    jobs = []
    for f in sorted(Path(json_dir).glob("*_foreign.json")):
        tag       = f.stem   
        base_lang = infer_base_language(f.name)
        jobs.append({
            "json_path": f,
            "tag":       tag,
            "base_lang": base_lang,
        })
    return jobs


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def split_into_sentences(text: str) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    parts = re.split(r"(?<=[\.\!\?\:;])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def pack_sentences(sentences: List[str], max_chars: int) -> List[str]:
    chunks, cur = [], ""
    for s in sentences:
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= max_chars:
            cur = cur + " " + s
        else:
            chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks

def silence(sr: int, ms: int) -> np.ndarray:
    return np.zeros(int(sr * ms / 1000.0), dtype=np.float32)


def run_job(job: dict, tts: TTS, sr: int):
    json_path = job["json_path"]
    tag       = job["tag"]
    base_lang = job["base_lang"]

    out_dir      = Path(BASE_OUT_DIR) / tag
    manifest_csv = Path(BASE_OUT_DIR) / f"{tag}.csv"

    # Load JSON entries
    with open(json_path, "r", encoding="utf-8") as fh:
        entries = json.load(fh)

    if LIMIT_ROWS and LIMIT_ROWS > 0:
        entries = entries[:LIMIT_ROWS]

    # Resume support
    done_indices: set[int] = set()
    rows_out: list[dict]   = []
    if manifest_csv.exists():
        try:
            existing     = pd.read_csv(manifest_csv)
            done_indices = set(existing["row_index"].tolist())
            rows_out     = existing.to_dict("records")
            print(f"[{tag}] Resuming — {len(done_indices)} rows already done")
        except Exception:
            pass

    remaining = [(i, e) for i, e in enumerate(entries) if i not in done_indices]
    if not remaining:
        print(f"[{tag}] Already complete — skipping")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"[{tag}]")
    print(f"  json      : {json_path.name}")
    print(f"  base_lang : {base_lang}")
    print(f"  rows todo : {len(remaining)}")
    print(f"{'='*60}")

    for i, entry in remaining:
        text = normalize_whitespace(entry.get("modified") or entry.get("original", ""))
        if not text:
            continue

        foreign_lang_name = entry.get("foreign_lang", "")
        # Always render under Finnish regardless of source language.
        language = base_lang  # always "fi" — set by infer_base_language

        sentences   = split_into_sentences(text)
        chunks      = pack_sentences(sentences, MAX_CHARS_PER_CHUNK) if sentences else [text]
        audio_parts = []

        for chunk in chunks:
            if SPEAKER_WAV is not None:
                wav = tts.tts(text=chunk, speaker_wav=SPEAKER_WAV, language=language)
            else:
                wav = tts.tts(text=chunk, speaker=SPEAKER_IDX, language=language)
            audio_parts.append(np.asarray(wav, dtype=np.float32))
            audio_parts.append(silence(sr, PAUSE_SILENCE_MS))

        full     = np.concatenate(audio_parts) if audio_parts else np.zeros(0, dtype=np.float32)
        out_path = out_dir / f"row_{i:04d}.wav"
        sf.write(str(out_path), full, sr)

        rows_out.append({
            "row_index":   i,
            "wav_path":    str(out_path),
            "foreign_lang": foreign_lang_name,
            "base_lang":   base_lang,
            "text":        text,
        })

        if (i + 1) % 10 == 0:
            pd.DataFrame(rows_out).to_csv(manifest_csv, index=False)
            print(f"  [{tag}] {i+1}/{len(entries)} done", flush=True)

    pd.DataFrame(rows_out).to_csv(manifest_csv, index=False)
    print(f"  [{tag}] ✓ {len(rows_out)} wav files -> {out_dir}")


def main():
    Path(BASE_OUT_DIR).mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(FOREIGN_JSON_DIR)
    if not jobs:
        print(f"No *_foreign.json files found in '{FOREIGN_JSON_DIR}'. "
              f"Run foreign_codeswitcher.py first.")
        return

    print(f"Found {len(jobs)} jobs:")
    for j in jobs:
        print(f"  {j['tag']:40s}  base_lang={j['base_lang']}")

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

    print("\n" + "="*60)
    print("ALL JOBS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()