import torch
import os
import re
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

CSV_PATH = "translations/jbb_behaviors_it_12b_codeswitch_strict_it.csv"
TEXT_COL = "codeswitch_en_it_strict"

OUT_DIR = "audio_xtts_cs_it"
MANIFEST_CSV = "audio_xtts_cs_it.csv"

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# Language code for XTTS (e.g., "it", "fr", "de", "es")
LANGUAGE = "it"

# one of these two
SPEAKER_WAV: Optional[str] = None
SPEAKER_IDX: Optional[str] = "Ana Florence"

LIMIT_ROWS = 0  # 0 to process all rows

# Chunking (helps with long rows)
MAX_CHARS_PER_CHUNK = 250
PAUSE_SILENCE_MS = 120

USE_GPU = torch.cuda.is_available()

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def split_into_sentences(text: str) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    parts = re.split(r"(?<=[\.\!\?\:;])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def pack_sentences(sentences: List[str], max_chars: int) -> List[str]:
    chunks = []
    cur = ""
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
    n = int(sr * (ms / 1000.0))
    return np.zeros(n, dtype=np.float32)


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if TEXT_COL not in df.columns:
        raise RuntimeError(f"Column '{TEXT_COL}' not found. Columns: {list(df.columns)}")

    if LIMIT_ROWS and LIMIT_ROWS > 0:
        df = df.head(LIMIT_ROWS)

    if SPEAKER_WAV is not None and not os.path.exists(SPEAKER_WAV):
        raise FileNotFoundError(f"SPEAKER_WAV not found: {SPEAKER_WAV}")

    # Load model
    tts = TTS(MODEL_NAME)
    tts.to("cuda")


    # sample rate 
    sr = 24000
    try:
        if hasattr(tts, "synthesizer") and hasattr(tts.synthesizer, "output_sample_rate"):
            sr = int(tts.synthesizer.output_sample_rate)
    except Exception:
        pass

    rows_out = []
    for i, row in df.iterrows():
        raw = row.get(TEXT_COL, "")
        if pd.isna(raw):
            continue
        text = normalize_whitespace(raw)
        if not text:
            continue

        sentences = split_into_sentences(text)
        chunks = pack_sentences(sentences, MAX_CHARS_PER_CHUNK) if sentences else [text]

        audio_parts = []

        for chunk in chunks:
            if SPEAKER_WAV is not None:
                wav = tts.tts(
                    text=chunk,
                    speaker_wav=SPEAKER_WAV,
                    language=LANGUAGE,
                )
            else:
                wav = tts.tts(
                    text=chunk,
                    speaker=SPEAKER_IDX,
                    language=LANGUAGE,
                )

            audio_parts.append(np.asarray(wav, dtype=np.float32))
            audio_parts.append(silence(sr, PAUSE_SILENCE_MS))

        full = np.concatenate(audio_parts) if audio_parts else np.zeros(0, dtype=np.float32)
        out_path = os.path.join(OUT_DIR, f"row_{i:04d}.wav")
        sf.write(out_path, full, sr)

        rows_out.append({"row_index": i, "wav_path": out_path, "text_col": TEXT_COL})

        if (i + 1) % 10 == 0:
            print(f"Done {i+1}/{len(df)}", flush=True)

    pd.DataFrame(rows_out).to_csv(MANIFEST_CSV, index=False)
    print(f"Wrote {len(rows_out)} wav files to {OUT_DIR}", flush=True)
    print(f"Manifest: {MANIFEST_CSV}", flush=True)


if __name__ == "__main__":
    main()