import os
import re
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

CSV_PATH = "translations/jbb_behaviors_it_12b.csv"
TEXT_COL = "Goal_it"

OUT_DIR = "audio_qwen3tts_it"
MANIFEST_CSV = "audio_qwen3tts_it_manifest.csv"

MODEL_ID = "/network/scratch/c/ceccatev/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-CustomVoice"
SPEAKER = "aiden"

LANGUAGE = "Italian" 

LIMIT_ROWS = 0 # 0 to process all

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

    assert os.path.isdir(MODEL_ID), f"MODEL_ID directory not found: {MODEL_ID}"

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if TEXT_COL not in df.columns:
        raise RuntimeError(f"Column '{TEXT_COL}' not found. Columns: {list(df.columns)}")
    
    if LIMIT_ROWS and LIMIT_ROWS > 0:
        df = df.head(LIMIT_ROWS)
    

    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation = "sdpa"
    )

    speakers = model.get_supported_speakers()
    if SPEAKER not in speakers:
        raise RuntimeError(f"Speaker '{SPEAKER}' not supported. Supported: {speakers}")

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
        out_sr = None

        for chunk in chunks:
            wavs, sr = model.generate_custom_voice(
                text=chunk,
                language=LANGUAGE,  
                speaker=SPEAKER,    
            )
            audio_parts.append(np.asarray(wavs[0], dtype=np.float32))
            audio_parts.append(silence(sr, PAUSE_SILENCE_MS))
            out_sr = sr

        full = np.concatenate(audio_parts) if audio_parts else np.zeros(0, dtype=np.float32)
        out_path = os.path.join(OUT_DIR, f"row_{i:04d}.wav")
        sf.write(out_path, full, out_sr if out_sr is not None else 24000)

        rows_out.append({"row_index": i, "wav_path": out_path, "text_col": TEXT_COL})

        if (i + 1) % 10 == 0:
            print(f"Done {i+1}/{len(df)}", flush=True)

    pd.DataFrame(rows_out).to_csv(MANIFEST_CSV, index=False)
    print(f"Wrote {len(rows_out)} wav files to {OUT_DIR}", flush=True)
    print(f"Manifest: {MANIFEST_CSV}", flush=True)


if __name__ == "__main__":
    main()