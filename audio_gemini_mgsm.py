import re
import wave
import struct
import os
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from google import genai
from google.genai import types


CSV_DIR = "mgsm/global_mgsm"
BASE_OUT_DIR = "audio_gemini_mgsm"

MODEL_NAME = "gemini-2.5-flash-preview-tts"

LIMIT_ROWS = 0   # 0 = all rows

TARGET_FILES = [
    "global_mgsm_de.csv",
    "global_mgsm_en.csv",
    "global_mgsm_es.csv",
    "global_mgsm_fr.csv",
    "global_mgsm_it.csv",
]

LANG_MAP = {
    "de": "de", "en": "en", "es": "es", "fr": "fr", "it": "it",
}

# Gemini TTS output: 24 kHz, mono, 16-bit PCM
SAMPLE_RATE = 24_000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes (int16)


def infer_lang_from_csv_name(csv_name: str) -> str:
    m = re.match(r"global_mgsm_([a-z]{2})\.csv$", Path(csv_name).name)
    if not m:
        raise ValueError(f"Could not infer language from filename: {csv_name}")
    code = m.group(1)
    if code not in LANG_MAP:
        raise ValueError(f"Unsupported language code '{code}': {csv_name}")
    return LANG_MAP[code]

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def flatten_text_for_tts(text: str) -> str:
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"[.!?]+", ",", text)
    text = re.sub(r",+", ",", text)
    text = re.sub(r"[\[\]\(\)\{\}<>]", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.rstrip(",")
    return text

def clean_text(text: str) -> str:
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2014", " - ").replace("\u2013", " - ")
    text = text.replace("\u2026", "...")
    text = re.sub(r"[\[\]\(\)\{\}<>]", " ", text)
    text = re.sub(r"([!?.,])\1+", r"\1", text)
    text = re.sub(r"\s+([,.?!])", r"\1", text)
    text = re.sub(r"([.?!,])([^\s])", r"\1 \2", text)
    return normalize_whitespace(text)

def pcm_to_wav(pcm_bytes: bytes, out_path: Path):
    """Write raw PCM bytes to a proper WAV file."""
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


import base64

def synthesize(client: genai.Client, text: str, language: str) -> Optional[bytes]:
    text = flatten_text_for_tts(text)
    if not text:
        return None

    voice_name = "Charon"
    lang_names = {"de": "German", "en": "English", "es": "Spanish", "fr": "French", "it": "Italian"}
    lang_label = lang_names.get(language, "English")

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            # For TTS models, keeping the instruction inside the content is often more stable
            contents=f"Narrate this in {lang_label}: {text}",
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name,
                        )
                    )
                ),
            ),
        )

        if not response.candidates or not response.candidates[0].content.parts:
            print(f"[DEBUG] Model returned no audio for text: {text[:30]}")
            return None

        part = response.candidates[0].content.parts[0]
        
        raw_data = part.inline_data.data
        if isinstance(raw_data, str):
            return base64.b64decode(raw_data)
        return raw_data

    except Exception as e:
        print(f"[WARN] Synthesis failed: {e}")
        return None


def build_jobs(csv_dir: str) -> list[dict]:
    jobs = []
    for filename in TARGET_FILES:
        csv_path = Path(csv_dir) / filename
        if not csv_path.exists():
            print(f"[WARN] Missing file, skipping: {csv_path}")
            continue
        lang = infer_lang_from_csv_name(filename)
        jobs.append({"csv_path": csv_path, "tag": Path(filename).stem, "language": lang})
    return jobs

def run_job(job: dict, client: genai.Client):
    csv_path = job["csv_path"]
    tag      = job["tag"]
    language = job["language"]

    out_dir      = Path(BASE_OUT_DIR) / tag
    manifest_csv = Path(BASE_OUT_DIR) / f"{tag}_manifest.csv"

    df = pd.read_csv(csv_path)
    if "question" not in df.columns:
        raise ValueError(f"'question' column not found in {csv_path}. Columns: {list(df.columns)}")

    if LIMIT_ROWS and LIMIT_ROWS > 0:
        df = df.iloc[:LIMIT_ROWS].copy()

    done_indices: set[int] = set()
    rows_out: list[dict] = []

    if manifest_csv.exists():
        try:
            existing    = pd.read_csv(manifest_csv)
            done_indices = set(existing["row_index"].tolist())
            rows_out    = existing.to_dict("records")
            print(f"[{tag}] Resuming — {len(done_indices)} rows already done")
        except Exception:
            pass

    remaining = [i for i in df.index.tolist() if i not in done_indices]
    if not remaining:
        print(f"[{tag}] Already complete — skipping")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"[{tag}]  language={language}  voice=Charon  rows={len(remaining)}")
    print(f"{'=' * 60}")

    for n, i in enumerate(remaining, start=1):
        raw_text   = str(df.at[i, "question"])
        clean      = clean_text(raw_text)
        if not clean:
            continue

        pcm = synthesize(client, clean, language)
        if pcm is None:
            continue

        out_path = out_dir / f"row_{i:04d}.wav"
        pcm_to_wav(pcm, out_path)

        rows_out.append({
            "row_index":  i,
            "wav_path":   str(out_path),
            "language":   language,
            "voice":      "Charon",
            "text_raw":   raw_text,
            "text_clean": clean,
            "source_csv": str(csv_path),
        })

        if n % 10 == 0:
            pd.DataFrame(rows_out).to_csv(manifest_csv, index=False)
            print(f"  [{tag}] {n}/{len(remaining)} done", flush=True)

    pd.DataFrame(rows_out).to_csv(manifest_csv, index=False)
    print(f"  [{tag}] ✓ {len(rows_out)} WAV files written")
    print(f"  [{tag}] ✓ audio → {out_dir}")


def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")

    Path(BASE_OUT_DIR).mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(CSV_DIR)
    if not jobs:
        print(f"No target CSV files found in '{CSV_DIR}'.")
        return

    print(f"Found {len(jobs)} jobs:")
    for j in jobs:
        print(f"  {j['tag']:25s} language={j['language']}  voice=Charon")

    client = genai.Client(api_key=api_key)

    for job in jobs:
        run_job(job, client)

    print("\n" + "=" * 60)
    print("ALL JOBS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()