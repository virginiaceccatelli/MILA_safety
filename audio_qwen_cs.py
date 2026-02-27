import torch
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import soundfile as sf
from qwen_tts import Qwen3TTSModel


CSV_PATH   = "translations/jbb_behaviors_it_12b_codeswitch_strict_it.csv"
TEXT_COL   = "codeswitch_en_it_strict"
ENGLISH_COL = "Goal"        # Original English text
FOREIGN_COL = "Goal_it"     # Foreign language text

OUT_DIR      = "codeswitch_qwen3tts_it"
MANIFEST_CSV = "audio_qwen3_cs_it.csv"

# Model variants:
#   "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"  preset speakers
#   "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"  smaller / faster
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# Matrix language 
MATRIX_LANGUAGE    = "Italian"   # full Qwen3 language name
SECONDARY_LANGUAGE = "English"   # full Qwen3 language name

# Preset speaker name. Available: Vivian, Ryan, Aiden, Dylan, Eric,
SPEAKER = "Aiden"

# Optional natural-language instruction appended to each generation call.
INSTRUCT: Optional[str] = "Speak naturally with a neutral tone."

LIMIT_ROWS = 0   # 0 = process all rows

# Pause (ms) inserted between language-switch boundaries
PHRASE_PAUSE_MS = 150

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def clean_word(word: str) -> str:
    return re.sub(r"[^\w\s-]", "", str(word).lower()).strip()


def identify_word_language(word: str, english_text: str, foreign_text: str) -> str:
    clean = clean_word(word)
    en_words      = {clean_word(w) for w in english_text.split()}
    foreign_words = {clean_word(w) for w in foreign_text.split()}

    in_english = clean in en_words
    in_foreign = clean in foreign_words

    if in_english and not in_foreign:
        return "en"
    elif in_foreign and not in_english:
        return "foreign"
    else:
        return "foreign"   # ambiguous or unknown → matrix language


def group_words_by_language(
    codeswitch_text: str,
    english_text: str,
    foreign_text: str,
) -> List[Tuple[str, str]]:
    words = codeswitch_text.split()
    if not words:
        return []

    groups: List[Tuple[str, str]] = []
    current_phrase = [words[0]]
    current_lang   = identify_word_language(words[0], english_text, foreign_text)

    for word in words[1:]:
        word_lang = identify_word_language(word, english_text, foreign_text)
        if word_lang == current_lang:
            current_phrase.append(word)
        else:
            groups.append((" ".join(current_phrase), current_lang))
            current_phrase = [word]
            current_lang   = word_lang

    if current_phrase:
        groups.append((" ".join(current_phrase), current_lang))

    return groups


def silence(sr: int, ms: int) -> np.ndarray:
    return np.zeros(int(sr * ms / 1000.0), dtype=np.float32)



def generate_codeswitch_audio(
    model: "Qwen3TTSModel",
    codeswitch_text: str,
    english_text: str,
    foreign_text: str,
    matrix_lang: str,
    secondary_lang: str,
    speaker: str,
    instruct: Optional[str],
    pause_ms: int,
) -> Tuple[np.ndarray, int]:
    phrases = group_words_by_language(codeswitch_text, english_text, foreign_text)

    print("    Code-switched phrases:")
    for phrase, lang in phrases:
        lang_name = secondary_lang if lang == "en" else matrix_lang
        print(f"      [{lang_name}] {phrase}")

    audio_parts: List[np.ndarray] = []
    sr: Optional[int] = None

    for i, (phrase, lang) in enumerate(phrases):
        tts_lang = secondary_lang if lang == "en" else matrix_lang

        try:
            wavs, sample_rate = model.generate_custom_voice(
                text=phrase,
                language=tts_lang,
                speaker=speaker,
                instruct=instruct,
            )
            if sr is None:
                sr = sample_rate

            audio_parts.append(np.asarray(wavs[0], dtype=np.float32))

            # Insert a short pause between language switches
            if i < len(phrases) - 1:
                audio_parts.append(silence(sample_rate, pause_ms))

        except Exception as exc:
            print(f"      ERROR generating [{tts_lang}] '{phrase}': {exc}")
            fallback_sr = sr if sr is not None else 24000
            audio_parts.append(silence(fallback_sr, 500))

    final_sr = sr if sr is not None else 24000
    if audio_parts:
        return np.concatenate(audio_parts), final_sr
    else:
        return np.zeros(0, dtype=np.float32), final_sr


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required_cols = [TEXT_COL, ENGLISH_COL, FOREIGN_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    if LIMIT_ROWS and LIMIT_ROWS > 0:
        df = df.head(LIMIT_ROWS)

    print("=" * 70)
    print("Code-Switched Qwen3-TTS Audio Generator")
    print("=" * 70)
    print(f"Model:              {MODEL_NAME}")
    print(f"Matrix language:    {MATRIX_LANGUAGE}")
    print(f"Secondary language: {SECONDARY_LANGUAGE}")
    print(f"Speaker:            {SPEAKER}")
    print(f"Processing:         {len(df)} rows")
    print("=" * 70)

    print("\nLoading Qwen3-TTS model…")
    device = "cuda:0" 
    dtype  = torch.bfloat16  # use float32 on CPU if bfloat16 unsupported

    try:
        model = Qwen3TTSModel.from_pretrained(
            MODEL_NAME,
            device_map=device,
            dtype=dtype,
            #attn_implementation=attn_impl,
        )
    except Exception:
        # Fall back to eager attention if flash-attn is not available
        model = Qwen3TTSModel.from_pretrained(
            MODEL_NAME,
            device_map=device,
            dtype=dtype,
            #attn_implementation="eager",
        )

    print(f"Model loaded on {device}\n")

    rows_out = []
    success  = 0
    failed   = 0

    for i, row in df.iterrows():
        codeswitch = row.get(TEXT_COL, "")
        english    = row.get(ENGLISH_COL, "")
        foreign    = row.get(FOREIGN_COL, "")

        if pd.isna(codeswitch) or pd.isna(english) or pd.isna(foreign):
            print(f"Row {i}: Skipping (missing data)")
            failed += 1
            continue

        codeswitch = normalize_whitespace(codeswitch)
        english    = normalize_whitespace(english)
        foreign    = normalize_whitespace(foreign)

        if not codeswitch:
            failed += 1
            continue

        print(f"\nRow {i}/{len(df) - 1}:")
        print(f"  Code-switched: {codeswitch[:80]}…")

        try:
            audio, sr = generate_codeswitch_audio(
                model=model,
                codeswitch_text=codeswitch,
                english_text=english,
                foreign_text=foreign,
                matrix_lang=MATRIX_LANGUAGE,
                secondary_lang=SECONDARY_LANGUAGE,
                speaker=SPEAKER,
                instruct=INSTRUCT,
                pause_ms=PHRASE_PAUSE_MS,
            )

            out_path = os.path.join(OUT_DIR, f"row_{i:04d}.wav")
            sf.write(out_path, audio, sr)

            duration = len(audio) / sr
            print(f"  ✓ Saved: {out_path} ({duration:.2f}s, {sr} Hz)")

            rows_out.append({
                "row_index": i,
                "wav_path":  out_path,
                "text_col":  TEXT_COL,
                "duration":  duration,
            })
            success += 1

        except Exception as exc:
            print(f"  ✗ ERROR: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

        if (i + 1) % 10 == 0:
            print(f"\n--- Progress: {i + 1}/{len(df)} ({success} ✓, {failed} ✗) ---")

    pd.DataFrame(rows_out).to_csv(MANIFEST_CSV, index=False)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"✓ Success: {success}/{len(df)}")
    print(f"✗ Failed:  {failed}/{len(df)}")
    print(f"Output:    {OUT_DIR}")
    print(f"Manifest:  {MANIFEST_CSV}")


if __name__ == "__main__":
    main()