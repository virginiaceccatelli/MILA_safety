import torch
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import soundfile as sf

_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)

torch.load = _torch_load_compat

from TTS.api import TTS

CSV_PATH = "translations/jbb_behaviors_de_12b_codeswitch_strict_de.csv"
TEXT_COL = "codeswitch_en_de_strict"
ENGLISH_COL = "Goal"  # Original English text
FOREIGN_COL = "Goal_de"  # Foreign language text

OUT_DIR = "codeswitch_xtts_de"
MANIFEST_CSV = "audio_xtts_cs_de.csv"

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

MATRIX_LANGUAGE = "de"
SECONDARY_LANGUAGE = "en"

SPEAKER_WAV: Optional[str] = None
SPEAKER_IDX: Optional[str] = "Ana Florence"

LIMIT_ROWS = 0

MAX_CHARS_PER_CHUNK = 500  # Higher limit
PHRASE_PAUSE_MS = 100  # Pause between language switches

USE_GPU = torch.cuda.is_available()


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def clean_word(word: str) -> str:
    return re.sub(r'[^\w\s-]', '', str(word).lower()).strip()


def identify_word_language(word: str, english_text: str, foreign_text: str) -> str:
    clean = clean_word(word)
    
    en_words = set(clean_word(w) for w in english_text.split())
    foreign_words = set(clean_word(w) for w in foreign_text.split())
    
    in_english = clean in en_words
    in_foreign = clean in foreign_words
    
    if in_english and not in_foreign:
        return 'en'
    elif in_foreign and not in_english:
        return 'foreign'
    elif in_english and in_foreign:
        return 'foreign'  # Default to matrix language
    else:
        return 'en'  # Unknown words default to matrix


def group_words_by_language(codeswitch_text, english_text, foreign_text):
    words = codeswitch_text.split()
    if not words:
        return []

    groups = []
    current_phrase = [words[0]]
    current_lang = identify_word_language(words[0], english_text, foreign_text)

    for word in words[1:]:
        wl = identify_word_language(word, english_text, foreign_text)

        # if unknown, don't force foreign; keep current language
        if wl not in ("en", "foreign"):
            wl = current_lang

        if wl == current_lang:
            current_phrase.append(word)
        else:
            groups.append((" ".join(current_phrase), current_lang))
            current_phrase = [word]
            current_lang = wl

    groups.append((" ".join(current_phrase), current_lang))
    return groups

def silence(sr: int, ms: int) -> np.ndarray:
    n = int(sr * (ms / 1000.0))
    return np.zeros(n, dtype=np.float32)

def generate_codeswitch_audio(
    tts: TTS,
    codeswitch_text: str,
    english_text: str,
    foreign_text: str,
    matrix_lang: str,
    secondary_lang: str,
    speaker_wav: Optional[str],
    speaker_idx: Optional[str],
    sr: int,
    pause_ms: int
) -> np.ndarray:

    # identify which words are in which language
    # group consecutive words by language 
    # generate each phrase with appropriate language 
    # concatenate with pauses at langauge switches 
    
    # Group words by language
    phrases = group_words_by_language(codeswitch_text, english_text, foreign_text)
    
    print(f"    Code-switched phrases:")
    for phrase, lang in phrases:
        lang_name = secondary_lang if lang == 'en' else matrix_lang
        print(f"      [{lang_name}] {phrase}")
    
    # Generate audio for each phrase
    audio_parts = []

    # gen_kwargs = dict(
    #     temperature=0.4,
    #     top_p=0.9,
    #     num_gpt_outputs=1,
    #     split_sentences=False, 
    # )

    for i, (phrase, lang) in enumerate(phrases):
        # Select appropriate language
        tts_lang = secondary_lang if lang == 'en' else matrix_lang
        raw = phrase.strip()
        phrase_for_tts = raw + ("." if len(raw) <= 6 and raw.isalpha() else "") + " "
        
        # Generate audio
        try:
            if speaker_wav is not None:
                wav = tts.tts(
                    text=phrase_for_tts,
                    speaker_wav=speaker_wav,
                    language=tts_lang,
                    split_sentences=False,
                    # **gen_kwargs,
                )
            else:
                wav = tts.tts(
                    text=phrase_for_tts,
                    speaker=speaker_idx,
                    language=tts_lang,
                    split_sentences=False,
                    # **gen_kwargs,
                )

            wav = np.asarray(wav, dtype=np.float32)
            audio_parts.append(wav)
                        
            # Add pause between phrases (language switches)
            if i < len(phrases) - 1:
                next_lang = phrases[i+1][1]
                if next_lang != lang:
                    audio_parts.append(silence(sr, pause_ms))
        
        except Exception as e:
            print(f"      ERROR generating [{tts_lang}] '{phrase}': {e}")
            # Add silence as fallback
            audio_parts.append(silence(sr, 500))
    
    # Concatenate all parts
    if audio_parts:
        return np.concatenate(audio_parts)
    else:
        return np.zeros(0, dtype=np.float32)


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    
    # Check required columns
    required = [TEXT_COL, ENGLISH_COL, FOREIGN_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    if LIMIT_ROWS and LIMIT_ROWS > 0:
        df = df.head(LIMIT_ROWS)

    if SPEAKER_WAV is not None and not os.path.exists(SPEAKER_WAV):
        raise FileNotFoundError(f"SPEAKER_WAV not found: {SPEAKER_WAV}")

    print("="*70)
    print("Code-Switched XTTS Audio Generator")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Matrix language: {MATRIX_LANGUAGE}")
    print(f"Secondary language: {SECONDARY_LANGUAGE}")
    print(f"Processing: {len(df)} rows")
    print("="*70)

    # Load model
    print("\nLoading TTS model...")
    tts = TTS(MODEL_NAME)
    if USE_GPU:
        tts.to("cuda")
        print("Using GPU")

    # Get sample rate
    sr = 24000
    # try:
    #     if hasattr(tts, "synthesizer") and hasattr(tts.synthesizer, "output_sample_rate"):
    #         sr = int(tts.synthesizer.output_sample_rate)
    # except Exception:
    #     pass
    
    print(f"Sample rate: {sr} Hz\n")

    rows_out = []
    success = 0
    failed = 0
    
    for i, row in df.iterrows():
        codeswitch = row.get(TEXT_COL, "")
        english = row.get(ENGLISH_COL, "")
        foreign = row.get(FOREIGN_COL, "")
        
        if pd.isna(codeswitch) or pd.isna(english) or pd.isna(foreign):
            print(f"Row {i}: Skipping (missing data)")
            failed += 1
            continue
        
        codeswitch = normalize_whitespace(codeswitch)
        english = normalize_whitespace(english)
        foreign = normalize_whitespace(foreign)
        
        if not codeswitch:
            failed += 1
            continue
        
        print(f"\nRow {i}/{len(df)-1}:")
        print(f"  Code-switched: {codeswitch[:60]}...")
        
        try:
            # Generate code-switched audio
            audio = generate_codeswitch_audio(
                tts=tts,
                codeswitch_text=codeswitch,
                english_text=english,
                foreign_text=foreign,
                matrix_lang=MATRIX_LANGUAGE,
                secondary_lang=SECONDARY_LANGUAGE,
                speaker_wav=SPEAKER_WAV,
                speaker_idx=SPEAKER_IDX,
                sr=sr,
                pause_ms=PHRASE_PAUSE_MS
            )
            
            # Save audio
            out_path = os.path.join(OUT_DIR, f"row_{i:04d}.wav")
            sf.write(out_path, audio, sr)
            
            duration = len(audio) / sr
            print(f"  ✓ Saved: {out_path} ({duration:.2f}s)")
            
            rows_out.append({
                "row_index": i,
                "wav_path": out_path,
                "text_col": TEXT_COL,
                "duration": duration
            })
            
            success += 1
        
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        
        if (i + 1) % 10 == 0:
            print(f"\n--- Progress: {i+1}/{len(df)} ({success} ✓, {failed} ✗) ---")

    # Save manifest
    pd.DataFrame(rows_out).to_csv(MANIFEST_CSV, index=False)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"✓ Success: {success}/{len(df)}")
    print(f"✗ Failed: {failed}/{len(df)}")
    print(f"Output: {OUT_DIR}")
    print(f"Manifest: {MANIFEST_CSV}")


if __name__ == "__main__":
    main()