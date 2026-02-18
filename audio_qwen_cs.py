import torch
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import soundfile as sf

_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)
torch.load = _torch_load_compat

from qwen_tts import Qwen3TTSModel

CSV_PATH    = "translations/jbb_behaviors_it_12b_codeswitch_strict_it.csv"
TEXT_COL    = "codeswitch_en_it_strict"
ENGLISH_COL = "Goal"
FOREIGN_COL = "Goal_it"

OUT_DIR      = "codeswitch_qwen3tts_it"
MANIFEST_CSV = "audio_qwen3tts_cs_it.csv"

MATRIX_LANGUAGE    = "it"   # foreign / matrix language
SECONDARY_LANGUAGE = "en"   # English

# One voice description used for BOTH languages so the voice stays consistent.
# The model will adapt pronunciation/accent per language automatically.
VOICE_INSTRUCT = (
    "Middle-aged male, neutral accent, flat and matter-of-fact delivery. "
    "No emotion, no laughter, no filler sounds. "
    "Reads text exactly as written."
)

# Short reference sentence used to synthesise the voice design clip.
VOICE_DESIGN_REF_TEXT_EN = "The report will be ready by tomorrow morning."
VOICE_DESIGN_REF_TEXT_FO = "Il rapporto sarà pronto domattina."

# If you have XTTS reference wavs, set these paths and they will be used
# instead of synthesising a new reference clip via VoiceDesign.
XTTS_REF_WAV_EN: Optional[str] = "XTTS_audio/audio_xtts_en/row_0000.wav"
XTTS_REF_WAV_FO: Optional[str] = "XTTS_audio/audio_xtts_it/row_0000.wav"

LIMIT_ROWS      = 0
PHRASE_PAUSE_MS = 100
USE_GPU         = True


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def clean_word(word: str) -> str:
    return re.sub(r"[^\w\s-]", "", str(word).lower()).strip()


def identify_word_language(word: str, english_text: str, foreign_text: str) -> str:
    clean         = clean_word(word)
    en_words      = set(clean_word(w) for w in english_text.split())
    foreign_words = set(clean_word(w) for w in foreign_text.split())

    in_english = clean in en_words
    in_foreign = clean in foreign_words

    if in_english and not in_foreign:  return "en"
    if in_foreign and not in_english:  return "foreign"
    if in_english and in_foreign:      return "foreign"  # ambiguous → matrix
    return "en"                                           # unknown → secondary


def group_words_by_language(cs_text: str, en_text: str, fo_text: str) -> List[Tuple[str, str]]:
    words = cs_text.split()
    if not words:
        return []

    groups: List[Tuple[str, str]] = []
    cur_words = [words[0]]
    cur_lang  = identify_word_language(words[0], en_text, fo_text)

    for word in words[1:]:
        wl = identify_word_language(word, en_text, fo_text)
        if wl not in ("en", "foreign"):
            wl = cur_lang
        if wl == cur_lang:
            cur_words.append(word)
        else:
            groups.append((" ".join(cur_words), cur_lang))
            cur_words = [word]
            cur_lang  = wl

    groups.append((" ".join(cur_words), cur_lang))
    return groups


def prepare_phrase(phrase: str) -> str:
    phrase = phrase.strip()
    if phrase and phrase[-1] not in ".!?":
        phrase += "."
    return phrase


def silence_array(sr: int, ms: int) -> np.ndarray:
    return np.zeros(int(sr * ms / 1000.0), dtype=np.float32)


def lang_code_to_qwen(code: str) -> str:
    return {"en": "English", "de": "German", "es": "Spanish",
            "fr": "French",  "it": "Italian"}.get(code, "Auto")


def build_voice_clone_prompts(
    design_model: Optional[Qwen3TTSModel],
    clone_model:  Qwen3TTSModel,
    sr: int,
) -> Dict[str, object]:
    prompts = {}

    configs = [
        ("en",      SECONDARY_LANGUAGE, XTTS_REF_WAV_EN, VOICE_DESIGN_REF_TEXT_EN),
        ("foreign", MATRIX_LANGUAGE,    XTTS_REF_WAV_FO, VOICE_DESIGN_REF_TEXT_FO),
    ]

    for lang_key, lang_code, ref_wav_path, ref_text in configs:
        qwen_lang = lang_code_to_qwen(lang_code)
        print(f"\n  Building voice prompt for [{qwen_lang}]...")

        if ref_wav_path and os.path.exists(ref_wav_path):
            # ── Use existing XTTS wav ─────────────────────────────────────────
            print(f"    Using XTTS reference: {ref_wav_path}")
            ref_audio_arg = ref_wav_path  # path string accepted by create_voice_clone_prompt

        else:
            # ── Synthesise reference clip via VoiceDesign ─────────────────────
            if design_model is None:
                raise RuntimeError(
                    f"No reference wav for [{lang_code}] and VoiceDesign model not loaded."
                )
            print(f"    Synthesising reference with VoiceDesign...")
            ref_wavs, design_sr = design_model.generate_voice_design(
                text=ref_text,
                language=qwen_lang,
                instruct=VOICE_INSTRUCT,
            )
            ref_path = os.path.join(OUT_DIR, f"voice_design_ref_{lang_code}.wav")
            sf.write(ref_path, ref_wavs[0], design_sr)
            print(f"    Saved design reference: {ref_path}")
            ref_audio_arg = (ref_wavs[0], design_sr)

        # ── Build reusable clone prompt ───────────────────────────────────────
        prompt = clone_model.create_voice_clone_prompt(
            ref_audio=ref_audio_arg,
            ref_text=ref_text,
        )
        prompts[lang_key] = prompt
        print(f"    ✓ Prompt ready for [{qwen_lang}]")

    return prompts


def generate_codeswitch_audio(
    clone_model:     Qwen3TTSModel,
    codeswitch_text: str,
    english_text:    str,
    foreign_text:    str,
    matrix_lang:     str,
    secondary_lang:  str,
    voice_prompts:   Dict[str, object],
    sr:              int,
    pause_ms:        int,
) -> np.ndarray:

    phrases = group_words_by_language(codeswitch_text, english_text, foreign_text)

    print("    Phrases:")
    for phrase, lang in phrases:
        label = secondary_lang if lang == "en" else matrix_lang
        print(f"      [{label}] {phrase}")

    audio_parts: List[np.ndarray] = []

    for i, (phrase, lang) in enumerate(phrases):
        lang_code    = secondary_lang if lang == "en" else matrix_lang
        qwen_lang    = lang_code_to_qwen(lang_code)
        text_for_tts = prepare_phrase(phrase)
        vcp          = voice_prompts[lang]  # per-language prompt

        try:
            wavs, out_sr = clone_model.generate_voice_clone(
                text=text_for_tts,
                language=qwen_lang,
                voice_clone_prompt=vcp,
            )

            if out_sr != sr:
                raise RuntimeError(f"Unexpected sample rate: {out_sr} (expected {sr})")

            wav = np.asarray(wavs[0], dtype=np.float32)
            audio_parts.append(wav)
            print(f"      ✓ [{qwen_lang}] '{text_for_tts}' ({len(wav)/sr:.2f}s)")

            # Pause only at language switches
            if i < len(phrases) - 1 and phrases[i + 1][1] != lang:
                audio_parts.append(silence_array(sr, pause_ms))

        except Exception as e:
            print(f"      ✗ ERROR [{qwen_lang}] '{phrase}': {e}")
            import traceback; traceback.print_exc()
            audio_parts.append(silence_array(sr, 500))

    return np.concatenate(audio_parts) if audio_parts else np.zeros(0, dtype=np.float32)


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    missing = [c for c in [TEXT_COL, ENGLISH_COL, FOREIGN_COL] if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    if LIMIT_ROWS and LIMIT_ROWS > 0:
        df = df.head(LIMIT_ROWS)

    print("=" * 70)
    print("Code-Switched Qwen3-TTS (VoiceDesign + Clone Prompt)")
    print("=" * 70)
    print(f"Matrix language:  {MATRIX_LANGUAGE}")
    print(f"Secondary lang:   {SECONDARY_LANGUAGE}")
    print(f"EN ref wav:       {XTTS_REF_WAV_EN or '(will synthesise with VoiceDesign)'}")
    print(f"FO ref wav:       {XTTS_REF_WAV_FO or '(will synthesise with VoiceDesign)'}")
    print(f"Rows:             {len(df)}")
    print("=" * 70)

    device_map = "cuda:0" if USE_GPU else "cpu"
    dtype      = torch.bfloat16 if USE_GPU else torch.float32

    # Only load VoiceDesign model if we actually need it
    need_design = (
        (not XTTS_REF_WAV_EN or not os.path.exists(XTTS_REF_WAV_EN)) or
        (not XTTS_REF_WAV_FO or not os.path.exists(XTTS_REF_WAV_FO))
    )

    design_model = None
    if need_design:
        print("\nLoading VoiceDesign model...")
        design_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            device_map=device_map,
            dtype=dtype,
            #attn_implementation="flash_attention_2",
        )
        print("✓ VoiceDesign model loaded")

    print("\nLoading Base (clone) model...")
    clone_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map=device_map,
        dtype=dtype,
        #attn_implementation="flash_attention_2",
    )
    print("✓ Base model loaded")

    sr = 24000

    # Build voice prompts once - reused for every row
    print("\nBuilding voice clone prompts...")
    voice_prompts = build_voice_clone_prompts(design_model, clone_model, sr)
    print("\n✓ Voice prompts ready - starting generation\n")

    # Free VoiceDesign from GPU memory now that prompts are built
    if design_model is not None:
        del design_model
        torch.cuda.empty_cache()

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

        print(f"\nRow {i}/{len(df)-1}:")
        print(f"  {codeswitch}")

        try:
            audio = generate_codeswitch_audio(
                clone_model=clone_model,
                codeswitch_text=codeswitch,
                english_text=english,
                foreign_text=foreign,
                matrix_lang=MATRIX_LANGUAGE,
                secondary_lang=SECONDARY_LANGUAGE,
                voice_prompts=voice_prompts,
                sr=sr,
                pause_ms=PHRASE_PAUSE_MS,
            )

            out_path = os.path.join(OUT_DIR, f"row_{i:04d}.wav")
            sf.write(out_path, audio, sr)

            duration = len(audio) / sr
            print(f"  ✓ {out_path} ({duration:.2f}s)")

            rows_out.append({
                "row_index":       i,
                "wav_path":        out_path,
                "codeswitch_text": codeswitch,
                "duration":        duration,
            })
            success += 1

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback; traceback.print_exc()
            failed += 1

        if (i + 1) % 10 == 0:
            print(f"\n--- Progress: {i+1}/{len(df)} ({success} ✓, {failed} ✗) ---")

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