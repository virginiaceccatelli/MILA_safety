import pandas as pd
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import textgrid
import argparse
from pathlib import Path
import json
import re
from typing import List, Dict, Optional, Tuple


def clean_word(word: str) -> str:
    """Remove punctuation and lowercase for matching."""
    return re.sub(r'[^\w\s-]', '', str(word).lower()).strip()


def load_mfa_alignment(textgrid_path: Path, audio_path: Path) -> Tuple[List[Dict], np.ndarray, int]:
    """Load MFA TextGrid and audio file."""
    tg = textgrid.TextGrid.fromFile(str(textgrid_path))
    
    # Find word tier
    word_tier = None
    for tier in tg:
        non_empty = sum(1 for interval in tier if str(interval.mark).strip())
        if non_empty > 0:
            if word_tier is None or len(tier) < len(word_tier):
                word_tier = tier
    
    if word_tier is None:
        word_tier = tg[0]
    
    words = []
    for interval in word_tier:
        mark = str(interval.mark).strip()
        if mark:
            words.append({
                'word': clean_word(mark),
                'original': mark,
                'start': float(interval.minTime),
                'end': float(interval.maxTime)
            })
    
    sample_rate, audio_data = wavfile.read(audio_path)
    return words, audio_data, int(sample_rate)


def identify_word_language(word: str, english_text: str, foreign_text: str) -> str:
    """Determine if word is English or foreign."""
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
        return 'unknown'


def find_word_in_alignment(word: str, alignment: List[Dict], prefer_after: Optional[int] = None) -> Optional[int]:
    """
    Find word in alignment, preferring matches after prefer_after but not requiring it.
    """
    target = clean_word(word)
    start = (prefer_after + 1) if prefer_after is not None else 0
    
    # Pass 1: Exact match forward from preferred position
    for i in range(start, len(alignment)):
        if alignment[i]['word'] == target:
            return i
    
    # Pass 2: Exact match before preferred position
    for i in range(0, start):
        if alignment[i]['word'] == target:
            return i
    
    # Pass 3: Partial match for short words
    if len(target) <= 4:
        for i in range(start, len(alignment)):
            if target.startswith(alignment[i]['word']) or alignment[i]['word'].startswith(target):
                return i
        for i in range(0, start):
            if target.startswith(alignment[i]['word']) or alignment[i]['word'].startswith(target):
                return i
    
    return None


def extract_audio_segment(audio: np.ndarray, start: float, end: float, sr: int) -> np.ndarray:
    """Extract audio segment using EXACT MFA boundaries - no trimming."""
    start_sample = max(0, int(start * sr))
    end_sample = min(len(audio), int(end * sr))
    
    return audio[start_sample:end_sample]


def apply_fade(segment: np.ndarray, sr: int, fade_ms: float = 3.0) -> np.ndarray:
    """Apply minimal fade to prevent clicks - doesn't remove content."""
    if segment.size == 0:
        return segment
    
    # Very short fade - max 20% of segment to preserve content
    fade_samples = min(int(sr * fade_ms / 1000.0), segment.shape[0] // 5)
    if fade_samples <= 0:
        return segment
    
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    
    result = segment.astype(np.float32, copy=True)
    result[:fade_samples] *= fade_in
    result[-fade_samples:] *= fade_out
    
    return result


def create_silence(sr: int, ms: float) -> np.ndarray:
    """Create silence."""
    return np.zeros(int(sr * ms / 1000.0), dtype=np.float32)


def create_codeswitch_audio(
    codeswitch_text: str,
    english_text: str,
    foreign_text: str,
    en_alignment: List[Dict],
    en_audio: np.ndarray,
    en_sr: int,
    foreign_alignment: List[Dict],
    foreign_audio: np.ndarray,
    foreign_sr: int,
    output_path: Path,
    target_lang: str,
    switch_pause_ms: float = 80.0,
    inter_word_pause_ms: float = 50.0
) -> Optional[Dict]:
    """Create code-switched audio."""
    
    print(f"\n{'='*70}")
    print(f"Creating code-switched audio")
    print(f"{'='*70}")
    print(f"Code-switched: {codeswitch_text}")
    
    # Resample if needed
    if foreign_sr != en_sr:
        print(f"\nResampling {target_lang} from {foreign_sr}Hz to {en_sr}Hz")
        foreign_audio = resample_poly(foreign_audio, en_sr, foreign_sr).astype(foreign_audio.dtype)
        foreign_sr = en_sr
    
    words = codeswitch_text.split()
    print(f"\nProcessing {len(words)} words:")
    
    en_last = None
    foreign_last = None
    
    segments = []
    metadata = []
    prev_lang = None
    
    for i, word in enumerate(words):
        expected = identify_word_language(word, english_text, foreign_text)
        print(f"  [{i:2d}] {word:20s} → {expected:8s}", end="")
        
        # Set up primary and fallback
        if expected == 'en':
            prim_ali, prim_aud, prim_last, prim_lang = en_alignment, en_audio, en_last, 'en'
            fall_ali, fall_aud, fall_last, fall_lang = foreign_alignment, foreign_audio, foreign_last, 'foreign'
        else:
            prim_ali, prim_aud, prim_last, prim_lang = foreign_alignment, foreign_audio, foreign_last, 'foreign'
            fall_ali, fall_aud, fall_last, fall_lang = en_alignment, en_audio, en_last, 'en'
        
        # Search primary
        ali_idx = find_word_in_alignment(word, prim_ali, prim_last)
        
        used_ali = prim_ali
        used_aud = prim_aud
        used_lang = prim_lang
        fallback_used = False
        
        # Validate match
        if ali_idx is not None:
            found = prim_ali[ali_idx]['word']
            target = clean_word(word)
            
            if found != target and not (len(target) <= 4 and target.startswith(found)):
                print(f" ✗ wrong:'{prim_ali[ali_idx]['original']}'", end="")
                ali_idx = None
        
        # Try fallback if needed
        if ali_idx is None:
            print(f" → {fall_lang}...", end="")
            ali_idx = find_word_in_alignment(word, fall_ali, fall_last)
            
            if ali_idx is not None:
                found = fall_ali[ali_idx]['word']
                target = clean_word(word)
                
                if found != target and not (len(target) <= 4 and target.startswith(found)):
                    print(f" ✗ wrong:'{fall_ali[ali_idx]['original']}'", end="")
                    ali_idx = None
                else:
                    used_ali = fall_ali
                    used_aud = fall_aud
                    used_lang = fall_lang
                    fallback_used = True
                    print(f" ✓", end="")
        
        # Extract or silence
        if ali_idx is not None:
            info = used_ali[ali_idx]
            
            # Use EXACT MFA boundaries - no trimming
            segment = extract_audio_segment(used_aud, info['start'], info['end'], en_sr)
            
            # Apply minimal fade only to prevent clicks
            segment = apply_fade(segment, en_sr, fade_ms=3.0)
            segments.append(segment)
            
            # Update correct last_idx
            if used_lang == 'en':
                en_last = ali_idx
            else:
                foreign_last = ali_idx
            
            metadata.append({
                'word': word,
                'expected_language': expected,
                'actual_language': used_lang,
                'fallback_used': fallback_used,
                'alignment_idx': ali_idx,
                'mfa_word': info['original'],
                'start': info['start'],
                'end': info['end']
            })
            
            fb = " [FB]" if fallback_used else ""
            print(f" ✓ {used_lang}[{ali_idx}]='{info['original']}' @{info['start']:.2f}s{fb}")
        else:
            print(f" ✗ NOT FOUND - silence")
            segments.append(create_silence(en_sr, 100.0))
            metadata.append({'word': word, 'expected_language': expected, 'error': 'not_found'})
            used_lang = expected
        
        # Add pause
        if i < len(words) - 1:
            next_expected = identify_word_language(words[i + 1], english_text, foreign_text)
            if next_expected != used_lang:
                segments.append(create_silence(en_sr, switch_pause_ms))
                print(f"      [Switch pause: {switch_pause_ms}ms]")
            else:
                segments.append(create_silence(en_sr, inter_word_pause_ms))
        
        prev_lang = used_lang
    
    if not segments:
        print("ERROR: No audio!")
        return None
    
    final = np.concatenate([s.astype(np.float32) for s in segments])
    
    # Convert dtype
    if np.issubdtype(en_audio.dtype, np.integer):
        info = np.iinfo(en_audio.dtype)
        final = np.clip(final, info.min, info.max).astype(en_audio.dtype)
    else:
        final = final.astype(en_audio.dtype)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output_path, en_sr, final)
    
    duration = len(final) / en_sr
    en_cnt = sum(1 for m in metadata if m.get('actual_language') == 'en')
    fo_cnt = sum(1 for m in metadata if m.get('actual_language') == 'foreign')
    fb_cnt = sum(1 for m in metadata if m.get('fallback_used'))
    err_cnt = sum(1 for m in metadata if 'error' in m)
    
    print(f"\n✓ Saved: {output_path}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Words: {len(words)} ({en_cnt} EN, {fo_cnt} {target_lang.upper()}, {err_cnt} errors, {fb_cnt} fallbacks)")
    
    return {
        'output_path': str(output_path),
        'duration': duration,
        'total_words': len(words),
        'english_words': en_cnt,
        'foreign_words': fo_cnt,
        'fallback_count': fb_cnt,
        'errors': err_cnt,
        'word_metadata': metadata
    }


def process_utterance(utt_id, cs_text, en_text, fo_text, lang, mfa_out, mfa_corp, out_dir):
    """Process one utterance."""
    print(f"\n{'='*70}\nProcessing: {utt_id}\n{'='*70}")
    
    en_tg = mfa_out / 'en' / 'speaker1' / f'{utt_id}.TextGrid'
    en_wav = mfa_corp / 'en' / 'speaker1' / f'{utt_id}.wav'
    
    if not en_tg.exists() or not en_wav.exists():
        print(f"ERROR: English files missing")
        return None
    
    print(f"Loading EN: {en_tg}")
    en_ali, en_aud, en_sr = load_mfa_alignment(en_tg, en_wav)
    print(f"  ✓ {len(en_ali)} words, {len(en_aud)/en_sr:.2f}s")
    
    fo_tg = mfa_out / lang / 'speaker1' / f'{utt_id}.TextGrid'
    fo_wav = mfa_corp / lang / 'speaker1' / f'{utt_id}.wav'
    
    if not fo_tg.exists() or not fo_wav.exists():
        print(f"ERROR: {lang} files missing")
        return None
    
    print(f"Loading {lang.upper()}: {fo_tg}")
    fo_ali, fo_aud, fo_sr = load_mfa_alignment(fo_tg, fo_wav)
    print(f"  ✓ {len(fo_ali)} words, {len(fo_aud)/fo_sr:.2f}s")
    
    out_wav = out_dir / f"{utt_id}_en_{lang}_strict.wav"
    
    result = create_codeswitch_audio(
        cs_text, en_text, fo_text,
        en_ali, en_aud, en_sr,
        fo_ali, fo_aud, fo_sr,
        out_wav, lang, 80.0, 50.0
    )
    
    if result:
        meta = {
            'utterance_id': utt_id,
            'language_pair': f'en_{lang}',
            'english_text': en_text,
            'foreign_text': fo_text,
            'codeswitch_text': cs_text,
            **result
        }
        
        meta_path = out_dir / f"{utt_id}_en_{lang}_strict.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"✓ Metadata: {meta_path}")
        return meta
    
    return None


def process_csv(csv, lang, mfa_out, mfa_corp, out_dir, max_rows):
    """Process CSV."""
    print("="*70)
    print("Code-Switched Audio Generator (FIXED)")
    print("="*70)
    print(f"CSV: {csv}\nLang: {lang}\nOutput: {out_dir}")
    print("="*70)
    
    df = pd.read_csv(csv)
    req = ['Goal', f'Goal_{lang}', f'codeswitch_en_{lang}_strict']
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing: {miss}")
    
    print(f"\n{len(df)} rows")
    if max_rows:
        df = df.head(max_rows)
        print(f"Processing {max_rows}")
    
    results = []
    ok, bad, skip = 0, 0, 0
    
    for idx, row in df.iterrows():
        en = row['Goal']
        fo = row[f'Goal_{lang}']
        cs = row[f'codeswitch_en_{lang}_strict']
        
        if pd.isna(cs) or pd.isna(en) or pd.isna(fo):
            print(f"\nRow {idx}: Skip (missing)")
            skip += 1
            continue
        
        utt = f"utt_{idx:04d}"
        
        try:
            res = process_utterance(utt, str(cs), str(en), str(fo), lang, mfa_out, mfa_corp, out_dir)
            if res:
                results.append(res)
                ok += 1
            else:
                bad += 1
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            bad += 1
        
        if (idx + 1) % 10 == 0:
            print(f"\n--- {idx+1}/{len(df)} ({ok} ✓, {bad} ✗, {skip} ⊘) ---")
    
    summ = out_dir / f'summary_{lang}.json'
    with open(summ, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}\nCOMPLETE\n{'='*70}")
    print(f"✓ {ok}/{len(df)}\n✗ {bad}/{len(df)}\n⊘ {skip}/{len(df)}\nSummary: {summ}")


def main():
    p = argparse.ArgumentParser(description='Generate code-switched audio (FIXED)')
    p.add_argument('--csv', required=True)
    p.add_argument('--target-lang', required=True, choices=['es', 'de', 'fr', 'it'])
    p.add_argument('--mfa-output', type=Path, default='mfa_output')
    p.add_argument('--mfa-corpus', type=Path, default='mfa_corpus')
    p.add_argument('--output-dir', type=Path, default='codeswitch_audio')
    p.add_argument('--max-rows', type=int)
    args = p.parse_args()
    
    process_csv(args.csv, args.target_lang, args.mfa_output, args.mfa_corpus, args.output_dir, args.max_rows)


if __name__ == "__main__":
    main()