import pandas as pd
import shutil
from pathlib import Path
import argparse
import wave
import subprocess

# Configuration
LANGS = ["de", "es", "fr", "it"]
CSV_TEMPLATE = "translations/jbb_behaviors_{lang}_12b.csv"
AUDIO_TEMPLATE = "XTTS_audio/audio_xtts_{lang}"
OUTPUT_BASE = "mfa_corpus"


def get_audio_info(audio_path):
    try:
        with wave.open(str(audio_path), 'rb') as wf:
            return {
                'channels': wf.getnchannels(),
                'sample_width': wf.getsampwidth(),
                'framerate': wf.getframerate(),
                'n_frames': wf.getnframes(),
                'duration': wf.getnframes() / wf.getframerate()
            }
    except Exception as e:
        return {'error': str(e)}


def prepare_corpus_for_language(lang, convert_audio=True, max_samples=None):
    # Setup paths
    csv_path = CSV_TEMPLATE.format(lang=lang)
    audio_dir = Path(AUDIO_TEMPLATE.format(lang=lang))
    output_dir = Path(OUTPUT_BASE) / lang / "speaker1"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: CSV file not found: {csv_path}")
        return {'error': 'CSV not found'}
    
    col_name = f"Goal_{lang}"
    if col_name not in df.columns:
        print(f"ERROR: Column '{col_name}' not found in CSV")
        print(f"Available columns: {list(df.columns)}")
        return {'error': f'Column {col_name} not found'}
    
    if max_samples:
        df = df.head(max_samples)
        print(f"Limited to {max_samples} samples")
    
    # audio directory
    if not audio_dir.exists():
        print(f"ERROR: Audio directory not found: {audio_dir}")
        return {'error': 'Audio directory not found'}
    
    # audio files
    audio_files = sorted(audio_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files in {audio_dir}")
    
    # Statistics
    stats = {
        'total_rows': len(df),
        'total_audio': len(audio_files),
        'matched': 0,
        'missing_audio': 0,
        'missing_text': 0,
        'converted': 0,
        'copied': 0,
        'errors': []
    }
    
    # Match audio files with text
    for idx, row in df.iterrows():
        text = row[col_name]
        
        if pd.isna(text) or str(text).strip() == '':
            stats['missing_text'] += 1
            continue
        
        # Try to find corresponding audio file
        expected_audio = audio_dir / f"row_{idx:04d}.wav"
        
        if not expected_audio.exists():
            stats['missing_audio'] += 1
            stats['errors'].append(f"Missing audio for row {idx}")
            continue
        
        # output filenames
        output_audio = output_dir / f"utt_{idx:04d}.wav"
        output_text = output_dir / f"utt_{idx:04d}.txt"
        
        # copy audio
        shutil.copy(expected_audio, output_audio)
        stats['copied'] += 1
        
        # text file
        output_text.write_text(str(text).strip())
        
        stats['matched'] += 1
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(df)} samples...")
    
    print(f"\nResults for {lang.upper()}:")
    print(f"  Total CSV rows: {stats['total_rows']}")
    print(f"  Total audio files: {stats['total_audio']}")
    print(f"  Successfully matched: {stats['matched']}")
    print(f"  Missing audio: {stats['missing_audio']}")
    print(f"  Missing text: {stats['missing_text']}")
    if convert_audio:
        print(f"  Converted: {stats['converted']}")
        print(f"  Copied as-is: {stats['copied']}")
    
    if stats['errors']:
        print(f"\nFirst 5 errors:")
        for err in stats['errors'][:5]:
            print(f"  - {err}")
    

# def verify_corpus(lang):
#     corpus_dir = Path(OUTPUT_BASE) / lang / "speaker1"
    
#     if not corpus_dir.exists():
#         print(f"Corpus directory not found: {corpus_dir}")
#         return False
    
#     wav_files = list(corpus_dir.glob("*.wav"))
#     txt_files = list(corpus_dir.glob("*.txt"))
    
#     print(f"\nVerification for {lang}:")
#     print(f"  WAV files: {len(wav_files)}")
#     print(f"  TXT files: {len(txt_files)}")
    
#     # Check pairing
#     unpaired = []
#     for wav in wav_files:
#         txt = wav.with_suffix('.txt')
#         if not txt.exists():
#             unpaired.append(wav.name)
    
#     if unpaired:
#         print(f"  Unpaired files: {len(unpaired)}")
#         print(f"  First 5: {unpaired[:5]}")
#         return False
#     else:
#         print(f"  All files properly paired ")
#         return True


def main():
    parser = argparse.ArgumentParser(
        description='Prepare MFA corpus from translations and XTTS audio'
    )
    parser.add_argument(
        '--lang',
        choices=LANGS + ['all'],
        default='all',
        help='Language to process (default: all)'
    )
    parser.add_argument(
        '--no-convert',
        action='store_true',
        help='Skip audio conversion (copy files as-is)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process per language'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing corpus'
    )
    
    args = parser.parse_args()
    
    # which languages to process
    langs_to_process = LANGS if args.lang == 'all' else [args.lang]
    
    if args.verify_only:
        for lang in langs_to_process:
            verify_corpus(lang)
        return
    
    all_stats = {}
    for lang in langs_to_process:
        stats = prepare_corpus_for_language(
            lang,
            convert_audio=not args.no_convert,
            max_samples=args.max_samples
        )
        all_stats[lang] = stats
        
        # Verify after preparation
        # verify_corpus(lang)
    
    # Summary
    for lang, stats in all_stats.items():
        if stats is None:
            print(f"{lang.upper()}: ERROR - Script returned None")
        elif 'error' in stats:
            print(f"{lang.upper()}: ERROR - {stats['error']}")
        else:
            print(f"{lang.upper()}: {stats['matched']} files prepared")

if __name__ == "__main__":
    main()