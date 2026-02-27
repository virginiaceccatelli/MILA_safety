import pandas as pd
import shutil
import subprocess
import tempfile
from pathlib import Path
import argparse
import wave

# Configuration
LANGS = ["de", "es", "fr", "it"]
CSV_TEMPLATE = "translations/jbb_behaviors_{lang}_12b.csv"
AUDIO_TEMPLATE = "XTTS_audio/audio_xtts_{lang}"
OUTPUT_BASE = "readalong_output"

# ReadAlong Studio uses BCP-47 language codes
LANG_CODES = {
    "de": "deu",
    "es": "spa",
    "fr": "fra",
    "it": "ita",
}

def get_audio_duration(audio_path):
    try:
        with wave.open(str(audio_path), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return None


def align_single_utterance(audio_path, text, lang_code, output_dir, utt_id):
    utt_folder = output_dir / utt_id
    utt_folder.mkdir(parents=True, exist_ok=True)

    # Write the text file inside that folder
    text_file = utt_folder / f"{utt_id}.txt"
    text_file.write_text(text.strip(), encoding="utf-8")

    READALONGS_BIN = "/home/mila/c/ceccatev/.conda/envs/readalong/bin/readalongs"

    cmd = [
        READALONGS_BIN,
        "align",
        "-f",             # Force overwrite
        str(text_file),   # Input text
        str(audio_path),  # Input audio
        str(utt_folder),  # Output directory
        "-l", lang_code,
        # "--output-formats", "readalong,smil",
    ]

    # for debugging
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return {
            "success": False,
            "error": f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}",
        }

    return {"success": True}

def process_language(lang, max_samples=None):
    csv_path = CSV_TEMPLATE.format(lang=lang)
    audio_dir = Path(AUDIO_TEMPLATE.format(lang=lang))
    output_dir = Path(OUTPUT_BASE) / lang
    output_dir.mkdir(parents=True, exist_ok=True)

    lang_code = LANG_CODES.get(lang)
    if not lang_code:
        print(f"ERROR: No language code mapping for '{lang}'")
        return {"error": "No lang code"}

    # Load CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: CSV not found: {csv_path}")
        return {"error": "CSV not found"}

    col_name = f"Goal_{lang}"
    if col_name not in df.columns:
        print(f"ERROR: Column '{col_name}' not found. Available: {list(df.columns)}")
        return {"error": f"Column {col_name} not found"}

    if max_samples:
        df = df.head(max_samples)
        print(f"Limited to {max_samples} samples")

    if not audio_dir.exists():
        print(f"ERROR: Audio directory not found: {audio_dir}")
        return {"error": "Audio directory not found"}

    audio_files = sorted(audio_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files in {audio_dir}")

    stats = {
        "total_rows": len(df),
        "total_audio": len(audio_files),
        "aligned": 0,
        "missing_audio": 0,
        "missing_text": 0,
        "failed": 0,
        "errors": [],
    }

    for idx, row in df.iterrows():
        text = row[col_name]

        if pd.isna(text) or str(text).strip() == "":
            stats["missing_text"] += 1
            continue

        expected_audio = audio_dir / f"row_{idx:04d}.wav"
        if not expected_audio.exists():
            stats["missing_audio"] += 1
            stats["errors"].append(f"Missing audio for row {idx}")
            continue

        utt_id = f"utt_{idx:04d}"
        result = align_single_utterance(
            audio_path=expected_audio,
            text=str(text),
            lang_code=lang_code,
            output_dir=output_dir,
            utt_id=utt_id,
        )

        if result["success"]:
            stats["aligned"] += 1
        else:
            stats["failed"] += 1
            stats["errors"].append(f"Row {idx}: {result['error']}")

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(df)} samples...")

    print(f"\nResults for {lang.upper()}:")
    print(f"  Total CSV rows:     {stats['total_rows']}")
    print(f"  Total audio files:  {stats['total_audio']}")
    print(f"  Successfully aligned: {stats['aligned']}")
    print(f"  Missing audio:      {stats['missing_audio']}")
    print(f"  Missing text:       {stats['missing_text']}")
    print(f"  Alignment failures: {stats['failed']}")

    if stats["errors"]:
        print(f"\nFirst 5 errors:")
        for err in stats["errors"][:5]:
            print(f"  - {err}")

    return stats


def verify_output(lang):
    """Check that each utterance directory contains a .readalong alignment file."""
    output_dir = Path(OUTPUT_BASE) / lang
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        return False

    utt_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    complete = 0
    incomplete = []

    for utt_dir in sorted(utt_dirs):
        readalong_files = list(utt_dir.glob("*.readalong"))
        if readalong_files:
            complete += 1
        else:
            incomplete.append(utt_dir.name)

    print(f"\nVerification for {lang.upper()}:")
    print(f"  Total utterance dirs:  {len(utt_dirs)}")
    print(f"  With .readalong file:  {complete}")
    print(f"  Missing alignment:     {len(incomplete)}")
    if incomplete:
        print(f"  First 5 incomplete: {incomplete[:5]}")
        return False

    print(f"  All utterances aligned ✓")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Forced alignment with ReadAlong Studio"
    )
    parser.add_argument(
        "--lang",
        choices=LANGS + ["all"],
        default="all",
        help="Language to process (default: all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process per language",
    )
    parser.add_argument(
        "--verify-only",
        action='store_true',
        help="Only verify existing output",
    )
    args = parser.parse_args()

    langs_to_process = LANGS if args.lang == "all" else [args.lang]

    if args.verify_only:
        for lang in langs_to_process:
            verify_output(lang)
        return

    all_stats = {}
    for lang in langs_to_process:
        print(f"\n{'='*50}")
        print(f"Processing: {lang.upper()}")
        print(f"{'='*50}")
        stats = process_language(lang, max_samples=args.max_samples)
        all_stats[lang] = stats

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for lang, stats in all_stats.items():
        if stats is None or "error" in stats:
            err = stats.get("error", "unknown") if stats else "None returned"
            print(f"  {lang.upper()}: ERROR — {err}")
        else:
            print(f"  {lang.upper()}: {stats['aligned']} aligned, {stats['failed']} failed")


if __name__ == "__main__":
    main()