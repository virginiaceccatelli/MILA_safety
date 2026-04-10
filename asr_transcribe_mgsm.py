import os
import csv
import logging
import whisper
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for sbatch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from datetime import datetime
from jiwer import cer, wer, Compose, ToLowerCase, RemovePunctuation, Strip

base_dir = "mgsm/audio_gemini_mgsm"
languages = ["de", "es", "it", "fr", "en"]
output_dir = "mgsm_asr_results"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

LANG_MAP = {"de": "de", "es": "es", "it": "it", "fr": "fr", "en": "en"}

log_path = os.path.join(output_dir, f"run_{timestamp}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),   # still visible in sbatch .out file
    ],
)
log = logging.getLogger()

log.info("Loading Whisper model...")
model = whisper.load_model("large")
log.info("Model loaded.")

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])

per_sample_csv_path = os.path.join(output_dir, f"per_sample_{timestamp}.csv")
summary_csv_path    = os.path.join(output_dir, f"summary_{timestamp}.csv")

per_sample_rows = []
overall_results  = {}

for lang in languages:
    audio_dir     = os.path.join(base_dir, f"global_mgsm_{lang}")
    manifest_path = os.path.join(base_dir, f"global_mgsm_{lang}_manifest.csv")

    if not os.path.isdir(audio_dir):
        log.warning(f"[{lang}] Audio dir not found: {audio_dir}")
        continue
    if not os.path.isfile(manifest_path):
        log.warning(f"[{lang}] Manifest not found: {manifest_path}")
        continue

    gt_map = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text      = row.get("text_clean", "").strip()
            audio_col = next(
                (c for c in ["audio_filepath", "file", "filename", "path"] if c in row),
                None,
            )
            stem = (
                os.path.splitext(os.path.basename(row[audio_col]))[0]
                if audio_col else None
            )
            gt_map[stem] = text

    use_index_fallback = any(k is None for k in gt_map)
    if use_index_fallback:
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader  = csv.DictReader(f)
            gt_list = [row.get("text_clean", "").strip() for row in reader]
        gt_map = {}

    wav_files = sorted(f for f in os.listdir(audio_dir) if f.endswith(".wav"))
    log.info(f"\n{'='*60}\n  Language: {lang.upper()}  |  Files: {len(wav_files)}\n{'='*60}")

    total_wer = total_cer = 0.0
    count = 0

    for fname in wav_files:
        stem       = os.path.splitext(fname)[0]
        audio_path = os.path.join(audio_dir, fname)

        if use_index_fallback:
            try:
                idx = int(stem.split("_")[-1])
            except ValueError:
                log.warning(f"  [{lang}] Cannot parse index from {fname}, skipping.")
                continue
            if idx < 0 or idx >= len(gt_list):
                log.warning(f"  [{lang}] Index {idx} out of range for {fname}, skipping.")
                continue
            ref = gt_list[idx]
        else:
            if stem not in gt_map:
                log.warning(f"  [{lang}] No GT entry for {stem}, skipping.")
                continue
            ref = gt_map[stem]

        result   = model.transcribe(audio_path, language=LANG_MAP[lang])
        hyp      = result["text"]
        ref_norm = transform(ref)
        hyp_norm = transform(hyp)

        if not ref_norm:
            log.warning(f"  [{lang}] Empty reference for {fname}, skipping.")
            continue

        w = wer(ref_norm, hyp_norm)
        c = cer(ref_norm, hyp_norm)
        total_wer += w
        total_cer += c
        count     += 1

        per_sample_rows.append({
            "lang": lang, "file": fname,
            "ref": ref_norm, "hyp": hyp_norm,
            "wer": round(w, 4), "cer": round(c, 4),
        })

        log.info(f"  {fname}  WER={w:.4f}  CER={c:.4f}")
        log.info(f"    REF: {ref_norm}")
        log.info(f"    HYP: {hyp_norm}")

    if count > 0:
        avg_wer = total_wer / count
        avg_cer = total_cer / count
        overall_results[lang] = {"wer": avg_wer, "cer": avg_cer, "n": count}
        log.info(f"\n  [{lang.upper()}] Avg WER={avg_wer:.4f}  Avg CER={avg_cer:.4f}  ({count} samples)")
    else:
        log.warning(f"\n  [{lang.upper()}] No valid samples processed.")


with open(per_sample_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["lang", "file", "wer", "cer", "ref", "hyp"])
    writer.writeheader()
    writer.writerows(per_sample_rows)
log.info(f"\nPer-sample results saved -> {per_sample_csv_path}")

macro_wer = sum(r["wer"] for r in overall_results.values()) / len(overall_results) if overall_results else 0
macro_cer = sum(r["cer"] for r in overall_results.values()) / len(overall_results) if overall_results else 0
total_n   = sum(r["n"]   for r in overall_results.values())

with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["lang", "n", "wer", "cer"])
    writer.writeheader()
    for lang, res in overall_results.items():
        writer.writerow({"lang": lang.upper(), "n": res["n"],
                         "wer": round(res["wer"], 4), "cer": round(res["cer"], 4)})
    writer.writerow({"lang": "MACRO", "n": total_n,
                     "wer": round(macro_wer, 4), "cer": round(macro_cer, 4)})
log.info(f"Summary saved -> {summary_csv_path}")

log.info(f"\n{'='*50}")
log.info("  GLOBAL SUMMARY")
log.info(f"{'='*50}")
log.info(f"  {'Lang':<8} {'N':>5}  {'WER':>8}  {'CER':>8}")
log.info(f"  {'-'*36}")
for lang, res in overall_results.items():
    log.info(f"  {lang.upper():<8} {res['n']:>5}  {res['wer']:>8.4f}  {res['cer']:>8.4f}")
log.info(f"  {'-'*36}")
log.info(f"  {'MACRO':<8} {total_n:>5}  {macro_wer:>8.4f}  {macro_cer:>8.4f}")
log.info(f"{'='*50}")

if overall_results:
    langs       = list(overall_results.keys())
    lang_labels = [l.upper() for l in langs]
    wers        = [overall_results[l]["wer"] * 100 for l in langs]
    cers        = [overall_results[l]["cer"] * 100 for l in langs]
    x           = np.arange(len(langs))
    bar_w       = 0.35

    # -- 1. Grouped bar chart: WER & CER per language --
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_wer = ax.bar(x - bar_w / 2, wers, bar_w, label="WER", color="#4C72B0", zorder=3)
    bars_cer = ax.bar(x + bar_w / 2, cers, bar_w, label="CER", color="#DD8452", zorder=3)

    ax.set_xlabel("Language", fontsize=12)
    ax.set_ylabel("Error Rate (%)", fontsize=12)
    ax.set_title("Whisper ASR — WER & CER per Language (MGSM)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(lang_labels, fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    # Macro average reference lines
    ax.axhline(macro_wer * 100, color="#4C72B0", linestyle=":", linewidth=1.4)
    ax.axhline(macro_cer * 100, color="#DD8452", linestyle=":", linewidth=1.4)

    for bar in bars_wer:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    for bar in bars_cer:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    bar_path = os.path.join(output_dir, f"wer_cer_bar_{timestamp}.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    log.info(f"Bar chart saved -> {bar_path}")

    # -- 2. Per-sample WER strip/scatter plot --
    fig, axes = plt.subplots(1, len(langs), figsize=(3 * len(langs), 4), sharey=True)
    if len(langs) == 1:
        axes = [axes]

    for ax, lang in zip(axes, langs):
        samples = [r for r in per_sample_rows if r["lang"] == lang]
        y_vals  = [s["wer"] * 100 for s in samples]
        jitter  = np.random.uniform(-0.15, 0.15, size=len(y_vals))
        ax.scatter(jitter, y_vals, alpha=0.55, s=20, color="#4C72B0", zorder=3)
        ax.axhline(float(np.mean(y_vals)), color="crimson", linewidth=1.5,
                   linestyle="--", zorder=4)
        ax.set_title(lang.upper(), fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        if ax == axes[0]:
            ax.set_ylabel("WER (%)", fontsize=11)

    fig.suptitle("Per-sample WER distribution  (dashed = mean)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, f"per_sample_wer_{timestamp}.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    log.info(f"Scatter plot saved -> {scatter_path}")

log.info("\nDone.")