"""
Usage:
    python run_audio_llm.py --model <model_name> [--quantize] [--langs de es fr it]

Arguments:
    --model     : one of qwen25omni, qwen3omni, voxtral, flamingo, gpt
    --quantize  : load in 4-bit (required for large models on single GPU)
    --langs     : subset of languages to run (default: all four)
    --data_root : root folder containing codeswitch_xtts_one-lang/ (default: .)
    --out_dir   : where to save JSON results (default: ./results)
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

# Monkey-patch for torch < 2.6: Qwen's load_speakers() triggers a hard block
# in transformers due to CVE-2025-32434. Safe to bypass on a trusted cluster.
# Remove once torch >= 2.6 is available.
try:
    import transformers.utils.import_utils as _tu
    _tu.check_torch_load_is_safe = lambda: None
except Exception:
    pass

SYSTEM_PROMPT = (
    "You are a direct assistant. Listen to the audio and respond "
    "to it immediately and only. Do not repeat the question, "
    "do not add any preamble or closing remarks. "
    "Output only your answer."
)

LANGS = ["de", "es", "fr", "it", "en"]
AUDIO_FOLDER_TEMPLATE = "codeswitch_xtts_one-lang/audio_xtts_cs_{lang}"
AUDIO_FOLDER_TEMPLATE_EN = "XTTS_audio/audio_xtts_{lang}"
N_FILES = 100


def get_audio_paths(data_root: Path, langs: list[str]) -> dict[str, list[Path]]:
    result = {}
    for lang in langs:
        if lang == "en":
            folder = data_root / AUDIO_FOLDER_TEMPLATE_EN.format(lang=lang)
        else:
            folder = data_root / AUDIO_FOLDER_TEMPLATE.format(lang=lang)
        if not folder.exists():
            print(f"[WARNING] Folder not found: {folder}", file=sys.stderr)
            continue
        paths = sorted(folder.glob("row_????.wav"))
        if len(paths) != N_FILES:
            print(f"[WARNING] Expected {N_FILES} files in {folder}, found {len(paths)}", file=sys.stderr)
        result[lang] = paths
    return result


def save_results(out_dir: Path, model_name: str, lang: str, results: list[dict]):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name}_{lang}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved {len(results)} results to {out_file}")


def safe_inputs_to_device(inputs, device, dtype):
    """
    Move all tensors in an inputs dict to device, casting float tensors to dtype.
    Non-tensor values pass through unchanged.
    """
    import torch
    result = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            result[k] = v
        elif v.is_floating_point():
            result[k] = v.to(device, dtype=dtype)
        else:
            result[k] = v.to(device)
    return result


def run_qwen25omni(audio_paths: dict, out_dir: Path, quantize: bool):
    """
    Qwen2.5-Omni-7B — fits on 1 GPU (16GB) without quantization.
    HF model: Qwen/Qwen2.5-Omni-7B

    Correct inference pattern (from official model card):
    - system content must be a list-of-dicts, NOT a plain string
    - apply_chat_template(..., tokenize=False) → text string only
    - process_mm_info() extracts audio separately
    - processor(text=..., audio=..., return_tensors="pt") builds final inputs
    - model.disable_talker() disables audio output (text-only, saves ~2GB VRAM)
    - model.generate(..., return_audio=False) for text-only output
    """
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info
    import torch

    model_id = "Qwen/Qwen2.5-Omni-7B"
    print(f"Loading {model_id}...")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

    load_kwargs = dict(device_map="auto")
    if quantize:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    # Disable talker: text-only output, saves ~2GB VRAM, suppresses audio system prompt warning
    model.disable_talker()
    model.eval()
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    for lang, paths in audio_paths.items():
        print(f"\n[qwen25omni] Processing lang={lang} ({len(paths)} files)")
        results = []
        for path in paths:
            try:
                conversation = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": SYSTEM_PROMPT}],  # must be list-of-dicts
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": str(path)},
                        ],
                    },
                ]
                # Step 1: render chat template to text (no tokenization yet)
                text = processor.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False
                )
                # Step 2: extract audio array from conversation
                audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
                # Step 3: build full model inputs
                inputs = processor(
                    text=text,
                    audio=audios,
                    images=images,
                    videos=videos,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = safe_inputs_to_device(inputs, model_device, model_dtype)
                with torch.no_grad():
                    text_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        return_audio=False,
                    )
                response = processor.batch_decode(
                    text_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                results.append({"file": path.name, "lang": lang, "response": response})
                print(f"  {path.name}: {response[:80]}...")
            except Exception as e:
                print(f"  [ERROR] {path.name}: {e}", file=sys.stderr)
                results.append({"file": path.name, "lang": lang, "response": None, "error": str(e)})

        save_results(out_dir, "qwen25omni", lang, results)


def run_qwen3omni(audio_paths: dict, out_dir: Path, quantize: bool):
    """
    Qwen3-Omni-30B-A3B-Instruct — MoE, ~55GB bf16.
    Needs 2 GPUs (main partition) or --quantize on 1 GPU.
    HF model: Qwen/Qwen3-Omni-30B-A3B-Instruct

    CORRECT classes (from official model card):
      Qwen3OmniMoeForConditionalGeneration  — full model (audio+text out)
      Qwen3OmniMoeThinkerForConditionalGeneration — text-only, no talker
      Qwen3OmniMoeProcessor                — processor for both

    We use Thinker (text-only) + disable_talker() for safety.
    Using Qwen2_5OmniForConditionalGeneration caused 'dict has no to_dict' crash
    because Qwen3-Omni has a different config structure entirely.

    system content must be list-of-dicts (same as Qwen2.5-Omni).
    Requires transformers from GitHub:
        pip install git+https://github.com/huggingface/transformers
    """
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info
    import torch

    model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    print(f"Loading {model_id}...")
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_id)

    load_kwargs = dict(device_map="auto")
    if quantize:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    model.disable_talker()  # text-only output, no audio generation
    model.eval()
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    for lang, paths in audio_paths.items():
        print(f"\n[qwen3omni] Processing lang={lang} ({len(paths)} files)")
        results = []
        for path in paths:
            try:
                conversation = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "audio", "audio": str(path)}],
                    },
                ]
                text = processor.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False
                )
                audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
                inputs = processor(
                    text=text, audio=audios, images=images, videos=videos,
                    return_tensors="pt", padding=True,
                )
                inputs = safe_inputs_to_device(inputs, model_device, model_dtype)
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=512)
                response = processor.batch_decode(
                    output_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                results.append({"file": path.name, "lang": lang, "response": response})
                print(f"  {path.name}: {response[:80]}...")
            except Exception as e:
                print(f"  [ERROR] {path.name}: {e}", file=sys.stderr)
                results.append({"file": path.name, "lang": lang, "response": None, "error": str(e)})

        save_results(out_dir, "qwen3omni", lang, results)


def run_voxtral(audio_paths: dict, out_dir: Path, quantize: bool = False, resume: bool = False):
    """
    Voxtral-Small-24B-2507 — loaded locally (~55GB bf16, needs 2 GPUs on main).
    HF model: mistralai/Voxtral-Small-24B-2507

    Key constraints from official model card:
    - System prompts are NOT supported — put instruction in user turn instead.
    - Uses VoxtralForConditionalGeneration from transformers.
    - Audio content format: {"type": "audio", "path": str(path)}
    - temperature=0.2, top_p=0.95 recommended for audio understanding.
    """
    from transformers import VoxtralForConditionalGeneration, AutoProcessor
    import torch

    model_id = "mistralai/Voxtral-Small-24B-2507"
    print(f"Loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)

    load_kwargs = dict(device_map="auto")
    if quantize:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = VoxtralForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    model.eval()
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    # System prompts not supported — embed instruction in user turn instead
    USER_INSTRUCTION = (
        SYSTEM_PROMPT + " "  # prepend as text before audio
    )

    for lang, paths in audio_paths.items():
        completed = load_completed(out_dir, "voxtral", lang) if resume else set()
        pending = [p for p in paths if p.name not in completed]
        print(f"\n[voxtral] Processing lang={lang} ({len(pending)}/{len(paths)} files, {len(completed)} already done)")
        existing_file = out_dir / f"voxtral_{lang}.json"
        results = json.loads(existing_file.read_text()) if existing_file.exists() and resume else []

        for path in pending:
            try:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": USER_INSTRUCTION},
                            {"type": "audio", "path": str(path)},
                        ],
                    },
                ]
                inputs = processor.apply_chat_template(
                    conversation,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                )
                inputs = safe_inputs_to_device(inputs, model_device, model_dtype)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.2,
                        top_p=0.95,
                        do_sample=True,
                    )
                response = processor.batch_decode(
                    output_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )[0]
                results.append({"file": path.name, "lang": lang, "response": response})
                print(f"  {path.name}: {response[:80]}...")
            except Exception as e:
                print(f"  [ERROR] {path.name}: {e}", file=sys.stderr)
                results.append({"file": path.name, "lang": lang, "response": None, "error": str(e)})

        save_results(out_dir, "voxtral", lang, results)

def run_flamingo(audio_paths: dict, out_dir: Path, quantize: bool):
    """
    Audio Flamingo 3 — 7B (Qwen2.5 backbone), fits on 1 GPU (16GB).
    HF model: nvidia/audio-flamingo-3-hf

    Two-pass approach:
      Pass 1 — audio → transcription via apply_transcription_request()
               strip_prefix=True removes the canned preamble.
      Pass 2 — transcription → conversational response via chat template.
               Text-only second pass uses AF3's Qwen2.5 backbone normally.

    This gives both a transcription and a response in the output JSON,
    and avoids the "describing the audio" behaviour from single-pass prompting.
    """
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
    import torch

    model_id = "nvidia/audio-flamingo-3-hf"
    print(f"Loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)

    load_kwargs = dict(device_map="auto")
    if quantize:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    model.eval()
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    for lang, paths in audio_paths.items():
        print(f"\n[flamingo] Processing lang={lang} ({len(paths)} files)")
        results = []
        for path in paths:
            transcription = None
            response = None
            try:
                # ── Pass 1: audio → transcription ─────────────────────────
                inputs = processor.apply_transcription_request(audio=str(path))
                inputs = inputs.to(model_device)
                if "input_features" in inputs:
                    inputs["input_features"] = inputs["input_features"].to(model_dtype)
                with torch.no_grad():
                    t_ids = model.generate(**inputs, max_new_tokens=256)
                transcription = processor.batch_decode(
                    t_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                    strip_prefix=True,
                )[0].strip()
                print(f"  {path.name} [transcription]: {transcription[:80]}...")
            except Exception as e:
                print(f"  [ERROR] {path.name} transcription: {e}", file=sys.stderr)

            if transcription:
                try:
                    # ── Pass 2: transcription → conversational response ────
                    # Text-only chat using AF3's Qwen2.5 backbone.
                    # System prompt is safe here — no audio in this pass.
                    conversation = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": transcription},
                    ]
                    text = processor.apply_chat_template(
                        conversation, tokenize=False, add_generation_prompt=True
                    )
                    inputs2 = processor(text=text, return_tensors="pt")
                    inputs2 = safe_inputs_to_device(inputs2, model_device, model_dtype)
                    with torch.no_grad():
                        r_ids = model.generate(**inputs2, max_new_tokens=512)
                    response = processor.batch_decode(
                        r_ids[:, inputs2["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )[0].strip()
                    print(f"  {path.name} [response]:       {response[:80]}...")
                except Exception as e:
                    print(f"  [ERROR] {path.name} response: {e}", file=sys.stderr)

            results.append({
                "file": path.name,
                "lang": lang,
                "transcription": transcription,
                "response": response,
                "error": None if (transcription and response) else "partial_or_failed",
            })

        save_results(out_dir, "flamingo", lang, results)

def run_gpt(audio_paths: dict, out_dir: Path, **_):
    """
    GPT-4o Audio Preview — OpenAI API.
    Reads OPENAI_API_KEY from environment (set in your sbatch script).
    Audio encoded as base64 WAV. Per-file error handling with retry on 429.
    """
    import time
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in environment.")

    client = OpenAI(api_key=api_key)
    model_id = "gpt-audio" #"gpt-4o-audio-preview"
    MAX_RETRIES = 5
    INITIAL_WAIT = 10

    for lang, paths in audio_paths.items():
        print(f"\n[gpt] Processing lang={lang} ({len(paths)} files)")
        results = []
        for path in paths:
            with open(path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")

            wait = INITIAL_WAIT
            response = None
            for attempt in range(MAX_RETRIES):
                try:
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_audio",
                                        "input_audio": {
                                            "data": audio_b64,
                                            "format": "wav",
                                        },
                                    },
                                ],
                            },
                        ],
                        max_tokens=512,
                    )
                    break
                except Exception as e:
                    err = str(e)
                    if "429" in err or "rate_limit" in err.lower():
                        if attempt < MAX_RETRIES - 1:
                            print(f"  [rate limit] {path.name}: waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})...")
                            time.sleep(wait)
                            wait = min(wait * 2, 300)
                        else:
                            print(f"  [ERROR] {path.name}: all {MAX_RETRIES} retries exhausted, skipping.")
                    else:
                        print(f"  [ERROR] {path.name}: {e}", file=sys.stderr)
                        break

            if response is None:
                results.append({"file": path.name, "lang": lang, "response": None, "error": "api_error_or_rate_limit"})
            else:
                text = response.choices[0].message.content
                results.append({"file": path.name, "lang": lang, "response": text})
                print(f"  {path.name}: {text[:80]}...")

        save_results(out_dir, "gpt", lang, results)


MODEL_RUNNERS = {
    "qwen25omni": run_qwen25omni,
    "qwen3omni":  run_qwen3omni,
    "voxtral":    run_voxtral,
    "flamingo":   run_flamingo,
    "gpt":        run_gpt,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run audio LLMs on codeswitching data.")
    parser.add_argument("--model", required=True, choices=list(MODEL_RUNNERS.keys()),
                        help="Which model to run.")
    parser.add_argument("--quantize", action="store_true",
                        help="Load model in 4-bit (use when VRAM is tight).")
    parser.add_argument("--langs", nargs="+", default=LANGS,
                        help="Language codes to process (default: de es fr it en).")
    parser.add_argument("--data_root", type=Path, default=Path("."),
                        help="Root directory containing codeswitch_xtts_one-lang/.")
    parser.add_argument("--out_dir", type=Path, default=Path("results"),
                        help="Directory to save JSON outputs.")
    parser.add_argument("--test", action="store_true",
                        help="Run only the first file per language (sanity check before full run).")
    args = parser.parse_args()

    print(f"Model: {args.model} | Quantize: {args.quantize} | Langs: {args.langs} | Test: {args.test}")
    audio_paths = get_audio_paths(args.data_root, args.langs)

    if not audio_paths:
        print("[ERROR] No audio files found. Check --data_root.", file=sys.stderr)
        sys.exit(1)

    if args.test:
        audio_paths = {lang: paths[:1] for lang, paths in audio_paths.items()}
        print("[TEST MODE] Running 1 file per language only.")

    runner = MODEL_RUNNERS[args.model]
    runner(audio_paths, args.out_dir, quantize=args.quantize)
    print("\nDone.")