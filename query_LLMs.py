import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

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

SINGLE_LANGS = ["de", "es", "fr", "it", "en"]
N_FILES = 250

FOREIGN_AUDIO_DIR = "audio_gemini_mgsm"
MGSM_CSV_DIR = "global_mgsm"


def single_lang_folder(data_root: Path, lang: str) -> Path:
    return data_root / "mgsm" / FOREIGN_AUDIO_DIR / f"global_mgsm_{lang}"


def single_lang_manifest_path(data_root: Path, lang: str) -> Path:
    return data_root / "mgsm" / FOREIGN_AUDIO_DIR / f"global_mgsm_{lang}_manifest.csv"


def single_lang_text_csv_path(data_root: Path, lang: str) -> Path:
    return data_root / "mgsm" / MGSM_CSV_DIR / f"global_mgsm_{lang}.csv"


def _sorted_audio_files(folder: Path) -> list[Path]:
    exts = ("*.wav", "*.mp3", "*.flac", "*.m4a")
    paths = []
    for pattern in exts:
        paths.extend(folder.glob(pattern))

    def sort_key(p: Path):
        stem = p.stem
        if stem.startswith("row_"):
            try:
                return (0, int(stem.split("_")[1]))
            except Exception:
                pass
        return (1, stem)

    return sorted(paths, key=sort_key)


def get_single_lang_paths(data_root: Path, langs: list[str]) -> dict[str, list[Path]]:
    result = {}
    for lang in langs:
        folder = single_lang_folder(data_root, lang)
        if not folder.exists():
            print(f"[WARNING] Folder not found: {folder}", file=sys.stderr)
            continue

        paths = _sorted_audio_files(folder)
        if len(paths) != N_FILES:
            print(
                f"[WARNING] Expected {N_FILES} files in {folder}, found {len(paths)}",
                file=sys.stderr
            )
        result[lang] = paths
    return result


def load_single_lang_manifest(data_root: Path, lang: str) -> Optional[pd.DataFrame]:
    path = single_lang_manifest_path(data_root, lang)
    if not path.exists():
        print(f"[WARNING] Manifest not found: {path}", file=sys.stderr)
        return None
    return pd.read_csv(path)


def load_single_lang_questions(data_root: Path, lang: str) -> dict[int, str]:
    path = single_lang_text_csv_path(data_root, lang)
    if not path.exists():
        print(f"[WARNING] Question CSV not found: {path}", file=sys.stderr)
        return {}

    df = pd.read_csv(path)

    candidate_cols = [
        "question",
        f"question_{lang}",
        "Question",
        f"Question_{lang}",
        "input",
        "prompt",
    ]

    q_col = next((c for c in candidate_cols if c in df.columns), None)
    if q_col is None:
        print(
            f"[WARNING] Could not find a question column in {path}. "
            f"Available columns: {list(df.columns)}",
            file=sys.stderr
        )
        return {}

    return {i: str(row[q_col]) for i, row in df.iterrows()}


def load_ground_truth_single(data_root: Path, lang: str) -> dict[int, str]:
    path = single_lang_text_csv_path(data_root, lang)
    if not path.exists():
        print(f"[WARNING] Ground-truth CSV not found: {path}", file=sys.stderr)
        return {}

    df = pd.read_csv(path)

    candidate_cols = [
        "answer",
        f"answer_{lang}",
        "Answer",
        "question",
        f"question_{lang}",
        "Question",
    ]

    gt_col = next((c for c in candidate_cols if c in df.columns), None)
    if gt_col is None:
        print(
            f"[WARNING] Could not find a ground-truth column in {path}. "
            f"Available columns: {list(df.columns)}",
            file=sys.stderr
        )
        return {}

    return {i: str(row[gt_col]) for i, row in df.iterrows()}


def row_index_from_filename(filename: str) -> int:
    stem = Path(filename).stem
    if stem.startswith("row_"):
        return int(stem.split("_")[1])
    raise ValueError(f"Cannot infer row index from filename: {filename}")


def save_results(out_dir: Path, model_name: str, tag: str, results: list[dict]):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name}_{tag}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved {len(results)} results to {out_file}")


def load_completed(out_dir: Path, model_name: str, tag: str) -> set[str]:
    path = out_dir / f"{model_name}_{tag}.json"
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        existing = json.load(f)
    return {r["file"] for r in existing if r.get("response") is not None}


def load_existing_results(out_dir: Path, model_name: str, tag: str) -> list[dict]:
    path = out_dir / f"{model_name}_{tag}.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def safe_inputs_to_device(inputs, device, dtype):
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


def attach_ground_truth(record: dict, filename: str, lang: str, data_root: Path) -> dict:
    row_idx = row_index_from_filename(filename)
    gt_map = load_ground_truth_single(data_root, lang)
    q_map = load_single_lang_questions(data_root, lang)
    record["ground_truth"] = gt_map.get(row_idx)
    record["question_text"] = q_map.get(row_idx)
    return record


def run_qwen25omni(audio_paths: dict, out_dir: Path, quantize: bool, data_root: Path, resume: bool = False):
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
    model.disable_talker()
    model.eval()
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    for lang, paths in audio_paths.items():
        completed = load_completed(out_dir, "qwen25omni", lang) if resume else set()
        pending = [p for p in paths if p.name not in completed]
        results = load_existing_results(out_dir, "qwen25omni", lang) if resume else []
        print(f"\n[qwen25omni] lang={lang} ({len(pending)}/{len(paths)} files)")

        for path in pending:
            try:
                conversation = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": [{"type": "audio", "audio": str(path)}]},
                ]
                text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
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
                    text_ids = model.generate(**inputs, max_new_tokens=512, return_audio=False)
                response = processor.batch_decode(
                    text_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                record = {"file": path.name, "tag": lang, "response": response}
                record = attach_ground_truth(record, path.name, lang, data_root)
                results.append(record)
                print(f"  {path.name}: {response[:80]}...")
            except Exception as e:
                print(f"  [ERROR] {path.name}: {e}", file=sys.stderr)
                results.append({"file": path.name, "tag": lang, "response": None, "error": str(e)})
            save_results(out_dir, "qwen25omni", lang, results)


def run_qwen3omni(audio_paths: dict, out_dir: Path, quantize: bool, data_root: Path, resume: bool = False):
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
    model.disable_talker()
    model.eval()
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    for lang, paths in audio_paths.items():
        completed = load_completed(out_dir, "qwen3omni", lang) if resume else set()
        pending = [p for p in paths if p.name not in completed]
        results = load_existing_results(out_dir, "qwen3omni", lang) if resume else []
        print(f"\n[qwen3omni] lang={lang} ({len(pending)}/{len(paths)} files)")

        for path in pending:
            try:
                conversation = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": [{"type": "audio", "audio": str(path)}]},
                ]
                text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
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
                    output_ids = model.generate(**inputs, max_new_tokens=512)
                response = processor.batch_decode(
                    output_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                record = {"file": path.name, "tag": lang, "response": response}
                record = attach_ground_truth(record, path.name, lang, data_root)
                results.append(record)
                print(f"  {path.name}: {response[:80]}...")
            except Exception as e:
                print(f"  [ERROR] {path.name}: {e}", file=sys.stderr)
                results.append({"file": path.name, "tag": lang, "response": None, "error": str(e)})
            save_results(out_dir, "qwen3omni", lang, results)


def run_voxtral(audio_paths: dict, out_dir: Path, quantize: bool, data_root: Path, resume: bool = False):
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

    for lang, paths in audio_paths.items():
        completed = load_completed(out_dir, "voxtral", lang) if resume else set()
        pending = [p for p in paths if p.name not in completed]
        results = load_existing_results(out_dir, "voxtral", lang) if resume else []
        print(f"\n[voxtral] lang={lang} ({len(pending)}/{len(paths)} files)")

        for path in pending:
            try:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "path": str(path)},
                            {"type": "text", "text": SYSTEM_PROMPT},
                        ],
                    },
                ]
                inputs = processor.apply_chat_template(conversation)
                inputs = inputs.to(model_device, dtype=model_dtype)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.2,
                        top_p=0.95,
                        do_sample=True,
                    )
                response = processor.batch_decode(
                    output_ids[:, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )[0]
                record = {"file": path.name, "tag": lang, "response": response}
                record = attach_ground_truth(record, path.name, lang, data_root)
                results.append(record)
                print(f"  {path.name}: {response[:80]}...")
            except Exception as e:
                print(f"  [ERROR] {path.name}: {e}", file=sys.stderr)
                results.append({"file": path.name, "tag": lang, "response": None, "error": str(e)})
            save_results(out_dir, "voxtral", lang, results)


def run_flamingo(audio_paths: dict, out_dir: Path, quantize: bool, data_root: Path, resume: bool = False):
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
        completed = load_completed(out_dir, "flamingo", lang) if resume else set()
        pending = [p for p in paths if p.name not in completed]
        results = load_existing_results(out_dir, "flamingo", lang) if resume else []
        print(f"\n[flamingo] lang={lang} ({len(pending)}/{len(paths)} files)")

        for path in pending:
            transcription = None
            response = None
            try:
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
                    conversation = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": transcription},
                    ]
                    text = processor.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    inputs2 = processor(text=text, return_tensors="pt")
                    inputs2 = safe_inputs_to_device(inputs2, model_device, model_dtype)
                    with torch.no_grad():
                        r_ids = model.generate(**inputs2, max_new_tokens=512)
                    response = processor.batch_decode(
                        r_ids[:, inputs2["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )[0].strip()
                    print(f"  {path.name} [response]: {response[:80]}...")
                except Exception as e:
                    print(f"  [ERROR] {path.name} response: {e}", file=sys.stderr)

            record = {
                "file": path.name,
                "tag": lang,
                "transcription": transcription,
                "response": response,
                "error": None if (transcription and response) else "partial_or_failed",
            }
            record = attach_ground_truth(record, path.name, lang, data_root)
            results.append(record)
            save_results(out_dir, "flamingo", lang, results)


def run_gpt(audio_paths: dict, out_dir: Path, data_root: Path, resume: bool = False, **_):
    import time
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in environment.")

    client = OpenAI(api_key=api_key)
    model_id = "gpt-4o-audio-preview"
    MAX_RETRIES = 5
    INITIAL_WAIT = 10

    for lang, paths in audio_paths.items():
        completed = load_completed(out_dir, "gpt", lang) if resume else set()
        pending = [p for p in paths if p.name not in completed]
        results = load_existing_results(out_dir, "gpt", lang) if resume else []
        print(f"\n[gpt] lang={lang} ({len(pending)}/{len(paths)} files)")

        for path in pending:
            with open(path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")

            wait = INITIAL_WAIT
            response_obj = None
            for attempt in range(MAX_RETRIES):
                try:
                    response_obj = client.chat.completions.create(
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
                            print(f"  [rate limit] {path.name}: waiting {wait}s...")
                            time.sleep(wait)
                            wait = min(wait * 2, 300)
                        else:
                            print(f"  [ERROR] {path.name}: all retries exhausted.")
                    else:
                        print(f"  [ERROR] {path.name}: {e}", file=sys.stderr)
                        break

            if response_obj is None:
                results.append({
                    "file": path.name,
                    "tag": lang,
                    "response": None,
                    "error": "api_error_or_rate_limit",
                })
            else:
                text = response_obj.choices[0].message.content
                record = {"file": path.name, "tag": lang, "response": text}
                record = attach_ground_truth(record, path.name, lang, data_root)
                results.append(record)
                print(f"  {path.name}: {text[:80]}...")
            save_results(out_dir, "gpt", lang, results)


def run_gemini(audio_paths: dict, out_dir: Path, data_root: Path, resume: bool = False, **_):
    import time
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set in environment.")

    client = genai.Client(api_key=api_key)
    model_id = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
    print(f"Using Gemini model: {model_id}")

    MAX_RETRIES = 5
    INITIAL_WAIT = 10

    for lang, paths in audio_paths.items():
        completed = load_completed(out_dir, "gemini", lang) if resume else set()
        pending = [p for p in paths if p.name not in completed]
        results = load_existing_results(out_dir, "gemini", lang) if resume else []
        print(f"\n[gemini] lang={lang} ({len(pending)}/{len(paths)} files)")

        for path in pending:
            with open(path, "rb") as f:
                audio_bytes = f.read()

            wait = INITIAL_WAIT
            response_text = None
            last_error = None

            for attempt in range(MAX_RETRIES):
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=[
                            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                            "Please respond to the audio.",
                        ],
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            max_output_tokens=8192,
                            thinking_config=types.ThinkingConfig(thinking_budget=2048),
                            temperature=0.0,
                        ),
                    )
                    if response.candidates and response.candidates[0].content.parts:
                        response_text = response.text
                    else:
                        finish_reason = response.candidates[0].finish_reason if response.candidates else "NO_CANDIDATES"
                        raise ValueError(f"Empty response from Gemini, finish_reason={finish_reason}")
                    break
                except Exception as e:
                    last_error = str(e)
                    err = last_error.lower()
                    if "429" in last_error or "quota" in err or "rate" in err:
                        if attempt < MAX_RETRIES - 1:
                            print(f"  [rate limit] {path.name}: waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})...")
                            time.sleep(wait)
                            wait = min(wait * 2, 300)
                        else:
                            print(f"  [ERROR] {path.name}: all retries exhausted.")
                    else:
                        print(f"  [ERROR] {path.name}: {e}", file=sys.stderr)
                        break

            if response_text is None:
                results.append({"file": path.name, "tag": lang, "response": None, "error": last_error})
            else:
                record = {"file": path.name, "tag": lang, "response": response_text}
                record = attach_ground_truth(record, path.name, lang, data_root)
                results.append(record)
                print(f"  {path.name}: {response_text[:80]}...")
            save_results(out_dir, "gemini", lang, results)


def run_salmonn(audio_paths: dict, out_dir: Path, quantize: bool, data_root: Path, resume: bool = False):
    import torch

    salmonn_repo = os.environ.get("SALMONN_REPO", str(data_root / "SALMONN"))
    if salmonn_repo not in sys.path:
        sys.path.insert(0, salmonn_repo)

    try:
        from model import SALMONN
    except ImportError as e:
        raise ImportError(f"Cannot import SALMONN from {salmonn_repo}. Error: {e}")

    ckpt_path = os.environ.get("SALMONN_CKPT", f"{salmonn_repo}/salmonn_v1.pth")
    beats_path = os.environ.get("SALMONN_BEATS_PATH", f"{salmonn_repo}/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
    whisper_path = os.environ.get("SALMONN_WHISPER_PATH", "openai/whisper-large-v2")
    vicuna_path = os.environ.get("SALMONN_LLM_PATH", "lmsys/vicuna-13b-v1.1")

    print("Loading SALMONN...")
    print(f"  ckpt:    {ckpt_path}")
    print(f"  beats:   {beats_path}")
    print(f"  whisper: {whisper_path}")
    print(f"  vicuna:  {vicuna_path}")
    print(f"  quantize (low_resource): {quantize}")

    model = SALMONN(
        ckpt=ckpt_path,
        whisper_path=whisper_path,
        beats_path=beats_path,
        vicuna_path=vicuna_path,
        low_resource=quantize,
    )
    model.eval()

    if torch.cuda.is_available():
        AUDIO_COMPONENTS = [
            "speech_encoder",
            "beats",
            "speech_Qformer",
            "speech_llama_proj",
            "ln_speech",
            "ln_audio",
            "second_btc_proj",
        ]
        for attr in AUDIO_COMPONENTS:
            if hasattr(model, attr):
                setattr(model, attr, getattr(model, attr).half().cuda())

        for _, param in model.named_parameters():
            if not param.is_cuda:
                param.data = param.data.half().cuda()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for lang, paths in audio_paths.items():
        completed = load_completed(out_dir, "salmonn", lang) if resume else set()
        pending = [p for p in paths if p.name not in completed]
        results = load_existing_results(out_dir, "salmonn", lang) if resume else []
        print(f"\n[salmonn] lang={lang} ({len(pending)}/{len(paths)} files)")

        for path in pending:
            try:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        response = model.generate(
                            wav_path=str(path),
                            prompt=SYSTEM_PROMPT,
                            num_beams=1,
                            do_sample=False,
                            min_length=1,
                            repetition_penalty=1.0,
                            length_penalty=1.0,
                            temperature=1.0,
                            device=device,
                        )[0]

                record = {"file": path.name, "tag": lang, "response": response}
                record = attach_ground_truth(record, path.name, lang, data_root)
                results.append(record)
                print(f"  {path.name}: {response[:80]}...")
            except Exception as e:
                print(f"  [ERROR] {path.name}: {e}", file=sys.stderr)
                results.append({"file": path.name, "tag": lang, "response": None, "error": str(e)})
            save_results(out_dir, "salmonn", lang, results)


def run_gemma(audio_paths: dict, out_dir: Path, quantize: bool, data_root: Path, resume: bool = False):
    from transformers import AutoProcessor, Gemma3nForConditionalGeneration
    import torch

    model_id = "google/gemma-3n-e4b-it"
    print(f"Loading {model_id}...")

    load_kwargs = dict(device_map="auto", torch_dtype=torch.bfloat16)
    if quantize:
        from transformers import BitsAndBytesConfig
        load_kwargs.pop("torch_dtype")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    model = Gemma3nForConditionalGeneration.from_pretrained(model_id, **load_kwargs).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    for lang, paths in audio_paths.items():
        completed = load_completed(out_dir, "gemma", lang) if resume else set()
        pending = [p for p in paths if p.name not in completed]
        results = load_existing_results(out_dir, "gemma", lang) if resume else []
        print(f"\n[gemma] lang={lang} ({len(pending)}/{len(paths)} files)")

        for path in pending:
            try:
                conversation = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": str(path)},
                        ],
                    },
                ]
                inputs = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)

                input_len = inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

                response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
                record = {"file": path.name, "tag": lang, "response": response}
                record = attach_ground_truth(record, path.name, lang, data_root)
                results.append(record)
                print(f"  {path.name}: {response[:80]}...")
            except Exception as e:
                print(f"  [ERROR] {path.name}: {e}", file=sys.stderr)
                results.append({"file": path.name, "tag": lang, "response": None, "error": str(e)})
            save_results(out_dir, "gemma", lang, results)
            
def run_gemma4(audio_paths: dict, out_dir: Path, data_root: Path, quantize: bool = False, resume: bool = False):
    import sys
    import numpy as np
    import torch
    import librosa
    from transformers import AutoProcessor, AutoModelForMultimodalLM

    model_id = "google/gemma-4-E4B-it"
    print(f"Loading {model_id}...")

    processor = AutoProcessor.from_pretrained(model_id)

    load_kwargs = {
        "device_map": "auto",
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }

    model = AutoModelForMultimodalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()

    for lang, paths in audio_paths.items():
        completed = load_completed(out_dir, "gemma4", lang) if resume else set()
        pending = [p for p in paths if p.name not in completed]
        results = load_existing_results(out_dir, "gemma4", lang) if resume else []

        print(f"\n[gemma4] lang={lang} ({len(pending)}/{len(paths)} files)")

        for path in pending:
            try:
                # Gemma audio expects mono 16kHz float audio.
                audio, sr = librosa.load(str(path), sr=16000, mono=True)
                audio = np.asarray(audio, dtype=np.float32)
                audio = np.clip(audio, -1.0, 1.0)

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": audio},
                            {
                                "type": "text",
                                "text": "Answer the question asked in the audio. Return only the final numeric answer."
                            },
                        ],
                    }
                ]

                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                )

                moved_inputs = {}
                for k, v in inputs.items():
                    if torch.is_tensor(v):
                        # Keep integer tensors as integers.
                        if v.dtype in (
                            torch.int8, torch.int16, torch.int32, torch.int64,
                            torch.uint8, torch.bool
                        ):
                            moved_inputs[k] = v.to(model.device)
                        else:
                            moved_inputs[k] = v.to(model.device, dtype=load_kwargs["dtype"])
                    else:
                        moved_inputs[k] = v

                input_len = moved_inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    output_ids = model.generate(
                        **moved_inputs,
                        max_new_tokens=64,
                        do_sample=False,
                    )

                generated_ids = output_ids[0][input_len:]
                raw_response = processor.decode(
                    generated_ids,
                    skip_special_tokens=True
                ).strip()

                record = {
                    "file": path.name,
                    "tag": lang,
                    "response": raw_response,
                }
                record = attach_ground_truth(record, path.name, lang, data_root)
                results.append(record)

                print(f"  {path.name}: {raw_response[:80]}...")

            except Exception as e:
                print(f"  [ERROR] {path.name}: {e}", file=sys.stderr)
                results.append({
                    "file": path.name,
                    "tag": lang,
                    "response": None,
                    "error": str(e),
                })

            save_results(out_dir, "gemma4", lang, results)


MODEL_RUNNERS = {
    "qwen25omni": run_qwen25omni,
    "qwen3omni": run_qwen3omni,
    "voxtral": run_voxtral,
    "flamingo": run_flamingo,
    "gpt": run_gpt,
    "gemini": run_gemini,
    "salmonn": run_salmonn,
    "gemma": run_gemma,
    "gemma4": run_gemma4,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run audio LLMs on monolingual MGSM audio data.")
    parser.add_argument("--model", required=True, choices=list(MODEL_RUNNERS.keys()))
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Load model in 4-bit (use when VRAM is tight).",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("."),
        help="Root directory containing audio_xtts_mgsm/ and global_mgsm/.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results_mgsm"),
        help="Directory to save JSON outputs.",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=None,
        help=(
            "Monolingual MGSM audio folders to process (e.g. de es fr it en). "
            "Looks under audio_xtts_mgsm/global_mgsm_{lang}/."
        ),
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run only the first file per language.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-completed files.",
    )
    args = parser.parse_args()

    print(f"Model: {args.model} | Quantize: {args.quantize} | Test: {args.test} | Resume: {args.resume}")

    langs = args.langs if args.langs is not None else SINGLE_LANGS
    audio_paths = get_single_lang_paths(args.data_root, langs)

    if not audio_paths:
        print(
            "[ERROR] No audio files found. Specify --langs and check --data_root.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.test:
        audio_paths = {lang: paths[:1] for lang, paths in audio_paths.items()}
        print("[TEST MODE] Running 1 file per language only.")

    runner = MODEL_RUNNERS[args.model]

    kwargs = {"data_root": args.data_root, "resume": args.resume}
    if args.model in ("qwen25omni", "qwen3omni", "voxtral", "flamingo", "salmonn", "gemma"):
        kwargs["quantize"] = args.quantize

    runner(audio_paths, args.out_dir, **kwargs)
    print("\nDone.")