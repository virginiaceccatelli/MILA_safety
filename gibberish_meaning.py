import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional
import pandas as pd

try:
    import transformers.utils.import_utils as _tu
    _tu.check_torch_load_is_safe = lambda: None
except Exception:
    pass


SYSTEM_PROMPT = """
You are analyzing a spoken utterance.
Return ONLY valid JSON with this exact schema:
{
  "transcript": "exact transcription of the utterance",
  "unrecognized_words": [
    {
      "word": "word or phrase from the utterance that you could not confidently recognize",
      "likely_meaning": "most likely meaning from context, or null if no likely meaning"
    }
  ]
}
Rules:
- Transcribe the utterance as exactly as possible.
- List only words or short phrases you did not confidently recognize.
- If there are no such words, return an empty list.
- If a word seems like nonsense/noise and you cannot infer a meaning, set likely_meaning to null.
- Do not answer the instruction or question in the utterance.
- Do not output markdown or any text outside the JSON.
""".strip()

SINGLE_LANGS = ["de", "es", "fr", "it", "en"]
N_FILES = 100

GIBBERISH_AUDIO_DIR = "gibberish_10/audio_xtts_gibberish"
GIBBERISH_META_DIR = "gibberish_10/output_augmented"
UNKNOWN_MEANING_STRINGS = {
    "", "null", "none", "unknown", "unk", "n/a", "na", "no meaning",
    "cannot infer", "can't infer", "unclear", "noise", "gibberish"
}


def single_lang_folder(data_root: Path, lang: str) -> Path:
    return data_root / GIBBERISH_AUDIO_DIR / f"original_{lang}_gibberish"


def single_lang_metadata_path(data_root: Path, lang: str) -> Path:
    return data_root / GIBBERISH_META_DIR / f"original_{lang}_gibberish.json"


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
                file=sys.stderr,
            )
        result[lang] = paths
    return result


def row_index_from_filename(filename: str) -> int:
    stem = Path(filename).stem
    if stem.startswith("row_"):
        return int(stem.split("_")[1])
    raise ValueError(f"Cannot infer row index from filename: {filename}")


def _json_load(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pick_text(entry: dict, *keys: str) -> Optional[str]:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str):
            return value
    return None


def load_gibberish_metadata(data_root: Path, lang: str) -> dict[int, dict]:
    path = single_lang_metadata_path(data_root, lang)
    if not path.exists():
        print(f"[WARNING] Metadata JSON not found: {path}", file=sys.stderr)
        return {}

    raw = _json_load(path)
    out: dict[int, dict] = {}

    if isinstance(raw, list):
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                continue
            idx = entry.get("row_idx", entry.get("row_index", i))
            out[int(idx)] = entry
        return out

    if isinstance(raw, dict):
        if "data" in raw and isinstance(raw["data"], list):
            for i, entry in enumerate(raw["data"]):
                if not isinstance(entry, dict):
                    continue
                idx = entry.get("row_idx", entry.get("row_index", i))
                out[int(idx)] = entry
            return out

        for key, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            if str(key).isdigit():
                idx = int(key)
            else:
                try:
                    idx = row_index_from_filename(str(key))
                except Exception:
                    idx = entry.get("row_idx", entry.get("row_index"))
                    if idx is None:
                        continue
            out[int(idx)] = entry
        return out

    print(f"[WARNING] Unsupported metadata structure in {path}", file=sys.stderr)
    return {}


def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[\W_]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize_words(s: Optional[str]) -> list[str]:
    return re.findall(r"[\wÀ-ÿ'-]+", s or "", flags=re.UNICODE)

def _find_implicit_meaning_from_transcript(
    transcript: str,
    original_text: str,
    modified_text: str,
    pseudo_text: str,
    diff_tag: Optional[str],
) -> Optional[str]:
    transcript_tokens = tokenize_words(transcript)
    original_tokens = tokenize_words(original_text)
    modified_tokens = tokenize_words(modified_text)
    pseudo_tokens = tokenize_words(pseudo_text)

    if not transcript_tokens or not modified_tokens or not pseudo_tokens:
        return None

    transcript_norm = [t.lower() for t in transcript_tokens]
    modified_norm = [t.lower() for t in modified_tokens]
    pseudo_norm = [t.lower() for t in pseudo_tokens]

    # If pseudo-word survives in transcript, this is not implicit normalization.
    if any(tok in transcript_norm for tok in pseudo_norm):
        return None

    # Only do the simple span extraction for insertions.
    if diff_tag != "insert" or len(pseudo_tokens) != 1:
        return None

    try:
        j = modified_norm.index(pseudo_norm[0])
    except ValueError:
        return None

    left_tok = modified_norm[j - 1] if j - 1 >= 0 else None
    right_tok = modified_norm[j + 1] if j + 1 < len(modified_norm) else None

    # Find left/right context in transcript and extract the words in between.
    left_idx = None
    right_idx = None

    if left_tok is not None:
        for i, tok in enumerate(transcript_norm):
            if tok == left_tok:
                left_idx = i

    if right_tok is not None:
        start = 0 if left_idx is None else left_idx + 1
        for i in range(start, len(transcript_norm)):
            if transcript_norm[i] == right_tok:
                right_idx = i
                break

    if left_idx is not None and right_idx is not None and right_idx > left_idx + 1:
        span = transcript_tokens[left_idx + 1:right_idx]
        if span:
            return " ".join(span)

    # If transcript collapses the pseudo-word entirely, return None.
    return None

def extract_pseudoword_spans(original_text: str, modified_text: str) -> list[dict]:
    original_tokens = tokenize_words(original_text)
    modified_tokens = tokenize_words(modified_text)
    sm = __import__('difflib').SequenceMatcher(a=[t.lower() for t in original_tokens], b=[t.lower() for t in modified_tokens])
    spans = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        mod_span = modified_tokens[j1:j2]
        orig_span = original_tokens[i1:i2]
        if not mod_span:
            continue
        spans.append({
            "pseudo_text": " ".join(mod_span),
            "pseudo_tokens": mod_span,
            "original_text": " ".join(orig_span) if orig_span else None,
            "original_tokens": orig_span,
            "diff_tag": tag,
        })
    return spans


def meaning_is_nullish(value) -> bool:
    if value is None:
        return True
    norm = normalize_text(str(value))
    return norm in UNKNOWN_MEANING_STRINGS


def parse_json_response(text: Optional[str]) -> dict:
    if not text:
        return {"transcript": None, "unrecognized_words": [], "parse_error": "empty_response"}

    stripped = text.strip()
    candidates = [stripped]
    if "```" in stripped:
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
        candidates = blocks + candidates
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if not isinstance(obj, dict):
                continue
            transcript = obj.get("transcript")
            unrec = obj.get("unrecognized_words", [])
            if not isinstance(unrec, list):
                unrec = []
            cleaned = []
            for item in unrec:
                if isinstance(item, str):
                    cleaned.append({"word": item, "likely_meaning": None})
                elif isinstance(item, dict):
                    cleaned.append({
                        "word": item.get("word") or item.get("token") or item.get("text"),
                        "likely_meaning": item.get("likely_meaning", item.get("meaning")),
                    })
            return {"transcript": transcript, "unrecognized_words": cleaned, "parse_error": None}
        except Exception:
            continue

    return {"transcript": stripped, "unrecognized_words": [], "parse_error": "json_parse_failed"}


def _best_similarity(a: str, b: str) -> float:
    import difflib
    return difflib.SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def _meaning_matches_expected(predicted_meaning: Optional[str], expected_meaning: Optional[str]) -> bool:
    if meaning_is_nullish(predicted_meaning) or not expected_meaning:
        return False
    p = normalize_text(predicted_meaning)
    e = normalize_text(expected_meaning)
    if not p or not e:
        return False
    if p == e or p in e or e in p:
        return True
    p_tokens = set(p.split())
    e_tokens = set(e.split())
    if p_tokens & e_tokens:
        return True
    return _best_similarity(p, e) >= 0.72


def evaluate_record_semantics(record: dict) -> dict:
    parsed = parse_json_response(record.get("response"))
    transcript = parsed.get("transcript") or ""
    unknowns = parsed.get("unrecognized_words", [])

    transcript_norm = normalize_text(transcript)
    unknown_words_norm = [normalize_text(x.get("word")) for x in unknowns if x.get("word")]

    pseudo_targets = record.get("pseudo_targets", []) or []
    item_results = []
    for target in pseudo_targets:
        pseudo_text = target.get("pseudo_text") or ""
        expected_meaning = target.get("original_text") or ""
        modified_text = record.get("modified_text") or ""
        pseudo_norm = normalize_text(pseudo_text)
        detected_in_transcript = bool(pseudo_norm and pseudo_norm in transcript_norm)
        detected_in_unknowns = any(
            uw == pseudo_norm or _best_similarity(uw, pseudo_norm) >= 0.8
            for uw in unknown_words_norm if uw
        )

        matched_unknown = None
        for item in unknowns:
            word = normalize_text(item.get("word"))
            if word and (word == pseudo_norm or _best_similarity(word, pseudo_norm) >= 0.8):
                matched_unknown = item
                break
        
        implicit_meaning = _find_implicit_meaning_from_transcript(
            transcript=transcript,
            original_text=record.get("original_text") or "",
            modified_text=record.get("modified_text") or "",
            pseudo_text=pseudo_text,
            diff_tag=target.get("diff_tag"),
        )

        meaning_assigned = bool(
            (matched_unknown and not meaning_is_nullish(matched_unknown.get("likely_meaning")))
            or implicit_meaning is not None
        )

        predicted_meaning = (
            matched_unknown.get("likely_meaning")
            if matched_unknown is not None
            else implicit_meaning
        )

        meaning_matches_original = (
            _meaning_matches_expected(predicted_meaning, expected_meaning)
            if expected_meaning
            else None
        )

        if detected_in_transcript:
            handling_type = "preserved"
        elif matched_unknown is not None:
            handling_type = "explicit_unknown"
        elif implicit_meaning is not None:
            handling_type = "implicit_normalization"
        elif normalize_text(transcript) != normalize_text(modified_text):
            handling_type = "substituted"
        else:
            handling_type = "ignored"

        item_results.append({
            "pseudo_text": pseudo_text,
            "expected_meaning": expected_meaning,
            "detected": detected_in_transcript or detected_in_unknowns,
            "detected_in_transcript": detected_in_transcript,
            "detected_in_unknowns": detected_in_unknowns,
            "meaning_assigned": meaning_assigned,
            "meaning_matches_original": meaning_matches_original,
            "predicted_meaning": predicted_meaning,
            "implicit_normalization": implicit_meaning is not None and matched_unknown is None,
            "handling_type": handling_type,
            "meaning_source": (
                "unknown_list"
                if matched_unknown is not None
                else "transcript_span"
                if implicit_meaning is not None
                else "none"
            ),
            "context_window_modified": modified_text,
            "context_window_transcript": transcript,
            "transcript_changed_around_pseudo": normalize_text(transcript) != normalize_text(modified_text),
        })

    return {
        "parsed_response": parsed,
        "pseudo_evaluation": item_results,
        "pseudo_detection_rate": (
            sum(1 for x in item_results if x["detected"]) / len(item_results)
            if item_results else None
        ),
        "meaning_attribution_rate": (
            sum(1 for x in item_results if x["meaning_assigned"]) / len(item_results)
            if item_results else None
        ),
        "meaning_match_rate": (
            sum(1 for x in item_results if x["meaning_matches_original"]) / len(item_results)
            if item_results else None
        ),
    }

def attach_ground_truth(record: dict, filename: str, lang: str, data_root: Path) -> dict:
    row_idx = row_index_from_filename(filename)
    meta_map = load_gibberish_metadata(data_root, lang)
    entry = meta_map.get(row_idx, {})

    original_text = _pick_text(entry, "original", "original_text", "clean", "source")
    modified_text = _pick_text(entry, "modified", "modified_text", "gibberish", "target")
    pseudo_targets = extract_pseudoword_spans(original_text or "", modified_text or "") if original_text and modified_text else []

    record["row_idx"] = row_idx
    record["original_text"] = original_text
    record["modified_text"] = modified_text
    record["pseudo_targets"] = pseudo_targets
    record.update(evaluate_record_semantics(record))
    return record


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
        existing = json.load(f)
    for r in existing:
        if "parsed_response" not in r:
            r.update(evaluate_record_semantics(r))
    return existing


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


def summarize_results_for_model(out_dir: Path, model_name: str, langs: list[str]) -> dict:
    summary = {"model": model_name, "langs": {}, "overall": {}}
    overall_pseudo = 0
    overall_detected = 0
    overall_meaning = 0
    overall_correct = 0

    for lang in langs:
        path = out_dir / f"{model_name}_{lang}.json"
        if not path.exists():
            continue
        rows = _json_load(path)
        pseudo_total = 0
        detected_total = 0
        meaning_total = 0
        correct_total = 0
        parse_failures = 0

        for row in rows:
            parsed = row.get("parsed_response") or parse_json_response(row.get("response"))
            if parsed.get("parse_error"):
                parse_failures += 1
            for item in row.get("pseudo_evaluation", []):
                pseudo_total += 1
                detected_total += int(bool(item.get("detected")))
                meaning_total += int(bool(item.get("meaning_assigned")))
                correct_total += int(bool(item.get("meaning_matches_original")))

        summary["langs"][lang] = {
            "n_rows": len(rows),
            "n_pseudo_targets": pseudo_total,
            "pseudo_detection_rate": detected_total / pseudo_total if pseudo_total else None,
            "meaning_attribution_rate": meaning_total / pseudo_total if pseudo_total else None,
            "meaning_match_rate": correct_total / pseudo_total if pseudo_total else None,
            "json_parse_failure_rate": parse_failures / len(rows) if rows else None,
        }
        overall_pseudo += pseudo_total
        overall_detected += detected_total
        overall_meaning += meaning_total
        overall_correct += correct_total

    summary["overall"] = {
        "n_pseudo_targets": overall_pseudo,
        "pseudo_detection_rate": overall_detected / overall_pseudo if overall_pseudo else None,
        "meaning_attribution_rate": overall_meaning / overall_pseudo if overall_pseudo else None,
        "meaning_match_rate": overall_correct / overall_pseudo if overall_pseudo else None,
    }

    out_path = out_dir / f"{model_name}_gibberish_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[summary] Saved {out_path}")
    return summary


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
                            "Return JSON exactly as instructed.",
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
    import librosa

    model_id = "google/gemma-3n-E4B-it"
    print(f"Loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)

    load_kwargs = dict(device_map="auto")
    if quantize:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = Gemma3nForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    model.eval()
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    for lang, paths in audio_paths.items():
        completed = load_completed(out_dir, "gemma", lang) if resume else set()
        pending = [p for p in paths if p.name not in completed]
        results = load_existing_results(out_dir, "gemma", lang) if resume else []
        print(f"\n[gemma] lang={lang} ({len(pending)}/{len(paths)} files)")

        for path in pending:
            try:
                audio_array, _ = librosa.load(str(path), sr=16000, mono=True)

                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": audio_array},
                            {"type": "text", "text": SYSTEM_PROMPT},
                        ],
                    },
                ]
                inputs = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model_device)

                inputs = {
                    k: v.to(model_dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                    for k, v in inputs.items()
                }

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


MODEL_RUNNERS = {
    "qwen25omni": run_qwen25omni,
    "qwen3omni": run_qwen3omni,
    "voxtral": run_voxtral,
    "flamingo": run_flamingo,
    "gpt": run_gpt,
    "gemini": run_gemini,
    "salmonn": run_salmonn,
    "gemma": run_gemma,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run audio LLMs on monolingual gibberish-audio data and score pseudo-word handling.")
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
        help="Root directory containing gibberish_10/.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("gibberish_meaning"),
        help="Directory to save JSON outputs.",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=None,
        help=(
            "Monolingual gibberish folders to process (e.g. de es fr it en). "
            "Looks under gibberish_10/audio_xtts_gibberish/original_{lang}_gibberish/."
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
    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Do not run the model; only recompute summaries from saved JSON outputs.",
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