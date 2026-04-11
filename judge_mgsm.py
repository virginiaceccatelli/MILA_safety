from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# MODELS = ["gemma", "flamingo", "gemini", "gpt", "qwen25omni", "qwen3omni", "salmonn", "voxtral"]
MODELS = ["gemma4"]

class GPT4MathJudge:
    SYSTEM_PROMPT = (
        "You are an expert mathematical evaluation judge.\n"
        "Given a QUESTION, a MODEL RESPONSE (which may include chain-of-thought reasoning), and the GROUND TRUTH ANSWER (a number), determine whether the model arrived at the correct final answer.\n\n"
        "Rules:\n"
        "1. Correct   -> The model's final answer matches the ground truth numerically (tolerate formatting differences like '$', ',', '.0', etc.).\n"
        "2. Incorrect -> The model's final answer does not match the ground truth.\n"
        "3. No Answer -> the model produced no answer / only gave reasoning without a conclusion.\n\n"
        "Focus on the FINAL answer the model commits to, not intermediate steps. "
        "If the model gives multiple answers, use the last one.\n\n"
        "Return a compact JSON with exactly these fields:\n"
        "{\"label\": \"Correct | Incorrect | No Answer\", \"extracted_answer\": \"<the number you extracted from the response, or null>\"}\n\n"
        "Return ONLY the JSON object, no markdown, no extra text."
    )

    USER_TEMPLATE = (
        "QUESTION:\n{question}\n\n"
        "MODEL RESPONSE:\n{response}\n\n"
        "GROUND TRUTH ANSWER:\n{ground_truth}"
    )

    def __init__(self, model_name: str = "gpt-4.1"):
        load_dotenv()
        self.model_name = model_name
        self.client = OpenAI()

    def judge(self, question: str, response: str, ground_truth: str, max_retries: int = 5) -> Tuple[str, str, str]:
        wait = 10
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": self.USER_TEMPLATE.format(
                                question=question,
                                response=response,
                                ground_truth=ground_truth,
                            ),
                        },
                    ],
                    temperature=0,
                    max_tokens=200,
                    response_format={"type": "json_object"},
                )
                text = completion.choices[0].message.content.strip()
                result = json.loads(text)
                return (
                    result.get("label", "ParseError"),
                    result.get("extracted_answer", None),
                    result.get("reason", ""),
                )

            except Exception as e:
                err = str(e)
                if "429" in err or "rate_limit" in err.lower():
                    if attempt < max_retries - 1:
                        print(f"    [rate limit] waiting {wait}s ...")
                        time.sleep(wait)
                        wait = min(wait * 2, 120)
                    else:
                        return "Error", None, err
                else:
                    return "Error", None, err

        return "Error", None, "max retries exceeded"


def row_index_from_filename(filename: str) -> int:
    return int(Path(filename).stem.split("_")[1])


def results_file_for_model(results_root: Path, model: str, tag: str) -> Path:
    return results_root / f"results_mgsm_{model}" / f"{model}_{tag}.json"


def discover_lang_tags(results_root: Path, model: str) -> list[str]:
    d = results_root / f"results_mgsm_{model}"
    if not d.exists():
        return []
    tags: list[str] = []
    for p in sorted(d.glob(f"{model}_*.json")):
        stem = p.stem
        if stem.endswith("_judged"):
            continue
        m = re.match(rf"^{re.escape(model)}_(.+)$", stem)
        if m:
            tags.append(m.group(1))
    return tags


def load_responses(results_root: Path, model: str, tag: str) -> list[dict]:
    path = results_file_for_model(results_root, model, tag)
    if not path.exists():
        print(f"[WARNING] Results file not found: {path}", file=sys.stderr)
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_judged(out_dir: Path, model: str, tag: str, results: list[dict]):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model}_{tag}_judged.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved {len(results)} judged results to {out_file}")


def load_completed_judge(out_dir: Path, model: str, tag: str) -> set[str]:
    path = out_dir / f"{model}_{tag}_judged.json"
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        existing = json.load(f)
    return {
        r["file"]
        for r in existing
        if r.get("judgement") is not None
        and r["judgement"].get("label") not in (None, "Error", "ParseError")
    }


def compute_summary(judge_dir: Path, models: list[str], lang_tags: list[str]) -> dict:
    summary: dict = {}
    for model in models:
        summary[model] = {}
        for tag in lang_tags:
            path = judge_dir / f"{model}_{tag}_judged.json"
            if not path.exists():
                continue

            with open(path, encoding="utf-8") as f:
                records = json.load(f)

            valid = [
                r for r in records
                if r.get("judgement") is not None
                and r["judgement"].get("label") not in ("Error", "ParseError")
            ]
            if not valid:
                continue

            n = len(valid)
            labels = [r["judgement"]["label"] for r in valid]
            no_response = sum(1 for r in records if r.get("response") is None)
            correct   = sum(1 for l in labels if l == "Correct")
            incorrect = sum(1 for l in labels if l == "Incorrect")

            summary[model][tag] = {
                "n_total": len(records),
                "n_judged": n,
                "n_no_response": no_response,
                "correct_rate":   round(correct   / n, 3),
                "incorrect_rate": round(incorrect / n, 3),
                "correct":   correct,
                "incorrect": incorrect,
            }

    return summary


def print_summary_table(summary: dict):
    print("\n" + "=" * 70)
    print("MATH JUDGE SUMMARY  (GPT-4.1 | temperature=0 | Correct / Incorrect)")
    print("=" * 70)
    header = f"{'Model':<14} {'Tag':<12} {'N':>5} {'Correct%':>9} {'Incorrect%':>11}"
    print(header)
    print("-" * 70)
    for model, tags in summary.items():
        for tag, m in tags.items():
            print(
                f"{model:<14} {tag:<12} {m['n_judged']:>5} "
                f"{m['correct_rate']*100:>8.1f}% "
                f"{m['incorrect_rate']*100:>10.1f}%"
            )
    print("=" * 70)


def run_judge(
    models: list[str],
    lang_tags: list[str] | None,
    results_root: Path,
    judge_dir: Path,
    test: bool,
    resume: bool,
    limit: int | None,
):
    judge = GPT4MathJudge(model_name="gpt-4.1")
    all_lang_tags_for_summary: set[str] = set(lang_tags or [])

    for model in models:
        tags = lang_tags if lang_tags else discover_lang_tags(results_root, model)
        for t in tags:
            all_lang_tags_for_summary.add(t)

        if not tags:
            print(f"\n[judge] {model}: no result files found under results_mgsm_{model}/")
            continue

        for tag in tags:
            print(f"\n[judge] model={model} tag={tag}")

            responses = load_responses(results_root, model, tag)
            if not responses:
                print("  [SKIP] No responses found.")
                continue

            if test:
                responses = responses[:1]
            if limit is not None:
                responses = responses[:limit]

            completed = load_completed_judge(judge_dir, model, tag) if resume else set()
            existing_path = judge_dir / f"{model}_{tag}_judged.json"
            judged_results = (
                json.loads(existing_path.read_text(encoding="utf-8"))
                if existing_path.exists() and resume
                else []
            )

            pending = [r for r in responses if r.get("file") not in completed]
            print(f"  pending={len(pending)} completed={len(completed)} total={len(responses)}")

            for record in pending:
                filename = record["file"]
                response_text = record.get("response")
                ground_truth = record.get("ground_truth")
                # Prefer question_text; fall back to transcription
                question = record.get("question_text") or record.get("transcription", "")

                print(f"  [process] model={model} tag={tag} file={filename}")

                if response_text is None:
                    print(f"    [SKIP] no_model_response")
                    judged_results.append({
                        **record,
                        "judgement": None,
                        "error": "no_model_response",
                    })
                    save_judged(judge_dir, model, tag, judged_results)
                    continue

                if ground_truth is None:
                    print(f"    [SKIP] no_ground_truth")
                    judged_results.append({
                        **record,
                        "judgement": None,
                        "error": "no_ground_truth",
                    })
                    save_judged(judge_dir, model, tag, judged_results)
                    continue

                label, extracted, reason = judge.judge(question, response_text, str(ground_truth))
                judged_results.append({
                    **record,
                    "judgement": {
                        "label": label,
                        "extracted_answer": extracted,
                        "ground_truth": ground_truth,
                        "reason": reason,
                    },
                })
                print(f"    [judge] label={label} extracted={extracted} gt={ground_truth} reason={reason[:100]}")
                save_judged(judge_dir, model, tag, judged_results)
                time.sleep(0.5)

    summary = compute_summary(
        judge_dir=judge_dir,
        models=models,
        lang_tags=sorted(all_lang_tags_for_summary),
    )
    judge_dir.mkdir(parents=True, exist_ok=True)
    summary_path = judge_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[judge] Summary saved to {summary_path}")
    print_summary_table(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-4.1 LLM-as-judge for MGSM math evaluation.")
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help="Models to judge. Default: all in MODELS list.")
    parser.add_argument(
        "--langs", nargs="+", default=None,
        help="Language tags to judge, e.g.: en de fr. If omitted, auto-discovers from results_mgsm_{model}/ folders.",
    )
    parser.add_argument("--results_root", type=Path, default=Path("."),
                        help="Root dir containing results_{model}/ subfolders. Default: current dir.")
    parser.add_argument("--judge_dir", type=Path, default=Path("judge_results_mgsm"),
                        help="Directory to save judged JSON outputs.")
    parser.add_argument("--test", action="store_true",
                        help="Judge only first record per model/tag.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-judged files.")
    parser.add_argument("--summary_only", action="store_true",
                        help="Skip judging; recompute summary from existing judged files.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N records per (model, tag).")
    args = parser.parse_args()

    if args.summary_only:
        tags = args.langs
        if tags is None:
            tags_set: set[str] = set()
            for m in args.models:
                tags_set.update(discover_lang_tags(args.results_root, m))
            tags = sorted(tags_set)
        summary = compute_summary(args.judge_dir, args.models, tags)
        args.judge_dir.mkdir(parents=True, exist_ok=True)
        with open(args.judge_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print_summary_table(summary)
    else:
        run_judge(
            models=args.models,
            lang_tags=args.langs,
            results_root=args.results_root,
            judge_dir=args.judge_dir,
            test=args.test,
            resume=args.resume,
            limit=args.limit,
        )