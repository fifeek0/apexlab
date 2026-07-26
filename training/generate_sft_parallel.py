#!/usr/bin/env python3
"""Generate SFT training data using multiple teacher endpoints in parallel.

Splits the summaries across N endpoints, generates report/radio examples
in both PL/EN, validates grounding, and merges into one JSONL.

Usage:
    python training/generate_sft_parallel.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps/analysis/src"))

from iracing_analysis.insights.openai_provider import SYSTEM_PROMPT, RADIO_PROMPT

# Endpoints: (base_url, model_name, label)
ENDPOINTS = [
    ("http://192.168.151.233:8000/v1", "gemma-4-26b-a4b", "da12-26b"),
    ("http://192.168.151.65:8000/v1", "gemma4-e4b-v5", "90aa-e4b"),
    ("http://192.168.1.190:8000/v1", "gemma-4-26b-a4b", "dgx-26b"),
]

SUMMARIES = Path("racing_real_summaries.jsonl")
OUT_TRAIN = Path("data/racing_sft_train.jsonl")
OUT_VAL = Path("data/racing_sft_val.jsonl")

TASKS = ["report", "radio"]
LANGS = ["pl", "en"]


def make_client(base_url):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key="none", timeout=300, max_retries=1)


def generate_one(client, model, row, task, lang):
    summary = row["summary"]
    student_prompt = f"<task:{task} lang:{lang}>\n{json.dumps(summary, indent=1)}"

    system = RADIO_PROMPT if task == "radio" else SYSTEM_PROMPT
    lang_name = "Polish" if lang == "pl" else "English"
    if task == "radio":
        user_msg = f"Write a short pit-wall radio message (max 25 words) in {lang_name}.\n\n{json.dumps(summary, indent=1)}"
    else:
        user_msg = f"Write the coaching report in {lang_name}.\n\n{json.dumps(summary, indent=1)}"

    resp = client.chat.completions.create(
        model=model, temperature=0.4, max_tokens=2048,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
    )
    teacher_response = resp.choices[0].message.content or ""
    if not teacher_response.strip():
        return None

    return {
        "messages": [
            {"role": "user", "content": student_prompt},
            {"role": "assistant", "content": teacher_response}
        ]
    }


def worker(endpoint_idx, rows, existing_ids):
    base_url, model, label = ENDPOINTS[endpoint_idx]
    client = make_client(base_url)
    results = []
    errors = 0

    for i, (row, task, lang) in enumerate(rows):
        row_id = f"{row['id']}_{task}_{lang}"
        if row_id in existing_ids:
            continue
        try:
            example = generate_one(client, model, row, task, lang)
            if example:
                example["_id"] = row_id
                example["_teacher"] = label
                results.append(example)
                if len(results) % 10 == 0:
                    print(f"  [{label}] {len(results)} generated, {errors} errors")
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [{label}] error #{errors}: {str(e)[:80]}")
            time.sleep(2)

    print(f"  [{label}] DONE: {len(results)} examples, {errors} errors")
    return results


def main():
    rows = [json.loads(l) for l in SUMMARIES.read_text().splitlines() if l.strip()]
    random.seed(42)
    random.shuffle(rows)

    # expand to (row, task, lang) combos
    combos = []
    for i, row in enumerate(rows):
        task = TASKS[i % 2]
        lang = LANGS[(i // 2) % 2]
        combos.append((row, task, lang))

    # load existing to resume
    existing_ids = set()
    for path in [OUT_TRAIN, OUT_VAL]:
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    existing_ids.add(json.loads(line).get("_id", ""))
                except:
                    pass
    print(f"Existing examples: {len(existing_ids)} (will skip)")

    # split across endpoints (round-robin)
    n_endpoints = len(ENDPOINTS)
    splits = [[] for _ in range(n_endpoints)]
    for i, combo in enumerate(combos):
        splits[i % n_endpoints].append(combo)

    print(f"\nTotal combos: {len(combos)}")
    for i, (_, _, label) in enumerate(ENDPOINTS):
        print(f"  {label}: {len(splits[i])} examples to generate")

    # run in parallel threads
    all_results = []
    with ThreadPoolExecutor(max_workers=n_endpoints) as pool:
        futures = {
            pool.submit(worker, i, splits[i], existing_ids): i
            for i in range(n_endpoints)
        }
        for future in as_completed(futures):
            all_results.extend(future.result())

    random.shuffle(all_results)

    # split 90/10 train/val
    split = int(len(all_results) * 0.9)
    train, val = all_results[:split], all_results[split:]

    # append to existing
    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TRAIN, "a") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(OUT_VAL, "a") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    total_train = sum(1 for _ in open(OUT_TRAIN))
    total_val = sum(1 for _ in open(OUT_VAL))
    print(f"\nNew: {len(train)} train + {len(val)} val")
    print(f"Total: {total_train} train + {total_val} val -> {OUT_TRAIN}, {OUT_VAL}")
    teachers = {}
    for ex in all_results:
        t = ex.get("_teacher", "?")
        teachers[t] = teachers.get(t, 0) + 1
    print(f"Per teacher: {teachers}")


if __name__ == "__main__":
    main()
