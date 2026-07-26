#!/usr/bin/env python3
"""Generate SFT data streaming to disk — survives interrupts."""

from __future__ import annotations

import json, random, sys, time, threading
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps/analysis/src"))
from iracing_analysis.insights.openai_provider import SYSTEM_PROMPT, RADIO_PROMPT

ENDPOINTS = [
    ("http://192.168.151.233:8000/v1", "gemma-4-26b-a4b", "da12"),
    # 90aa excluded — context too short
    ("http://192.168.1.190:8000/v1", "gemma-4-26b-a4b", "dgx"),
]

OUT = Path("data/racing_sft_all.jsonl")
SUMMARIES = Path("racing_real_summaries.jsonl")
TASKS = ["report", "radio"]
LANGS = ["pl", "en"]
lock = threading.Lock()


def seen_ids():
    if not OUT.exists():
        return set()
    ids = set()
    for line in OUT.read_text().splitlines():
        try:
            ids.add(json.loads(line).get("_id", ""))
        except:
            pass
    return ids


def generate(client, model, summary, task, lang):
    system = RADIO_PROMPT if task == "radio" else SYSTEM_PROMPT
    ln = "Polish" if lang == "pl" else "English"
    grounding_rules = """
CRITICAL RULES:
- If signals.tyre_data_available is false, do NOT mention tyres, tires, pressure, or temperature AT ALL. Not even to say data is unavailable. Simply skip the topic entirely.
- Every number you write MUST come from the JSON summary. Do not invent, round differently, or estimate any number.
- Only mention corners that appear in the 'corners' list.
"""
    if task == "radio":
        user = f"Write a short pit-wall radio message (max 25 words) in {ln}.{grounding_rules}\n{json.dumps(summary, indent=1)}"
    else:
        user = f"Write the coaching report in {ln}.{grounding_rules}\n{json.dumps(summary, indent=1)}"

    resp = client.chat.completions.create(
        model=model, temperature=0.4, max_tokens=2048,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}]
    )
    return (resp.choices[0].message.content or "").strip()


def worker(ep_idx, combos, existing):
    url, model, label = ENDPOINTS[ep_idx]
    client = OpenAI(base_url=url, api_key="none", timeout=600, max_retries=1)
    done, errors = 0, 0

    for row, task, lang in combos:
        rid = f"{row['id']}_{task}_{lang}"
        if rid in existing:
            continue
        try:
            text = generate(client, model, row["summary"], task, lang)
            if not text:
                continue
            prompt = f"<task:{task} lang:{lang}>\n{json.dumps(row['summary'], indent=1)}"
            entry = json.dumps({
                "_id": rid, "_teacher": label,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": text}
                ]
            }, ensure_ascii=False)
            with lock:
                with open(OUT, "a") as f:
                    f.write(entry + "\n")
            done += 1
            if done % 5 == 0:
                print(f"  [{label}] {done} done, {errors} err", flush=True)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [{label}] err: {str(e)[:60]}", flush=True)
            time.sleep(3)

    print(f"  [{label}] FINISHED: {done} ok, {errors} err", flush=True)


def main():
    rows = [json.loads(l) for l in SUMMARIES.read_text().splitlines() if l.strip()]
    random.seed(42)
    random.shuffle(rows)

    combos = [(row, TASKS[i % 2], LANGS[(i // 2) % 2]) for i, row in enumerate(rows)]
    existing = seen_ids()
    print(f"Summaries: {len(rows)}, combos: {len(combos)}, existing: {len(existing)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    splits = [[] for _ in ENDPOINTS]
    for i, c in enumerate(combos):
        splits[i % len(ENDPOINTS)].append(c)

    threads = []
    for i in range(len(ENDPOINTS)):
        t = threading.Thread(target=worker, args=(i, splits[i], existing))
        t.start()
        threads.append(t)
        print(f"Started {ENDPOINTS[i][2]}: {len(splits[i])} combos")

    for t in threads:
        t.join()

    total = sum(1 for _ in open(OUT)) if OUT.exists() else 0
    print(f"\nTotal: {total} examples in {OUT}")


if __name__ == "__main__":
    main()
