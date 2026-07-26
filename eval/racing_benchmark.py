#!/usr/bin/env python3
"""Racing-coach LLM benchmark: grounding + LLM-as-judge.

Compares models on the racing-coach task using real telemetry summaries.
Measures grounding (corner validity, tyre gate, language, numbers, length)
and optionally coaching quality via an LLM judge.

Usage:
    # Benchmark 26B teacher (baseline)
    python eval/racing_benchmark.py \
        --endpoints '{"gemma26b": "http://192.168.1.190:8000/v1"}' \
        --model-names '{"gemma26b": "gemma-4-26b-a4b"}' \
        --n 40

    # Compare teacher vs student
    python eval/racing_benchmark.py \
        --endpoints '{"teacher": "http://192.168.1.190:8000/v1", "student": "http://localhost:8080/v1"}' \
        --model-names '{"teacher": "gemma-4-26b-a4b", "student": "gemma-4-e2b-racing-v1"}' \
        --judge-endpoint teacher --n 40

    # No-judge mode (grounding only, fast)
    python eval/racing_benchmark.py \
        --endpoints '{"gemma26b": "http://192.168.1.190:8000/v1"}' \
        --model-names '{"gemma26b": "gemma-4-26b-a4b"}' \
        --no-judge --n 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from racing_scorers import score_all  # noqa: E402
from stats import bootstrap_ci, compare_models  # noqa: E402

SUMMARIES_PATH = Path(__file__).resolve().parent.parent / "racing_real_summaries.jsonl"

SYSTEM_PROMPT_REPORT = (
    Path(__file__).resolve().parent.parent
    / "apps/analysis/src/iracing_analysis/insights/openai_provider.py"
)

RADIO_PROMPT = """\
You are a race engineer on the pit-wall radio. From the JSON lap summary,
say ONE short radio message to the driver (max 25 words): the delta, the one
or two corners that matter most and what to change. Speak the language given
in the 'language' field. No preamble, no markdown — just the radio line.
"""

JUDGE_PROMPT = """\
You are evaluating a sim-racing coaching response. The coach received a JSON
telemetry summary and was asked to write a {task} in {language}.

Score the response on a 1-5 scale:
5 = Specific, quantitative, actionable advice addressing the actual data
4 = Good advice, minor imprecision or missing one key point
3 = Generic/vague advice or partially addresses the data
2 = Mostly irrelevant or generic sim-racing tips
1 = Off-topic, wrong language, or fabricated data

Respond with ONLY a JSON object: {{"score": N, "reason": "one sentence"}}

Summary (input):
{summary}

Response (to evaluate):
{response}
"""


# ---------------------------------------------------------------------------
# benchmark set builder
# ---------------------------------------------------------------------------


def build_benchmark_set(
    jsonl_path: Path, n: int = 40, seed: int = 42
) -> list[dict]:
    """Sample n scenarios from real summaries, balanced across tasks/languages."""
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    # filter out rows with no corners (flat-out ovals produce less interesting evals)
    usable = [r for r in rows if r.get("summary", {}).get("corners")]
    rng = random.Random(seed)
    rng.shuffle(usable)
    usable = usable[:n]

    bench = []
    tasks = ["report", "radio"]
    langs = ["pl", "en"]
    for i, row in enumerate(usable):
        task = tasks[i % 2]
        lang = langs[(i // 2) % 2]
        summary = row["summary"]
        prompt = f"<task:{task} lang:{lang}>\n{json.dumps(summary, indent=1)}"
        bench.append({
            "id": f"racing-{i:03d}",
            "task": task,
            "language": lang,
            "discipline": row.get("discipline", "road"),
            "track": row.get("track", ""),
            "summary": summary,
            "prompt": prompt,
        })
    return bench


# ---------------------------------------------------------------------------
# model calling
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    id: str
    endpoint_name: str
    task: str
    language: str
    prompt: str
    response: str = ""
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    tokens: int = 0
    error: str = ""
    grounding: dict = field(default_factory=dict)
    judge_score: int | None = None
    judge_reason: str = ""


async def call_model(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    system: str,
    user: str,
    timeout: float = 120.0,
) -> tuple[str, float, float, int]:
    t0 = time.monotonic()
    resp = await client.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "temperature": 0.4,
            "max_tokens": 2048,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=timeout,
    )
    total = (time.monotonic() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"] or ""
    tokens = data.get("usage", {}).get("completion_tokens", len(text.split()))
    return text, 0.0, total, tokens


async def call_judge(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    task: str,
    language: str,
    summary: dict,
    response: str,
) -> tuple[int, str]:
    prompt = JUDGE_PROMPT.format(
        task=task, language=language,
        summary=json.dumps(summary, indent=1)[:3000],
        response=response[:2000],
    )
    text, _, _, _ = await call_model(client, base_url, model, "You are an evaluation judge.", prompt)
    try:
        obj = json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
        return int(obj["score"]), str(obj.get("reason", ""))
    except Exception:
        import re
        m = re.search(r'"score"\s*:\s*(\d)', text)
        if m:
            return int(m.group(1)), text[:200]
        return 0, f"parse error: {text[:200]}"


# ---------------------------------------------------------------------------
# main benchmark loop
# ---------------------------------------------------------------------------


def _get_system_prompt(task: str) -> str:
    if task == "radio":
        return RADIO_PROMPT
    # extract SYSTEM_PROMPT from the openai_provider module
    try:
        src = SYSTEM_PROMPT_REPORT.read_text()
        start = src.index('SYSTEM_PROMPT = """\\\n') + len('SYSTEM_PROMPT = """\\\n')
        end = src.index('\n"""', start)
        return src[start:end]
    except Exception:
        return "You are a professional race engineer analysing iRacing telemetry."


async def run_benchmark(
    endpoints: dict[str, str],
    model_names: dict[str, str],
    bench: list[dict],
    judge_endpoint: str | None = None,
    concurrency: int = 2,
) -> list[ModelResult]:
    results: list[ModelResult] = []
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        async def process(item: dict, ep_name: str):
            async with sem:
                url = endpoints[ep_name]
                model = model_names.get(ep_name, "")
                system = _get_system_prompt(item["task"])
                r = ModelResult(
                    id=item["id"], endpoint_name=ep_name,
                    task=item["task"], language=item["language"],
                    prompt=item["prompt"],
                )
                try:
                    r.response, r.ttft_ms, r.total_ms, r.tokens = await call_model(
                        client, url, model, system, item["prompt"]
                    )
                except Exception as e:
                    r.error = str(e)[:300]

                if r.response and not r.error:
                    r.grounding = score_all(
                        r.response, item["summary"], item["task"], item["language"]
                    )

                if r.response and not r.error and judge_endpoint:
                    j_url = endpoints[judge_endpoint]
                    j_model = model_names.get(judge_endpoint, "")
                    try:
                        r.judge_score, r.judge_reason = await call_judge(
                            client, j_url, j_model,
                            item["task"], item["language"],
                            item["summary"], r.response,
                        )
                    except Exception as e:
                        r.judge_reason = f"judge error: {e}"

                results.append(r)
                status = "✓" if r.grounding.get("grounding_pass") else "✗"
                judge = f" judge={r.judge_score}" if r.judge_score else ""
                print(f"  {status} {r.id} {ep_name} {r.task}/{r.language}"
                      f" {r.total_ms:.0f}ms{judge}")

        tasks = []
        for item in bench:
            for ep_name in endpoints:
                tasks.append(process(item, ep_name))
        await asyncio.gather(*tasks)

    return results


def summarize(results: list[ModelResult], endpoints: list[str]) -> dict:
    summary = {"generated_at": datetime.now().isoformat(), "endpoints": {}}

    for ep in endpoints:
        ep_results = [r for r in results if r.endpoint_name == ep and not r.error]
        if not ep_results:
            continue

        grounding_pass = [1.0 if r.grounding.get("grounding_pass") else 0.0 for r in ep_results]
        acc, lo, hi = bootstrap_ci(grounding_pass)

        per_check = {}
        for check_name in ("corner_grounding", "tyre_gate", "language", "number_grounding", "radio_length"):
            vals = [1.0 if r.grounding.get("checks", {}).get(check_name, {}).get("pass") else 0.0
                    for r in ep_results]
            per_check[check_name] = round(sum(vals) / len(vals) * 100, 1) if vals else 0

        judge_scores = [r.judge_score for r in ep_results if r.judge_score is not None]

        ep_summary = {
            "n": len(ep_results),
            "errors": sum(1 for r in results if r.endpoint_name == ep and r.error),
            "grounding_pass_rate": round(acc * 100, 1),
            "grounding_ci_95": (round(lo * 100, 1), round(hi * 100, 1)),
            "per_check": per_check,
            "avg_latency_ms": round(sum(r.total_ms for r in ep_results) / len(ep_results)),
        }
        if judge_scores:
            ep_summary["judge_mean"] = round(sum(judge_scores) / len(judge_scores), 2)
            ep_summary["judge_distribution"] = {
                str(i): sum(1 for s in judge_scores if s == i) for i in range(1, 6)
            }
        summary["endpoints"][ep] = ep_summary

    # paired comparison if exactly 2 endpoints
    if len(endpoints) == 2:
        a_name, b_name = endpoints
        a_by_id = {r.id: 1.0 if r.grounding.get("grounding_pass") else 0.0
                   for r in results if r.endpoint_name == a_name and not r.error}
        b_by_id = {r.id: 1.0 if r.grounding.get("grounding_pass") else 0.0
                   for r in results if r.endpoint_name == b_name and not r.error}
        cmp = compare_models(a_by_id, b_by_id)
        summary["comparison_grounding"] = {
            "a": a_name, "b": b_name,
            "delta": round(cmp["delta"] * 100, 1) if cmp["delta"] is not None else None,
            "p_value": round(cmp["p_value"], 4) if cmp["p_value"] is not None else None,
            "significant": cmp.get("significant", False),
        }

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoints", required=True, help="JSON: {name: url}")
    parser.add_argument("--model-names", required=True, help="JSON: {name: model_id}")
    parser.add_argument("--n", type=int, default=40, help="benchmark size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summaries", default=str(SUMMARIES_PATH))
    parser.add_argument("--judge-endpoint", default=None,
                        help="endpoint name to use as LLM judge")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--out", default=None, help="save detailed results JSON")
    args = parser.parse_args()

    endpoints = json.loads(args.endpoints)
    model_names = json.loads(args.model_names)
    judge = None if args.no_judge else args.judge_endpoint

    print(f"Building benchmark set (n={args.n}) from {args.summaries}")
    bench = build_benchmark_set(Path(args.summaries), n=args.n, seed=args.seed)
    print(f"  tasks: {sum(1 for b in bench if b['task']=='report')} report, "
          f"{sum(1 for b in bench if b['task']=='radio')} radio")
    print(f"  languages: {sum(1 for b in bench if b['language']=='pl')} PL, "
          f"{sum(1 for b in bench if b['language']=='en')} EN\n")

    print(f"Running {len(bench)} × {len(endpoints)} = {len(bench) * len(endpoints)} calls...")
    results = asyncio.run(run_benchmark(endpoints, model_names, bench, judge, args.concurrency))

    print("\n" + "=" * 60)
    summary = summarize(results, list(endpoints.keys()))
    for ep, data in summary["endpoints"].items():
        print(f"\n{ep}:")
        print(f"  Grounding pass rate: {data['grounding_pass_rate']}% "
              f"(95% CI: {data['grounding_ci_95'][0]}–{data['grounding_ci_95'][1]}%)")
        for check, rate in data["per_check"].items():
            print(f"    {check:25s} {rate:5.1f}%")
        if "judge_mean" in data:
            print(f"  Judge mean: {data['judge_mean']}/5  dist: {data['judge_distribution']}")
        print(f"  Avg latency: {data['avg_latency_ms']}ms, errors: {data['errors']}")

    if "comparison_grounding" in summary:
        c = summary["comparison_grounding"]
        sig = "SIGNIFICANT" if c["significant"] else "not significant"
        print(f"\nPaired comparison ({c['a']} vs {c['b']}): "
              f"Δ={c['delta']}pp, p={c['p_value']} ({sig})")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "summary": summary,
            "results": [asdict(r) for r in results],
        }, indent=2, default=str))
        print(f"\nDetailed results saved to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
