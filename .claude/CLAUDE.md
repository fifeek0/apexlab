# Apex Lab — iRacing Telemetry Analysis

## Repo structure

```
packages/iracing-core/     shared library (parsing, alignment, delta, library, watcher, live)
apps/analysis/             post-session analysis app (PySide6 GUI + analysis lib + AI insights)
apps/overlay/              live Bloops-style overlay (delta bar, input trace, gear, braking cues)
eval/                      benchmark harness (racing_benchmark.py, racing_scorers.py, stats.py)
training/                  fine-tuning scripts (train_racing_coach.py, generate_sft_streaming.py)
configs/                   training configs (gemma4_e2b_racing_lora_v*.yaml)
data/                      SFT datasets (racing_sft_v*_train/val.jsonl)
tools/                     harvest_campaign.py (CSV-driven G61 data collection)
docs/                      documentation
```

## Key conventions

- Language of code: English
- Language of docs: Polish (user) / English (wiki, README)
- Communication with user: Polish
- Tests: pytest, offscreen Qt (`QT_QPA_PLATFORM=offscreen` in conftest.py)
- Run tests: `.venv/bin/python -m pytest`

## Infrastructure

| Host | Alias | IP | Model | Use |
|------|-------|----|-------|-----|
| DGX Spark (main) | `ssh dgx` | 192.168.1.190 | Gemma 4 26B (vLLM, port 8000) | Training + serving |
| DGX Spark #2 | `ssh da12` | 192.168.151.233 | Gemma 4 26B | SFT generation |
| DGX Spark #3 | `ssh 90aa` | 192.168.151.65 | Gemma 4 E4B | SFT generation |
| Gaming PC | — | 192.168.1.232 | Ollama (E2B fine-tuned) | Local inference |
| Mac (dev) | localhost | — | Ollama (benchmarks) | Development |

### DGX Spark critical rules

- **Never run training alongside 26B vLLM** — OOM guaranteed (80+30 > 128 GB)
- **Always reboot before training** — GPU memory leak after docker stop (see below)
- **Start 26B with**: `ssh dgx "bash ~/start_gemma4_vllm.sh --mtp --expose"`
- **Before touching DGX, ask the user** — don't stop/start containers without confirmation

### GPU memory leak (GB10 unified memory)

DGX Spark has no separate VRAM — GPU and CPU share 128 GB. After `docker stop`, NVIDIA UVM driver keeps ~50-70 GB in invisible pools (not in `/proc/meminfo` categories). `free -g` shows 50-70 GB "used" with no processes consuming it. **Reboot is the only reliable fix** (`sudo rmmod nvidia_uvm` is faster but needs sudo + stops everything). See `docs/llm-training-wiki.md` for full analysis.

## LLM Training

**Current best model: v5** (75% grounding, 100% language, 90% numbers)
**Confirmed after v6/v7/v8:** more data doesn't help — v5's 302 examples is the sweet spot. Deploy v5 with system prompt for tyre gate.

After any training-related task (new version trained, benchmark run, dataset change, config change), **update the wiki**:

### Auto-update checklist for `docs/llm-training-wiki.md`

After training a new model version:
1. Add a row to the **Experiment Log** table with version, train examples, key change, and all metrics
2. Update **"What each version taught us"** with the lesson from this run
3. If the new version is the best, update **"Best config"** section and the summary at the top

After generating new SFT data:
1. Update **Dataset Engineering** section with new counts and teacher stats
2. Update the **data quality** table if filter rates changed

After a benchmark run:
1. Update **Baseline Benchmarks** if testing a new base model
2. Add results to the experiment log

After discovering a new bug/workaround:
1. Add to **Gemma 4 specific bugs** or **DGX Spark Specifics**
2. Add to **Key Lessons** if it's a general principle

After changing the dataset pipeline:
1. Update **Filtering pipeline** code snippet
2. Update **Recommended dataset composition**

## Garage 61 API

- Token: `~/.iracing_analysis/g61_token` (never committed, never read its value)
- Resolve: `--token` > `$G61_TOKEN` > token file
- Rate limit: 1 req/2s (polite default)
- Scope: own + teammates + followed drivers (no global access)
- Docs say "don't use internal endpoints" — we use only official `/api/v1/*`
- Team join requests sent to ~20 open teams for data diversity

## Harvest

- CSV spec: `iracing_telemetry_dataset.csv` (single source of truth for what to collect)
- Run: `python tools/harvest_campaign.py [--dry-run] [--map-only]`
- Resumable: ULID manifest in `~/.iracing_analysis/g61_cache/`
- Current library: ~2469 laps, ~2359 analysis pairs, 35+ car models, 5 disciplines

## Fine-tuning quick reference

```bash
# Generate SFT data (da12 + dgx)
python training/generate_sft_streaming.py

# Filter + build dataset
# (see docs/llm-training-wiki.md "Filtering pipeline")

# Train (stop 26B first, reboot for clean memory)
ssh dgx "docker stop vllm-gemma4"
ssh dgx "sudo reboot"
# after reboot:
ssh dgx "cd ~/racing-coach-training && HF_HOME=~/hf_cache HF_TOKEN=... BNB_CUDA_VERSION=130 \
  python3 training/train_racing_coach.py --config configs/... --train-data ... --val-data ..."

# Merge + GGUF + deploy
ssh dgx "python3 training/merge_adapter.py --base-model google/gemma-4-e2b-it --adapter .../final --output .../merged"
ssh dgx "python3 llama.cpp/convert_hf_to_gguf.py .../merged --outfile ...-f16.gguf --outtype f16"
ssh dgx "llama-quantize ...-f16.gguf ...-Q4_K_M.gguf Q4_K_M"
scp dgx:...-Q4_K_M.gguf ~/models/
ollama create racing-coach-vN -f Modelfile

# Benchmark
python eval/racing_benchmark.py --endpoints '{"name": "http://localhost:11434/v1"}' \
  --model-names '{"name": "racing-coach-vN"}' --no-judge --n 40

# Restore 26B
ssh dgx "bash ~/start_gemma4_vllm.sh --mtp --expose"
```
