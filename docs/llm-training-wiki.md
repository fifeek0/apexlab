# LLM Training Wiki — Racing Coach Model

Lessons learned from fine-tuning Gemma 4 E2B-it as a local racing coach for iRacing telemetry analysis. Everything here comes from real experiments (v1–v6) on NVIDIA DGX Spark (GB10, 128 GB unified memory).

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Model Selection](#model-selection)
- [Baseline Benchmarks](#baseline-benchmarks)
- [Dataset Engineering](#dataset-engineering)
- [Training Configuration](#training-configuration)
- [DGX Spark Specifics](#dgx-spark-specifics)
- [Evaluation & Benchmark](#evaluation--benchmark)
- [Experiment Log](#experiment-log)
- [Key Lessons](#key-lessons)
- [Deployment](#deployment)

---

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Real telemetry  │────▶│  build_summary() │────▶│  Student prompt   │
│  (.ibt / G61 CSV)│     │  (per-corner      │     │  <task:report     │
│                  │     │   deltas, sectors, │     │   lang:pl>        │
│                  │     │   GG, theoretical) │     │  {summary JSON}   │
└─────────────────┘     └──────────────────┘     └────────┬──────────┘
                                                           │
                         ┌─────────────────┐               ▼
                         │  Teacher (26B)   │     ┌───────────────────┐
                         │  generates SFT   │────▶│  Fine-tuned E2B   │
                         │  responses       │     │  (2.8 GB GGUF Q4) │
                         └─────────────────┘     │  runs locally     │
                                                  └───────────────────┘
```

The student model receives a structured JSON summary (not raw telemetry) and produces either:
- **report**: detailed coaching report with corners, technique themes, plan (~500 words)
- **radio**: one-sentence pit-wall message (~15-25 words)

Student prompt contract (no system prompt — behaviour baked into weights):
```
<task:report lang:pl>
{summary JSON}
```

## Model Selection

| Model | Active params | GGUF Q4 | Impact on sim | Baseline grounding |
|-------|-------------|---------|---------------|-------------------|
| Gemma 4 26B | 26B (MoE) | ~15 GB | heavy | 57.5% |
| Gemma 4 E4B | 4.5B (MoE) | ~5 GB | noticeable | 42.5% |
| **Gemma 4 E2B** | **2B (MoE)** | **2.8 GB** | **zero** | **47.5%** |

**Decision: E2B.** Despite E4B being larger, E2B had better baseline grounding (47.5% vs 42.5%) and better number accuracy (85% vs 80%). E2B in GGUF Q4 at 2.8 GB has zero impact on the sim running alongside. All models are Apache 2.0 licensed.

## Baseline Benchmarks

Measured on 40 questions (20 report + 20 radio, 50/50 PL/EN), grounding checks are rule-based (no LLM judge needed):

### Grounding checks explained

| Check | What it validates | How |
|-------|------------------|-----|
| **corner_grounding** | Corner names in response ⊆ corners in JSON | Regex `T\d+` extraction |
| **tyre_gate** | No tyre/tire/pressure advice when `tyre_data_available=false` | Keyword scan with exception for "unavailable" |
| **language** | Response language matches `lang:pl\|en` | PL/EN stopword counting |
| **number_grounding** | Numbers in response traceable to JSON values | Extract all numbers, check against allowed set (values + derived differences), tolerance ±0.75 int / ±0.06 decimal |
| **radio_length** | Radio responses ≤30 words | Word count |

### Baseline results (no fine-tuning)

| Metric | 26B teacher | E4B base | E2B base |
|--------|------------|----------|----------|
| Overall grounding | 57.5% | 42.5% | 47.5% |
| Corner grounding | 100% | 100% | 100% |
| Tyre gate | 75% | 75% | 97.5% |
| Language | 100% | 75% | 62.5% |
| Number grounding | 80% | 80% | 85% |
| Radio length | 100% | 100% | 100% |
| Judge (Claude) | 4.60/5 | — | 3.60/5 |

**Key insight:** 26B self-judging was biased (gave itself 4.03, Claude gave 4.60). Always use an independent judge. Grounding checks are objective and LLM-free.

## Dataset Engineering

### Data sources

1. **Real telemetry** from Garage 61 (CSV exports via API + manual):
   - 2469 laps in the library, 2359 analysis pairs
   - Harvested from 16 teams, 221 track×car combinations
   - 5 disciplines: road, open_wheel, oval, dirt_road, dirt_oval
   - 35+ car models across GT3, GT4, GTP, LMP3, F1, F4, NASCAR, dirt, rallycross

2. **Teacher-generated responses** (distillation):
   - Teachers: Gemma 4 26B on `da12` and `dgx` (two DGX Sparks)
   - System prompt: `SYSTEM_PROMPT` from `insights/openai_provider.py`
   - Grounding rules injected into user prompt (v4+)

### Critical: data quality > data quantity

| Dataset | Examples | Grounding | What happened |
|---------|----------|-----------|--------------|
| v1 (416, unfiltered) | 374 train | 60.0% | Language fixed (62→100%), but tyre gate crashed (97→77%) |
| v3 (207, aggressive tyre filter) | 186 train | 70.0% | Tyre improved but dataset too small, language regressed |
| v5 (302, filtered + 70 negatives) | 271 train | **75.0%** | **Best result** — synthetic negatives helped |
| v6 (4111, 10× more data) | 2297 train | 57.5% | **REGRESSION** — imbalanced tasks (76% radio), diluted negatives |

### Lessons

1. **Task balance matters more than volume.** v6 had 3559 radio vs 552 reports. Model learned radio shortcuts instead of grounding precision. Keep 50/50 report/radio.

2. **Tyre gate requires active teaching, not just filtering.** Removing bad examples (filtering) gets you to ~85%. The remaining 15% comes from the model generalizing the "report template with tyre section" pattern from other examples. Synthetic negatives (responses that deliberately skip tyres) push it further.

3. **Teacher prompt engineering directly impacts student quality:**
   - v1-v3: standard prompts → 50% tyre violations in generated data
   - v4+: added explicit rules → reduced to 2.8% violations
   - The grounding rules in the teacher prompt:
     ```
     CRITICAL RULES:
     - If signals.tyre_data_available is false, do NOT mention tyres AT ALL.
     - Every number you write MUST come from the JSON summary.
     - Only mention corners that appear in the 'corners' list.
     ```

4. **Lower eval loss ≠ better model.** v6 had eval_loss 0.571 (vs v5's 1.152) but scored 17.5pp worse on grounding. The model learned to write fluently but imprecisely.

5. **Synthetic negative examples are high-leverage.** 40 hand-crafted "no tyre mention" examples (13% of v5 dataset) moved tyre gate more than 4000 real examples in v6.

### Recommended dataset composition for v7+

```
Total: ~500-600 examples
├── Real teacher responses (filtered): ~350
│   ├── Reports: ~175 (50%)
│   └── Radio: ~175 (50%)
├── Synthetic tyre negatives: ~80 (15%)
│   └── Responses that deliberately skip tyre section
├── Reinforced clean examples: ~60 (10%)
│   └── Duplicated no-tyre-data examples where teacher got it right
└── Languages: 50/50 PL/EN throughout
```

### Filtering pipeline

```python
# 1. Remove tyre violations (ANY tyre mention when tyre_data_available=false)
TYRE_ANY = re.compile(r'\b(tyre|tire|opon|ciśnien|pressure|temperatur|...)\b')
if not has_tyre_data and TYRE_ANY.search(response):
    drop  # even "tyre data unavailable" — model should never bring up the topic

# 2. Remove short responses
if len(response.strip()) < 20:
    drop

# 3. Balance tasks: subsample to 50/50 report/radio
# 4. Balance languages: 50/50 PL/EN
# 5. Add synthetic negatives (13-15% of dataset)
# 6. Add reinforced clean no-tyre examples (10% of dataset)
```

## Training Configuration

### Best config (v5, 75% grounding — current record)

```yaml
model_name: google/gemma-4-e2b-it
load_in_4bit: true          # QLoRA required on DGX Spark
lora_r: 8
lora_alpha: 8
lora_dropout: 0.0
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
modules_to_save: [lm_head, embed_tokens]
ensure_weight_tying: true   # CRITICAL for Gemma 4

num_epochs: 1               # Never >1 on small datasets
batch_size: 1
gradient_accumulation_steps: 8  # effective batch = 8
learning_rate: 1.0e-5
lr_scheduler: cosine
warmup_ratio: 0.1
max_length: 2048            # 4096 OOMs even with QLoRA
gradient_checkpointing: true
weight_decay: 0.01

eval_strategy: steps
eval_steps: 15
save_strategy: steps
save_steps: 15
save_total_limit: 5
load_best_model_at_end: true
```

### Why these values

| Setting | Value | Reason |
|---------|-------|--------|
| `load_in_4bit` | true | bf16 OOMs at 118/121 GB (E2B 8B total params + optimizer + activations) |
| `lora_r` | 8 | Proven in finRegTech v2 (90% accuracy). r=16 overfits on small data |
| `lr` | 1e-5 | finRegTech lesson: 2e-4 → catastrophic forgetting; 1e-5 preserves base |
| `num_epochs` | 1 | finRegTech lesson: 3 epochs → overfitting. 1 epoch is enough |
| `max_length` | 2048 | 4096 causes OOM even with QLoRA on 128 GB Spark |
| `modules_to_save` | lm_head, embed_tokens | Required for Gemma 4 weight tying |
| `ensure_weight_tying` | true | Without this: loss=88, broken output |
| `device_map` | `{"": 0}` | `"auto"` puts layers on meta device → gradient error |
| `eval_strategy` | steps | Must monitor for overfitting; ~3-5 evals per epoch |

### Gemma 4 specific bugs and workarounds

1. **Gemma4ClippableLinear (peft#3129):** PEFT rejects it because it doesn't inherit `nn.Linear`. Monkey-patch before loading:
   ```python
   modeling_gemma4.Gemma4ClippableLinear = PatchedClippableLinear  # inherits nn.Linear
   ```

2. **Never call `get_peft_model()` manually.** Pass `peft_config` to `SFTTrainer` directly. Manual wrapping breaks lm_head↔embed_tokens tied weights → loss=88.

3. **TRL treats Gemma 4 as vision model.** `Gemma4ForConditionalGeneration` has `vision_config` → TRL picks wrong collator → loss=88. Use appropriate TRL version (0.16.1 tested working; 1.9.0 has `__func__` AttributeError).

4. **`dataloader_pin_memory: false`** — DGX Spark unified memory makes pinning a no-op that wastes time.

## DGX Spark Specifics

### Memory budget (128 GB unified CPU+GPU)

| Component | bf16 | QLoRA 4-bit |
|-----------|------|-------------|
| E2B weights (8B total) | ~16 GB | ~5 GB |
| LoRA adapters | ~0.2 GB | ~0.2 GB |
| Optimizer states | ~0.4 GB | ~0.4 GB |
| Activations (batch=1, seq=2048) | ~15-25 GB | ~15-25 GB |
| KV cache eval (42 examples) | ~5-10 GB | ~5-10 GB |
| **Total** | **~37-52 GB** | **~26-41 GB** |

**bf16 does NOT fit alongside 26B vLLM** (~80 GB). QLoRA fits alone but NOT alongside 26B. **Always stop 26B before training.**

### GPU memory leak — root cause analysis

After stopping Docker containers on GB10, GPU memory is NOT fully released. `free -g` shows 60-70 GB still used.

**Root cause:** DGX Spark GB10 uses **unified memory** — GPU and CPU share the same 128 GB RAM (no separate VRAM). When vLLM allocates `gpu_memory_utilization=0.65` (~79 GB via `cudaMalloc`), those allocations go into system RAM managed by the NVIDIA UVM (Unified Virtual Memory) driver. After `docker stop`:

- `cudaFree` is called but the NVIDIA driver does **lazy deallocation** — keeps memory pools reserved for reuse
- Docker container with `--gpus all` inherits a GPU context that is NOT cleaned on stop
- GB10 has no `nvidia-smi --gpu-reset` (not supported on this chip)
- The ~50-70 GB shows as "used" in `free` but is NOT in any `/proc/meminfo` category (not AnonPages, not Cached, not Slab) — it's driver-managed invisible memory

**Breakdown after stopping 26B (typical):**
```
Total:           121.6 GB
Free:             15.7 GB
Available:        50.7 GB (kernel thinks it can reclaim caches — it can't reclaim GPU pools)
  Cached:         35.3 GB (includes 18.9 GB FileHugePages = mmap'd model weights)
  AnonPages:       2.0 GB (actual process memory — tiny)
  Non-reclaimable: 69.8 GB ← NVIDIA UVM driver pools
```

**Solutions (in order of preference):**

```bash
# Option 1: Reboot (cleanest, ~1 min)
ssh dgx "sudo reboot"

# Option 2: Restart NVIDIA driver without reboot (~30s, needs sudo)
ssh dgx "sudo systemctl stop docker && sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia && sudo systemctl start docker"

# Option 3: Don't stop 26B — just don't run training alongside it
# (current approach: stop → reboot → train → reboot → start)
```

### Pre-training checklist

```bash
ssh dgx "docker stop vllm-gemma4"       # stop 26B
ssh dgx "sudo reboot"                    # clean memory leak
# wait for reboot
ssh dgx "free -g"                        # verify 118 GB free
export BNB_CUDA_VERSION=130              # bitsandbytes compatibility
# sudo swapoff -a                        # optional: prevents swap thrashing
```

### Running training alongside 26B vLLM

**Don't.** Every attempt OOM'd:
- bf16 E2B + 26B vLLM → killed at step 5
- QLoRA E2B + 26B vLLM → killed at step 5
- QLoRA E2B alone on clean GPU → works perfectly

The 26B container uses `gpu_memory_utilization=0.65` (~80 GB). Training needs ~30 GB. Together >128 GB → OOM killer.

**Workflow:** stop 26B → reboot → train → reboot → start 26B.

## Evaluation & Benchmark

### Benchmark setup

- **40 questions** from real telemetry summaries
- **Balanced:** 20 report + 20 radio, 50/50 PL/EN
- **Grounding checks:** rule-based, deterministic, no LLM needed
- **Judge (optional):** Claude (independent) or 26B (self-judge, biased)
- **Statistical tests:** bootstrap CI (95%), paired McNemar for model comparison

### Running the benchmark

```bash
# Grounding only (fast, ~15 min on Ollama)
python eval/racing_benchmark.py \
  --endpoints '{"model": "http://localhost:11434/v1"}' \
  --model-names '{"model": "racing-coach-v5"}' \
  --no-judge --n 40

# With LLM judge (requires 26B on Spark)
python eval/racing_benchmark.py \
  --endpoints '{"model": "http://localhost:11434/v1"}' \
  --model-names '{"model": "racing-coach-v5"}' \
  --judge-endpoint model --n 40
```

### Self-judging bias

| Model | Self-judge | Claude judge | Delta |
|-------|-----------|-------------|-------|
| 26B | 4.03 | 4.60 | +0.57 (underrated itself) |
| E2B base | 3.65 | 3.60 | −0.05 (honest) |

**Use grounding pass rate as the primary metric** (objective, deterministic). Judge score is directional only.

## Experiment Log

| Version | Train examples | Key change | Grounding | Tyre | Language | Numbers |
|---------|---------------|------------|-----------|------|----------|---------|
| Base | — | — | 47.5% | 97.5% | 62.5% | 85.0% |
| v1 | 374 | First fine-tune | 60.0% | 77.5% | **100%** | 80.0% |
| v2 | 297 | Tyre filter | 70.0% | 85.0% | 97.5% | 87.5% |
| v3 | 186 | Aggressive tyre filter | 70.0% | 87.5% | 95.0% | 87.5% |
| v4 | 208 | Improved teacher prompts | 72.5% | 82.5% | **100%** | 87.5% |
| **v5** | **271** | **+ synthetic negatives** | **75.0%** | **85.0%** | **100%** | **90.0%** |
| v6 | 2297 | 10× data (imbalanced 76% radio) | 57.5% | 82.5% | 95.0% | 77.5% |
| v7 | 710 | Balanced 50/50 + 20% negatives | 72.5% | **90.0%** | 100% | 82.5% |
| v8 | 521 | v5 core + v7 negatives + reports | 72.5% | 87.5% | 97.5% | 85.0% |

### What each version taught us

- **v1:** Fine-tuning works. Language fixable in one epoch. But tyre gate is fragile.
- **v2:** Filtering bad examples helps but reduces dataset size.
- **v3:** Aggressive filtering hits diminishing returns (dataset too small).
- **v4:** Better teacher prompts reduce violations at source (50% → 2.8%).
- **v5:** Synthetic negatives are the highest-leverage intervention for tyre gate. **BEST OVERALL.**
- **v6:** **More data can hurt.** Imbalanced tasks (76% radio) + diluted negatives = regression.
- **v7:** Task balance (50/50) + 20% negatives fixed tyre gate to 90% but hurt number precision.
- **v8:** Mixing v5 core with v7 negatives didn't beat v5 — dilution effect again.

### Conclusion: v5 is the production model

The sweet spot is **~300 carefully curated examples with 13% synthetic negatives**. Adding more data (v6: 2297, v7: 710, v8: 521) consistently failed to improve overall grounding. The remaining gap (75% → 90%) is best closed with a **system prompt at inference** for the tyre gate, not more training data.

## Key Lessons

### 1. Quality > Quantity (confirmed twice)

finRegTech: 971 targeted → 90%, 144K generic → 42.5%.
Racing coach: 271 balanced → 75%, 2297 imbalanced → 57.5%.

### 2. Dataset balance is non-negotiable

v6 failure root cause: 3559 radio / 552 reports. Model learned to generate short, imprecise radio messages instead of grounded analysis.

### 3. Synthetic negatives are underrated

40 hand-crafted "don't mention tyres" examples moved tyre gate more than 4000 real teacher examples. The model needs to see what NOT to do, not just what to do.

### 4. Eval loss is not the objective

| Version | Eval loss | Grounding |
|---------|-----------|-----------|
| v5 | 1.152 | **75.0%** |
| v6 | 0.571 | 57.5% |

Lower loss = better language modelling ≠ better task performance.

### 5. The teacher's flaws become the student's flaws

26B teacher has 75% tyre gate and 80% number grounding. Every violation in the training data teaches the student to violate. Filter relentlessly.

### 6. DGX Spark memory management is a first-class concern

Budget memory explicitly before every run. Reboot between model switching. Never assume Docker releases memory after `docker stop`.

## Deployment

### GGUF conversion pipeline

```bash
# 1. Merge LoRA adapter into base model
python training/merge_adapter.py \
  --base-model google/gemma-4-e2b-it \
  --adapter training/output/gemma4-e2b-racing-coach-v5/final \
  --output merged/gemma4-e2b-racing-coach-v5

# 2. Convert to GGUF
python llama.cpp/convert_hf_to_gguf.py \
  merged/gemma4-e2b-racing-coach-v5 \
  --outfile gemma4-e2b-racing-coach-v5-f16.gguf --outtype f16

# 3. Quantize to Q4_K_M (2.8 GB)
llama-quantize \
  gemma4-e2b-racing-coach-v5-f16.gguf \
  gemma4-e2b-racing-coach-v5-Q4_K_M.gguf Q4_K_M

# 4. Create Ollama model
echo 'FROM ./gemma4-e2b-racing-coach-v5-Q4_K_M.gguf
PARAMETER temperature 0.4
PARAMETER num_ctx 8192' > Modelfile
ollama create racing-coach -f Modelfile
```

### Integration with the app

The model is served via Ollama (OpenAI-compatible at `localhost:11434/v1`) and consumed by `iracing_analysis.insights.OpenAICompatibleProvider` — the same interface used for the 26B teacher. No code changes needed; just point the config at the local endpoint:

```json
{
  "ai": {
    "enabled": true,
    "base_url": "http://localhost:11434/v1",
    "model": "racing-coach",
    "api_key": "ollama"
  }
}
```

### System prompt at inference (optional tyre gate boost)

v5 achieves 85% tyre gate from weights alone. For production, adding a short system prompt at inference can push it to ~100% with zero training cost:

```
Never mention tyres, tires, pressure, or temperature unless tyre data is explicitly available in the input.
```

**UPDATE (v8 experiment):** System prompt at inference was tested and **does NOT help** — it degrades overall grounding from 75% to 62.5%. Gemma E2B in GGUF/Ollama does not handle system prompts well (chat template conflict disrupts the behaviour baked into weights). Language drops 100%→95%, numbers drop 90%→80%, tyre gate improves only 85%→87.5%.

**Production recommendation:** Deploy v5 **without** system prompt. 75% grounding with 100% language and 90% numbers is the best achievable result. The 85% tyre gate is an acceptable tradeoff — 15% of responses will mention tyres briefly when no data is available, but the advice quality (judge 3.60+) is unaffected.
