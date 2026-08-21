# MP Shear-Modulus Rescue — Run Guide

Five-arm controlled comparison of stagnation-rescue strategies for Bayesian
optimization, on a Materials Project shear-modulus benchmark that passed the
rescuable-stagnation adequacy gate.

---

## 0. What's in this package

| File | Role |
|------|------|
| `mp_oracle.py` | Dataset load, Magpie featurization, RF oracle, VRH physics channel (Channel B), distance-bounded admission |
| `mp_harness.py` | Five-arm BO loop; two-representation threading (Magpie vector + composition dict); all instrumentation |
| `mp_llm_proposal.py` | LLM arm — shear-modulus prompt, 59-element vocabulary, audited API calls |
| `mp_runner.py` | Multi-split driver; per-split oracle + admission threshold; feature caching; stats + reporting |
| `exploration_digest.py` | Shared digest builder (carried over unchanged; guarantees LLM and digest arms see identical information) |
| `mp_shear_metallic.csv` | The cleaned 1,827-composition dataset (you supply this — from Drive) |
| `requirements.txt` | Pinned dependencies |

All five `.py` files **must sit in the same directory** — they import each other.

---

## 1. Environment (Python 3.12)

In PyCharm: **Settings → Project → Python Interpreter → Add Interpreter →
Virtualenv**, base interpreter Python **3.12** (not 3.13/3.14).

Then in PyCharm's terminal:

```
pip install -r requirements.txt
```

## 2. Data & keys

- Place **`mp_shear_metallic.csv`** in the project folder (same directory as the
  `.py` files). With it present, no Materials Project API call is made.
- Set the LLM key (new terminal after `setx` so it loads):

```
setx ANTHROPIC_API_KEY "sk-ant-..."
```

- `MP_API_KEY` is only needed if the CSV is absent and data must be re-pulled.

## 3. Preflight — MOCK run first (do not skip)

Confirms the whole pipeline end-to-end and builds the feature cache
(~7 min featurization, one time) WITHOUT spending API budget:

```
python -c "from mp_runner import run_experiment; run_experiment(n_splits=1, n_bo_seeds=1, n_iterations=10, force_mock_llm=True)"
```

Success = it prints a summary and writes `results/mp_shear_v1/`. If it errors
on an import, a file is missing or misplaced — fix before proceeding. After
this, `mp_features_cache.parquet` exists and later runs skip featurization.

## 4. The real run

Edit nothing — defaults are the pre-registered configuration. From the
project directory:

```
python mp_runner.py
```

- 5 arms × 5 splits × 7 seeds × 20 iterations = **175 runs**
- The LLM arm runs first on split 0, so the **fail-fast guard** aborts within
  minutes if the key is bad or the response format drifted
- Real LLM calls: budget for ~hundreds of API requests
- Outputs land in `results/mp_shear_v1/`

Run it when you can watch it through; on a laptop this is a multi-hour job.
Consider running it in a terminal you won't close.

## 5. What gets written

```
results/mp_shear_v1/
├── config.json          pre-registration snapshot (all parameters)
├── summary.json         finals per arm; paired Holm-corrected tests;
│                        oracle_ranking (Spearman CI); admission_by_arm
│                        (per-arm reject reasons); cost_by_arm (gen time)
├── run_detail/          one JSON per (arm, split, seed): best_history,
│                        calibration_log (GP mu/sigma/y), candidate_tags,
│                        inject_events, trajectory_log, stagnation_trace
└── llm_traces/          per-run LLM reasoning + raw candidates
```

## 6. Confirm it's a REAL run

After launch, open `results/mp_shear_v1/config.json` and check the console
said `LLM arm using REAL API key`. A mock run completing successfully is the
one silent failure mode that matters — the summary will say `llm_mode: MOCK`
if the key wasn't picked up.

---

## Pre-registered configuration (for the methods section)

- Objective: maximize shear modulus G (GPa), metallic compositions ≥3 elements
- Oracle: RandomForest (300 trees), per-split, leakage-free; ranking is the
  operative quality (Spearman ≈ 0.85), value R² ≈ 0.43 stated as a limitation
- Channel B: Voigt-Reuss-Hill G from elemental moduli (independent, unfitted)
- Admission: Euclidean-to-nearest-train distance in scaled Magpie space ≤ 90th
  percentile of pool distances (recomputed per split), then near-duplicate
  dedup against observed ∪ pool ∪ **train** (train included to block
  memorization)
- Budget: 8 initial + 20 BO iterations; inject 3/event, 4× oversample,
  cooldown 3; stagnation window 5, threshold 0.02
- LLM: claude-sonnet-4-6, temperature 1.0, max_tokens 2048
- Replication: 5 split seeds × 7 BO seeds
- Claim scope: rescue-on-benchmark; NOT a discovery claim

## After the run — analysis layer (not yet built)

The run records everything needed; the analysis module (to be built) will
produce: paired stats with effect sizes, **GP calibration** (±1σ/±2σ coverage,
NLL — from `calibration_log`), dual-channel agreement, **per-arm admission
rejection rates** (already in `summary.json`), **mutation/LLM region overlap**,
**off-pool ranking validity** via Channel B, and **per-arm compute cost**. None
require re-running.

## Notes / gotchas

- First featurization takes ~7 min then caches to `mp_features_cache.parquet`.
  Delete that file if you change the dataset.
- Python 3.12 only. On 3.13/3.14 the wheels may not resolve.
- If the preflight or run hangs on the first LLM call, it's almost certainly
  the API key or a network/proxy issue — the fail-fast guard covers a bad key,
  but a stalled connection needs a client timeout (add if you hit it).
