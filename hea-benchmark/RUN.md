# LLM-Guided BO Stagnation Rescue — Deployment Guide

Everything needed to run the experiment is in this folder.

## 1. Setup

Place the modules in a `src/` subdirectory — the code resolves the dataset
at `../data/processed/` relative to the .py files:

```
project/
├── src/                          ← ALL .py files go here
├── data/processed/hea_hardness_working.csv
├── results/                      (created automatically)
└── requirements.txt
```

```bash
mkdir -p src data/processed
mv *.py src/
mv hea_hardness_working.csv data/processed/
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd src        # all commands below run from src/
```

Version pins matter: BoTorch/GPyTorch change GP-fitting behavior across
releases; keep torch/gpytorch/botorch as a matched set.

## 2. Pre-flight (required — do not skip)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python3 preflight_api_test.py
```

Makes ~3 real API calls and verifies the full parse → admission path.
Exit 0 = safe to launch. This is the cheap check that prevents wasting a
multi-hour run on a broken key or a drifted response format.

## 3. The main experiment

Edit the bottom of `four_arm_runner_v2.py`: set `force_mock_llm=False`.
Then:

```bash
python3 four_arm_runner_v2.py
```

- **Arms (5):** none / random / digest / mutation / llm
- **Design:** 3 splits × 7 BO seeds × 5 arms × 50 iterations (105 runs)
- **Runtime:** ~3.75 h CPU (GP fits run to convergence by design)
- **Fail-fast:** aborts within minutes if the key is set but the LLM
  returns nothing usable, so a broken key cannot waste the run
- **Outputs:** `results/four_arm_v2/` — summary.json, summary_table.csv,
  config.json (full pre-registered configuration), run_detail/,
  llm_traces/, figures/

## 4. Analysis

```bash
python3 -c "from analysis import run_full_analysis; run_full_analysis('../results/four_arm_v2')"
```

Produces: Holm-corrected paired tests with rank-biserial effect sizes and
bootstrap CIs; per-split direction-consistency; oracle-error (78 HV MAE)
contextualization; GP calibration on acquisition-selected points
(coverage + NLL); dual-channel oracle-reliability characterization;
LLM reasoning-trace analysis.

## The five arms

| Arm      | Information                          | Policy |
|----------|--------------------------------------|--------|
| none     | —                                    | no injection (baseline) |
| random   | none                                 | random valid compositions |
| digest   | identical to LLM (shared object)     | rank clusters by query count; uniformly sample the least-explored half; zero hyperparameters, no optimization |
| mutation | top-k incumbents (local)             | Gaussian perturbation σ=0.05 |
| llm      | identical to digest (shared object)  | open-ended reasoning (claude-sonnet-4-6, T=1.0, logged) |

**The identical-information guarantee is structural:** both the LLM prompt
and the digest arm consume one `exploration_digest.build_digest()` object
(same clusters, same 8-cluster truncation, counts AND best-HV). The digest
arm's rule reads only the counts — the information is matched; the use of
it is the experimental contrast.

## Reproducibility notes (for methods)

- **Pairing:** within each (split, seed) cell all arms share the oracle,
  the split, and the initial design; paired tests align explicitly on
  (split, seed) keys and error on mismatch.
- **Pre-registration:** config.json snapshots every hyperparameter at run
  start.
- **Claim scope:** rescue-on-benchmark, not discovery. Injected candidates
  are scored by the RF oracle and have no experimental ground truth; the
  dual-channel physics cross-check characterizes where the oracle is
  trustworthy. Differences below one oracle MAE (78 HV) are interpreted
  as optimization-behavior findings, not physical hardness claims.
