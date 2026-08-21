# Rescuable Stagnation — Code and Data

Code and data accompanying:

> **Rescuable Stagnation: A Diagnostic for Benchmark Suitability in LLM-Guided
> Materials Optimization**s
> Muhammad Asad. (Independent Researcher)

This repository reproduces both benchmarks in the paper: the **Materials Project
shear-modulus benchmark** (main text) and the **high-entropy-alloy hardness
benchmark** (Electronic Supplementary Information). It contains the source code,
the processed datasets, the raw results (JSON/CSV outputs and LLM traces), and
the figure-generating scripts.

---

## What the paper is about (one paragraph)

Large language models are increasingly placed inside closed-loop materials
optimization to "rescue" a stalled Bayesian-optimization run. This work
introduces a **benchmark-qualification criterion**: a benchmark can only
meaningfully test a rescue strategy if it stalls *while headroom to the pool
ceiling remains* (the "rescuable stagnation" regime). The Materials Project
benchmark is constructed to satisfy this criterion and yields a clean aggregate
null — the LLM arm explores distinctly but shows no outcome advantage at the
tested budget. The high-entropy-alloy benchmark is a retrospective example that
*fails* the criterion (it saturates near the base-pool ceiling), and is used to
motivate why qualification matters.

---

## Repository layout

The two benchmarks are independent projects, each with its own `RUN.md`,
`requirements.txt`, and pinned environment. They are kept as two top-level
folders:

```
rescuable-stagnation/
├── README.md              ← this file (start here)
├── LICENSE                ← open-source license for the code
├── .gitignore
│
├── mp-benchmark/          ← Materials Project shear modulus  (MAIN TEXT)
│   ├── RUN.md             ← authoritative step-by-step guide
│   ├── requirements.txt
│   ├── *.py               ← flat layout; all scripts import each other
│   ├── mp_shear_metallic.csv        (dataset; see Data provenance)
│   ├── mp_features_cache.parquet    (Magpie feature cache; regenerable)
│   ├── figures/
│   └── results/mp_shear_v1/         (JSON/CSV outputs, run_detail/, llm_traces/)
│
└── hea-benchmark/         ← High-entropy-alloy hardness  (ESI)
    ├── RUN.md             ← authoritative step-by-step guide
    ├── requirements.txt
    ├── src/               ← all scripts live here; they import each other
    ├── data/processed/hea_hardness_working.csv
    └── results/four_arm_v2/         (JSON/CSV outputs, run_detail/, llm_traces/, figures/)
```

Each folder's **`RUN.md` is authoritative** for that benchmark — exact
> commands, the pre-flight step, run times, and the pre-registered
> configuration. This top-level README is an overview and a map from paper
> results to the files that produce them.

---

## Quick start

Each benchmark has its own environment (the pinned versions differ — see the two
`requirements.txt`). From inside a benchmark folder:

```bash
# 1. Python 3.12 (required — NOT 3.13/3.14; matminer/pymatgen/botorch/gpytorch
#    wheels lag newer Python, and botorch/gpytorch drift changes GP fits)
python3.12 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# 2. install pinned dependencies
pip install -r requirements.txt

# 3. set the Anthropic key (LLM arm only; all else runs without it)
export ANTHROPIC_API_KEY="sk-ant-..."   # Windows: setx ANTHROPIC_API_KEY "sk-ant-..."

# 4. follow that folder's RUN.md (each has a required pre-flight/mock step first)
```

> **API key.** The LLM arm calls the Anthropic API and reads the key from the
> environment variable `ANTHROPIC_API_KEY`; it is **never** stored in code. Every
> other arm (`none`, `random`, `mutation`, `digest`) and **every analysis and
> figure script** runs directly from the committed `results/` files, so all
> reported numbers and all figures can be verified **without an API key and
> without any paid API calls**.

Both benchmarks are pinned to **Python 3.12**. Keep `torch` / `gpytorch` /
`botorch` as a matched set (botorch 0.18.x expects gpytorch 1.15.x) — version
drift there silently changes GP fits, which drive every acquisition decision.

---

## Where each result in the paper comes from

Tables and figures are produced from the committed `results/` files by the
scripts below, so any reported value can be checked without re-running the
optimization. See each folder's `RUN.md` for the exact output filenames.

### Materials Project benchmark — main text (`mp-benchmark/`)

Run: `python mp_runner.py` → writes `results/mp_shear_v1/`.
Design: 5 arms × 5 splits × 7 seeds × 20 iterations = **175 runs**.

| Paper item | Produced by | Reads / writes |
|---|---|---|
| Main run — per-split (pool=731, ceiling, admit_thresh, dedup), arm finals, Holm tests | `python mp_runner.py` | writes `results/mp_shear_v1/` |
| Full analysis — headline finals, GP calibration (n=3500), admission-by-arm, rescuable fraction, region overlap, cost-benefit, LLM outliers, oracle ranking (Spearman 0.858) | `python mp_analysis.py` | writes `results/mp_shear_v1/analysis_report.json` |
| Per-split oracle MAE / R² (mean 11.10 GPa, R² 0.520) | `python check_mae.py` | reads feature cache; feeds `ORACLE_MAE` in `mp_analysis.py` |
| Tail check — peak reality (split2 seed3, G=172.4) + LLM-only region G vs Channel B; 100% above dataset mean | `python verify_tail.py` | reads `results/mp_shear_v1/`, feature cache |
| Rationale quality / concept density (Spearman −0.206; 4/122 improved) | `python reasoning_quality.py` | writes `reasoning_labels.json` |
| Rationale coherence (5.78/8 Sonnet; per-criterion) | `python reasoning_coherence.py results/mp_shear_v1 --runs 3` | writes `reasoning_coherence.json` |
| Between-model Haiku run (generation) + its coherence (5.59/8) | `python run_capability_arm.py --model claude-haiku-4-5-20251001 --out results/haiku_v1 --splits 5` | writes `results/haiku_v1/` |
| Cross-model comparison — descriptor overlap 0.708, coverage, novelty, Jaccard, region stability | `python compare_models.py results/mp_shear_v1 results/haiku_v1` | writes `results/haiku_v1/cross_model_comparison.json` |
| Cross-model paired difference CIs — G Δ+0.018, coherence Δ+0.187, coverage, novelty | `python compare_models_ci.py results/mp_shear_v1 results/haiku_v1` | writes `results/haiku_v1/cross_model_difference_ci.json` |
| Per-event rescue effectiveness — k=3 (21.3%, llm 2.490) and k=5 (27.0%, llm 4.072) | `python rescue_effectiveness.py results/mp_shear_v1 --figure` | reads `run_detail/` |
| All main-text figures | `python make_figures.py` | writes `figures/` |

Core modules: `mp_harness.py` (five-arm BO loop), `mp_oracle.py` (RF oracle +
Voigt–Reuss–Hill Channel B + distance-bounded admission), `mp_llm_proposal.py`
(LLM arm), `exploration_digest.py` (shared digest control).
Data: `mp_shear_metallic.csv` (1,827 compositions); `mp_features_cache.parquet`
is a regenerable Magpie cache (~7 min to rebuild if deleted).

### High-entropy-alloy benchmark — ESI (`hea-benchmark/`)

Run: set `force_mock_llm=False` in `src/four_arm_runner_v2.py`, then `python3 four_arm_runner_v2.py` → writes `results/four_arm_v2/`.
Analysis: `python3 -c "from analysis import run_full_analysis; run_full_analysis('../results/four_arm_v2')"` (see the table for the two standalone follow-up analyses).
Design: 5 arms × 3 splits × 7 seeds × 50 iterations = **105 runs**.

| ESI item | Produced by | Reads |
|---|---|---|
| S1 — dataset, oracle, design | (dataset shipped directly; build step not separately logged) | `data/processed/hea_hardness_working.csv` |
| Main run — arm finals, paired Holm tests, convergence | `four_arm_runner_v2.py` | writes `results/four_arm_v2/` (`summary.json`, `summary_table.csv`, `figures/`) |
| S2.1 — aggregate outcome, per-split direction check, rescue-frequency, GP calibration | `analysis.run_full_analysis` | `results/four_arm_v2/summary.json`, `run_detail/`, `calibration.json` |
| Table — paired comparisons on HEA (Wilcoxon, Holm) | `analysis.run_full_analysis` | `summary.json`, `run_detail/` |
| Dual-channel oracle (RF + Toda–Caraballo SSH) | `analysis.run_full_analysis` (writes `dual_channel.json`) | `dual_channel.json` |
| Extrapolation check (distance vs RF; +0.208; physics-favoured 659.5 vs 469.4) | `oracle_extrapolation_analysis.py ../results/four_arm_v2` | writes `oracle_extrapolation.json` |
| S2.3 — silent proposal loss at featurization (2976→1777; B 99%, Re 86%…) | `pickup_failure_analysis.py ../results/four_arm_v2` | writes `pickup_failure.json`; reads `llm_events.csv` |
| S2.2 — retrospective ceiling check (751.2 / 764.0 / 749.7) | `pickup_failure_analysis.py ../results/four_arm_v2` | (printed by the ceiling-check stage) |
| LLM trace / per-event escape (5 of 301) | `analysis.run_full_analysis` | `llm_events.csv`, `llm_traces/` |

Core modules: `experiment_harness_v2.py` (five-arm BO loop),
`canonical_oracle.py` (oracle: RF trained on the 166-alloy training partition;
optimization pool = held-out val + hidden = 112), `llm_proposal.py` (LLM arm),
`exploration_digest.py` (shared digest control). Pre-flight:
`preflight_api_test.py` (makes ~3 real API calls to verify the parse→admission
path before a multi-hour run).

> Some scripts in `src/` support robustness checks and ablations that are
> described but not part of the headline runs — e.g. `constrained_bo.py`
> (constrained-BO ablation), `novelty_metric.py` (exploration/novelty
> characterization).
> They are included for completeness and are not required to reproduce the
> reported results.

---

## Data provenance and licensing

- **Materials Project data** (`mp-benchmark/mp_shear_metallic.csv`): derived from
  the [Materials Project](https://materialsproject.org/) via its API, then
  filtered and Magpie-featurized as described in the paper (Methods). Observe the
  Materials Project data license and citation requirements when reusing. If the
  CSV is absent, `mp-api` re-pulls the source (requires `MP_API_KEY`).
- **High-entropy-alloy data** (`hea-benchmark/data/processed/hea_hardness_working.csv`):
  built from the multi-principal-element-alloy dataset of **Borg et al.,
  *Sci. Data* 7, 430 (2020)**. Cite that source when reusing the underlying
  measurements.
- **LLM traces** (`results/*/llm_traces/`, `llm_events.csv`): generated in this
  study; released to substantiate the rationale-coherence and per-event escape
  analyses reported in the paper.

---

## Reproducibility notes

- **Pinned environments, Python 3.12.** The two `requirements.txt` files pin the
  exact versions the pipelines were verified against (they differ slightly: the
  MP project additionally needs `matminer`, `pymatgen`, `pyarrow`, and `mp-api`).
- **Pre-registration.** Each run writes a `config.json` snapshotting every
  hyperparameter at run start; within each (split, seed) cell all arms share the
  oracle, split, and initial design, and paired tests align on (split, seed).
- **Determinism.** All arms except the LLM arm are deterministic given the
  split/seed grid. The **LLM arm depends on a hosted model** and is therefore not
  bitwise reproducible; the committed LLM traces are provided so its behavior and
  every downstream analysis can be inspected exactly as reported.
- **Verify without a re-run.** Analysis and figure scripts read from `results/`
  and do not re-run the optimization, so every reported number and figure can be
  regenerated in seconds with no API access.
- **Claim scope.** Rescue-on-benchmark, not discovery. Injected candidates are
  scored by the RF oracle and have no experimental ground truth; differences
  below one oracle MAE (78 HV for the HEA benchmark) are interpreted as
  optimization-behavior findings, not physical hardness claims.

---

## Citation

If you use this code or data, please cite the paper (and, for the underlying
datasets, the sources under *Data provenance* above):

```bibtex
@article{asad_rescuable_stagnation,
  author  = {Asad, Muhammad},
  title   = {Rescuable Stagnation: A Diagnostic for Benchmark Suitability
             in LLM-Guided Materials Optimization},
  journal = {Digital Discovery},
  year    = {2026},
  note    = {Code and data: <INSERT Zenodo DOI here>}
}
```

---

## License

Code is released under the terms in [`LICENSE`](LICENSE). Underlying datasets
remain subject to the licenses of their original sources (see *Data provenance*).
