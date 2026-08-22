# Prospective benchmark screening (the go/no-go gate)

This folder contains the **pre-run adequacy screening** for the Materials
Project shear-modulus benchmark — the step that decides, *before* the
optimization comparison is built and run, whether the benchmark can actually
test a rescue hypothesis. This is the prospective "rescuable-stagnation"
criterion that is the central methodological contribution of the paper.

These scripts are run **first**. Only after the benchmark passes the gate is the
full five-arm experiment (in `mp-benchmark/`) built and run. They are separate
from the post-hoc analysis in `mp_analysis.py`: that script reports rescuable
events *from the completed experiment*, whereas the scripts here estimate
rescuability *from a short probe before any rescue arm exists*.

---

## Why a pre-run gate is needed

A rescue experiment is only interpretable if the optimizer stalls **while
meaningful improvement is still available**. Two degenerate regimes silently
destroy the experiment, at opposite ends of oracle quality:

- **Saturated** — a weak oracle or small pool makes the loop stall *at* the pool
  ceiling. Rescue fires, but there is nothing left to find.
- **Never stalls** — a strong oracle or large pool makes the loop climb steadily
  and never trigger a rescue at all.

The hypothesis is testable only in the middle band. These scripts measure,
in advance, whether that band exists for this benchmark — and refuse to
proceed if it does not.

---

## The three scripts

Run them in this order. Each is self-contained, needs no rescue-experiment
results, and (except for the initial data pull) makes no API calls.

### 1. `oracle_diagnostics.py` — is the oracle a fair judge?

Answers three questions that `R^2` alone cannot:

- **Ranking**: does the oracle rank higher-`G` compositions above lower-`G`
  ones? For a benchmark *objective*, rank fidelity (Spearman), not calibrated
  value accuracy, is what matters.
- **Model choice**: does a cheap model swap (Gradient Boosting) beat Random
  Forest on the same splits? (It does not — RF is retained.)
- **Error vs. distance** (the decisive check): does oracle error grow with
  distance from the training data? If it does, exploratory arms (LLM, digest)
  would be scored more noisily than local arms (mutation) — the exact confound
  that muddied the earlier work.

```bash
python oracle_diagnostics.py        # expects mp_shear_metallic.csv in the folder
```

Recorded output (`oracle_diagnostics_output.txt`): 1,827 compositions;
RandomForest Spearman **0.849** (≥ 0.7, a discriminating landscape); error–
distance coupling **+0.259**, with quartile errors 8.9 / 10.9 / 10.7 / 17.8 GPa.
Verdict: **BUILD WITH CARE** — ranking is sound, but error rises in the far
tail, so candidates must be bounded to the reliable region (see script 2).

### 2. `admission_threshold.py` — where is the oracle reliable?

Given the distance coupling found above, this locates — empirically — the
distance bound within which the oracle's error stays flat, and expresses it as
a single enforceable quantity: Euclidean distance to the nearest training point
in standardized Magpie-feature space. It sweeps candidate thresholds and reports
mean error inside each and the fraction of the pool retained, so the cutoff is a
**pre-registerable** design decision rather than a post-hoc choice.

```bash
python admission_threshold.py       # expects mp_shear_metallic.csv in the folder
```

Recorded output (`admission_threshold_output.txt`): recommended threshold
**distance ≤ 6.42** (the 90th percentile of candidate distances), mean error
inside **11.3 GPa** vs. 9.9 GPa in the closest half, **90%** of the pool
admitted. This is the admission rule the full experiment enforces (its
per-split thresholds, 6.33–6.80, bracket this value).

### 3. `MP_benchmark_screening.py` — THE go/no-go gate

The gate itself. It builds the dataset and oracle, then runs a short
**baseline-only probe** (the `none` arm — no rescue), sweeping optimization
budgets (20 / 40 / 60 / 80 iterations, 3 seeds each). For each budget it counts
how many stagnation events occur **while the gap to the oracle-computed pool
ceiling still exceeds one oracle MAE** — i.e., stalls a rescue could actually
improve. Passing "has headroom" and "stalls sometimes" *separately* is not
enough; the gate requires them to co-occur.

Verdict logic per budget:

- `rescuable@1MAE ≥ 3` → **GOOD**
- stalls but at the ceiling → **SATURATED**
- rarely stalls → **never stalls**s

Recorded output (`mp_screening_results.json`): every budget returns
verdict **GOOD**, with rescuable-at-1-MAE counts of 8.0 / 15.3 / 18.7 / 18.7.
**Decision: GO** — the loop stalls with real headroom remaining, so the rescue
hypothesis is testable on this benchmark. Only after this GO was the full
five-arm experiment built.

---

## Important: screening numbers are a pre-run estimate, not the final result

The screening stage is a fast, coarse probe (a single split, three seeds, no
per-split oracles), so its numbers differ from — and should never be confused
with — the final experiment:

| Quantity | Screening probe (this folder) | Final experiment (`mp-benchmark/`) |
|---|---|---|
| Oracle R² | 0.43 | 0.52 |
| Oracle MAE | 13.1 GPa | 11.1 GPa |
| Rescuable events | probe-budget counts (8–19) | ~3 per run, 86–88% of stalls |

The screening values establish only that the benchmark **passed the gate**
(GO). All oracle-quality and rescuable-fraction numbers reported as results in
the paper come from the final experiment, not from this probe.

---

## Files in this folder

| File | Role |
|---|---|
| `MP_benchmark_screening.py` | The prospective go/no-go gate (rescuable-stagnation probe) |
| `oracle_diagnostics.py` | Oracle adequacy: ranking, model choice, error-vs-distance |
| `admission_threshold.py` | Empirical, pre-registerable distance-admission cutoff |
| `mp_screening_results.json` | Recorded gate output (verdicts + rescuable counts) |
| `oracle_diagnostics_output.txt` | Recorded oracle-diagnostics console output |
| `admission_threshold_output.txt` | Recorded admission-threshold console output |

## Reproducing

`oracle_diagnostics.py` and `admission_threshold.py` run directly from the
committed `mp_shear_metallic.csv` with no API key. `MP_benchmark_screening.py`
needs an `MP_API_KEY` only on its first run to pull Materials Project elasticity
data; once `mp_shear_metallic.csv` is cached it runs key-free. Version pins are
in the benchmark's `requirements.txt` (Python 3.12).
