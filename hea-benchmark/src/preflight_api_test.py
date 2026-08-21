"""
Pre-flight API Test (C1-new)
============================
Run this ONCE, with a real ANTHROPIC_API_KEY set, BEFORE launching the
expensive 3×7 four-arm experiment. It exercises the real LLM code path
that the mock never touches:

    build_prompt → call_llm (real API, retries, JSON-fence stripping,
    audit logging) → element-sanity parse → shared admit_candidate()

and confirms that at least some real LLM proposals survive admission and
become usable feature vectors. If this fails, the full run would waste
time and money hitting the same failure repeatedly.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 preflight_api_test.py

Exit code 0 = ready to run; nonzero = fix before running.
"""

import os
import sys
import warnings
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from canonical_oracle import (
    load_working_dataset, make_splits, get_feature_cols,
    get_composition_cols,
)
from novelty_metric import descriptor_covariance, default_min_novelty
from admission      import admit_candidate
from llm_proposal   import llm_propose_compositions, LLM_TEMPERATURE, LLM_MODEL


def main():
    print("=" * 64)
    print("  PRE-FLIGHT API TEST (C1-new)")
    print("=" * 64)

    # ── Key present? ──────────────────────────────────────────────────────
    key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not key.startswith('sk-ant-'):
        print("  ✗ ANTHROPIC_API_KEY not set or malformed.")
        print("    export ANTHROPIC_API_KEY=sk-ant-... then rerun.")
        return 1
    print(f"  ✓ API key present")
    print(f"  Model: {LLM_MODEL}   Temperature: {LLM_TEMPERATURE}\n")

    # ── Build a realistic stagnation context from the working data ────────
    df   = load_working_dataset()
    fc   = get_feature_cols(df)
    ae   = get_composition_cols(fc)
    X    = df[fc].values

    cov_inv = descriptor_covariance(X, fc)
    min_nov = default_min_novelty(X, fc, cov_inv)

    splits = make_splits(df, random_seed=0)
    # Simulate a loop that has queried ~15 Cantor-ish points and stalled
    rng    = np.random.default_rng(0)
    idx    = rng.choice(len(splits['train']), size=15, replace=False)
    obs_X  = splits['train'][fc].values[idx]
    obs_y  = splits['train']['HV'].values[idx]

    print(f"  Simulated context: {len(obs_X)} observed points, "
          f"best HV so far = {obs_y.max():.0f}\n")

    # ── Make N real calls, verify parse → admission ───────────────────────
    n_calls        = 3
    n_request      = 12          # matches LLM_OVERSAMPLE × inject_n
    total_proposed = 0
    total_admitted = 0
    log            = []

    for c in range(1, n_calls + 1):
        print(f"  Call {c}/{n_calls} ...", end=" ", flush=True)
        try:
            raw = llm_propose_compositions(
                observed_X=obs_X, observed_y=obs_y, feature_cols=fc,
                available_elements=ae, stagnation_count=8,
                iteration=20, budget=50, intervention_log=log,
                n_request=n_request,
            )
        except Exception as e:
            print(f"✗ EXCEPTION: {e}")
            return 2

        if not raw:
            print("✗ returned 0 parseable candidates")
            # inspect the audit trail
            if log and log[-1].get('audit'):
                last = log[-1]['audit']['attempts'][-1]
                print(f"    last attempt: {last}")
            continue

        # Run each through the SHARED admission path
        reference = np.vstack([obs_X, splits['val'][fc].values])
        admitted  = 0
        reasons   = {}
        for comp in raw:
            res = admit_candidate(comp, fc, reference, cov_inv, min_nov)
            if res.admitted:
                admitted += 1
                reference = np.vstack([reference, res.vec])
            else:
                reasons[res.reason] = reasons.get(res.reason, 0) + 1

        total_proposed += len(raw)
        total_admitted += admitted
        print(f"✓ {len(raw)} parsed, {admitted} admitted"
              + (f"  (rejects: {reasons})" if reasons else ""))
        # Show one reasoning snippet to confirm semantic content
        if log and log[-1].get('reasoning'):
            print(f"    reasoning: {log[-1]['reasoning'][:90]}...")

    # ── Verdict ───────────────────────────────────────────────────────────
    print()
    print("-" * 64)
    print(f"  Total: {total_proposed} parsed, {total_admitted} admitted "
          f"across {n_calls} calls")
    if total_admitted == 0:
        print("  ✗ FAIL — no real LLM proposal survived admission.")
        print("    Inspect the prompt / parsing before the full run.")
        return 3
    if total_proposed < n_calls:            # fewer than ~1 per call
        print("  ⚠ WARNING — very low parse yield; check prompt formatting.")
        return 4
    print("  ✓ READY — real LLM path parses and admits. Safe to launch")
    print("    the full 3×7 run with force_mock_llm=False.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
