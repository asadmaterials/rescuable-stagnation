"""
Pickup-Failure Analysis (post-run, no rerun needed)
===================================================
The oracle-extrapolation hypothesis FAILED its test: the RF does not
penalise distant candidates (Spearman(distance, RF) = +0.21, positive),
and the RF's own top-scoring LLM proposals reach ~828 HV — well above the
~755 HV the search actually achieved.

That reframes the question. The oracle liked many LLM candidates. So:

    Why did candidates the RF scores at 780-828 HV never improve the
    incumbent?

There are only three possible failure points, and the run logged all of
them. This script attributes the loss to each in turn:

    STAGE 1  PROPOSED  -> did it survive descriptor computation?
                          (Re-containing alloys produce NaN delta_H/omega
                           and are silently dropped BEFORE admission)
    STAGE 2  ADMITTED  -> did it pass simplex/dedup/novelty screening?
    STAGE 3  PICKED    -> did the acquisition function ever query it?
    STAGE 4  IMPROVED  -> when queried, did it beat the incumbent?

It also answers the acquisition question directly using the recorded
inj_diag EI diagnostics: when an injected candidate was NOT picked, what
was its EI rank, and what did the GP think of it (mu, sigma)?

Usage (from src/):
    python pickup_failure_analysis.py
    python pickup_failure_analysis.py ../results/four_arm_v2
"""

import os
import sys
import json
import glob
import warnings
import numpy as np
from collections import Counter

warnings.filterwarnings('ignore')

from canonical_oracle import (
    load_working_dataset, make_splits, CanonicalOracle,
    get_feature_cols, get_composition_cols,
)
from data_pipeline import compute_descriptors


def descriptors_ok(comp):
    """True if the composition yields finite descriptors (else it is dropped)."""
    try:
        d = compute_descriptors(comp)
    except Exception:
        return False
    for v in d.values():
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return False
    return True


def main(save_dir='../results/four_arm_v2'):
    print("=" * 72)
    print("  PICKUP-FAILURE ANALYSIS")
    print("  The oracle liked many LLM candidates. Where were they lost?")
    print("=" * 72)

    df = load_working_dataset()
    fc = get_feature_cols(df)
    cc = get_composition_cols(fc)

    # ── STAGE 1: proposal -> descriptor survival (the Re problem) ─────────
    print("\n  STAGE 1 — PROPOSED -> descriptor-valid")
    print("  " + "-" * 68)
    props, elem_fail, elem_all = [], Counter(), Counter()
    for fn in sorted(glob.glob(f'{save_dir}/llm_traces/*.json')):
        try:
            events = json.load(open(fn, encoding='utf-8'))
        except Exception:
            continue
        for ev in events:
            for comp in ev.get('raw_candidates', []):
                if not isinstance(comp, dict) or not comp:
                    continue
                ok = descriptors_ok(comp)
                props.append(ok)
                for el in comp:
                    elem_all[el] += 1
                    if not ok:
                        elem_fail[el] += 1

    n_prop = len(props)
    n_ok   = sum(props)
    if n_prop:
        print(f"    proposed: {n_prop}   descriptor-valid: {n_ok} "
              f"({100*n_ok/n_prop:.0f}%)   DROPPED: {n_prop-n_ok} "
              f"({100*(n_prop-n_ok)/n_prop:.0f}%)")
        print("\n    Elements most implicated in dropped proposals:")
        print(f"      {'element':<9}{'in_failed':>11}{'in_all':>9}{'fail_rate':>11}")
        rates = [(el, elem_fail[el], elem_all[el], elem_fail[el]/elem_all[el])
                 for el in elem_all if elem_all[el] >= 5]
        for el, f_, a_, r in sorted(rates, key=lambda t: -t[3])[:8]:
            flag = "  <-- always fails" if r > 0.95 else ""
            print(f"      {el:<9}{f_:>11}{a_:>9}{r:>10.0%}{flag}")
        print("\n    NOTE: an element with ~100% fail rate has missing entries in the")
        print("    descriptor tables (e.g. H_mix pairs). Every alloy the LLM proposed")
        print("    containing it was discarded BEFORE admission — an information loss")
        print("    that penalises the LLM arm specifically, since the heuristic arms")
        print("    sample from the dataset's own element distribution.")

    # ── STAGES 2-4: admitted -> picked -> improved, per arm ───────────────
    print("\n  STAGES 2-4 — ADMITTED -> PICKED -> IMPROVED (per arm)")
    print("  " + "-" * 68)
    arms = ['random', 'digest', 'mutation', 'llm']
    print(f"    {'arm':<10}{'admitted':>10}{'picked':>8}{'pickup':>8}"
          f"{'improved':>10}{'hit_rate':>10}")
    per_arm = {}
    for arm in arms:
        adm = pick = imp = 0
        for fn in sorted(glob.glob(f'{save_dir}/run_detail/{arm}_*.json')):
            try:
                d = json.load(open(fn, encoding='utf-8'))
            except Exception:
                continue
            tags = d.get('candidate_tags', {}) or {}
            # candidate_tags may serialise as a dict {pool_idx: tag} or as a
            # list of tags depending on JSON round-tripping — handle both.
            tag_iter = tags.values() if isinstance(tags, dict) else tags
            for t in tag_iter:
                if not isinstance(t, dict):
                    continue
                adm += 1
                if t.get('queried'):
                    pick += 1
                    if t.get('improved'):
                        imp += 1
        per_arm[arm] = (adm, pick, imp)
        if adm:
            print(f"    {arm:<10}{adm:>10}{pick:>8}{pick/adm:>8.2f}"
                  f"{imp:>10}{(imp/pick if pick else 0):>10.3f}")

    # ── The acquisition question: EI rank of unpicked injected candidates ─
    print("\n  WHY WEREN'T THEY PICKED? (recorded EI diagnostics)")
    print("  " + "-" * 68)
    print(f"    {'arm':<10}{'steps':>8}{'med_EI_rank':>13}{'med_mu':>10}"
          f"{'med_sigma':>11}{'picked_inj':>12}")
    for arm in arms:
        ranks, mus, sds, picked_flags = [], [], [], []
        for fn in sorted(glob.glob(f'{save_dir}/run_detail/{arm}_*.json')):
            try:
                d = json.load(open(fn, encoding='utf-8'))
            except Exception:
                continue
            for step in d.get('trajectory_log', []):
                dg = step.get('inj_diag')
                if not dg:
                    continue
                ranks.append(dg.get('ei_rank', np.nan))
                mus.append(dg.get('mu', np.nan))
                sds.append(dg.get('sigma', np.nan))
                picked_flags.append(bool(dg.get('picked_an_injected')))
        if ranks:
            print(f"    {arm:<10}{len(ranks):>8}{np.nanmedian(ranks):>13.0f}"
                  f"{np.nanmedian(mus):>10.1f}{np.nanmedian(sds):>11.1f}"
                  f"{np.mean(picked_flags):>12.2f}")
    print("\n    med_EI_rank = rank of the BEST injected-but-unqueried candidate")
    print("    among all available candidates (1 = would be picked next).")
    print("    A high median rank means the GP consistently preferred pool")
    print("    points over injected ones; a low rank means injected candidates")
    print("    were near-misses that a slightly different acquisition would take.")

    # ── Ceiling check: what was actually reachable? ───────────────────────
    print("\n  CEILING CHECK — was there headroom to find?")
    print("  " + "-" * 68)
    for sp_seed in range(3):
        splits = make_splits(df, random_seed=sp_seed)
        oracle = CanonicalOracle(splits['train'], fc)
        pool = np.vstack([splits['val'][fc].values, splits['hidden'][fc].values])
        preds = oracle.model.predict(oracle.scaler.transform(pool))
        print(f"    split {sp_seed}: pool best (RF) = {preds.max():.1f} HV   "
              f"pool mean = {preds.mean():.1f} HV")
    print("\n    Compare with the ~755 HV all arms reached. If pool-best is close")
    print("    to 755, the search essentially SOLVED the pool and no rescue")
    print("    method could have differentiated — the benchmark lacks headroom.")

    out = {
        'proposed': n_prop, 'descriptor_valid': n_ok,
        'dropped_fraction': (n_prop - n_ok) / n_prop if n_prop else None,
        'element_fail_rates': {el: elem_fail[el]/elem_all[el]
                               for el in elem_all if elem_all[el] >= 5},
        'per_arm_admitted_picked_improved': per_arm,
    }
    with open(f'{save_dir}/pickup_failure.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved → {save_dir}/pickup_failure.json")
    print("=" * 72)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '../results/four_arm_v2')
