"""
Tail-Behaviour Verification
===========================
Two checks that decide whether the LLM's tail behaviour is a real,
reasoning-driven signal or an artifact. Reads existing results — no re-run.

CHECK 1 — Is the 172.4 peak real?
    Pull split2/seed3 (the LLM's best run). Trace the trajectory: did an
    INJECTED (LLM-proposed) candidate get queried and actually raise the
    incumbent to the peak? Or did the run reach it via ordinary pool points
    (in which case the peak is not attributable to the LLM)?

CHECK 2 — Are the LLM-only regions genuinely high-G?
    Section 5 listed regions (Re-W, Os-Re, ...) the LLM explored but mutation
    never did. Score representative compositions from those regions with the
    oracle and Channel B. If they rank high, the LLM moved toward genuinely
    stiff chemistry, not merely different chemistry.

Usage (project folder):
    python verify_tail.py
    python verify_tail.py results/mp_shear_v1 2 3     # dir split seed
"""

import os
import sys
import json
import glob
import warnings
import numpy as np

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════
# CHECK 1 — peak attribution
# ══════════════════════════════════════════════════════════════════════════

def check_peak(save_dir, split, seed):
    print("=" * 68)
    print(f"  CHECK 1 — is the LLM peak real?  (split{split} seed{seed})")
    print("=" * 68)
    fn = f'{save_dir}/run_detail/llm_split{split}_seed{seed}.json'
    if not os.path.exists(fn):
        print(f"  {fn} not found"); return
    d = json.load(open(fn, encoding='utf-8'))
    hist = d['best_history']
    print(f"  final best G      : {d['final_best']:.1f}")
    print(f"  incumbent path    : {hist[0]:.1f} -> {max(hist):.1f} "
          f"(improved {sum(1 for i in range(1,len(hist)) if hist[i]>hist[i-1]+1e-9)} times)")

    # candidate_tags is a list of tag dicts; find injected candidates that
    # were queried and improved the incumbent
    tags = d.get('candidate_tags', [])
    inj_queried = [t for t in tags if t.get('queried')]
    inj_improved = [t for t in tags if t.get('improved')]
    print(f"\n  injected candidates: {len(tags)}")
    print(f"    queried by the GP : {len(inj_queried)}")
    print(f"    improved incumbent: {len(inj_improved)}")

    # trajectory: mark the iteration where the peak was reached, and whether
    # that pick was injected
    traj = d.get('trajectory_log', [])
    peak = max((t['hv'] for t in traj), default=None)
    print("\n  trajectory around the peak:")
    for t in traj:
        mark = ""
        if t.get('is_injected'):
            mark += " [INJECTED]"
        if peak is not None and abs(t['hv'] - peak) < 1e-6:
            mark += " <== PEAK"
        if mark:
            print(f"    iter {t['iter']:>2}: G={t['hv']:.1f}  {t.get('formula','')}{mark}")

    # verdict
    peak_iter = max(traj, key=lambda t: t['hv']) if traj else None
    if peak_iter and peak_iter.get('is_injected'):
        print("\n  → The peak was reached by an INJECTED (LLM-proposed) candidate.")
        print("    The tail-behaviour claim is directly supported: LLM reasoning")
        print("    produced the best point in the run.")
    else:
        print("\n  → The peak was reached by a POOL point, not an injected one.")
        print("    The LLM run scored high, but the peak is NOT attributable to")
        print("    an LLM proposal — state this honestly; the tail claim weakens.")

    # tie the improving injected candidate back to its reasoning
    if inj_improved:
        ev_ids = set(t.get('event') for t in inj_improved)
        traces_fn = f'{save_dir}/llm_traces/split{split}_seed{seed}.json'
        if os.path.exists(traces_fn):
            tr = json.load(open(traces_fn, encoding='utf-8'))
            print("\n  reasoning behind improving injected candidate(s):")
            for rec in tr:
                if rec.get('reasoning'):
                    print(f"    reasoning: {rec['reasoning'][:220]}")
                    if rec.get('mechanism'):
                        print(f"    mechanism: {rec['mechanism'][:140]}")
                    cands = rec.get('raw_candidates', [])
                    if cands:
                        print(f"    proposed: {cands[:3]}")
                    break


# ══════════════════════════════════════════════════════════════════════════
# CHECK 2 — are LLM-only regions high-G?
# ══════════════════════════════════════════════════════════════════════════

def check_regions(save_dir):
    print("\n" + "=" * 68)
    print("  CHECK 2 — are the LLM-only regions genuinely high-G?")
    print("=" * 68)

    # gather every LLM-injected composition, group by dominant 2-element region
    def top2(comp):
        parts = sorted(((e, f) for e, f in comp.items() if f > 0.01),
                       key=lambda p: -p[1])
        return "-".join(sorted(e for e, _ in parts[:2]))

    llm_comps, mut_regions = {}, set()
    for fn in glob.glob(f'{save_dir}/run_detail/*.json'):
        d = json.load(open(fn, encoding='utf-8'))
        for ev in d.get('inject_events', []):
            for comp in ev.get('compositions', []):
                k = top2(comp)
                if d['arm'] == 'llm':
                    llm_comps.setdefault(k, []).append(comp)
                elif d['arm'] == 'mutation':
                    mut_regions.add(k)

    llm_only = {k: v for k, v in llm_comps.items() if k not in mut_regions}
    if not llm_only:
        print("  no LLM-only regions found"); return

    # score representative compositions with the oracle + Channel B
    try:
        import mp_oracle as MO
        from mp_oracle import MPOracle, FEATURE_COLS, composition_to_vector, channel_b_vrh
        from mp_runner import prepare_dataset, make_splits
        df, _ = prepare_dataset()
        tr_idx, pool_idx = make_splits(len(df), seed=0)
        oracle = MPOracle(df, FEATURE_COLS, tr_idx, pool_idx, seed=0)
        dataset_mean = float(df['G'].mean())
        have_oracle = True
    except Exception as e:
        print(f"  (oracle unavailable: {e}; showing Channel B only)")
        from mp_oracle import channel_b_vrh
        have_oracle = False
        dataset_mean = None

    print(f"  dataset mean G: {dataset_mean:.1f} GPa" if dataset_mean else "")
    print(f"\n  {'region':<12}{'n':>4}{'oracle G':>11}{'Channel B':>12}  example")
    scored = []
    for region, comps in sorted(llm_only.items(),
                                key=lambda kv: -len(kv[1]))[:15]:
        rep = comps[0]
        og = np.nan
        if have_oracle:
            v = composition_to_vector(rep, FEATURE_COLS)
            if v is not None:
                og = oracle.query(v)
        b = channel_b_vrh(rep)
        scored.append((region, og, b))
        ex = " ".join(f"{e}{f:.2f}" for e, f in
                      sorted(rep.items(), key=lambda p: -p[1])[:4])
        print(f"  {region:<12}{len(comps):>4}{og:>11.1f}{(b if b else 0):>12.1f}  {ex}")

    if have_oracle and dataset_mean:
        ogs = [s[1] for s in scored if not np.isnan(s[1])]
        if ogs:
            frac_above = np.mean(np.array(ogs) > dataset_mean)
            print(f"\n  LLM-only regions scoring above dataset mean "
                  f"({dataset_mean:.0f} GPa): {frac_above:.0%}")
            if frac_above > 0.6:
                print("  → The LLM moved toward genuinely HIGH-G chemistry, not just")
                print("    different chemistry. The tail-behaviour claim is supported.")
            else:
                print("  → The LLM-only regions are not systematically high-G; its")
                print("    exploration was different but not clearly better-directed.")


def main(save_dir='results/mp_shear_v1', split=2, seed=3):
    check_peak(save_dir, split, seed)
    check_regions(save_dir)
    print("\n" + "=" * 68)


if __name__ == '__main__':
    args = sys.argv[1:]
    sd = args[0] if len(args) > 0 else 'results/mp_shear_v1'
    sp = int(args[1]) if len(args) > 1 else 2
    se = int(args[2]) if len(args) > 2 else 3
    main(sd, sp, se)
