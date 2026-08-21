"""
Capability-Comparison Runner  —  LLM arm only, configurable model
=================================================================
Runs ONLY the llm arm, across all 5 splits, using the SAME oracle, admission
rule, pool, seeds, and budget as the main run — the only thing that changes is
the model. Writes the same run_detail/ + llm_traces/ schema as mp_runner, so
mp_analysis.py and reasoning_coherence.py run on the output UNCHANGED.

WHY THIS DESIGN (per review):
  - LLM arm only: the four heuristic arms are model-independent and already
    done. Re-running them would waste budget and change nothing.
  - ALL 5 splits, not a hand-picked one: selecting the split where the frontier
    model looked best would be cherry-picking. All splits removes that.
  - Same protocol: identical oracle/admission/seeds/budget so the model is the
    only variable.
  - This is a ROBUSTNESS comparison across two capability levels, NOT a
    capability-scaling law (two points is not a gradient). Interpret
    agnostically: robust / collapses-to-heuristic / different-but-distinct are
    all publishable.

USAGE (needs ANTHROPIC_API_KEY; get the current Haiku id from the Anthropic
console — do not trust a hard-coded guess):

    python run_capability_arm.py --model claude-haiku-XXXX --out results/haiku_v1
    python run_capability_arm.py --model claude-haiku-XXXX --out results/haiku_v1 --splits 5

The Sonnet baseline already lives in results/mp_shear_v1; this writes a
separate folder so nothing is overwritten. Compare with compare_models.py.
"""

import os
import sys
import json
import glob
import datetime
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import mp_oracle as MO
import mp_harness as H
import mp_llm_proposal as LP
from mp_oracle import MPOracle, FEATURE_COLS
from mp_runner import prepare_dataset, make_splits, dedup_tolerance, _clean


def run_capability_arm(model, out_dir, n_splits=5, n_bo_seeds=7,
                       n_initial=8, n_iterations=20, inject_n=3):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f'{out_dir}/run_detail', exist_ok=True)
    os.makedirs(f'{out_dir}/llm_traces', exist_ok=True)

    # set the model for this run, and FREEZE all sampling params explicitly
    # (reviewer concern 8): the comparison is only "model-only" if temperature,
    # max_tokens, etc. are identical to the baseline. Pin them here rather than
    # inherit possibly-drifting module defaults, and record them in config.
    LP.LLM_MODEL       = model
    LP.LLM_TEMPERATURE = 1.0      # frozen — identical to the Sonnet baseline
    LP.LLM_MAX_TOKENS  = 2048     # frozen
    FROZEN_SAMPLING = {'temperature': LP.LLM_TEMPERATURE,
                       'max_tokens': LP.LLM_MAX_TOKENS,
                       'top_p': 'API default (unset, identical both runs)',
                       'system_prompt_sha': __import__('hashlib').sha256(
                           LP.SYSTEM_PROMPT.encode()).hexdigest()[:16]}
    print(f"  model set to: {LP.LLM_MODEL}")
    print(f"  frozen sampling: T={LP.LLM_TEMPERATURE} "
          f"max_tokens={LP.LLM_MAX_TOKENS} "
          f"prompt_sha={FROZEN_SAMPLING['system_prompt_sha']}")

    if not os.environ.get('ANTHROPIC_API_KEY', '').startswith('sk-ant-'):
        print("  ANTHROPIC_API_KEY not set/invalid — aborting (this run needs "
              "the real API).")
        return

    df, comps = prepare_dataset()
    X_all = df[FEATURE_COLS].values
    y_all = df['G'].values
    elements = MO.dataset_elements(df)
    print(f"  dataset: {len(df)} compositions, {len(elements)} elements")

    config = {
        'timestamp': datetime.datetime.now().isoformat(),
        'run_type': 'capability-comparison (LLM arm only)',
        'model': model, 'arms': ['llm'],
        'n_splits': n_splits, 'n_bo_seeds': n_bo_seeds,
        'n_initial': n_initial, 'n_iterations': n_iterations,
        'inject_n': inject_n, 'llm_temperature': LP.LLM_TEMPERATURE,
        'frozen_sampling': FROZEN_SAMPLING,
        'note': ('Same oracle/admission/seeds/budget as the main run; only the '
                 'model differs. All sampling params frozen identical to the '
                 'baseline. Robustness comparison across two capability '
                 'levels, NOT a scaling law.'),
    }
    json.dump(config, open(f'{out_dir}/config.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)

    results = []
    for sp in range(n_splits):
        tr_idx, pool_idx = make_splits(len(df), seed=sp)
        oracle = MPOracle(df, FEATURE_COLS, tr_idx, pool_idx, seed=sp)
        pool_X = X_all[pool_idx]
        pool_comps = [comps[i] for i in pool_idx]
        dtol = dedup_tolerance(oracle, pool_X)
        print(f"  Split {sp}: pool={len(pool_X)} "
              f"admit_thresh={oracle.admission_threshold:.2f}")

        for seed in range(n_bo_seeds):
            res = H.run_arm(
                arm='llm', oracle=oracle, pool_vectors=pool_X,
                pool_comps=pool_comps, elements=elements, dedup_tol=dtol,
                n_initial=n_initial, n_iterations=n_iterations,
                inject_n=inject_n, random_seed=seed)
            res['split_seed'] = sp
            if res['terminated_early']:
                raise RuntimeError(f"pool exhausted: split{sp} seed{seed}")
            results.append(res)

            # fail-fast on the very first run: a real model must yield candidates
            if sp == 0 and seed == 0:
                got = any(r.get('raw_candidates') for r in res['intervention_log'])
                fired = len(res['intervention_log']) > 0
                if fired and not got:
                    raise RuntimeError(
                        f"model {model} produced zero usable candidates on the "
                        f"first run — bad model id, key, or response format. "
                        f"Aborting before a wasted run.")

            # persist same schema as mp_runner
            if res['intervention_log']:
                json.dump(_clean(res['intervention_log']),
                          open(f'{out_dir}/llm_traces/split{sp}_seed{seed}.json',
                               'w', encoding='utf-8'), indent=2, ensure_ascii=False)
            detail = {
                'arm': 'llm', 'split_seed': sp, 'bo_seed': seed,
                'final_best': res['final_best'],
                'best_history': res['best_history'],
                'calibration_log': res['calibration_log'],
                'candidate_tags': res['candidate_tags'],
                'inject_events': res['inject_events'],
                'trajectory_log': res['trajectory_log'],
                'stagnation_trace': res['stagnation_trace']}
            json.dump(_clean(detail),
                      open(f'{out_dir}/run_detail/llm_split{sp}_seed{seed}.json',
                           'w', encoding='utf-8'), ensure_ascii=False)

        finals = [r['final_best'] for r in results if r['split_seed'] == sp]
        print(f"    llm final: {np.mean(finals):.1f} ± {np.std(finals):.1f}")

    finals = np.array([r['final_best'] for r in results])
    summary = {'model': model, 'n_runs': len(finals),
               'final_mean': float(finals.mean()),
               'final_std': float(finals.std()),
               'finals': finals.tolist()}
    json.dump(summary, open(f'{out_dir}/summary.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print(f"\n  Saved → {out_dir}/   (model={model})")
    print(f"  llm final G across all splits: {finals.mean():.1f} ± {finals.std():.1f}")
    print("\n  Next: python compare_models.py results/mp_shear_v1 "
          f"{out_dir}")
    return summary


if __name__ == '__main__':
    model = None; out = 'results/capability_v1'
    ns = 5
    if '--model' in sys.argv:
        model = sys.argv[sys.argv.index('--model') + 1]
    if '--out' in sys.argv:
        out = sys.argv[sys.argv.index('--out') + 1]
    if '--splits' in sys.argv:
        ns = int(sys.argv[sys.argv.index('--splits') + 1])
    if not model:
        print("ERROR: pass --model <id> (get the current Haiku id from the "
              "Anthropic console; do not guess).")
        sys.exit(1)
    run_capability_arm(model, out, n_splits=ns)
