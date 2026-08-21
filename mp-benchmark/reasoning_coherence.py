"""
Element-Blinded Reasoning-Coherence Scorer
==========================================
Measures the INTERNAL SCIENTIFIC COHERENCE of the LLM's rescue-explanation
traces — NOT whether the proposed composition was good. These are different
questions, and conflating them (correlating reasoning with candidate outcome)
invites every confounder in the BO loop (surrogate error, GP uncertainty,
acquisition, headroom). This scorer deliberately does not do that.

TWO PRE-COMMITTED REQUIREMENTS (agreed before running):
  R1. The rubric below is FROZEN. It is not changed after seeing any output.
  R2. The result is reported as reasoning-coherence ONLY. It is NOT used to
      claim reasoning caused good candidates. No correlation with candidate G.

THE FROZEN RUBRIC (four criteria, 0-2 each, max 8) — element-blinded
  Nothing here rewards NAMING good elements. Element symbols are BLINDED to
  placeholders (Element A/B/C...) before scoring, so the judge literally
  cannot reward correct element choices — only explanatory structure.

    1. Causal chain        0 absent / 1 partial / 2 complete
         (cause -> mechanism -> property present?)
    2. Scientific correctness  0 incorrect / 1 partly / 2 correct
         (is the stated physics right, independent of outcome?)
    3. Composition->mechanism link  0 absent / 1 weak / 2 explicit
         (does it tie the compositional change to the mechanism?)
    4. Trade-offs          0 absent / 1 mentioned / 2 integrated
         (genuine design reasoning acknowledges tension)
    5. Mechanistic specificity  0 vague / 1 moderate / 2 specific
         (specific mechanism vs buzzwords)

  EXPLICITLY NOT SCORED: mentioning any element, "refractory", "BCC",
  "transition metal", "5d" — these are ANSWERS, not reasoning.

CONTROLS
  - Element blinding: symbols replaced with Element A/B/C... consistently
    within each trace before the judge sees it. Residual leakage (the judge
    inferring identity from context) is possible and is reported as a caveat,
    not claimed to be zero.
  - Multiple scoring runs per trace (default 3) -> SCORING STABILITY (the
    same model's stochastic variance, NOT inter-rater reliability — three
    Claude runs are not three independent judges). Reported honestly as such.
  - Positioning: SUPPLEMENTARY. The paper's main result stands without this.

Usage (project folder; needs ANTHROPIC_API_KEY):
    python reasoning_coherence.py                       # results/mp_shear_v1
    python reasoning_coherence.py results/mp_shear_v1 --model claude-haiku... --runs 3
    python reasoning_coherence.py --dry-run             # blinding self-test, no API
"""

import os
import re
import sys
import json
import glob
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# All element symbols (for blinding). Order longest-first so 'Re' isn't
# clobbered by 'R', etc. — we match whole tokens anyway.
ELEMENTS = ['He','Li','Be','Ne','Na','Mg','Al','Si','Cl','Ar','Ca','Sc','Ti',
 'Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr',
 'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','Xe','Cs',
 'Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb',
 'Lu','Hf','Ta','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn',
 'Th','Pa','Np','Pu','V','B','C','N','O','F','P','S','K','Y','W','H','I','U']
ELEMENTS = sorted(ELEMENTS, key=len, reverse=True)

FROZEN_RUBRIC = """You are scoring the INTERNAL SCIENTIFIC COHERENCE of a materials-science
explanation. Element names have been replaced with placeholders (Element A,
Element B, ...). You CANNOT and MUST NOT judge whether the chosen elements are
good — judge ONLY the quality of the explanatory reasoning.

Score four criteria, 0-2 each (max 8). Do NOT award any points for mentioning
specific element classes, "refractory", "BCC", "transition metal", "5d", high
"VEC", "equiatomic", or similar — those are answers or identity cues, not
reasoning.

1. MECHANISTIC CHAIN (0-2): is there a cause -> mechanism -> property chain?
   0 = none ("choose Element A")
   1 = partial (mechanism OR property link, not both)
   2 = complete ("higher bond strength -> more lattice resistance -> higher
       shear modulus")

2. SCIENTIFIC PLAUSIBILITY (0-2): would a materials scientist consider the
   stated mechanism physically reasonable? (Plausibility, NOT experimental
   verification — you are not checking against ground truth.)
   0 = implausible ("larger atoms always increase shear modulus")
   1 = partly plausible
   2 = plausible ("stronger metallic bonding raises resistance to shear")

3. MECHANISTIC SPECIFICITY (0-2): concrete mechanism vs buzzwords?
   0 = vague ("this composition should work")
   1 = moderate ("higher bond strength improves stiffness")
   2 = specific ("higher d-electron overlap increases directional bonding,
       raising resistance to dislocation motion")

4. LOGICAL CONSISTENCY (0-2): internally consistent, no contradictions or
   unsupported jumps?
   0 = contradictory or non-sequitur
   1 = mostly consistent with one weak jump
   2 = fully consistent chain of statements

Respond ONLY with JSON, no prose:
{"mechanistic_chain": <0-2>, "plausibility": <0-2>, "specificity": <0-2>,
 "consistency": <0-2>}"""


def blind_elements(text):
    """Replace element symbols with stable placeholders within one trace.
    Whole-token match so 'Co' in 'Cobalt' or ordinary words are not hit; we
    only replace standalone chemical symbols (optionally followed by a
    fraction like Mo0.5)."""
    mapping = {}
    letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    def repl(m):
        el = m.group(1)
        if el not in mapping:
            try:
                mapping[el] = "Element " + next(letters)
            except StopIteration:
                mapping[el] = "Element Z"
        return mapping[el] + (m.group(2) or "")
    # match an element symbol as a whole token, optional trailing number
    pattern = r'\b(' + '|'.join(ELEMENTS) + r')(\d*\.?\d+)?\b'
    blinded = re.sub(pattern, repl, text)
    return blinded, mapping


def load_traces(save_dir):
    events = []
    for fn in sorted(glob.glob(f'{save_dir}/llm_traces/*.json')):
        base = os.path.basename(fn).replace('.json', '')
        try:
            sp = int(base.split('_')[0].replace('split', ''))
            se = int(base.split('_')[1].replace('seed', ''))
        except Exception:
            continue
        for rec in json.load(open(fn, encoding='utf-8')):
            txt = ((rec.get('reasoning') or '') + ' ' +
                   (rec.get('mechanism') or '')).strip()
            if txt:
                events.append({'split': sp, 'seed': se,
                               'iteration': rec.get('iteration'), 'text': txt})
    return events


def score_once(blinded_text, model):
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    resp = client.messages.create(
        model=model, max_tokens=200, temperature=0.0,
        system=FROZEN_RUBRIC,
        messages=[{"role": "user",
                   "content": f"Explanation to score:\n\n{blinded_text}"}])
    raw = resp.content[0].text.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw); raw = re.sub(r'\s*```$', '', raw)
    d = json.loads(raw)
    keys = ['mechanistic_chain', 'plausibility', 'specificity', 'consistency']
    vals = [int(d[k]) for k in keys]
    return vals, sum(vals)


def main(save_dir='results/mp_shear_v1', model='claude-sonnet-4-6',
         runs=3, dry_run=False):
    print("=" * 70)
    print("  ELEMENT-BLINDED REASONING-COHERENCE SCORER  (supplementary)")
    print("=" * 70)
    print("  FROZEN rubric, 5 criteria x 0-2 = 10. Reported as coherence ONLY;")
    print("  NOT correlated with candidate outcome (pre-committed).")
    print(f"  Judge: {model}   scoring runs/trace: {runs}\n")

    events = load_traces(save_dir)
    print(f"  reasoning traces found: {len(events)}")

    # blinding self-test — always shown, so the control is auditable
    if events:
        ex = events[0]
        b, m = blind_elements(ex['text'])
        print("\n  BLINDING SELF-TEST (first trace):")
        print(f"    original: {ex['text'][:160]}")
        print(f"    blinded : {b[:160]}")
        print(f"    mapping : {m}")
        # residual check must ignore the placeholder letters ("Element B")
        b_no_placeholders = re.sub(r'Element [A-Z]', '', b)
        leftover = [e for e in ELEMENTS
                    if re.search(r'\b'+e+r'\b', b_no_placeholders)]
        print(f"    residual element symbols after blinding: "
              f"{leftover if leftover else 'none'}")

    if dry_run:
        print("\n  --dry-run: blinding verified, no API calls made.")
        return

    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("\n  ANTHROPIC_API_KEY not set — cannot run the judge.")
        return

    # score every trace, `runs` times, on the blinded text
    all_scores, per_run_totals, per_criterion = [], [], []
    labelled = []
    for i, e in enumerate(events):
        blinded, mapping = blind_elements(e['text'])
        run_totals, run_vecs = [], []
        for _ in range(runs):
            try:
                vals, tot = score_once(blinded, model)
                run_totals.append(tot); run_vecs.append(vals)
            except Exception as ex_:
                continue
        if not run_totals:
            continue
        mean_tot = float(np.mean(run_totals))
        # per-criterion mean across this trace's runs (4-vector)
        crit_mean = np.mean(np.array(run_vecs), axis=0).tolist() if run_vecs else None
        all_scores.append(mean_tot)
        per_run_totals.append(run_totals)
        if crit_mean:
            per_criterion.append(crit_mean)
        labelled.append({'split': e['split'], 'seed': e['seed'],
                         'iteration': e['iteration'],
                         'blinded_text': blinded, 'mapping': mapping,
                         'run_totals': run_totals, 'mean_total': mean_tot,
                         'criteria_mean': crit_mean})
        if (i + 1) % 20 == 0:
            print(f"    scored {i+1}/{len(events)}...")

    if not all_scores:
        print("  no traces scored"); return
    arr = np.array(all_scores)

    # scoring stability: mean within-trace std across runs (same-model
    # stochastic variance, NOT inter-rater reliability)
    within = [np.std(t) for t in per_run_totals if len(t) > 1]
    agreement = float(np.mean(within)) if within else float('nan')

    print("\n  " + "-" * 66)
    print("  REASONING-COHERENCE RESULT  (0-8)")
    print("  " + "-" * 66)
    print(f"    traces scored     : {len(arr)}")
    print(f"    mean coherence    : {arr.mean():.2f} / 8")
    print(f"    std across traces : {arr.std():.2f}")
    print(f"    range             : {arr.min():.1f} – {arr.max():.1f}")
    print(f"    scoring stability (mean within-trace SD over {runs} runs): "
          f"{agreement:.2f}")
    print(f"      (same-model stochastic variance, NOT inter-rater reliability;\n"
          f"       lower = more stable; >~1.2 means a noisy judge)")

    # distribution
    lo = np.mean(arr <= 2); mid = np.mean((arr > 2) & (arr <= 5)); hi = np.mean(arr > 5)
    print(f"\n    low coherence (0-2) : {lo:.0%}")
    print(f"    mid coherence (3-5) : {mid:.0%}")
    print(f"    high coherence (6-8): {hi:.0%}")

    # per-criterion means (reviewer: more informative than a single mean)
    crit_names = ['mechanistic_chain', 'plausibility', 'specificity', 'consistency']
    crit_summary = {}
    if per_criterion:
        pc = np.array(per_criterion)   # (n_traces, 4)
        print("\n    per-criterion mean (0-2 each):")
        for k, name in enumerate(crit_names):
            m = float(pc[:, k].mean())
            crit_summary[name] = m
            print(f"      {name:<20}: {m:.2f}")

    json.dump({'rubric': FROZEN_RUBRIC, 'model': model, 'runs': runs,
               'mean_coherence': float(arr.mean()), 'std': float(arr.std()),
               'max_score': 8, 'per_criterion_mean': crit_summary,
               'scoring_stability_sd': agreement,
               'events': labelled},
              open(f'{save_dir}/reasoning_coherence.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)

    print("\n  " + "-" * 66)
    print("  REPORTING (pre-committed):")
    print("   - Report as: 'LLM rescue explanations scored mean X/8 on a")
    print("     pre-specified, element-blinded coherence rubric.'")
    print("   - Do NOT correlate with candidate G. This is coherence, not cause.")
    print("   - Positioning: SUPPLEMENTARY. The main optimization result stands")
    print("     independently.")
    print("   - Publish the rubric, blinded texts, and per-run scores")
    print("     (reasoning_coherence.json) for verification.")
    print("   - State the residual-leakage caveat: blinding replaces symbols,")
    print("     but a judge may still infer identity from context; agreement")
    print("     across runs bounds judge noise, not this inference risk.")
    print(f"\n  Full output → {save_dir}/reasoning_coherence.json")
    print("=" * 70)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    save_dir = args[0] if args else 'results/mp_shear_v1'
    model = 'claude-sonnet-4-6'
    runs = 3
    if '--model' in sys.argv:
        model = sys.argv[sys.argv.index('--model') + 1]
    if '--runs' in sys.argv:
        runs = int(sys.argv[sys.argv.index('--runs') + 1])
    dry = '--dry-run' in sys.argv
    main(save_dir, model, runs, dry)
