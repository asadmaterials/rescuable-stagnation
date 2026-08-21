"""
MP LLM Proposal Module — shear-modulus rescue
=============================================
Port of llm_proposal.py to the Materials Project shear-modulus benchmark.
Structure is identical to the Borg version (context builders, de-anchored
prompt, audited API call with retries, raw-dict return with harness-side
admission). Only the DOMAIN changes.

WHAT CHANGED FROM THE BORG VERSION
  - Inputs are COMPOSITION DICTS + objective values, not feature vectors.
    (The MP harness threads compositions alongside Magpie vectors; the LLM
    only ever needed compositions anyway, so this is cleaner.)
  - Objective is shear modulus G (GPa), not Vickers hardness.
  - Framing is metallic multi-component compositions (>=3 elements), not
    as-cast HEAs.
  - Element vocabulary is the 59 in-dataset elements (decision A). The LLM
    is told the exact allowed set; anything outside it, or too far from the
    training data, is rejected harness-side by the distance-bounded
    admission gate — the same shared path every arm uses.

PRESERVED (methodological invariants):
  - A1: returns RAW composition dicts; admission is harness-side, shared.
  - A6: receives the TRUE stagnation count, iteration, and budget.
  - C3: prompt de-anchored — the JSON example uses placeholder tokens, no
        concrete numbers, so the model does not regress toward examples.
  - C4: temperature explicit; every request/response audited (hash + raw).
  - C5: whole-run exploration digest via the SHARED build_digest object,
        so the LLM and the digest arm receive identical information.
"""

import os
import re
import json
import hashlib
import numpy as np

# ── Declared experimental parameters (C4) — unchanged from Borg ───────────
LLM_MODEL       = "claude-sonnet-4-6"
LLM_TEMPERATURE = 1.0
LLM_MAX_TOKENS  = 2048
LLM_MAX_RETRIES = 3

OBJECTIVE_NAME  = "shear modulus"
OBJECTIVE_UNIT  = "GPa"


# ══════════════════════════════════════════════════════════════════════════
# context builders  (composition-based)
# ══════════════════════════════════════════════════════════════════════════

def _comp_string(comp, top=6):
    """Readable composition string from a dict, e.g. 'Fe0.25 Ni0.25 ...'."""
    parts = sorted(((e, f) for e, f in comp.items() if f > 0.01),
                   key=lambda p: -p[1])
    return " ".join(f"{e}{f:.2f}" for e, f in parts[:top])


def build_exploration_digest(obs_comps, obs_y, max_lines=8):
    """
    C5 — whole-run memory for the prompt, via the SHARED digest builder in
    the harness (build_digest_from_comps). The digest arm reads the SAME
    object's fields, so identical information is guaranteed by construction.
    """
    from mp_harness import build_digest_from_comps, render_digest_G
    digest = build_digest_from_comps(obs_comps, obs_y, max_clusters=max_lines)
    return render_digest_G(digest)


def build_recent_history(obs_comps, obs_y, n_recent=15):
    """Top-5 and recent-window detail (composition-based)."""
    lines = ["TOP 5 COMPOSITIONS SO FAR:"]
    top = np.argsort(obs_y)[::-1][:5]
    for r, i in enumerate(top, 1):
        lines.append(f"  #{r} {OBJECTIVE_UNIT}={obs_y[i]:.0f}  "
                     f"[{_comp_string(obs_comps[i])}]")
    lines.append("")
    lines.append(f"MOST RECENT {min(n_recent, len(obs_y))} QUERIES "
                 f"(most recent last):")
    for i in range(max(0, len(obs_y) - n_recent), len(obs_y)):
        lines.append(f"  {OBJECTIVE_UNIT}={obs_y[i]:>6.0f}  "
                     f"[{_comp_string(obs_comps[i])}]")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# prompt  (C3 — de-anchored)
# ══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert materials scientist specializing in the elastic and mechanical properties of metallic alloys and intermetallic compounds. You assist a Bayesian optimization loop searching for metallic compositions with maximum shear modulus.

Your role: analyze the optimization history, reason about why the search has stagnated, and propose new compositional regions that are physically meaningful for high shear modulus (stiff, strongly-bonded metallic systems).

Respond ONLY with a valid JSON object — no preamble, no markdown fences, no text outside the JSON."""


def build_prompt(recent_history, exploration_digest, available_elements,
                 best_g, stagnation_count, iteration, budget, n_request=3):
    elements_str = ", ".join(available_elements)
    return f"""The Bayesian optimization loop is stagnating and needs rescue.

RUN CONTEXT:
- Target: maximize {OBJECTIVE_NAME} ({OBJECTIVE_UNIT}) of metallic compositions (3 or more elements)
- Available elements (use ONLY these): {elements_str}
- Current best {OBJECTIVE_UNIT}: {best_g:.0f}
- Consecutive iterations without improvement: {stagnation_count}
- Iteration {iteration} of {budget} total budget

FULL-RUN EXPLORATION SUMMARY (all regions tried so far):
{exploration_digest}

{recent_history}

TASK:
Propose exactly {n_request} DISTINCT new metallic compositions that explore a
DIFFERENT compositional region than where the search is stuck. Make them
genuinely different from each other, not minor variants. Requirements:
1. Physically motivated for high shear modulus (cite the bonding/structural
   reason you expect stiffness — e.g. strong directional bonding, short
   bond lengths, high valence-electron density, refractory character)
2. 3 or more elements, all drawn ONLY from the available list above
3. Molar fractions must sum exactly to 1.0
4. Avoid regions the exploration summary shows are already exhausted

Respond with ONLY this JSON structure ({n_request} entries in
"candidates"). Use your own values everywhere a placeholder appears —
do not copy placeholder names:
{{
  "reasoning": "<2-3 sentences: why stagnation, what new region and why>",
  "strengthening_mechanism": "<the primary physical reason for high stiffness>",
  "candidates": [
    {{
      "composition": {{"<El>": <fraction>, "<El>": <fraction>, ...}},
      "rationale": "<one sentence for this composition>"
    }}
  ]
}}"""


# ══════════════════════════════════════════════════════════════════════════
# LLM call  (C4 — explicit temperature, full audit)  — unchanged from Borg
# ══════════════════════════════════════════════════════════════════════════

def call_llm(prompt, model=LLM_MODEL, temperature=LLM_TEMPERATURE,
             max_tokens=LLM_MAX_TOKENS, max_retries=LLM_MAX_RETRIES):
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=api_key)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    audit = {'prompt_hash': prompt_hash, 'model': model,
             'temperature': temperature, 'attempts': []}

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.messages.create(
                model=model, max_tokens=max_tokens, temperature=temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}])
            raw = resp.content[0].text.strip()
            audit['attempts'].append({'attempt': attempt, 'raw_response': raw})
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            return json.loads(raw), audit
        except json.JSONDecodeError as e:
            audit['attempts'].append({'attempt': attempt, 'error': f'json: {e}'})
            if attempt == max_retries:
                return None, audit
        except Exception as e:
            audit['attempts'].append({'attempt': attempt, 'error': str(e)})
            return None, audit


# ══════════════════════════════════════════════════════════════════════════
# main entry — returns RAW composition dicts (admission is harness-side)
# ══════════════════════════════════════════════════════════════════════════

def llm_propose_compositions(obs_comps, obs_y, available_elements,
                             stagnation_count, iteration, budget,
                             intervention_log, n_request=3):
    """
    Ask the LLM for rescue candidates. Returns raw composition dicts
    {element: fraction}. Only JSON parsing and element-symbol sanity here;
    simplex/distance/dedup admission is the harness's shared path.

    Signature matches the MP harness's generate_llm call: composition dicts
    in, raw composition dicts out.
    """
    best_g = float(np.max(obs_y))
    digest = build_exploration_digest(obs_comps, obs_y)
    recent = build_recent_history(obs_comps, obs_y)
    prompt = build_prompt(recent, digest, available_elements, best_g,
                          stagnation_count, iteration, budget,
                          n_request=n_request)

    parsed, audit = call_llm(prompt)

    record = {'iteration': iteration, 'stagnation_count': stagnation_count,
              'best_g': best_g, 'audit': audit, 'reasoning': None,
              'mechanism': None, 'raw_candidates': []}

    if parsed is None:
        intervention_log.append(record)
        return []

    record['reasoning'] = parsed.get('reasoning', '')
    record['mechanism'] = parsed.get('strengthening_mechanism', '')

    comps = []
    for cand in parsed.get('candidates', []):
        comp = cand.get('composition', {})
        # LLM-response sanity only: known in-vocabulary elements, positive
        # fractions. Everything else (distance gate, dedup) is harness-side.
        comp = {el: float(v) for el, v in comp.items()
                if el in available_elements and isinstance(v, (int, float))
                and v > 0}
        if len(comp) >= 3:              # metallic multi-component: >=3 elements
            comps.append(comp)
            record['raw_candidates'].append(comp)

    intervention_log.append(record)
    return comps
