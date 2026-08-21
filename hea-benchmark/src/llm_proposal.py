"""
LLM Proposal Module — v2
=========================
Wave-1 rewrite of the LLM arm's proposal step. Changes from v1:

  A1  — Returns RAW composition dicts. Admission (nan/simplex/dedup/
        stability-annotation) now happens in the harness through the
        same admit_candidate() path as every other arm. The old hard
        passes_stability() gate on LLM candidates is GONE.
  A6  — Receives the TRUE consecutive-stagnation count plus current
        iteration and total budget, so the LLM reasons over accurate
        run context (previously it received a constant).
  C3  — Prompt de-anchored: the JSON format example contains NO concrete
        composition numbers (placeholder tokens only), preventing the
        LLM from regressing toward example values.
  C4  — Temperature set EXPLICITLY (declared experimental parameter).
        Every request/response is logged (prompt hash, temperature,
        raw text) for auditability.
  C5  — Prompt includes a compact whole-run exploration digest (per
        dominant-element-cluster: queries, best HV) so the LLM has
        global memory beyond the recent-15 window and can avoid
        re-proposing exhausted regions.
"""

import os
import re
import json
import hashlib
import numpy as np

from canonical_oracle import get_composition_cols, DESCRIPTOR_COLS

# ── Declared experimental parameters (C4) ─────────────────────────────────────
LLM_MODEL       = "claude-sonnet-4-6"
LLM_TEMPERATURE = 1.0     # default sampling; variance across seeds is a
                          # measured component of the LLM arm (stated in methods)
LLM_MAX_TOKENS  = 2048    # fix #3: A2-new asks for up to inject_n×oversample
                          # (=12) candidates + rationales + reasoning; 1024
                          # risked JSON truncation → parse failure → degraded
                          # LLM arm. 2048 gives headroom.
LLM_MAX_RETRIES = 3


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def _comp_string(x, feature_cols, comp_cols, top=6):
    """Readable composition string for one feature vector."""
    parts = [(c, x[feature_cols.index(c)]) for c in comp_cols
             if x[feature_cols.index(c)] > 0.01]
    parts.sort(key=lambda p: -p[1])
    return " ".join(f"{c}{v:.2f}" for c, v in parts[:top])


def build_exploration_digest(obs_X, obs_y, feature_cols, max_lines=8):
    """
    C5 — compact whole-run memory for the prompt.

    Delegates to exploration_digest.build_digest() — THE single source of
    truth — then renders it. The digest-guided control arm reads the SAME
    object's fields. This guarantees by construction that both arms receive
    identical information; they differ only in what they do with it.
    """
    from exploration_digest import build_digest, render_digest, MAX_CLUSTERS
    comp_cols = get_composition_cols(feature_cols)
    digest = build_digest(obs_X, obs_y, feature_cols, comp_cols,
                          max_clusters=max_lines or MAX_CLUSTERS)
    return render_digest(digest)


def build_recent_history(obs_X, obs_y, feature_cols, n_recent=15):
    """Recent-window detail (as v1) — kept alongside the global digest."""
    comp_cols = get_composition_cols(feature_cols)
    lines = []

    # Top 5 overall
    top = np.argsort(obs_y)[::-1][:5]
    lines.append("TOP 5 COMPOSITIONS SO FAR:")
    for r, i in enumerate(top, 1):
        lines.append(f"  #{r} HV={obs_y[i]:.0f}  "
                     f"[{_comp_string(obs_X[i], feature_cols, comp_cols)}]")

    lines.append("")
    lines.append(f"MOST RECENT {min(n_recent, len(obs_y))} QUERIES "
                 f"(most recent last):")
    for i in range(max(0, len(obs_y) - n_recent), len(obs_y)):
        lines.append(f"  HV={obs_y[i]:>6.0f}  "
                     f"[{_comp_string(obs_X[i], feature_cols, comp_cols)}]")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT (C3 — de-anchored: no concrete numbers in the format example)
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert materials scientist specializing in high-entropy alloys (HEAs) and computational alloy design. You assist a Bayesian optimization loop searching for HEA compositions with maximum Vickers hardness (HV).

Your role: analyze the optimization history, reason about why the search has stagnated, and propose new compositional regions that are physically meaningful.

Respond ONLY with a valid JSON object — no preamble, no markdown fences, no text outside the JSON."""


def build_prompt(recent_history, exploration_digest, available_elements,
                 best_hv, stagnation_count, iteration, budget, n_request=3):
    elements_str = ", ".join(available_elements)
    return f"""The Bayesian optimization loop is stagnating and needs rescue.

RUN CONTEXT:
- Target: maximize Vickers hardness (HV) of as-cast HEAs (4-6 elements)
- Available elements: {elements_str}
- Current best HV: {best_hv:.0f}
- Consecutive iterations without improvement: {stagnation_count}
- Iteration {iteration} of {budget} total budget

FULL-RUN EXPLORATION SUMMARY (all regions tried so far):
{exploration_digest}

{recent_history}

TASK:
Propose exactly {n_request} DISTINCT new HEA compositions that explore a
DIFFERENT compositional region than where the search is stuck. Make them
genuinely different from each other, not minor variants. Requirements:
1. Physically motivated (cite the strengthening mechanism you expect)
2. 4-6 elements from the available list
3. Molar fractions must sum exactly to 1.0
4. Avoid regions the exploration summary shows are already exhausted

Respond with ONLY this JSON structure ({n_request} entries in
"candidates"). Use your own values everywhere a placeholder appears —
do not copy placeholder names:
{{
  "reasoning": "<2-3 sentences: why stagnation, what new region and why>",
  "strengthening_mechanism": "<the primary hardening mechanism expected>",
  "candidates": [
    {{
      "composition": {{"<El>": <fraction>, "<El>": <fraction>, ...}},
      "rationale": "<one sentence for this composition>"
    }}
  ]
}}"""


# ═══════════════════════════════════════════════════════════════════════════════
# LLM CALL (C4 — explicit temperature, full request/response logging)
# ═══════════════════════════════════════════════════════════════════════════════

def call_llm(prompt, model=LLM_MODEL, temperature=LLM_TEMPERATURE,
             max_tokens=LLM_MAX_TOKENS, max_retries=LLM_MAX_RETRIES):
    """
    Call the Claude API. Returns (parsed_dict | None, audit_record).
    The audit record captures prompt hash, temperature, and raw response
    text for every attempt — reproducibility requirement C4.
    """
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
                model       = model,
                max_tokens  = max_tokens,
                temperature = temperature,
                system      = SYSTEM_PROMPT,
                messages    = [{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            audit['attempts'].append({'attempt': attempt,
                                      'raw_response': raw})
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            return json.loads(raw), audit

        except json.JSONDecodeError as e:
            audit['attempts'].append({'attempt': attempt,
                                      'error': f'json: {e}'})
            if attempt == max_retries:
                return None, audit
        except Exception as e:
            audit['attempts'].append({'attempt': attempt,
                                      'error': str(e)})
            return None, audit


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY — returns RAW composition dicts (A1: admission is harness-side)
# ═══════════════════════════════════════════════════════════════════════════════

def llm_propose_compositions(
    observed_X, observed_y, feature_cols, available_elements,
    stagnation_count, iteration, budget, intervention_log,
    n_request=3,
):
    """
    Ask the LLM for rescue candidates. Returns a list of raw composition
    dicts {element: fraction}. NO admission logic here (A1) — only JSON
    parsing and element-symbol sanity (LLM-response-specific concerns).

    A2-new: n_request controls how many candidates are requested so the
    LLM arm has the same admission headroom (oversampling) as the other
    arms, equalizing realized injection counts in expectation.

    Everything is logged to intervention_log: run context, prompt hash,
    reasoning, raw candidates, audit trail.
    """
    best_hv = float(np.max(observed_y))

    digest  = build_exploration_digest(observed_X, observed_y, feature_cols)
    recent  = build_recent_history(observed_X, observed_y, feature_cols)
    prompt  = build_prompt(recent, digest, available_elements,
                           best_hv, stagnation_count, iteration, budget,
                           n_request=n_request)

    parsed, audit = call_llm(prompt)

    record = {
        'iteration'        : iteration,
        'stagnation_count' : stagnation_count,
        'best_hv'          : best_hv,
        'audit'            : audit,
        'reasoning'        : None,
        'mechanism'        : None,
        'raw_candidates'   : [],
    }

    if parsed is None:
        intervention_log.append(record)
        return []

    record['reasoning'] = parsed.get('reasoning', '')
    record['mechanism'] = parsed.get('strengthening_mechanism', '')

    comps = []
    for cand in parsed.get('candidates', []):
        comp = cand.get('composition', {})
        # LLM-response-specific sanity only: known element symbols,
        # positive fractions. (Simplex, descriptors, dedup, stability
        # annotation all happen in the shared admission path.)
        comp = {el: float(v) for el, v in comp.items()
                if el in available_elements and isinstance(v, (int, float))
                and v > 0}
        if len(comp) >= 2:
            comps.append(comp)
            record['raw_candidates'].append(comp)

    intervention_log.append(record)
    return comps
