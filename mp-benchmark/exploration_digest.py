"""
Exploration Digest — THE single source of truth
================================================
Both the LLM arm and the digest-guided control consume the exploration
digest through THIS module and nothing else.

Why this module exists (methodological, not stylistic):

    The paper's central control claim is:
        "The information available to both arms was identical.
         The difference is purely what they did with it."

    That claim must be guaranteed BY CONSTRUCTION, not by two separate
    implementations happening to agree today. Previously the cluster key,
    the aggregation, and the 8-cluster truncation were implemented three
    times (once in llm_proposal, twice in the harness). Any drift between
    them would have silently falsified the claim while every test passed.

    Now: one build_digest() call produces one digest object. The LLM arm
    renders it to text for the prompt; the digest-guided arm reads its
    fields. Same object, same truncation, same numbers, by construction.

WHAT THE DIGEST CONTAINS (identical for both arms):
    per dominant-element cluster (top-2 elements by fraction):
        - n     : number of queries in that cluster
        - best  : best HV observed in that cluster
    truncated to the MAX_CLUSTERS most-queried clusters.

HOW EACH ARM USES IT (this is where they differ, by design):
    LLM arm    : receives the rendered text (counts AND best-HV) and
                 reasons over it however it chooses.
    digest arm : reads ONLY the query counts and ignores the best-HV
                 values, applying a fixed rule (see harness). The values
                 are present and available to it; the rule does not use
                 them. That asymmetry of USE — not of ACCESS — is the
                 experimental contrast.
"""

import numpy as np

MAX_CLUSTERS = 8          # the truncation both arms see


def dominant_cluster(x, feature_cols, comp_cols, thresh=0.01):
    """Cluster key = the 2 most abundant elements, sorted, e.g. 'Al-Mo'."""
    parts = [(c, x[feature_cols.index(c)]) for c in comp_cols
             if x[feature_cols.index(c)] > thresh]
    parts.sort(key=lambda p: -p[1])
    return "-".join(sorted(p[0] for p in parts[:2]))


def build_digest(obs_X, obs_y, feature_cols, comp_cols,
                 max_clusters: int = MAX_CLUSTERS) -> dict:
    """
    Build THE exploration digest. Called once per rescue event; the same
    returned object is what both the LLM arm and the digest arm consume.

    Returns
    -------
    dict with:
      'visible'   : {cluster_key: {'n': int, 'best': float}} — truncated to
                    the max_clusters most-queried clusters (what both arms see)
      'n_hidden'  : how many further clusters exist beyond the truncation
      'n_total'   : total clusters observed
      'seen_elements' : set of elements appearing in any visible cluster key
    """
    clusters = {}
    for i in range(len(obs_X)):
        key = dominant_cluster(obs_X[i], feature_cols, comp_cols)
        if key not in clusters:
            clusters[key] = {'n': 0, 'best': -np.inf}
        clusters[key]['n']   += 1
        clusters[key]['best'] = max(clusters[key]['best'], float(obs_y[i]))

    ordered = sorted(clusters.items(), key=lambda kv: -kv[1]['n'])
    visible = dict(ordered[:max_clusters])

    seen = set()
    for key in visible:
        seen.update(key.split('-'))

    return {
        'visible'       : visible,
        'n_hidden'      : max(0, len(clusters) - len(visible)),
        'n_total'       : len(clusters),
        'seen_elements' : seen,
    }


def render_digest(digest: dict) -> str:
    """
    Render the digest to the text the LLM sees in its prompt.

    This is a pure formatting of the SAME object the digest arm reads —
    the LLM gets no field the control lacks.
    """
    lines = [f"  {key:<10} : {v['n']:>3} queries, best HV {v['best']:.0f}"
             for key, v in digest['visible'].items()]
    if digest['n_hidden']:
        lines.append(f"  (+{digest['n_hidden']} smaller clusters)")
    return "\n".join(lines)
