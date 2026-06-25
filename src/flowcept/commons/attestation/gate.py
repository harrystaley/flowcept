"""Source-attestation gate: consult tier at retrieval time before content is acted on.

Architectural separation (important):

- The TIER is a *fact* about the content's provenance, computed at ingest by the
  annotator and stored on the task document
  (``attestation_tier = {"value": "S"|"W"|"N", ...}``).
- The GATE POLICY -- which tiers to disallow (hard) and how much to down-weight each
  tier (soft) -- lives in Flowcept configuration, NOT on the data. The same
  annotated corpus is therefore re-gateable under different policies without
  re-annotation, which is what the evaluation sweep requires.

Two knobs, both Flowcept policy:

- ``hard_blocked_tiers`` (hard / disallow): tiers excluded from retrieval, applied as a
  DAO ``$match`` exclusion via :func:`gate_filter`. Default blocks ``{"N"}``.
- ``soft_weight_factors`` (soft / adjust-weight): per-tier multiplier applied to the
  retrieval similarity score via :func:`gate_rerank`, where the similarity lives
  (the retriever/wrapper -- MongoDB does not compute similarity, so the soft
  re-rank cannot live in the DAO ``$sort``). Default ``{"S":1.0,"W":0.5,"N":0.0}``.
"""
from typing import Any, Callable, Dict, List, Optional

from flowcept.commons.attestation.tier import Tier


_DEFAULT_SOFT_WEIGHT_FACTORS = {Tier.STRONG: 1.0, Tier.WEAK: 0.5, Tier.NONE: 0.0}
_DEFAULT_HARD_BLOCKED_TIERS = [Tier.NONE]


def gate_filter(query_filter: Optional[Dict[str, Any]],
                hard_blocked_tiers: Optional[List[str]] = None
                ) -> Dict[str, Any]:
    """Augment a DAO query filter to exclude blocked attestation tiers (HARD / disallow).

    The exclusion is applied in the retrieval query itself (a ``$match`` condition),
    so blocked-tier content is never retrieved -- not retrieved-then-filtered.

    Parameters
    ----------
    query_filter : dict or None
        The existing DAO ``$match`` filter (may be None).
    hard_blocked_tiers : list of str, optional
        Tiers to disallow. POLICY from Flowcept config, not from the data.
        Defaults to blocking T_N.

    Returns
    -------
    dict
        The filter with an ``attestation_tier.value`` ``$nin`` exclusion added.
        Items lacking a tier are treated as T_N (fail-closed) and thus excluded
        whenever T_N is blocked.
    """
    blocked = list(hard_blocked_tiers) if hard_blocked_tiers is not None else list(_DEFAULT_HARD_BLOCKED_TIERS)
    new_filter = dict(query_filter) if query_filter else {}
    # Fail-closed: documents with no tier are treated as T_N. When N is blocked,
    # exclude both explicit-N and tier-absent documents.
    if Tier.NONE in blocked:
        new_filter["$and"] = new_filter.get("$and", []) + [
            {"attestation_tier.value": {"$nin": blocked}},
            {"attestation_tier.value": {"$exists": True}},
        ]
    else:
        new_filter["attestation_tier.value"] = {"$nin": blocked}
    return new_filter


def _default_score(item: Any) -> float:
    """Read a retrieval similarity score off an item. Swappable wiring point.

    The item/score shape is retriever-specific. Default convention: ``item["score"]``.
    Pin this to the actual retriever's representation when wiring retrieval.
    """
    if isinstance(item, dict):
        return float(item.get("score", 0.0))
    return float(getattr(item, "score", 0.0))


def _default_tier(item: Any) -> str:
    """Read the attestation tier value off an item. Fail-closed to T_N if absent."""
    tier_obj = item.get("attestation_tier") if isinstance(item, dict) else getattr(item, "attestation_tier", None)
    if isinstance(tier_obj, dict):
        return tier_obj.get("value", Tier.NONE)
    return Tier.NONE


def gate_rerank(
        result: Any,
        soft_weight_factors: Optional[Dict[str, float]] = None,
        drop_zero: bool = True,
        score_getter: Callable[[Any], float] = _default_score,
        tier_getter: Callable[[Any], str] = _default_tier,
) -> Any:
    """Re-rank retrieved items by ``similarity * tier_weight`` (SOFT / adjust-weight).

    Multiplicative down-weighting: an item's retrieval score is scaled by the weight
    its tier maps to. Weights are POLICY from Flowcept config, not stored on the data;
    the item supplies only its tier, the weight is looked up here.

    Operates on the wrapper ``result`` (where similarity scores live), since the
    MongoDB pipeline does not compute similarity. Non-collection results
    (dict/string/response objects from non-retrieval tasks) are returned unchanged
    -- the gate is a no-op for anything that is not a list of retrieval items.

    Parameters
    ----------
    result : Any
        The wrapped function's return. Gated only if it is a list of items; else
        returned unchanged.
    soft_weight_factors : dict, optional
        Tier -> multiplier. POLICY from Flowcept config. Default
        ``{"S":1.0,"W":0.5,"N":0.0}``.
    drop_zero : bool
        If True, items whose adjusted score is 0 (e.g. T_N at weight 0.0) are
        removed -- giving the soft mode a hard-ish floor when a tier weight is 0.
    score_getter, tier_getter : callable
        Accessors for the retriever-specific item shape (swappable wiring points).

    Returns
    -------
    Any
        Re-ranked list (highest adjusted score first), or ``result`` unchanged if
        it is not a gateable collection.
    """
    if not isinstance(result, list) or not result:
        return result

    weights = dict(soft_weight_factors) if soft_weight_factors is not None else dict(_DEFAULT_SOFT_WEIGHT_FACTORS)

    scored = []
    for item in result:
        tier = tier_getter(item)  # tier from the DATA
        w = weights.get(tier, weights.get(Tier.NONE, 0.0))  # weight from CONFIG
        adjusted = score_getter(item) * w
        scored.append((adjusted, item))

    if drop_zero:
        scored = [(s, it) for (s, it) in scored if s > 0.0]

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [it for (_s, it) in scored]