"""Retrieval task module."""

from collections import Counter
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from flowcept.commons.attestation.gate import gate_rerank, _default_score, _default_tier
from flowcept.commons.flowcept_logger import FlowceptLogger
from flowcept.configs import (
    ATTESTATION_GATE_ENABLED,
    ATTESTATION_GATE_MODE,
    ATTESTATION_SOFT_WEIGHT_FACTORS,
    INSTRUMENTATION_ENABLED,
)
from flowcept.instrumentation.task_capture import FlowceptTask


def _tier_counts(items, tier_getter):
    """Get the tier histogram of a candidate list as plain str->int."""
    counts = Counter(tier_getter(it) for it in items)
    return {str(k): int(v) for k, v in counts.items()}


def gate_retrieval(
        candidates: List[Any],
        query: Optional[Any] = None,
        soft_weight_factors: Optional[Dict[str, float]] = None,
        score_getter: Callable[[Any], float] = _default_score,
        tier_getter: Callable[[Any], str] = _default_tier,
        workflow_id: Optional[str] = None,
        campaign_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        parent_task_id: Optional[str] = None,
        activity_id: str = "gated_retrieval",
        extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """
    Gate a retrieval's candidates on source-attestation and capture it as a task.

    The candidates are the retriever's similarity-ordered items, each carrying a
    score and an attestation_tier (read via the getters). The tier is a fact on the
    item; the weights/mode are policy from Flowcept config, so the same call serves
    both the gated run and the gate-off baseline. The captured task keeps the tier
    distribution before and after the gate, so provenance records what the gate did,
    not only the result. Returns the gated list (unchanged when the gate is off).
    """
    weights = soft_weight_factors if soft_weight_factors is not None else ATTESTATION_SOFT_WEIGHT_FACTORS
    active = bool(ATTESTATION_GATE_ENABLED) and ATTESTATION_GATE_MODE == "soft"

    before = _tier_counts(candidates, tier_getter)
    if active:
        gated = gate_rerank(
            candidates,
            soft_weight_factors=weights,
            score_getter=score_getter,
            tier_getter=tier_getter,
        )
    else:
        gated = candidates
    after = _tier_counts(gated, tier_getter)

    gate_record = {
        "enabled": active,
        "mode": ATTESTATION_GATE_MODE,
        "weights": dict(weights) if weights is not None else None,
        "n_in": len(candidates),
        "n_out": len(gated),
        "tiers_in": before,
        "tiers_out": after,
    }
    if extra_metadata:
        gate_record.update(extra_metadata)

    if INSTRUMENTATION_ENABLED:
        try:
            FlowceptTask(
                activity_id=activity_id,
                subtype="retrieval_task",
                used={"query": query} if query is not None else None,
                generated={"tiers_out": after, "n_out": len(gated)},
                custom_metadata={"attestation_gate": gate_record},
                workflow_id=workflow_id,
                campaign_id=campaign_id,
                agent_id=agent_id,
                parent_task_id=parent_task_id,
            )
        except Exception as e:
            # Don't let a capture failure break retrieval.
            FlowceptLogger().exception(e)

    return gated


def flowcept_retrieval_task(func=None, **decorator_kwargs):
    """Get gated retrieval task."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if not isinstance(result, list):
                return result
            return gate_retrieval(
                result,
                score_getter=decorator_kwargs.get("score_getter", _default_score),
                tier_getter=decorator_kwargs.get("tier_getter", _default_tier),
                activity_id=decorator_kwargs.get("activity_id", "gated_retrieval"),
                extra_metadata=decorator_kwargs.get("extra_metadata"),
            )

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)