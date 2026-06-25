"""Tier annotation hook for the document inserter.

``annotate_tier`` is called as a sibling of ``TaskObject.enrich_task_dict`` in the
document inserter -- inside ``if ENRICH_MESSAGES:`` but OUTSIDE the telemetry
branch -- so that EVERY enriched message carrying provenance evidence is tiered,
not only messages that happen to carry telemetry. (Criticality tagging is nested
in the telemetry branch; tier annotation must not be, since retrieved RAG content
has provenance but typically no telemetry.)
"""
from typing import Any, Dict, Optional

from flowcept.commons.attestation.tier import (
    AttestationValidator,
    compute_tier,
)


def _extract_evidence(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pull provenance evidence out of an ingested message.

    Evidence rides with the content at ingest. By convention it is placed under
    ``message["attestation_evidence"]``; adapters that carry provenance inside the
    ``used`` payload may instead surface it there. Returns None when no provenance
    is present (the content is then tiered T_N).
    """
    if "attestation_evidence" in message and message["attestation_evidence"]:
        return message["attestation_evidence"]
    used = message.get("used")
    if isinstance(used, dict) and used.get("_attestation_evidence"):
        return used["_attestation_evidence"]
    return None


def annotate_tier(
        message: Dict[str, Any],
        trust_roots: Any,
        validator: Optional[AttestationValidator] = None,
) -> None:
    """Compute and attach the source-attestation tier to ``message`` in place.

    Writes ``message["attestation_tier"]`` as a structured dict (see
    ``compute_tier``). Mirrors how criticality writes ``message["tags"]``: a field
    set on the message dict, persisted by the existing inserter buffer.
    """
    evidence = _extract_evidence(message)
    message["attestation_tier"] = compute_tier(evidence, trust_roots, validator)