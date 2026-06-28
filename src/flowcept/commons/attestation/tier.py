"""Source-attestation tier computation for ingested task content.

This module computes a *source-attestation tier* for content entering the
provenance substrate. The tier is collected at ingest, carried on the task
document, and consulted at the agent-task gate before retrieved content is acted
upon.

Tier semantics (trust boundary is T_S vs. {T_W, T_N}):

- ``S`` (strong): a cryptographic provenance claim (e.g. C2PA manifest, PKI
  signature, Sigstore bundle, or a TPM 2.0 attestation / TPM-sealed signing key)
  is present AND validates against a configured trust root. Forgery-resistant: an
  attacker cannot reach this tier without a key that chains to a trusted root.
- ``W`` (weak): a provenance claim is present but is NOT trust-anchored -- either an
  unsigned-but-resolvable source handle, or a signature that does not validate
  against any configured trust root (e.g. self-signed, or a TPM quote whose EK/AIK
  chain we do not hold). Identity-asserting but forgeable.
- ``N`` (none): no provenance claim, or a claim that fails to parse/resolve.

Note on TPM 2.0: the tier is decided by the same rule as any other claim --
*does it validate against a configured trust root, and does it bind the content?*
A TPM-sealed signing key whose certificate chains to a trusted CA -> T_S. A bare
platform quote (attests the machine, not a content binding) or a TPM whose EK/AIK
chain is not in the trust set -> T_W. "TPM" alone does not imply T_S; the
validation outcome does. Validator backends are pluggable by attestation type
(``c2pa``, ``pki``, ``sigstore``, ``tpm``).

Non-cryptographic (structural) attestation uses a resolvable source ``handle``
instead of a ``claim``: the handle validates iff it is a member of the configured
manifest of genuine source handles (the structural trust root), yielding T_W
(resolution/custody binding -- it proves the content is a genuine corpus entry,
not who authored it). T_S over such content requires author-level signing (e.g.
C2PA) layered on top, at which point the ``claim`` path applies.
"""
from typing import Any, Dict, Optional, Protocol


class Tier:
    """Source-attestation tier constants."""

    STRONG = "S"
    WEAK = "W"
    NONE = "N"


class AttestationValidator(Protocol):
    """Interface for a cryptographic-attestation validator backend.

    Implementations decide whether a provenance claim chains to one of the
    supplied trust roots. Backends are selected by configuration so a deployment
    can use C2PA, PKI, Sigstore, or TPM (EK/AIK chain) validation without
    changing call sites.
    """

    def validate(self, claim: Any, trust_roots: Any) -> bool:
        """Return True iff ``claim`` validates against a root in ``trust_roots``."""
        ...


def resolve_handle(handle: Any, trust_roots: Any = None) -> bool:
    """Return True iff a non-cryptographic provenance handle resolves to a real origin.

    Structural attestation: the handle resolves iff it is a member of the
    configured manifest of genuine source handles (the structural trust root).
    When no manifest is supplied, falls back to a presence check -- the
    pre-manifest behaviour, retained for callers that pass no manifest.
    """
    if handle is None or handle == "":
        return False
    manifest = _handle_manifest(trust_roots)
    if manifest is None:
        return True
    return handle in manifest


def _handle_manifest(trust_roots: Any):
    """Extract a set of genuine source handles from ``trust_roots``, or None.

    Accepts a set/list of handle strings, or a dict carrying a ``handle_manifest``
    key. Returns None when no handle manifest is present, so ``resolve_handle``
    falls back to presence-only (the PKI path passes root-dicts, which carry no
    handle manifest, so it is unaffected).
    """
    if trust_roots is None:
        return None
    if isinstance(trust_roots, dict):
        m = trust_roots.get("handle_manifest")
        return set(m) if m is not None else None
    if isinstance(trust_roots, (set, frozenset)):
        return set(trust_roots)
    if isinstance(trust_roots, (list, tuple)):
        if trust_roots and all(isinstance(x, str) for x in trust_roots):
            return set(trust_roots)
        return None
    return None


def compute_tier(
        evidence: Optional[Dict[str, Any]],
        trust_roots: Any,
        validator: Optional[AttestationValidator] = None,
) -> Dict[str, Any]:
    """Compute the source-attestation tier for an item's provenance ``evidence``.

    Parameters
    ----------
    evidence : dict or None
        Provenance carried with the content at ingest. Recognised keys:
        ``claim`` (a cryptographic attestation, e.g. a C2PA manifest) and
        ``handle`` (a non-cryptographic but resolvable source identifier).
    trust_roots : Any
        Configured set of trusted roots the validator checks a claim against.
        For the structural (handle) path this carries the genuine-source manifest
        (a set/list of handles, or a dict with a ``handle_manifest`` key).
    validator : AttestationValidator, optional
        Backend that decides whether a cryptographic claim chains to a trust root.
        If a claim is present but no validator is configured, the claim is treated
        as present-but-unverified (T_W), never T_S.

    Returns
    -------
    dict
        Structured tier:
        ``{"value": "S"|"W"|"N", "basis": <str>, "validated_against": <root id or None>}``.
    """
    if not evidence:
        return {"value": Tier.NONE, "basis": "no_evidence", "validated_against": None}

    claim = evidence.get("claim")
    if claim is not None:
        if validator is not None and validator.validate(claim, trust_roots):
            return {
                "value": Tier.STRONG,
                "basis": evidence.get("claim_type", "cryptographic_claim"),
                "validated_against": evidence.get("trust_root_id"),
            }
        # Present but not trust-anchored (e.g. self-signed) -> weak, never strong.
        return {
            "value": Tier.WEAK,
            "basis": evidence.get("claim_type", "cryptographic_claim"),
            "validated_against": None,
            "reason": "untrusted_or_unverified_signer",
        }

    handle = evidence.get("handle")
    if handle is not None and resolve_handle(handle, trust_roots):
        return {
            "value": Tier.WEAK,
            "basis": "resolvable_handle",
            "validated_against": None,
        }

    return {"value": Tier.NONE, "basis": "unresolvable_or_absent", "validated_against": None}