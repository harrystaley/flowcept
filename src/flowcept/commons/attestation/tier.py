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


def resolve_handle(handle: Any) -> bool:
    """Return True iff a non-cryptographic provenance handle resolves to a real origin.

    For the static evaluation this is an offline check (e.g. the handle is present
    in a bundled manifest of known source identifiers). Online resolution is the
    re-validation hook's responsibility and is out of scope here.
    """
    return handle is not None and handle != ""


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
    if handle is not None and resolve_handle(handle):
        return {
            "value": Tier.WEAK,
            "basis": "resolvable_handle",
            "validated_against": None,
        }

    return {"value": Tier.NONE, "basis": "unresolvable_or_absent", "validated_against": None}