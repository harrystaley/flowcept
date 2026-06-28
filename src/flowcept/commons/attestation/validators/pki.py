"""PKI (ed25519) attestation-validator backend.

A concrete :class:`AttestationValidator` that decides whether a cryptographic
provenance claim is both *authentic* (the signature verifies over the bound
content hash) and *trust-anchored* (the signing key is one of the configured
trust roots). Only a claim that is both reaches T_S; a claim that verifies under
a key outside the trust set is authentic-but-untrusted and is left for the tier
logic to treat as T_W.

Claim shape (produced by the eval signer / a real signing pipeline)::

    {
        "alg": "ed25519",
        "pubkey": "<hex ed25519 public key, 32 bytes>",
        "signature": "<hex ed25519 signature, 64 bytes>",
        "payload_hash": "<hex sha256 of the attested content>",
    }

Trust roots: an iterable of trusted ed25519 public keys (hex). The signing key
must appear here for the claim to be trust-anchored.
"""
from typing import Any, Iterable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


def _trust_root_pubkeys(trust_roots: Any) -> set:
    """Normalise the configured trust roots into a set of hex ed25519 pubkeys.

    Accepts an iterable of hex strings, or of dicts carrying a ``pubkey`` (hex)
    -- so a trust-root config entry like ``{"id": "eval-root-0", "pubkey": "..."}``
    works directly. Entries without a usable pubkey are ignored.
    """
    roots = set()
    if trust_roots is None:
        return roots
    if isinstance(trust_roots, (str, bytes)):
        trust_roots = [trust_roots]
    for r in trust_roots:
        if isinstance(r, dict):
            pk = r.get("pubkey")
            if pk:
                roots.add(pk.lower())
        elif isinstance(r, (str, bytes)):
            roots.add((r.decode() if isinstance(r, bytes) else r).lower())
    return roots


class PkiValidator:
    """Validate an ed25519 PKI claim: authentic signature AND trust-anchored key."""

    def validate(self, claim: Any, trust_roots: Any) -> bool:
        """Return True iff ``claim`` verifies AND its key chains to a trust root.

        Authenticity: the ed25519 signature must verify over the claim's
        ``payload_hash`` under its ``pubkey``. Trust-anchoring: that ``pubkey``
        must be one of ``trust_roots``. Both are required for True (-> T_S). A
        claim that is malformed, fails verification, or whose key is not a trust
        root returns False (the tier logic then assigns T_W/T_N as appropriate).
        """
        if not isinstance(claim, dict):
            return False
        if claim.get("alg", "ed25519") != "ed25519":
            return False
        pubkey_hex = claim.get("pubkey")
        sig_hex = claim.get("signature")
        payload_hash_hex = claim.get("payload_hash")
        if not (pubkey_hex and sig_hex and payload_hash_hex):
            return False

        # Trust-anchoring: the signing key must be a configured trust root.
        if pubkey_hex.lower() not in _trust_root_pubkeys(trust_roots):
            return False

        # Authenticity: the signature must verify over the bound payload hash.
        try:
            pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex))
            pub.verify(bytes.fromhex(sig_hex), bytes.fromhex(payload_hash_hex))
        except (InvalidSignature, ValueError):
            return False
        return True