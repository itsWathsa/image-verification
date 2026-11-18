"""Offline C2PA verification, separate from image-generation/editing detectors.

Uses the official Content Authenticity SDK. Signed statements are claims by a
signer, not independent proof that a scene or its described history is true.
"""

import importlib
import io
import json
import re
from urllib.parse import unquote


OFFLINE_SETTINGS = {
    "verify": {
        "verify_after_reading": True,
        "verify_trust": True,
        "verify_timestamp_trust": True,
        "remote_manifest_fetch": False,
        "ocsp_fetch": False,
    },
    # Context preserves Some([]); RestrictedResolver matches no host. None/null
    # would allow all. See SDK context.rs and http/restricted.rs implementation.
    "core": {"allowed_network_hosts": []},
}
LIMITATION = (
    "C2PA checks signed claims and their binding to this file; it does not prove "
    "the depicted scene is true or that every edit is recorded. Missing credentials "
    "do not imply AI generation or editing. Remote manifests, trust-list updates, "
    "and online certificate-revocation checks are not fetched."
)


def _base(status, summary, **extra):
    result = {
        "status": status,
        "summary": summary,
        "manifest_present": None,
        "signature_valid": None,
        "asset_binding_valid": None,
        "signer_trust": "unknown",
        "signer": {},
        "claims": {},
        "actions": [],
        "checks": [],
        "details": {},
        "limitations": LIMITATION,
        "network_access": "disabled",
    }
    result.update(extra)
    return result


def _text(value, limit=500):
    return str(value)[:limit] if value is not None else None


def _interpret_report(report):
    """Interpret SDK verification output; never infer validity from metadata."""
    manifests = report.get("manifests", {})
    active_id = report.get("active_manifest")
    if not active_id and not manifests:
        return _base("no_manifest", "No embedded C2PA credentials were found.", manifest_present=False)
    if not isinstance(manifests, dict) or not isinstance(manifests.get(active_id), dict):
        return _base("inconclusive", "Credentials were reported, but no usable active manifest was returned.", manifest_present=bool(manifests))

    active = manifests[active_id]
    state = str(report.get("validation_state", "")).lower()
    results = report.get("validation_results", {})
    active_results = results.get("activeManifest", {}) if isinstance(results, dict) else {}
    checks = []
    for category in ("success", "informational", "failure"):
        for item in active_results.get(category, []) if isinstance(active_results, dict) else []:
            if isinstance(item, dict):
                checks.append({"kind": category, "code": _text(item.get("code")),
                               "explanation": _text(item.get("explanation")), "url": _text(item.get("url"), 1000)})
    success = {item["code"] for item in checks if item["kind"] == "success"}
    failed = {item["code"] for item in checks if item["kind"] == "failure"}
    # The flat legacy field normally omits success details. Never turn an empty
    # list into a successful verification or ignore an explicit contrary failure.
    legacy = report.get("validation_status", [])
    legacy_unknown = False
    ingredient_issues = []
    ingredient_integrity_uncertain = False
    for item in legacy if isinstance(legacy, list) else []:
        if isinstance(item, dict) and item.get("code") and item["code"] not in success:
            code = str(item["code"])
            entry = {"kind": "reported", "code": _text(item.get("code")),
                     "explanation": _text(item.get("explanation")), "url": _text(item.get("url"), 1000)}
            scope = re.search(r"(?:^|[/=])c2pa/([^/?#]+)(?:/|$)", unquote(str(item.get("url", ""))))
            if scope is not None and scope.group(1) != str(active_id):
                # Legacy output flattens ingredient and active-manifest issues.
                # An ingredient failure cannot negate this asset's own binding.
                entry["kind"] = "ingredient_reported"
                ingredient_issues.append(entry)
                checks.append(entry)
                if code != "signingCredential.untrusted":
                    ingredient_integrity_uncertain = True
                continue
            if code == "signingCredential.untrusted" or any(
                token in code.lower() for token in ("mismatch", ".missing", ".invalid", ".expired", ".revoked", ".malformed")
            ):
                failed.add(code)
            else:
                legacy_unknown = True
            checks.append(entry)

    signature = "claimSignature.validated" in success
    binding = bool(success & {"assertion.dataHash.match", "assertion.bmffHash.match", "assertion.boxesHash.match", "assertion.collectionHash.match"})
    trusted = (state == "trusted" and "signingCredential.trusted" in success
               and "signingCredential.untrusted" not in failed)
    hard_failures = failed - {"signingCredential.untrusted"}
    if state == "invalid" or hard_failures:
        status, summary = "validation_failed", "C2PA validation reported a failure. Do not rely on these credentials as validated history."
    elif ingredient_integrity_uncertain:
        status, summary = "inconclusive", "The active asset has separate validation results, but ingredient-history issues prevent a validated-history conclusion."
    elif state in {"valid", "trusted"} and signature and binding and not legacy_unknown:
        if trusted and not ingredient_issues:
            status, summary = "validated", "Signature and file binding validated; the SDK reported a trusted signer under its configured trust policy."
        elif trusted:
            status, summary = "valid_untrusted", "The active signature, file binding, and signer trust validated, but ingredient signer trust is not established."
        else:
            status, summary = "valid_untrusted", "Signature and file binding validated, but signer trust is not established."
    else:
        status, summary = "inconclusive", "Embedded claims were found, but successful signature and file-binding validation could not both be established."

    info = active.get("signature_info", {})
    signer = {key: _text(info[key]) for key in ("issuer", "common_name", "time", "cert_serial_number", "alg") if isinstance(info, dict) and key in info}
    claims = {key: _text(active[key], 1000) for key in ("title", "format", "claim_generator", "instance_id") if key in active}
    actions = []
    assertions = active.get("assertions", [])
    for assertion in assertions if isinstance(assertions, list) else []:
        if not isinstance(assertion, dict) or not str(assertion.get("label", "")).startswith("c2pa.actions"):
            continue
        assertion_data = assertion.get("data", {})
        for action in assertion_data.get("actions", []) if isinstance(assertion_data, dict) else []:
            if isinstance(action, dict) and len(actions) < 100:
                actions.append({key: _text(action[key], 1000) for key in ("action", "when", "softwareAgent", "digitalSourceType") if key in action})

    return _base(status, summary, manifest_present=True,
                 signature_valid=False if any(code and code.startswith("claimSignature.") and "mismatch" in code for code in failed) else signature or None,
                 asset_binding_valid=False if any(code and code.startswith("assertion.") and "Hash.mismatch" in code for code in failed) else binding or None,
                 signer_trust="trusted" if trusted else "not_established",
                 signer=signer, claims=claims, actions=actions, checks=checks,
                 details={"active_manifest": active_id, "manifest_count": len(manifests),
                          "validation_state": report.get("validation_state"), "validation_results": results,
                          "validation_status": legacy, "ingredient_validation_issues": ingredient_issues,
                          "claims_are_signer_statements": True})


def check_provenance(data, image_format):
    """Validate embedded JPEG/PNG credentials without external network access.

    Returns a JSON-serializable dictionary. SDK/runtime failures remain
    inconclusive and cannot become AI-generation or edited labels.
    """
    mime = {"JPEG": "image/jpeg", "PNG": "image/png"}.get(str(image_format).upper())
    if mime is None or not data or len(data) > 20 * 1024 * 1024:
        return _base("inconclusive", "Supply a readable JPEG or PNG of at most 20 MB.")
    try:
        sdk = importlib.import_module("c2pa")
    except Exception as error:
        return _base("unavailable", "The C2PA verification runtime could not load; provenance remains inconclusive.",
                     details={"error": f"{type(error).__name__}: {error}"[:2000]})
    try:
        with sdk.Context.from_dict(OFFLINE_SETTINGS) as context:
            with sdk.Reader(mime, io.BytesIO(data), context=context) as reader:
                raw = reader.json()
                if len(raw) > 8 * 1024 * 1024:
                    return _base("inconclusive", "The C2PA report exceeds this app's processing limit.", manifest_present=True)
                report = json.loads(raw)
                if not isinstance(report, dict):
                    raise ValueError("Unexpected C2PA report structure")
                return _interpret_report(report)
    except Exception as error:
        if isinstance(error, sdk.C2paError.ManifestNotFound):
            return _base("no_manifest", "No embedded C2PA credentials were found. Remote credentials were not fetched.", manifest_present=False)
        if isinstance(error, sdk.C2paError.RemoteManifest):
            return _base("inconclusive", "This file references remote credentials. They were not fetched because network access is disabled.",
                         details={"remote_manifest_not_fetched": True})
        return _base("inconclusive", "C2PA verification could not complete. No authenticity conclusion was produced.",
                     details={"error": f"{type(error).__name__}: {error}"[:2000]})
