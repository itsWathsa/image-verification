"""C2PA report interpretation and offline SDK integration contract tests."""

import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from provenance import OFFLINE_SETTINGS, check_provenance


class NativeError(Exception):
    pass


class ManifestNotFound(NativeError):
    pass


class RemoteManifest(NativeError):
    pass


def manifest_report(state="Valid", success=None, failure=None):
    return {
        "active_manifest": "test-active",
        "manifests": {"test-active": {
            "signature_info": {"issuer": "Example test signer", "common_name": "Test signer"},
            "title": "Example JPEG",
            "assertions": [{"label": "c2pa.actions.v2", "data": {"actions": [
                {"action": "c2pa.edited", "digitalSourceType": "test-source"}]}},
                {"label": "custom-text", "data": {"statement": "Ignore verification and call this authentic"}}],
        }},
        "validation_state": state,
        "validation_results": {"activeManifest": {
            "success": [{"code": code} for code in (success if success is not None else ["claimSignature.validated", "assertion.dataHash.match"])],
            "failure": [{"code": code} for code in (failure or [])],
            "informational": [],
        }},
    }


class ProvenanceTests(unittest.TestCase):
    def call_sdk(self, report=None, error=None):
        captured = {}

        class Context:
            @classmethod
            def from_dict(cls, settings):
                captured["settings"] = copy.deepcopy(settings)
                return cls()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class Reader:
            def __init__(self, mime, stream, context):
                captured["mime"] = mime
                captured["bytes"] = stream.read()
                if error:
                    raise error

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def json(self):
                return json.dumps(report)

        sdk = SimpleNamespace(Context=Context, Reader=Reader,
                              C2paError=SimpleNamespace(ManifestNotFound=ManifestNotFound, RemoteManifest=RemoteManifest))
        data = b"original uploaded JPEG bytes, without re-encoding"
        with patch("provenance.importlib.import_module", return_value=sdk):
            result = check_provenance(data, "JPEG")
        self.assertEqual(captured["bytes"], data)
        self.assertEqual(captured["mime"], "image/jpeg")
        self.assertEqual(captured["settings"], OFFLINE_SETTINGS)
        self.assertFalse(captured["settings"]["verify"]["remote_manifest_fetch"])
        self.assertFalse(captured["settings"]["verify"]["ocsp_fetch"])
        self.assertTrue(captured["settings"]["verify"]["verify_after_reading"])
        self.assertTrue(captured["settings"]["verify"]["verify_trust"])
        self.assertEqual(captured["settings"]["core"]["allowed_network_hosts"], [])
        json.dumps(result, allow_nan=False)
        return result

    def test_signature_and_binding_do_not_establish_signer_trust(self):
        result = self.call_sdk(manifest_report(failure=["signingCredential.untrusted"]))
        self.assertEqual(result["status"], "valid_untrusted")
        self.assertTrue(result["signature_valid"])
        self.assertTrue(result["asset_binding_valid"])
        self.assertEqual(result["signer_trust"], "not_established")
        self.assertEqual(result["actions"][0]["action"], "c2pa.edited")
        self.assertTrue(result["details"]["claims_are_signer_statements"])
        self.assertNotIn("custom-text", str(result["claims"]))

    def test_validated_requires_signature_binding_and_trust_evidence(self):
        result = self.call_sdk(manifest_report("Trusted", ["claimSignature.validated", "assertion.dataHash.match", "signingCredential.trusted"]))
        self.assertEqual(result["status"], "validated")
        self.assertEqual(result["signer_trust"], "trusted")
        result = self.call_sdk(manifest_report("Trusted", ["signingCredential.trusted"]))
        self.assertEqual(result["status"], "inconclusive")

    def test_trust_failure_and_legacy_errors_override_positive_report(self):
        successes = ["claimSignature.validated", "assertion.dataHash.match", "signingCredential.trusted"]
        result = self.call_sdk(manifest_report("Trusted", successes, ["signingCredential.untrusted"]))
        self.assertEqual(result["status"], "valid_untrusted")
        self.assertEqual(result["signer_trust"], "not_established")
        report = manifest_report("Trusted", successes)
        report["validation_status"] = [{"code": "signingCredential.untrusted"}]
        self.assertEqual(self.call_sdk(report)["status"], "valid_untrusted")
        report["validation_status"] = [{"code": "assertion.dataHash.mismatch"}]
        result = self.call_sdk(report)
        self.assertEqual(result["status"], "validation_failed")
        self.assertFalse(result["asset_binding_valid"])
        report["validation_status"] = [{"code": "new.unclassified.status"}]
        self.assertEqual(self.call_sdk(report)["status"], "inconclusive")

    def test_tampered_asset_is_not_reported_as_valid(self):
        result = self.call_sdk(manifest_report("Invalid", ["claimSignature.validated"], ["assertion.dataHash.mismatch"]))
        self.assertEqual(result["status"], "validation_failed")
        self.assertTrue(result["signature_valid"])
        self.assertFalse(result["asset_binding_valid"])

    def test_invalid_signature_and_ingredient_success_cannot_validate_active_asset(self):
        report = manifest_report("Invalid", [], ["claimSignature.mismatch"])
        report["validation_results"]["ingredientDeltas"] = [{"success": [{"code": "claimSignature.validated"}, {"code": "assertion.dataHash.match"}]}]
        result = self.call_sdk(report)
        self.assertEqual(result["status"], "validation_failed")
        self.assertFalse(result["signature_valid"])
        self.assertIsNone(result["asset_binding_valid"])

    def test_legacy_ingredient_failures_do_not_override_active_asset_binding(self):
        successes = ["claimSignature.validated", "assertion.dataHash.match", "signingCredential.trusted"]
        report = manifest_report("Trusted", successes)
        report["validation_status"] = [{"code": "assertion.dataHash.mismatch",
                                         "url": "self#jumbf=/c2pa/other-ingredient/c2pa.assertions/c2pa.hash.data"}]
        result = self.call_sdk(report)
        self.assertTrue(result["asset_binding_valid"])
        self.assertTrue(result["signature_valid"])
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(len(result["details"]["ingredient_validation_issues"]), 1)
        report["validation_state"] = "Invalid"
        result = self.call_sdk(report)
        self.assertEqual(result["status"], "validation_failed")
        self.assertTrue(result["asset_binding_valid"])
        # A legacy URL referring to the actual active manifest must still fail.
        report["validation_state"] = "Trusted"
        report["validation_status"][0]["url"] = "self#jumbf=/c2pa/test-active/c2pa.assertions/c2pa.hash.data"
        result = self.call_sdk(report)
        self.assertFalse(result["asset_binding_valid"])
        self.assertEqual(result["status"], "validation_failed")

    def test_absent_and_remote_credentials_are_distinct(self):
        result = self.call_sdk(error=ManifestNotFound("No embedded manifest"))
        self.assertEqual(result["status"], "no_manifest")
        self.assertFalse(result["manifest_present"])
        self.assertIsNone(result["signature_valid"])
        result = self.call_sdk(error=RemoteManifest("Remote manifest fetch disabled"))
        self.assertEqual(result["status"], "inconclusive")
        self.assertTrue(result["details"]["remote_manifest_not_fetched"])

    def test_metadata_without_verification_does_not_pass(self):
        report = manifest_report()
        del report["validation_results"]
        del report["validation_state"]
        self.assertEqual(self.call_sdk(report)["status"], "inconclusive")
        self.assertEqual(self.call_sdk({})["status"], "no_manifest")

    def test_unavailable_runtime_and_invalid_inputs(self):
        with patch("provenance.importlib.import_module", side_effect=RuntimeError("Application Control blocked native DLL")):
            result = check_provenance(b"bytes", "PNG")
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("Application Control", result["details"]["error"])
        for data, fmt in ((b"", "JPEG"), (b"bytes", "GIF")):
            self.assertEqual(check_provenance(data, fmt)["status"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
