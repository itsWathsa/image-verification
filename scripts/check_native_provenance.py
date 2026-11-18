"""Exercise real offline C2PA verification on a pinned official test fixture.

Run from the repository root after installing requirements.txt:
    python scripts/check_native_provenance.py --output native-provenance-results.json

Only the official fixture is downloaded. A damaged copy is derived in memory;
no user image, signing key, trust anchor, or external manifest is used.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import urllib.request

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provenance import check_provenance  # noqa: E402


FIXTURE_COMMIT = "5765f372b29101969db804dcbde49508a50f06e2"
FIXTURE_URL = f"https://raw.githubusercontent.com/contentauth/c2pa-python/{FIXTURE_COMMIT}/tests/fixtures/C.jpg"
FIXTURE_SHA256 = "4cbac1af485155f878d12a610da851c2938f2ce886189c61d9da7a0a43bd00b5"


def alter_quantization_table(data):
    """Modify covered JPEG bytes while preserving the embedded signed manifest."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("Fixture is not JPEG")
    position = 2
    while position + 4 < len(data):
        if data[position] != 0xFF:
            raise ValueError("Unexpected JPEG segment structure")
        while data[position] == 0xFF:
            position += 1
        marker = data[position]
        position += 1
        if marker in (0xDA, 0xD9):  # Stop before scan/end if no DQT found.
            break
        length = int.from_bytes(data[position:position + 2], "big")
        if length < 2 or position + length > len(data):
            raise ValueError("Invalid fixture segment length")
        if marker == 0xDB:  # Quantization-table byte is outside APP11/C2PA.
            table = position + 2
            if data[table] >> 4 != 0:
                raise ValueError("Expected an 8-bit quantization table")
            edited = bytearray(data)
            coefficient = table + 1
            edited[coefficient] = data[coefficient] + 1 if data[coefficient] < 255 else 254
            return bytes(edited)
        position += length
    raise ValueError("No quantization table in fixture")


def run():
    with tempfile.TemporaryDirectory(prefix="image-verification-c2pa-") as directory:
        with urllib.request.urlopen(FIXTURE_URL, timeout=45) as response:
            signed = response.read(2 * 1024 * 1024 + 1)
        if hashlib.sha256(signed).hexdigest() != FIXTURE_SHA256:
            raise AssertionError("Official fixture hash mismatch; verification aborted")
        (Path(directory) / "signed.jpg").write_bytes(signed)
        tampered = alter_quantization_table(signed)
        # Prove the damaged fixture remains a decodable image rather than testing
        # only a JPEG parsing failure, and that its metadata was not re-encoded.
        for data in (signed, tampered):
            with Image.open(io.BytesIO(data)) as image:
                image.load()
        unsigned_buffer = io.BytesIO()
        Image.new("RGB", (64, 64), "blue").save(unsigned_buffer, format="JPEG")
        results = {
            "fixture_url": FIXTURE_URL,
            "fixture_sha256": FIXTURE_SHA256,
            "signed": check_provenance(signed, "JPEG"),
            "tampered": check_provenance(tampered, "JPEG"),
            "unsigned": check_provenance(unsigned_buffer.getvalue(), "JPEG"),
        }

    signed_result = results["signed"]
    assert signed_result["signature_valid"] is True, signed_result
    assert signed_result["asset_binding_valid"] is True, signed_result
    assert signed_result["status"] in {"validated", "valid_untrusted", "validation_failed"}, signed_result
    # Certificate lifetime/trust can change with the runner date. The fixture is
    # not added to our trust policy; no public-trust expectation is asserted.
    damaged_result = results["tampered"]
    assert damaged_result["status"] == "validation_failed", damaged_result
    assert damaged_result["asset_binding_valid"] is False, damaged_result
    assert any(item.get("code") == "assertion.dataHash.mismatch" for item in damaged_result["checks"]), damaged_result
    assert results["unsigned"]["status"] == "no_manifest", results["unsigned"]
    assert all(results[key]["network_access"] == "disabled" for key in ("signed", "tampered", "unsigned"))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Optional JSON output path for workflow artifacts")
    args = parser.parse_args()
    report = run()
    serialized = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(serialized + "\n", encoding="utf-8")
    print(json.dumps({name: {key: report[name][key] for key in (
        "status", "signature_valid", "asset_binding_valid", "signer_trust")}
        for name in ("signed", "tampered", "unsigned")}, indent=2))
    print("PASS: native C2PA signed-asset binding, tampered-asset rejection, and unsigned handling.")
