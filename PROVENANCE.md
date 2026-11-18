# C2PA signed-history checks

`provenance.check_provenance(data, image_format)` uses **c2pa-python 0.39.0**, the official Content Authenticity SDK, to verify embedded JPEG/PNG Content Credentials. It receives the exact uploaded bytes: do not pass a resized, rotated, or re-encoded preview, because that would change the file binding being checked.

The SDK verifies while reading. The app reports cryptographic signature validation, the asset's hash binding, and signer trust separately. It does not classify an image as AI-generated or edited merely because credentials are absent, unverifiable, or invalid.

## Result meanings

| Status | Meaning |
|---|---|
| `no_manifest` | No embedded credentials found. Remote credentials have not been fetched. |
| `valid_untrusted` | SDK validation succeeded for signature/file binding, but signer trust is not established. |
| `validated` | Signature and file binding succeeded and the SDK reported a trusted signer under its configured trust policy. This is not proof that signed statements are truthful. |
| `validation_failed` | SDK validation reported an active-manifest failure or invalid state. It does not identify who changed a file or why validation failed. |
| `inconclusive` | A complete verification conclusion could not be established, including remote-only credentials or incomplete validation output. |
| `unavailable` | SDK or native runtime cannot load. No provenance judgment is made. |

The app displays limited active-manifest claims/actions as statements attributed to the signer. It does not execute embedded instructions or convert declared editing actions into independently verified facts. Other manifests/ingredients may have their own validation status; their success codes do not establish the active asset's validity.

Legacy flat validation issues are scoped by their manifest URL when available. A hash/signature failure on an ingredient is retained as an ingredient-history issue and cannot turn the active asset's successful binding into a failed binding. Ingredient integrity concerns still prevent a validated-history conclusion; an SDK-wide `Invalid` state remains a validation failure. Unscoped legacy issues are handled conservatively.

## Offline and trust limits

Each Reader gets its own Context with `verify_after_reading`, `verify_trust`, and `verify_timestamp_trust` enabled. `remote_manifest_fetch` and `ocsp_fetch` are disabled, and `core.allowed_network_hosts` is an empty allow-list. The app does not download remote manifests, trust lists, revocation responses, or linked resources. It does not add test certificates or silently trust a self-signed signer.

The explicit empty host list is different from an omitted/null list. The SDK's [Context resolver construction](https://github.com/contentauth/c2pa-rs/blob/main/sdk/src/context.rs) preserves `Some([])` and passes it to [RestrictedResolver](https://github.com/contentauth/c2pa-rs/blob/main/sdk/src/http/restricted.rs); its host-matching implementation matches no entries and returns false. The implementation, rather than an inconsistent older comment about an “empty” list, establishes the deny-all behavior. Both synchronous and asynchronous context paths apply it. Native verification passed locally with this configuration; cloud execution still needs verification.

Trust is therefore assessed only under the SDK's locally available configuration. A trust result is not a claim that the app maintains an up-to-date public trust list. Online revocation freshness and unavailable remote material are not established. A signature protects the signed statements/file binding; it cannot guarantee every edit was recorded or establish whether the photographed event happened.

## Runtime and verification

Install `c2pa-python==0.39.0` with the full app dependencies. It includes a native library; preview/file-details functionality and the independent detectors can remain usable when C2PA cannot load.

An initial import on this Windows PC reported an Application Control block on `c2pa_c.dll` (`WinError 4551`). A subsequent run loaded the SDK successfully without changing security settings; the earlier block was no longer reproducible. **Real native validation now passes locally:** the pinned official signed fixture reports valid signature/file binding with trust not established; its modified copy reports `assertion.dataHash.mismatch` and failed binding; an unsigned JPEG reports `no_manifest`. The recorded report is `evaluation-private/native-provenance-results.json`.

The supplied Gemini bottle-removal image also reports `valid_untrusted`, with valid signature/file binding. Its embedded signer fields name Google LLC / Google Media Processing Services and its declared actions include opened/resized activity with an algorithmic-media source type. These remain signed statements under an unestablished signer-trust policy, not proof of every edit. Real Streamlit execution of all four checks and a session rerun passed on this local example. Cloud inference/validation after deployment still needs verification.

Tests in `test_provenance.py` validate SDK context configuration, exact-byte input, status interpretation, signature/binding/trust separation, tampered reports, missing and remote credentials, and runtime failures using SDK doubles. These tests do not substitute for a native signature-validation test. Run:

```powershell
.\.venv\Scripts\python.exe -m unittest test_provenance -v
```

Primary references: [official Python SDK usage](https://github.com/contentauth/c2pa-python/blob/main/docs/usage.md), [per-instance settings](https://github.com/contentauth/c2pa-python/blob/main/docs/context-settings.md), [Reader JSON reference](https://contentauth.github.io/json-manifest-reference/reader-ref), and [C2PA validation status codes](https://spec.c2pa.org/specifications/specifications/2.3/specs/ContentCredentials.html).

## Native verification command

`scripts/check_native_provenance.py` downloads the official SDK's 138 KB `tests/fixtures/C.jpg` from pinned commit `5765f372b29101969db804dcbde49508a50f06e2`, verifies its SHA-256, and checks it using this app's offline Context. The upstream `TestReader.test_stream_read_get_validation_state` test expects this signed fixture's state to be `Valid` without added test trust roots.

The script also checks a new unsigned Pillow JPEG and changes a JPEG quantization-table coefficient in a copy of the signed fixture, preserving the embedded C2PA bytes and confirming that both files still decode. Native verification must report the signed file's signature and asset binding as valid, and reject the modified file with `assertion.dataHash.mismatch`. It does not assert that the test signer is publicly trusted. Certificate-time issues can produce an overall validation failure while signature and asset binding remain separately visible.

Run this on a supported environment (for example Linux GitHub Actions) after dependency installation:

```text
python scripts/check_native_provenance.py --output native-provenance-results.json
```

The downloaded fixture is kept only in a temporary directory. The JSON output contains official-fixture reports, not private user images or keys. The complete script passed on the subsequent local Windows run. Linux/cloud execution remains a separate deployment check.
