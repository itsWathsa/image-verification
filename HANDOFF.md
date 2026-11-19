# Saved checkpoint - 19 November 2026

Work stopped at the user's request. Resume the COVERAGE edit-detection task next session; no dataset evaluation or training should continue unattended.

## Current application

The last completed main-branch update is `d061bcc`: the FiveK research results and brightness/contrast measurements in original-image comparison. Detectra remains the AI-use detector and TruFor remains the edit detector. The FiveK adjustment candidate was not deployed because it failed external false-positive and synthetic-adjustment recall checks. `SYSTEM_REPORT.pdf` describes that completed version, not the unfinished COVERAGE work.

## COVERAGE work saved today

- User requested: use https://github.com/wenbihan/coverage to improve **single-image Edit detection**, rather than only original-image comparison.
- COVERAGE is a copy-move benchmark with 100 original/forged image pairs and region masks. It does not establish whole-image brightness/contrast-edit labels. Original terms restrict use to noncommercial research; cite Wen et al., ICIP 2016.
- `copy_move.py`: unfinished single-image SIFT matching, affine-consistency and pixel/high-pass correlation inspection. It produces a repeated-region evidence map and index. It is **not connected to the app and has no COVERAGE accuracy evaluation**. One synthetic pasted-region smoke check worked; that is not accuracy evidence. Similar genuine objects can cause false alarms.
- `edit_detector.py`: accepts an optional `max_edge` argument (32-768) to support future controlled resolution experiments. Its default stays 384. No model weights, conclusions or app buttons changed in this checkpoint.
- `coverage_dataset.py`: downloads an exploratory processed-release cache, keeps original/forged photo IDs together, checks exact cross-partition duplication and creates 60 development / 20 validation / 20 test photo groups.

## Dataset access and limitations

The author OneDrive API required authentication. The HiFi-IFDL authors link a 5.6 GB benchmark archive containing native COVERAGE, but streaming it locally was slow and eventually failed before reaching COVERAGE. No helper remains running.

A complete 138,777,718-byte Parquet mirror was downloaded locally from `ThaneJoss/COVERAGE`. SHA-256: `8079bd790c028c20e733028c70d74ec12bee58718770118454ae6c2672bf0cdd`. It packages the ForgeryVCR **832 x 832 preprocessed** release; it is not the original native TIFF release. Original image terms still apply.

Windows Application Control blocked PyArrow's Parquet component and a separately installed DuckDB 1.4.3 reader. Neither security settings nor runtime package requirements were changed. DuckDB is an unused research-only local installation; it is not required by the app.

The dataset viewer successfully provided 200 images and 100 masks, saved under `.models/coverage/images/`, with a manifest and file hashes under `.models/coverage/manifest.json`. **Viewer image URLs end in JPEG, so these assets can be re-encoded even though the cache files are saved as PNG. Do not present this cache as original PNG/TIFF pixels or use its forensic results as native-dataset results.** Check mask encoding, alignment and semantics too. The cached manifest has been corrected to record this caveat.

The other large combined Hugging Face mirror supports row retrieval but filter/search failed; sparse metadata queries encountered connection errors and rate limits. Stop/retry cautiously rather than repeatedly querying the full dataset. No native-source evaluation was completed, and no test accuracy or localization metrics exist for this work.

## Resume from here

1. Obtain a verified source edition with correct image/mask correspondence. Prefer native COVERAGE; a Linux evaluation environment may avoid the local blocked Parquet readers. A processed-edition study must state all resizing/re-encoding explicitly.
2. Audit the known mask-dimension issues (IDs 41, 48, 55, 56, 57, 58, 59, 61, 95), original/forged labels, exact duplicates, and whether masks mark pasted regions or both source and destination. Do not silently stretch masks or choose mask polarity to improve results.
3. Preserve photo groups; tune only on development/validation. Compare TruFor 384 against a bounded higher-detail variant and the repeated-region inspector. Freeze settings before opening held-out test results. Measure image-level false alarms/recall and localization precision/recall/IoU, with resolution/time/memory costs.
4. Integrate only a supported improvement into **Edit detection**, preserving uncertainty, independent AI-use results, failure isolation and per-upload state. Do not claim copy-move performance covers global tonal edits, AI removals or every manipulation.
5. Add meaningful regression tests, run the relevant app checks, document measured results, update the report, and publish the completed change after review.

Exploration scripts are archived in `research/coverage_exploration/`; they are not application runtime code. Large downloaded datasets, credentials, virtual environments, private input images and private diagnostics remain local and excluded from Git. The existing Streamlit server was left running; only development work was stopped.

Sources: [COVERAGE authors](https://github.com/wenbihan/coverage), [paper](https://stefan.winklerbros.net/Publications/icip2016b.pdf), [HiFi-IFDL research archive link](https://github.com/CHELSEA234/HiFi_IFDL), [processed mirror](https://huggingface.co/datasets/ThaneJoss/COVERAGE), [reported mask corrections](https://github.com/SunnyHaze/IML-Dataset-Corrections).
