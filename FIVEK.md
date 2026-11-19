# Photographic-adjustment research

MIT-Adobe FiveK provides real input/retouched pairs for studying photographic tonal adjustments. It is a closer match to brightness, contrast and colour retouching than a splicing-only dataset. There is no single best dataset for every type of image edit.

The experiment uses the 5,000-pair sRGB release linked in the Retinexformer research repository, not its enhancement model. FiveK was collected by Bychkovsky, Paris, Chan and Durand for CVPR 2011, *Learning Photographic Global Tonal Adjustment with a Database of Input / Output Image Pairs*. The original dataset contains RAW photographs and five expert renditions; this processed release contains one input and one target per photo.

Sources: [original FiveK dataset](https://data.csail.mit.edu/graphics/fivek/), [researcher-provided processed release](https://github.com/caiyuanhao1998/Retinexformer). Original research-only image licenses are retained in `research-licenses/FiveK/`. The repository's software licenses do not replace these image licenses. The dataset is not redistributed with the app.

## Reproduce on Windows

Use the existing environment and dependencies. These commands are research tasks; the Streamlit app never downloads this dataset or trains from uploads.

```powershell
cd "C:\Users\Hasini\OneDrive\Desktop\AASL\image-verification"
.\.venv\Scripts\python.exe fivek_retouch.py download
.\.venv\Scripts\python.exe fivek_retouch.py prepare
.\.venv\Scripts\python.exe fivek_retouch.py train
.\.venv\Scripts\python.exe fivek_retouch.py evaluate
```

The download is 658,823,577 bytes. SHA-256: `1c2c79f92c103c337517c8fff32ecc434d08c0287ecf49325447344945d05251`.

## Protocol

- Preserve the processed release's official 500-photo test split. Select 500 of its 4,500 training photo IDs for validation, leaving 4,000 for training (seed 20261119).
- Keep input, expert target, synthetic variants and repeated controls from each photo together. Exclude exact duplicate families crossing partitions, pixel-identical input/target pairs, and mismatched pair dimensions. The completed audit found no exact duplicate families and excluded one unchanged pair and one size-mismatched pair, both from training. Final counts are 3,998 training, 500 validation and 500 test photo IDs; four correlated examples per ID. Undeclared near-duplicate scenes can remain.
- Add one deterministic synthetic adjustment per photo: brightness, contrast, gamma, saturation or combined brightness/contrast. Single-operation factors are 0.65, 0.8, 1.2 or 1.4; the combined operation uses brightness 1.25 and contrast 1.2.
- For each photo, export two input controls, the expert target and the synthetic variant with the same JPEG quality (sampled 80 to 98). This reduces an export-setting shortcut without eliminating all source/compression artifacts. Four examples per photo are correlated, not four independent photographs.
- Extract a 64 x 64 RGB thumbnail and 272 native-level histogram-frequency/intensity features. Neither metadata nor filenames are model inputs.
- Train the small paired CNN/frequency classifier from scratch with binary cross-entropy plus a 0.15-weight paired-ranking term. AdamW uses learning rate 0.0005 and weight decay 0.02. Feature standardization is fitted on training data only, with standard deviations floored at 0.0001 and normalized values clipped to [-5,5]. Horizontal flips preserve tone labels.
- Select the checkpoint by minimum validation binary cross-entropy, at most 100 epochs with patience 15. Freeze the validation-control 95th-percentile high-adjustment threshold before testing. The 5% validation false-alarm target is not a promised false-alarm rate on other images.
- Report expert-retouch and synthetic-adjustment test results separately, plus source-transfer checks. Previously inspected phone examples are diagnostics, not an independent accuracy benchmark; they are not used for training or checkpoint/threshold selection.

Before evaluating the test set, a minimum release gate was recorded: detect at least half the expert-retouched and half the synthetic-adjustment test positives at the frozen threshold; flag at most 10% of external camera controls; and correctly handle all four known phone diagnostics. Passing this small gate would still not establish general forensic accuracy. A failed candidate remains research-only, without choosing a more flattering threshold after testing.

Cached dataset, manifest, weights, training history and private diagnostics are under `.models/fivek/`, excluded from Git. Commands refuse to silently replace prepared data, a trained checkpoint, or an evaluated test report. Preserve completed runs before starting a different experiment.

## Interpretation

An expert target is a known retouched rendition. A FiveK input is a particular camera-RAW rendering, **not a guarantee of an untouched camera JPEG**. Modern phones already adjust tone, contrast and colour. A classifier can learn a rendering style rather than an edit history, and subtle edits can resemble normal capture variation. Dataset accuracy cannot be presented as accuracy on all uploads.

Conventional adjustments are positive labels for this experiment and negative labels for the separate AI-use task. A retouching score must never be substituted for an AI-use score. TruFor continues to inspect local inconsistencies; original-image comparison can measure tonal differences when a reference is available.

## Completed results and release decision

The 38,713-parameter candidate trained for 37 epochs on CPU; validation selected epoch 22. The frozen high-adjustment threshold is **0.5395969748**, chosen before testing. It is a research threshold, not an app setting or an authenticity probability.

| Held-out task | Examples | Accuracy at 0.50 | Adjustment recall at frozen threshold | Control false-positive rate at frozen threshold |
|---|---:|---:|---:|---:|
| FiveK expert retouch versus input | 1,000 from 500 photo IDs | 93.00% | 90.00% (450/500) | 4.80% (24/500) |
| Synthetic tonal adjustments versus controls | 1,000 from the same 500 IDs | 66.10% | 37.60% (188/500) | 4.80% (24/500) |
| Combined FiveK experiment | 2,000 correlated examples | 79.55% | 63.80% (638/1,000) | 4.80% (48/1,000 repeated controls) |
| External Columbia originals / combined tonal variants | 366 from 183 photos | 75.68% | 97.81% (179/183) | **45.90% (84/183)** |

The paired controls are repeated in the combined experiment, so its 48 false flags refer to 24 distinct FiveK input photos. Exact photo IDs are disjoint across training, validation and test, but related/near-duplicate scenes have not been exhaustively excluded. The Columbia transfer check shares the old dataset's related subimages and is not 183 independent scenes. It is nevertheless strong evidence of source sensitivity. The Columbia images are not part of this candidate's training or threshold selection.

All three known original phone diagnostics were unflagged, and the previously inspected brightness/contrast-edited phone image was flagged. Four known examples are not a general accuracy test and do not outweigh the large external false-alarm rate. No threshold was changed after these results.

**Decision: do not integrate the candidate into Streamlit.** It failed the recorded release gate on synthetic-edit recall and external camera false positives. The attractive 93% result describes expert-retouch style recognition within this processed FiveK release; it is not accuracy for general edit detection, and is not an improvement claim over TruFor on a shared benchmark.

The production improvement from this work is explicit brightness/contrast measurement in **Compare original**, with valid-overlap handling and measurements included in JSON exports. The Edit detection panel now explains why local inconsistency analysis can miss global adjustments. Detectra and TruFor weights/thresholds remain unchanged. See [aggregate results](FIVEK_RESULTS.json) for the full measured evidence; private images, individual diagnostics, model weights and training history remain in the ignored cache.

For a future single-image classifier, use more representative camera-rendered JPEG controls and real conventional edits from multiple phones/software tools; pair/group them by original photo, device and scene. Reserve a new source-disjoint test set before selecting another candidate. The FiveK test set has now been inspected and must not be repeatedly used to tune and claim fresh independent accuracy. Do not use more aggressive thresholds just to make a known example pass.
