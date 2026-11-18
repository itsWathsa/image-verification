# Image Verification ? System Report

Prepared: 18 November 2026. Presentation version.

This report describes the current three-check application. Automated outputs may be incorrect; clear conclusions are qualified estimates rather than guaranteed forensic judgments.


## 1. Application overview and operation

A Python web application for AI-use assessment, edit inspection, and original-image comparison.

### Features

- **AI detection:** a qualified answer about signs of AI generation or AI-assisted editing.
- **Edit detection:** a qualified answer about possible editing, with anomaly and reliability maps.
- **Compare original:** align a supplied original and highlight differences.
- **File details:** preview, format, stored dimensions, and available camera metadata.
- Persistent session results, CSV/PNG downloads, and a combined JSON report.

Automated assessments may be incorrect. Uncertain cases remain unable to determine. Missing metadata never triggers an AI or edited label. The application does not provide a definitive forensic verdict.

### Windows setup

```powershell
cd "C:\Users\Hasini\OneDrive\Desktop\AASL\image-verification"
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501 if needed. Stop with Ctrl+C. Activation is not required. Use requirements-basic.txt only for preview/file details. Full dependencies are needed for analysis. Supported content: JPEG/PNG, up to 20 MiB and 20 megapixels.

### Result interpretation

AI scores <=0.10 indicate likely not fully AI-generated, but do not exclude AI-assisted editing. Scores >=0.90 indicate likely AI-generated or AI-edited. Intermediate scores cannot determine AI use. Scores are not calibrated probabilities.

For TruFor, <=0.10 is displayed as no strong editing evidence detected; >=0.90 as possible editing detected; intermediate scores cannot determine editing. These are exploratory presentation bands, not independently validated edit/no-edit thresholds. Low scores may miss edits. Maps analyze a maximum 384-pixel edge and may lose small details.

Reference comparison finds differences relative to the supplied original. Marked area is not an editing probability. Lighting, compression, reconstruction, and alignment error can contribute. Weak alignment produces an inconclusive result.

### Tests and deployment

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

See DEPLOYMENT.md for hosting. Source: https://github.com/itsWathsa/image-verification. The app runs models on the server; the browser needs no local Python or Colab. The latest source requires a cloud rebuild and browser smoke check after changes.

Inputs and results stay under data/ and evaluation-private/ locally and are excluded from Git. The app does not explicitly save hosted uploads or call an external image-inference API. Model weights download on first use. Review reports before sharing sensitive information.

Methods and licenses: Detectra-OpenFake c84565e (CC-BY-NC-4.0); TruFor ae54475df6f41a491d7615100feb19263dec13f7 (informational/nonprofit license retained under trufor_vendor/). OpenCV performs reference alignment. See TRUFOR.md, REFERENCE_COMPARISON.md and SYSTEM_REPORT.md.


## 2. Editing method and limitations

The second app question now has its own **Analyze possible edits** button. It runs TruFor independently of Detectra-OpenFake and displays:

- A whole-image anomaly score, not an authenticity probability.
- A red/blue anomaly overlay using a fixed 0–1 color scale.
- A reliability map: bright is higher predicted reliability; dark needs caution.
- CSV score and plain anomaly/reliability PNG downloads, plus an optional on-screen overlay.

The application displays qualified assessments; no edit/no-edit threshold has been independently validated. A camera photo can be edited, a generated image can produce anomalies, and compression or normal image processing can mislead the model. A low score does not establish absence of edits.

### Runtime and preprocessing

Install `requirements.txt` and run Streamlit as described in README.md. The first edit analysis downloads the approximately 261 MB official weights ZIP and checks its published MD5 (`7bee48f3476c75616c3c5721ab256ff8`). The model is cached in `.models/trufor/`; it is not uploaded to GitHub. You can pre-download and check loading with:

```powershell
.\.venv\Scripts\python.exe edit_detector.py
```

The adapter uses CPU PyTorch 2.10, strict checkpoint loading, and weights-only deserialization with narrowly scoped NumPy scalar/dtype constructors needed by the original checkpoint. It does not use unrestricted checkpoint unpickling. Inference is serialized within the process to limit peak concurrent memory.

To limit memory on the demo host, images are proportionally reduced to a maximum edge of **384 pixels** before analysis. This is an explicit deviation from native-resolution inference and can erase small edit traces or introduce resizing artifacts. Maps apply to the analyzed dimensions shown in the app. Input is RGB float32 divided by **256**, matching the official test dataset, followed by the model's internal ImageNet preprocessing. Anomaly values use softmax class 1; reliability and whole-image outputs use sigmoid. Maps are not dynamically rescaled to make weak signals appear strong.

The inference architecture is vendored under `trufor_vendor/`, from official source revision `ae54475df6f41a491d7615100feb19263dec13f7`. Only imports and inference layer helpers were adapted. See `trufor_vendor/SOURCE.md` and retained license files. The timm training dependency is replaced with evaluation-only DropPath (identity during evaluation), tuple conversion, and PyTorch truncated-normal initialization; no topology or forward attention layers were replaced.

### Result interpretation

The interface uses exploratory presentation bands: <=0.10 no strong editing evidence detected; >=0.90 possible editing detected; otherwise unable to determine. These are not calibrated or independently validated edit/no-edit thresholds. Outputs may be incorrect. A low score must not be presented as confirmed unedited.

Independent evaluation with untouched originals and known edit masks is required. Measure false positives, missed edits, localization precision/recall/IoU and sensitivity to the resolution cap. Do not tune thresholds to already-inspected cases.

### Attribution and license

TruFor: Fabrizio Guillaro, Davide Cozzolino, Avneesh Sud, Nicholas Dufour, Luisa Verdoliva, CVPR 2023. GRIP-UNINA's license permits informational/nonprofit use and requires retained notices and attribution; review `trufor_vendor/LICENSE.txt` and `LICENSE_CMX.txt`. This project is an educational prototype.

Sources: [official repository](https://github.com/grip-unina/TruFor), [official inference instructions](https://github.com/grip-unina/TruFor/blob/main/TruFor_train_test/README.md), [paper](https://arxiv.org/abs/2212.10957).


## 3. Original-image comparison

Upload the version to examine as the main image. In **Compare original**, select **I have an original/reference image**, upload the original, and click **Compare with original**.

The comparison is independent of Detectra and TruFor. It does not infer the reference's authenticity, identify the editing software, or establish whether a depicted event happened. It is useful when a known original exists; it cannot solve the same problem from a single unknown image.

### Method

Each decoded, orientation-corrected RGB image is reduced to at most 1000 pixels on its longest edge. SIFT feature matching in both directions and RANSAC estimate a homography. The comparison rejects insufficient matches, weak spatial coverage, implausible/mirrored transforms, excessive reprojection residual, or insufficient overlap. A homography assumes a shared geometric relationship; parallax and generative reconstruction can violate it even when some features align.

Only overlapping pixels are compared. A small fixed Gaussian blur reduces single-pixel noise, and a fixed per-channel absolute-difference setting marks pixels above 32 on the 0–255 scale. Small isolated components are removed. These are inspection settings, not trained or calibrated edit-detection thresholds. Non-overlapping borders appear gray and are excluded from the marked-area fraction. Marked area is **not an editing probability or a validated estimate of edited area**.

The interface displays the aligned reference, target, a plain difference map, and an optional red overlay. Plain maps avoid confusing an image's original colors with added highlights. Lighting, compression, resampling, geometric mismatch, and reconstruction may all produce differences.

### Checks

Tests cover identical images, resized/JPEG-recompressed copies, a controlled splice, unrelated or featureless inputs, and invalid transforms. These establish implementation behavior, not general forensic accuracy. Local input files and derived comparison maps remain excluded from Git.

Method reference: [OpenCV feature matching and homography](https://docs.opencv.org/4.13.0/d7/dff/tutorial_feature_homography.html).


## 4. Hosting and verification

Host the Streamlit app from GitHub repository itsWathsa/image-verification, branch main, entrypoint app.py, using Python 3.11.

Include app.py, conclusions.py, detector.py, edit_detector.py, reference_compare.py, requirements.txt, .streamlit/config.toml, and the complete trufor_vendor/ directory with licenses. Exclude private inputs/results, virtual environments, and model caches.

After a push, watch Streamlit Community Cloud build logs and refresh the site. Confirm tabs AI detection, Edit detection, Compare original, and File details. Run each analysis, download a report, then confirm results persist. Replacing the primary image must clear all results; replacing only the reference clears its comparison.

The generation model downloads about 30 MB, and TruFor downloads about 261 MB on first use. CPU time, memory, and first-run delays depend on the host. No latency or uptime guarantee is made. Model downloads can recur after a restart.

The GitHub Actions regression workflow uses Linux Python 3.11. Passing tests do not establish detection accuracy or confirm the live deployment. Verify the interface in a browser after rebuilding.

The application provides automated estimates that may be incorrect. It does not confirm that an image is authentic or unedited. No paid hosting plan is configured.


## 5. Technology versions and AI preprocessing

| Technology | Requirement | Role |
|---|---|---|
| Python | Cloud 3.11 | Server application |
| Streamlit | >=1.40,<2 | Browser interface and session state |
| Pillow | >=11,<13 | Decoding, preview and orientation |
| NumPy | >=2,<3 | Pixel arrays |
| ONNX Runtime | >=1.20,<2 | Generation classification on CPU |
| huggingface-hub | >=0.34,<1 | Download pinned generation model |
| PyTorch CPU | 2.10.0+cpu | TruFor inference |
| OpenCV headless | >=4.10,<5 | Reference alignment and differences |
| GitHub Actions | Linux Python 3.11 | Automated regression checks |

Detectra-OpenFake revision c84565e is a pretrained model; its weights have not been fine-tuned in this project. A separate CIFAKE CNN was trained and evaluated as described in section 9, but was not promoted to the application. Detectra RGB input is resized with bilinear interpolation so its shorter edge is 440 pixels, then center-cropped to 384 x 384. Bytes are divided by 255 and normalized with ImageNet mean [0.485,0.456,0.406] and standard deviation [0.229,0.224,0.225]. A float32 1 x 3 x 384 x 384 tensor enters ONNX Runtime. Sigmoid converts the output logit to the displayed score, which is not a calibrated probability. The author's score scaling and full multi-crop procedure are not applied. Extreme aspect ratios producing an edge above 16,000 pixels are rejected.

## 6. Output guide

| Check | Range/condition | Answer |
|---|---|---|
| AI | 0 to 0.10 inclusive | Likely not fully AI-generated; AI-assisted edits may still be present |
| AI | Greater than 0.10 and less than 0.90 | Unable to determine whether AI was used |
| AI | 0.90 to 1 inclusive | Likely AI-generated or AI-edited |
| Edits | 0 to 0.10 inclusive | No strong evidence of editing detected; not confirmed unedited |
| Edits | Greater than 0.10 and less than 0.90 | Unable to determine whether the image was edited |
| Edits | 0.90 to 1 inclusive | Possible editing detected |
| Reference | Positive marked-area fraction | Differences detected relative to supplied original |
| Reference | Zero marked-area fraction | No significant differences detected at the inspection settings |
| Any check | Runtime/validation failure | Unavailable or unable to determine |

Scores range from zero to one; neither those numbers nor the marked-area percentage are authenticity probabilities. The editing bands are exploratory UI rules, not established forensic operating thresholds. Full-resolution details may be missed because the models resize/crop inputs. Metadata is informational only.

Results are retained for the latest image in the session. Changing the main image clears all results; changing only the reference clears its comparison. Downloads include CSV results, image maps/overlays, and a combined JSON report with qualified conclusions and explicit unrun/failed statuses. The SHA-256 identifier in the report checks exact byte consistency, not camera origin.

## 7. Development challenges

1. **Missing camera metadata:** EXIF fields could be absent after transfer/export. Metadata was kept informational and was never used as an authenticity label.
2. **Defining separate targets:** Full AI generation, AI-assisted editing, and camera origin are different questions. The interface separates the checks and avoids claiming an AI-edited photograph must be fully generated.
3. **Initial classifier suitability:** The first model performed poorly on some newer generated images and original photos. Its class mapping was checked and a broader-generation model was evaluated before integration. Diagnostic successes do not establish general accuracy.
4. **Windows library restrictions:** Native components were initially blocked by Application Control. Actual errors were diagnosed; supported runtimes and cloud execution were used without disabling security controls.
5. **Dependency compatibility:** CPU framework builds, image libraries, and older checkpoints needed compatible dependencies. PyTorch was pinned and unnecessary training-related dependencies were avoided.
6. **Safe checkpoint loading:** The official checkpoint included NumPy scalar values. Weights-only deserialization was retained with narrowly allowed primitive constructors and strict model-state matching.
7. **Resource constraints:** CPU inference and model loading can be slow or memory-intensive. Lazy loading, model caches, serialized TruFor inference and explicit size caps reduce resource use, at the cost of fine-detail sensitivity.
8. **Missed edits and false positives:** No detector reliably handles every image. Limitations remain documented; display wording does not change the underlying scores or repair missed detections.
9. **Image registration:** Reference and examined images can differ in scale, geometry, lighting and reconstructed content. Mutual matching, RANSAC, transform checks and overlap exclusion were added; weak alignment remains inconclusive.
10. **Map readability:** Original image colors made overlays hard to interpret. Separate fixed-scale anomaly and reliability maps were introduced.
11. **Session reruns:** Results previously disappeared after other actions. Per-upload session state and explicit invalidation preserve relevant results across reruns and downloads.
12. **Presentation:** Technical scores previously dominated conclusions. The interface now presents a concise qualified answer first, with methods and limitations under expandable details.

## 8. Evaluation and professional use

Regression tests verify implementation behavior, including image errors, conclusion boundaries, output mapping, failure isolation, state persistence and geometric comparisons. They do not establish population accuracy. Real model inference and controlled examples were also tested during development.

Before claiming reliable accuracy, collect independent original photos, generated images from multiple tools, conventional edits and AI edits. Record generation/edit history separately and obtain pixel masks for localization evaluation. Keep originals and their derivatives in the same split. Choose thresholds on validation data, then freeze them before testing. Measure false positives, missed edits, precision, recall, F1, coverage/inconclusive rate, calibration and localization IoU. Do not select thresholds to make already-inspected examples pass.

The application is suitable for a supervised demonstration of its implemented workflow and documented limitations. Production or forensic use still requires broader validation, capacity testing and appropriate access/privacy controls. No service uptime or guaranteed authenticity classification is claimed.

## 9. CIFAKE training experiment

The user-selected CIFAKE dataset was downloaded from the author's repository at revision e112a942abaecd02b6b1f6f646c807d56be8fb62. It consists of 32 x 32 images from CIFAR-10 and Stable Diffusion 1.4. Original dataset counts are 100,000 training images and 20,000 test images. The preparation audit removed 1,336 exact duplicate records, prevented exact-image overlap across partitions, and reserved validation data from the official training pool. Final counts were 88,832 training, 9,870 validation and 19,962 test images. Near-duplicate leakage has not been exhaustively audited.

A separate PyTorch CNN was trained from scratch for 15 epochs on CPU in approximately 11.5 minutes. It uses convolutional channels 16/32/64, a 64-unit dense layer, dropout 0.25, AdamW and a cosine learning-rate schedule. RGB pixels are bilinearly resized to 32 x 32 and divided by 255. REAL maps to 0 and generated to 1. Checkpoint selection used minimum validation binary cross-entropy; the official test set was excluded from training, model selection and threshold fitting. This is not a reproduction of the paper's exact architecture.

| Evaluation | Current Detectra model | CIFAKE candidate |
|---|---:|---:|
| Same 500-image CIFAKE test subset, accuracy at 0.50 | 50.0% | 93.6% |
| Full 19,962-image audited CIFAKE test, accuracy at 0.50 | Not evaluated | 94.37% |
| Six known user-labeled development cases, correct at 0.50 | 6/6 | 1/6 |

On the full CIFAKE test, the candidate achieved 93.93% generated-class precision, 94.85% recall, 94.39% F1, ROC-AUC 0.9868 and a 6.11% REAL-class false-positive rate. At the unchanged 0.10/0.90 bands, decision coverage was 83.89%; 3,215 images were inconclusive and accuracy among indicated images was 98.39%. Accuracy among indicated images excludes inconclusive cases and is not overall accuracy.

The candidate missed all three generated examples in the six known development cases at the 0.50 cutoff. These cases are too small and previously inspected to estimate target-population accuracy, but the regression is sufficient to withhold deployment. The website continues to use Detectra-OpenFake. No general-use accuracy improvement is claimed from this experiment. CIFAKE does not train edit localization, and the application's TruFor model is unchanged.

Training code, exact commands, source attribution, licensing and limitations are in CIFAKE.md; aggregate results are in CIFAKE_RESULTS.json. The downloaded dataset, raw predictions, training history and trained candidate weights are stored locally under the ignored .models/cifake/ cache. A future candidate needs source-diverse modern-generator training data and an independent target test before promotion.

Sources: [author dataset repository](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images), [Bird and Lotfi paper](https://arxiv.org/abs/2303.14126). Real-image source: Krizhevsky and Hinton, Learning Multiple Layers of Features from Tiny Images, 2009. The dataset's MIT notice is retained in the cached source archive.
