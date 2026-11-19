# Image Verification - System Report

Updated: 19 November 2026. Presentation version.

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

Detectra-OpenFake revision c84565e is the only active AI detector; its pretrained weights have not been fine-tuned in this project. Standard RGB input is resized with bilinear interpolation so its shorter edge is 440 pixels, then center-cropped to 384 x 384. Bytes are divided by 255 and normalized with ImageNet mean [0.485,0.456,0.406] and standard deviation [0.229,0.224,0.225]. A float32 1 x 3 x 384 x 384 tensor enters ONNX Runtime. Sigmoid converts the output logit to the displayed score, which is not a calibrated probability. Automatic additional views are described in section 9; the author's score scaling is not applied. Extreme aspect ratios producing an edge above 16,000 pixels are rejected.

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

## 9. AI involvement and automatic closer inspection

The active AI model remains Detectra-OpenFake. Its task is to look for AI involvement in fully generated images and camera photos modified using AI. A positive signal does not distinguish these origins or locate a changed region. Original camera images and conventional brightness, contrast, crop and compression edits are negative examples for AI-use evaluation. TruFor remains a separate localization check; an editing anomaly does not establish AI involvement.

There is one Analyze image pixels action and no mode selector. Initial raw scores in [0.50, 0.95) automatically trigger two more center crops using shorter-edge sizes 576 and 768 pixels. Each crop is 384 x 384 with the existing normalization; the highest raw sigmoid score is retained. Low initial scores and scores >=0.95 skip rescanning. This uses the same pinned weights. The author-inspired gate and scales do not reproduce the author's calibrated procedure because Platt scaling is not applied. CSV/JSON exports retain the method, initial score and view scores. Saved results from the previous selectable-mode workflow are cleared before using this pipeline.

Real inference covered 25 related development cases: three originals, eighteen conventional transformations, three fully generated images and one AI-edited camera image. The conventional transformations used brightness factors 0.7/1.3, contrast factors 0.7/1.3, a crop removing 10% from each border and JPEG quality 70. Creation labels came from known history, not inferred metadata.

| Creation type | Initial-view reference AI indications | Automatic-rescan AI indications |
|---|---:|---:|
| 3 camera originals | 0 | 0 |
| 18 conventional edits | 0 | 2 false alarms |
| 3 fully generated images | 3 | 3 |
| 1 AI-edited camera image | 0, inconclusive | 1 |

The known AI-edited example changed from 0.500714 to 0.983172. Two conventional transformations changed from inconclusive to incorrect AI indications. The initial view left five conventional cases inconclusive; rescanning left three inconclusive. Ungated zooming also incorrectly flagged an original. Automatic rescanning is now the requested default pipeline; these results demonstrate a sensitivity/false-alarm tradeoff rather than validated generalization accuracy. The initial-view column is a development reference, not a second user-selectable approach. These examples are previously inspected and share original-image families; they are not an independent test set.

The real-model Streamlit check verified automatic rescanning on the first click, the absence of a mode selector and identical scores on repeated clicks. Regression tests separately cover preprocessing, score aggregation, rescan gating, result persistence and session isolation. Aggregate diagnostics are in AI_DETECTION_RESULTS.json, without private image filenames. AI_DETECTION.md describes the exact procedure and limitations.

The evaluate_ai_use.py tool accepts a CSV with path, label, group_id and split. It compares the initial-view reference and automatic-rescan scores for development measurement, reports each creation type, and records both forced-0.50 metrics and the application's 0.10/0.90 answer counts. Declared original families cannot cross training, validation, development and test splits. The evaluator does not detect every undeclared duplicate or train the model.

Further fine-tuning requires a compatible native training checkpoint and a larger source-diverse labeled dataset including AI removals, insertions and replacements, full generations, originals and conventional edits. The current quantized ONNX export is used for inference; no Detectra weights were retrained in this update. It remains necessary to separate original families and generator/source groups, select settings on validation data, then evaluate on untouched test data. Detection of all AI involvement cannot be guaranteed.

The separate CIFAKE CNN was removed from active source and archived locally with its weights, code and results. The CIFAKE dataset cache is retained for possible future research and is not an active model or website input source. The historical experiment's 94.37% CIFAKE result is not an accuracy claim for the active Detectra model. There is now one active AI detector and the separate TruFor edit detector.

Sources: [pinned model card](https://huggingface.co/s-gehring/detectra-openfake/blob/c84565e5bf8969d95908e12844f551e77ae87105/README.md), [author training code](https://github.com/s-gehring/ai-watermark-addon/tree/main/tools/train). Model license: CC-BY-NC-4.0.

## 10. Columbia dataset and photographic-adjustment experiment

The user-proposed Columbia Uncompressed Image Splicing Detection Evaluation Dataset contains 183 authentic images and 180 manually spliced images from four cameras. The supplied splices have no post-processing, so they are suitable for studying copy-and-paste manipulation but do not provide brightness/contrast-edit labels. The author-linked archive was downloaded for noncommercial research, with the source README and attribution retained locally. Source photos and archives are not redistributed in the repository.

For a separate enhancement experiment, the 183 authentic photographs were used to generate individual brightness and contrast factors 0.65/0.8/1.2/1.4 and gamma factors 0.7/0.85/1.2/1.5. Each adjusted example had a control with matching JPEG quality, sampled from 65 to 98. This created 4,392 derived examples, not 4,392 independent photographs. Twenty-three original-photo families were grouped before splitting; related subimages and adjustments remain together. Cross-camera scene overlap and undeclared near duplicates have not been exhaustively audited.

| Partition | Original families | Source photos | Derived examples |
|---|---:|---:|---:|
| Training | 13 | 101 | 2,424 |
| Validation | 5 | 37 | 888 |
| Test | 5 | 45 | 1,080 |

The final research candidate, tonal-frequency-mlp-v2, uses 272 native-level histogram-frequency and intensity-summary features with a small 16/8-unit MLP. It does not use EXIF, filenames or reference images as prediction inputs. Feature normalization is fitted on training data only; normalized values are clipped to [-5,5]. AdamW, dropout 0.5 and validation binary cross-entropy select the checkpoint. An earlier larger histogram MLP overfit validation data and was rejected before the test was evaluated. The final checkpoint was selected at epoch 17 and training stopped at epoch 42. No test or user-phone examples were used in checkpoint selection.

At a fixed 0.50 cutoff, validation accuracy was 60.92% and test accuracy 60.56%. Test adjustment precision was 62.03%, recall 54.44%, ROC-AUC 0.6501 and control false-positive rate 33.33%. A separate threshold of 0.609775 was selected from the validation negative 95th percentile; it detected 129/540 test adjustments and incorrectly flagged 46/540 controls. Its test recall was therefore 23.89% with an 8.52% false-positive rate, demonstrating that a validation target is not a population guarantee.

On four known phone diagnostics, the three originals were not flagged at the validation-selected threshold, but the known brightness/contrast-edited photo was also missed (score 0.354249). That image may contain combined operations, while the synthetic training examples cover individual operations. The dataset is small, mostly indoor and limited to four cameras; repeated controls and related crops cannot be treated as independent observations for broad accuracy claims.

Decision: the candidate is not integrated into Streamlit. This experiment did not establish a useful enhancement-detection improvement. The active Detectra and TruFor models, weights, thresholds and user workflow remain unchanged. This result does not establish that Columbia is unsuitable for splicing or that a different enhancement approach could not perform better. More representative real-retouching pairs, diverse camera processing and untouched phone/combined-retouch test data are needed for further enhancement work.

Code and exact commands are in COLUMBIA.md; aggregate evidence is in COLUMBIA_RESULTS.json. Source images, manifests, trained weights and histories are in the ignored .models/columbia/ cache. The enhancement experiment is separate from AI-generation detection and does not turn conventional edits into AI-positive labels.

Attribution: Columbia DVMM Laboratory; Y.-F. Hsu and S.-F. Chang, Detecting Image Splicing Using Geometry Invariants and Camera Characteristics Consistency, ICME 2006. Sources: [dataset description](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/), [author terms/download](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/dlform.html). Use is subject to the dataset's noncommercial research terms.

## 11. FiveK retouching research and tonal-comparison improvement

MIT-Adobe FiveK was selected as a closer match for photographic brightness, contrast and colour retouching than a splicing-only dataset. The downloaded sRGB release is linked by the Retinexformer researchers and contains 5,000 input/target pairs. The original research-only licenses are retained under research-licenses/FiveK/. This project uses those image pairs, not the Retinexformer enhancement model. FiveK's input renderings are not certified untouched camera JPEGs, so a classifier may learn rendering style rather than edit history.

The source archive is 658,823,577 bytes with SHA-256 1c2c79f92c103c337517c8fff32ecc434d08c0287ecf49325447344945d05251. The release's official 500 test photo IDs were preserved; 500 of its training IDs were held out for validation. One unchanged pair and one pair with different dimensions were excluded from training. No exact decoded-image duplicate families were found across different photo IDs. Final counts were 3,998 training, 500 validation and 500 test photos. Near-duplicate scenes have not been exhaustively ruled out.

Each photo supplied an expert target, one explicit brightness/contrast/gamma/saturation/combined adjustment and two matched controls, exported using the same JPEG quality within the photo. There were 19,992 derived examples, not 19,992 independent photographs. All versions of a photo remained in one split. Inputs combined a 64 x 64 RGB thumbnail with 272 histogram-frequency/intensity features. A small 38,713-parameter CNN/frequency network was trained from scratch using binary cross-entropy plus paired ranking, AdamW and dropout. Training-only statistics standardized features; metadata, private filenames and reference images were not prediction inputs.

Validation binary cross-entropy selected epoch 22; training stopped at epoch 37. A high-adjustment score threshold of 0.5395969748 was frozen from the validation-control 95th percentile before testing. A release gate was also recorded before testing: at least 50% recall for both expert and synthetic adjustments, at most 10% false positives on external camera controls, and correct handling of four known phone diagnostics. These are research decisions, not existing application thresholds.

| Task | Accuracy at 0.50 | Recall at frozen threshold | Control false-positive rate at frozen threshold |
|---|---:|---:|---:|
| FiveK expert retouch, 500 held-out pairs | 93.00% | 90.00% | 4.80% |
| FiveK synthetic adjustments, same 500 photo IDs | 66.10% | 37.60% | 4.80% |
| Combined FiveK examples | 79.55% | 63.80% | 4.80% |
| External Columbia originals / combined tonal variants, 183 source photos | 75.68% | 97.81% | 45.90% |

At the frozen threshold, 450/500 expert targets and 188/500 synthetic adjustments were detected; 24/500 FiveK input photos were falsely flagged. Repeated controls are counted twice in the combined table, not independent photos. In the external check, 179/183 combined tonal variants were detected, but 84/183 authentic camera controls were falsely flagged. Columbia includes related subimages, so it is not 183 independent scenes; nevertheless, the false alarms reveal substantial source sensitivity. Those images were not part of this candidate's training or threshold selection.

The three known original phone diagnostics remained unflagged, and the known brightness/contrast-edited phone image was flagged. These four previously inspected cases are diagnostic evidence only. No threshold was adjusted after testing. The candidate failed its release gate on synthetic-edit recall and external false alarms, and is not loaded by Streamlit. Its 93% expert-retouch result must not be reported as the active app's edit-detection accuracy or as a direct improvement over TruFor on a shared benchmark. Repeated tuning against the now-inspected FiveK test would require a fresh source-disjoint test set for a new independent accuracy claim.

The application improvement implemented in this update is reference-based tonal measurement. After successful alignment, valid blurred overlapping pixels supply display luma (0.299 R + 0.587 G + 0.114 B). The app displays the candidate/reference mean difference on a 0-255 scale and the ratio of luma standard deviations. A contrast ratio is withheld for nearly flat references. Non-overlapping borders are excluded. Mean shifts of at least two levels or contrast-spread shifts of at least 5% contribute to the comparison summary even if no region exceeds the separate 32-level highlighting threshold. These are observable pair differences, not learned editing probabilities or proof of software editing.

On the supplied original/tonally edited phone pair, comparison measured +24.8 brightness levels, a 1.30x contrast-spread ratio and a 33.6% marked overlap area. The app test confirmed these measurements and the differences-detected conclusion. The Edit detection panel now explains that local inconsistency analysis may miss global adjustments and points to comparison when an original is available. JSON exports retain the tonal measurements. Detectra and TruFor weights, score bands and independent checks remain unchanged.

All 39 regression tests passed, including new split/class-mapping checks and tonal-comparison cases below the region-marking threshold, flat references and overlap exclusion. These tests establish implementation behavior, not detection accuracy. FIVEK.md contains the reproducible Windows commands and protocol; FIVEK_RESULTS.json contains aggregate evidence. Dataset files, trained research weights, histories and private diagnostics stay in the ignored local cache. The public app does not train on uploads or download FiveK.

Attribution: Vladimir Bychkovsky, Sylvain Paris, Eric Chan and Fredo Durand, Learning Photographic Global Tonal Adjustment with a Database of Input / Output Image Pairs, CVPR 2011. Sources: [MIT-Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/), [processed release linked by the Retinexformer researchers](https://github.com/caiyuanhao1998/Retinexformer). Use is subject to the original research-only licenses.
