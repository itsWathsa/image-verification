# Columbia photographic-adjustment experiment

The user-selected [Columbia Uncompressed Image Splicing Detection Evaluation Dataset](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/) contains 183 authentic photographs and 180 manually spliced images from four cameras. Its spliced images have no post-processing, so they are not examples of brightness, contrast or gamma adjustments. The dataset is suitable for studying splicing; its authentic images can also seed a controlled adjustment experiment.

## What this experiment uses

`columbia_enhancement.py` uses the 183 authentic images only. It generates brightness and contrast factors 0.65, 0.8, 1.2 and 1.4, and gamma factors 0.7, 0.85, 1.2 and 1.5. Each modified image is paired with a control exported using the same JPEG quality, sampled from 65 to 98. This avoids an obvious uncompressed-original versus JPEG-edited shortcut. Labels describe these generated operations, not Columbia's original authentic/spliced classification.

Nine subimages from the same original photograph are grouped together using the filename prefix before `_sub`. Original-photo families are split before training. Related subimages and their derived controls/adjustments stay in one split. Cross-camera scene overlap and undeclared near duplicates have not been exhaustively audited. Repeated controls are not independent new photographs.

The final candidate is `tonal-frequency-mlp-v2`, separate from the AI detector and TruFor. It computes native-level RGB/grayscale histograms and feeds 272 histogram-frequency/intensity-summary features to a 16/8-unit MLP with dropout 0.5. At most approximately 262,144 pixels are sampled without resizing/interpolation. Normalization is fitted on the training partition only and normalized values are clipped to [-5,5]. Checkpoint selection uses validation binary cross-entropy. A positive threshold is selected from validation negatives; this does not guarantee a population false-positive rate. The test partition is excluded from training and checkpoint/threshold selection. An initial larger histogram MLP overfit validation data and was rejected before test evaluation; its source, weights and history are retained in the local cache.

This is an exploratory adjustment-classification task. A naturally bright photo or a camera's own processing may resemble a manually enhanced photo. Small or reversible edits may leave no reliable detectable trace. A score is not proof that a person edited a file, and does not establish AI involvement.

## Reproduce on Windows

Install the project's `requirements.txt`. Download the source archive from the [author-linked download page](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/dlform.html) and retain its terms and README. The current local cache stores the author's folder download as `.models/columbia/source.zip`, containing `README.txt`, `4cam_auth.tar.bz2` and `4cam_splc.tar.bz2`.

```powershell
.\.venv\Scripts\python.exe columbia_enhancement.py prepare
.\.venv\Scripts\python.exe columbia_enhancement.py train
.\.venv\Scripts\python.exe columbia_enhancement.py evaluate
```

The commands protect existing experiment artifacts against accidental overwrite. They do not download images automatically or train from website uploads. The cache contains extracted authentic images, features, a split manifest, source provenance, weights and measured results, all excluded from Git. Archive members are read individually rather than extracting arbitrary archive paths. The author's download page provides a public download link; no personal-details form was submitted during this experiment.

The model is not connected to the Streamlit application by this experiment. A candidate must be evaluated before integration; the earlier bottle-photo example is held out of training and model selection. Synthetic-operation performance from this small four-camera dataset alone is not evidence of broad performance on modern phone photos or editing tools. No improvement is claimed until results support it.

## Completed results: 19 November 2026

The downloaded archive SHA-256 is `fb6a3dd39f3e6c89fc2c7c4505f3423a853fe32d5199ed9f28a92e19fabd56a9`. It was obtained through the author's download page; this hash records our downloaded archive, not an author-signed checksum. All source images, masks and dataset archives stay local and are not redistributed in the repository.

| Partition | Original-photo families | Source photographs | Derived examples |
|---|---:|---:|---:|
| Training | 13 | 101 | 2,424 |
| Validation | 5 | 37 | 888 |
| Test | 5 | 45 | 1,080 |

Each source photograph produces twelve adjusted examples and twelve matched export controls. Repeated controls and related subimages must not be counted as independent photographs for statistical claims. Transformations are individual brightness, contrast or gamma operations; combined/complex retouching is not covered by these synthetic training labels.

The final model's minimum validation loss selected epoch 17; training stopped at epoch 42 after 25 epochs without improvement. At a fixed 0.50 cutoff:

| Measure | Validation | Test |
|---|---:|---:|
| Accuracy | 60.92% | 60.56% |
| Adjustment precision | 66.67% | 62.03% |
| Adjustment recall | 43.69% | 54.44% |
| Control false-positive rate | 21.85% | 33.33% |
| ROC-AUC | 0.6485 | 0.6501 |

A threshold of 0.609775 was separately selected from the validation negative 95th percentile. On test examples it detected 129/540 adjustments (23.89% recall) and incorrectly flagged 46/540 controls (8.52%). Thus the validation target did not guarantee a 5% false-positive rate on the test partition.

The three previously inspected original phone photos were not flagged at that threshold, but the user's known brightness/contrast-edited phone image was also missed (score 0.354249). These four cases were not used in training, validation selection or threshold selection, but remain known development diagnostics rather than a large independent phone test set.

**Decision: not integrated into the app.** This experiment did not establish a useful improvement for the missing brightness/contrast task. The current AI detector and TruFor weights, thresholds and runtime remain unchanged. This result does not establish that the dataset is unusable for splicing or that more capable enhancement models could not perform better. The dataset as supplied lacks real brightness/contrast edit labels; its small, mostly indoor, four-camera authentic subset is a limited basis for this extension. Aggregate evidence is in `COLUMBIA_RESULTS.json`; local raw predictions, trained weights and manifests remain under `.models/columbia/`.

For the next enhancement experiment, prefer verified real retouching pairs from diverse cameras/tools and matched encoding pipelines. Reserve independent modern phone photographs and combined retouches for testing. Columbia's original spliced images remain appropriate for a separate TruFor splicing benchmark. Their edgemasks need careful interpretation: authentic-image masks mark salient object boundaries, not genuine tampering, and camera-origin colors in spliced masks are not ordinary binary edit masks.

## Attribution and terms

The dataset is free for noncommercial purposes including academic research under the author's [terms](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/dlform.html). Acknowledge Columbia's dataset and cite Y.-F. Hsu and S.-F. Chang, *Detecting Image Splicing Using Geometry Invariants and Camera Characteristics Consistency*, ICME 2006. Retain the source README and attribution when sharing derived research materials. This dataset has different terms from CIFAKE, Detectra and TruFor; the app's existing model licenses remain unchanged.
