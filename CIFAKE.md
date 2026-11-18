# CIFAKE training experiment

This project includes a separate, reproducible AI-generation classification experiment using the user-selected CIFAKE dataset. It trains a small CNN from scratch; it does not fine-tune the pretrained Detectra-OpenFake model. The application's edit-localization model is independent of this experiment.

## Dataset and scope

Source: [Jordan Bird's CIFAKE repository](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images), pinned to `e112a942abaecd02b6b1f6f646c807d56be8fb62`.

The official dataset contains 60,000 REAL images from CIFAR-10 and 60,000 FAKE images generated with Stable Diffusion 1.4. It provides an official 100,000-image training split and 20,000-image test split. The downloaded images are 32 x 32 RGB pixels. Performance on this domain is not an estimate of accuracy on Gemini, other generators, high-resolution phone photographs, graphics, or local edits. No image-origin model establishes whether a depicted event actually happened.

## Reproduce on Windows

Run these commands from the project folder after installing `requirements.txt` using the README. No additional packages or Kaggle credentials are required; the archive comes directly from the author's public GitHub repository.

```powershell
.\.venv\Scripts\python.exe cifake_experiment.py prepare
.\.venv\Scripts\python.exe cifake_experiment.py train --epochs 15 --batch-size 256 --threads 2
.\.venv\Scripts\python.exe cifake_experiment.py evaluate
.\.venv\Scripts\python.exe cifake_experiment.py compare --per-class 250
.\.venv\Scripts\python.exe cifake_experiment.py target
```

Training runs locally on CPU and may take several minutes or longer. An interrupted or completed experiment's existing checkpoint is protected against overwrite. Use a separate workspace for a fresh experiment. Do not repeatedly adjust settings based on official test results. The `target` command uses the existing six user-labeled images under `data/` when available; these are known development diagnostics, not an independent target test set.

## Isolation and audit

The archive is read without extracting arbitrary paths. The preparation step checks the expected class counts and 32 x 32 dimensions. It hashes decoded RGB pixels, removes contradictory labels, keeps one representative per exact duplicate group, and prevents exact duplicates appearing in multiple partitions. Where an identical image is present in official train and test, the official test representative takes precedence. Near duplicates and semantic-family overlap are not exhaustively audited.

Validation is sampled deterministically from the official training pool after the audit. Class mapping is explicit: REAL = 0, FAKE = 1. Checkpoint selection uses the lowest validation binary cross-entropy, with patience four. The official test set is excluded from training, checkpoint selection and threshold fitting. The existing 0.10/0.90 presentation bands are reported unchanged; they are not calibrated by this experiment.

Seed: `20261118`. Audited partitions: 88,832 training images, 9,870 validation images, 19,962 test images. The audit removed 1,336 redundant records, with no conflicting labels. Archive SHA-256: `df9a280df27336041a13812435261668ca151562155cbfff7e655dbeeae5516c` (computed for the downloaded archive, not an author-published signature).

## Model and outputs

`cifake_model.py` defines `cifake-cnn-v1`: three convolution/ReLU/max-pooling blocks (16, 32, 64 channels), a 64-unit dense layer, 0.25 dropout and one binary output logit. RGB input is resized bilinearly to 32 x 32, divided by 255 and converted to CHW float32. The generated-class score is the sigmoid of the output logit. No EXIF or filenames are model features. The training baseline uses AdamW and a cosine learning-rate schedule, with no image augmentation. It is not an exact reproduction of the authors' architecture or claimed paper results.

Artifacts are written under `.models/cifake/`, excluded from Git:

- `dataset.zip`, `prepared.npz`, `data_audit.json`: source archive, decoded images and split provenance.
- `candidate.pt`, `training_history.json`: weights and validation-only selection history.
- `test_metrics.json`, `test_scores.csv`: candidate results on the audited official test split.
- `baseline_comparison.json`, `baseline_subset_scores.csv`, `candidate_subset_scores.csv`: paired comparison against the app's existing Detectra model on the same seeded subset of 250 examples per class.
- `target_diagnostic.json`: separate scores on the six known user examples.

The test report includes accuracy, AI precision/recall/F1, false-positive rate, ROC-AUC, inconclusive count and decision coverage. Scores are not calibrated authenticity probabilities. Uploaded app images are not added to this dataset or used for automatic training.

## Promotion rule

Keep the current app detector while evaluating this candidate. A better CIFAKE test score alone is insufficient for replacing the detector used on general uploads. Compare both models on the same CIFAKE subset and inspect regressions on known target examples. An independent source-diverse target set is still required before claiming improved general-use accuracy. Do not average the two models' scores without separately validating the combined method.

## Completed training run: 18 November 2026

The 15-epoch CPU run completed in approximately 11.5 minutes, excluding preparation and evaluation. The checkpoint with the lowest validation loss was epoch 15. On 19,962 audited official test images, with a fixed 0.50 cutoff:

| Measure | Result |
|---|---:|
| Accuracy | 94.37% |
| Generated-class precision | 93.93% |
| Generated-class recall | 94.85% |
| Generated-class F1 | 94.39% |
| REAL-class false-positive rate | 6.11% |
| ROC-AUC | 0.9868 |

At the unchanged 0.10/0.90 presentation bands, 83.89% of images received an indication, 3,215 were inconclusive, and accuracy among indicated images was 98.39%. That last number excludes inconclusive images and must not be described as overall accuracy. The evaluator's `camera` field represents the CIFAKE REAL label here; it does not establish physical camera provenance.

On the same seeded subset of 500 official test images (250 per class), Detectra-OpenFake scored 50.0% accuracy and the CIFAKE candidate scored 93.6% at 0.50. Detectra assigned every subset image to its low-score class, missing all generated examples. The candidate detected 233/250 generated images and incorrectly flagged 15/250 REAL images. This demonstrates an improvement on this low-resolution benchmark, with a simultaneous regression on the target diagnostics below. It does not demonstrate that the candidate is a better general-purpose detector. Aggregate results and provenance are recorded in `CIFAKE_RESULTS.json`.

**Deployment decision: not promoted.** On the six previously inspected user-labeled development examples, the current detector classified 6/6 correctly at 0.50, while the CIFAKE candidate classified 1/6 correctly and missed all three generated examples. With the app's 0.10/0.90 bands, the candidate gave three indications, two incorrect, and abstained on three examples. These are small development diagnostics, not population accuracy estimates, but they demonstrate a regression that rules out replacing the default with this candidate. The trained weights remain local at `.models/cifake/candidate.pt`; the website continues to use Detectra-OpenFake.

The dataset was used to train and assess a real candidate; no general-use accuracy improvement was demonstrated. Next experiments require a source-diverse set including modern generators and independent phone photos. Any new model or combined method must be validated before deployment.

## Attribution and licensing

Cite Jordan J. Bird and Ahmad Lotfi, *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*, IEEE Access, 2024; [paper/preprint](https://arxiv.org/abs/2303.14126). Cite Alex Krizhevsky and Geoffrey Hinton, *Learning Multiple Layers of Features from Tiny Images*, 2009, for the CIFAR-10 real images. The dataset author's README supplies an MIT license; retain that notice when redistributing the dataset. The source README remains inside the cached archive. Dataset licensing is separate from Detectra's CC-BY-NC-4.0 model license and TruFor's retained licenses.
