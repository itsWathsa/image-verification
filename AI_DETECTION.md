# Detecting AI involvement

The active application has one AI detector: Detectra-OpenFake, pinned to revision `c84565e` (full revision `c84565e5bf8969d95908e12844f551e77ae87105`). TruFor is a separate edit-localization model. It cannot establish that an edit was made with AI.

## Target labels

The AI-detection goal is to find AI involvement in either an entirely generated image or a camera photograph modified using AI. An AI indication does not distinguish these two origins or locate the modified region.

| Creation history | Evaluation label | AI involvement |
|---|---|---|
| Original camera image | `camera_original` | Negative |
| Camera image with ordinary crop, brightness, contrast or compression changes | `camera_conventional_edit` | Negative |
| Fully AI-generated image | `fully_ai_generated` | Positive |
| Camera image modified by AI, such as generative object removal | `camera_ai_edited` | Positive |

Labels describe verified creation history, not appearance, filenames or EXIF. An ordinary edited photo is not automatically an AI-edited photo. No detector or finite test can guarantee detection of every generator and editing method.

## Standard analysis and optional closer inspection

Standard analysis retains the existing preprocessing and weights: bilinear resize to a 440-pixel shorter edge, 384 x 384 center crop, RGB bytes divided by 255 and ImageNet normalization. The ONNX logit is converted with sigmoid. Scores <=0.10 receive a low-AI-signal answer, scores >=0.90 an AI-use indication, and intermediate scores remain inconclusive. These are exploratory bands, not calibrated authenticity probabilities.

The **Inspect additional views** checkbox is off by default. When enabled, initial raw scores in [0.50, 0.95) trigger two more 384-pixel center crops from shorter edges 576 and 768; the highest score is retained. Low initial scores and initial scores >=0.95 are not rescanned. All views use the same pinned model. This is an inference extension, not new model training. View scores, method and baseline are retained in the CSV and JSON reports. Changing the checkbox clears only the AI result so an old result is not shown under a different setting.

The model author's published procedure inspired the scales and gate. Our variant keeps raw sigmoid scores and the app's existing answer bands. It does not apply the author's Platt scaling, reproduce their calibration, or inherit their reported false-positive rates. Checking more views can increase false alarms. Center crops can also miss small or peripheral AI edits.

## Development checks: 19 November 2026

Real inference ran on three known camera originals, eighteen conventional transformations of those originals, three known generated images, and one known AI-edited camera image. Transformations were brightness factors 0.7/1.3, contrast factors 0.7/1.3, cropping 10% from each border, and JPEG quality 70. These 25 cases are related development examples, not independent validation or test data. No weights or thresholds were fitted on them.

| Cases | Standard AI indications | Optional AI indications | Observation |
|---|---:|---:|---|
| 3 camera originals | 0 | 0 | Initial low scores skip rescanning |
| 18 conventional edits | 0 | 2 | Optional rescans introduced two false alarms; standard mode left five cases inconclusive |
| 3 fully generated images | 3 | 3 | Already high scores skip rescanning |
| 1 AI-edited camera image | 0 | 1 | Score changed from 0.500714 to 0.983172 |

An ungated experiment that zoomed every image also incorrectly flagged an original at 0.929212. These failures are why rescanning is optional, not a claimed overall accuracy improvement. Aggregate metrics are in `AI_DETECTION_RESULTS.json`; private source images and per-image results remain outside Git. The real-model Streamlit check covers both modes and invalidation.

## Evaluate a labeled dataset

Copy `ai_use_manifest.example.csv`, replace its example paths with your images, and assign labels from known creation history. Paths are relative to the CSV. Give every original and its edited derivatives the same `group_id` and keep that group in one split. Different originals and generators are needed for independent testing; the existing development examples cannot be reused as an untouched test set.

After downloading the model with `python detector.py --download`, run:

```powershell
.\.venv\Scripts\python.exe evaluate_ai_use.py path\to\manifest.csv --split development --output evaluation-private\ai-use\development.json
.\.venv\Scripts\python.exe evaluate_ai_use.py path\to\manifest.csv --split test --output evaluation-private\ai-use\test.json
```

The evaluator compares the initial view with optional rescanning, reports each creation type separately, and records false AI indications, inconclusive cases, accuracy, precision, recall, F1 and ROC-AUC where defined. The forced 0.50 metrics are separate from the app's 0.10/0.90 answer counts. It rejects a declared original family crossing splits; it cannot discover all undeclared near duplicates. Dataset uploads do not automatically train the model.

## Further training

Fine-tuning for AI-edited photographs requires a larger labeled set containing AI object removal, insertion and background replacement from multiple tools, plus original photographs and conventional edits as negatives. Include whole generated images from multiple generators. Hold out original families and generator/source groups for validation and testing; retain region masks if edit localization will be evaluated.

Our current checkpoint is a quantized ONNX inference export. It is not trained by this app's ONNX Runtime adapter. A future fine-tuning run needs a compatible native training checkpoint and training pipeline; starting again from the author's base model is not the same as resuming our exact exported checkpoint. No Detectra weights were retrained in this update. Prefer measuring a candidate against this frozen baseline before replacing it; keep both missed AI edits and false alarms visible in the report.

The unsuccessful CIFAKE CNN experiment was removed from the active source and its weights, code and results were archived locally under `evaluation-private/cifake-archive-2026-11-19/`. CIFAKE is a dataset of small real and fully generated images, not a dataset of camera photos edited using AI. Its cache remains available locally; it is not used by the website.

Source and attribution: [pinned model card](https://huggingface.co/s-gehring/detectra-openfake/blob/c84565e5bf8969d95908e12844f551e77ae87105/README.md), [author training code](https://github.com/s-gehring/ai-watermark-addon/tree/main/tools/train). Model weights are CC-BY-NC-4.0, noncommercial. Detectra v3 by TensorFold; Community Forensics by Park and Owens; OpenFake by Livernoche and colleagues. Retain the upstream licensing and attribution when using or distributing the model.
