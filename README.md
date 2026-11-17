# Image verification lab

A Streamlit web app that previews JPEG/PNG images, displays file details, and runs an experimental pixel-based AI-generation detector. Users of the hosted website only need a browser; VS Code and Colab are not required.

## Current version

- `app.py`: upload, preview, file details, analysis, and CSV score download.
- `detector.py`: Detectra-OpenFake ONNX detector, pinned revision `c84565e`.
- `requirements.txt`: complete CPU model and web-app dependencies for cloud hosting.
- `requirements-basic.txt`: optional local file-details-only dependencies.
- `.streamlit/config.toml`: upload limit and server settings.
- `evaluate.py`: descriptive metrics for labeled score CSV files.
- `test_app.py`, `test_evaluate.py`: regression checks.
- `DEPLOYMENT.md`: cloud deployment instructions.

The current app uses the integrated ONNX detector. PyTorch, Transformers, and Colab are not needed. The earlier SDXL implementation remains in GitHub history.

## Interpretation

Camera origin and local editing are separate questions. A camera photo can also be edited. Local edit detection is **not implemented**, and its result remains inconclusive.

The AI score is an uncalibrated model output, not a probability of authenticity. Exploratory bands are <=0.10 camera-photo-like, >=0.90 AI-like, and otherwise inconclusive. The integrated detector matched all six supplied diagnostic labels, including three Gemini images and three originals, without tuning the thresholds. Those known examples do not establish general accuracy. Do not rely on this experimental detector for authenticity decisions. Missing metadata never affects inference.

The model is [s-gehring/detectra-openfake](https://huggingface.co/s-gehring/detectra-openfake), listed as CC-BY-NC-4.0 for noncommercial use. Attribution: s-gehring, OpenFake authors, TensorFold, and Community Forensics authors as described in the model card. The implementation uses a single center crop: shorter edge 440 (bilinear), crop 384, ImageNet normalization, ONNX logits, then raw sigmoid. It does not implement the author's score scaling or full multi-crop procedure.

## Run locally on Windows

Open PowerShell to run the full app:

```powershell
cd "C:\Users\Hasini\OneDrive\Desktop\AASL\image-verification"
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The first analysis downloads about 30 MB of pinned model weights and caches them in `.models/`. Later analyses reuse the in-memory model. The ONNX implementation was verified locally on this Windows PC without changing its security settings. The old SDXL/PyTorch runtime block is no longer in the app's execution path. To pre-download weights, run `.\.venv\Scripts\python.exe detector.py --download`. For preview only, `requirements-basic.txt` remains available.

## Test and evaluate

Copy an original JPEG phone photo to your PC and upload it. HEIC is unsupported. Camera metadata may be missing; this is not evidence of generation or editing. Try a PNG screenshot and an invalid text file renamed `.jpg` to check graceful handling.

```powershell
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe evaluate.py evaluation-private\reported_scores.csv --output evaluation-private\baseline_metrics.json
```

Evaluation CSV columns are `filename,label,ai_score`, with reference labels `camera` or `generated`. The evaluator does not train a model or select thresholds. Input images remain under `data/`; local results remain under `evaluation-private/`. Both are excluded from Git. `test results.txt` preserves the initial manually recorded scores.

The integrated model reproduced the earlier six-image comparison scores. The actual Streamlit analyze-button flow was checked using a real model and the supplied dog image. Full results remain private under `evaluation-private/integrated_results.json`. Cloud execution after the update still needs verification. Collect a larger independent dataset before claiming general accuracy or tuning thresholds.

## Hosting and privacy

See [DEPLOYMENT.md](DEPLOYMENT.md). The repository is https://github.com/itsWathsa/image-verification and the app is deployed through Streamlit Community Cloud. Hosted uploads are processed on that server. The app does not explicitly save uploaded images or call an external image inference API. Hugging Face supplies model weights. Upload limits are 20 MB and 20 megapixels.

## Later work

Compare broader-generation detectors on independent labeled data; retain an inconclusive outcome. Evaluate TruFor for edit localization and C2PA for signed provenance as separate checks. Neither missing provenance nor missing EXIF proves AI generation or editing.
