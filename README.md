# Image verification lab

A Streamlit web app that previews JPEG/PNG images, displays file details, and runs an experimental pixel-based AI-generation detector. Users of the hosted website only need a browser; VS Code and Colab are not required.

## Current version

- `app.py`: upload, preview, file details, analysis, and CSV score download.
- `detector.py`: Organika SDXL detector, pinned revision `b37fede`.
- `requirements.txt`: complete CPU model and web-app dependencies for cloud hosting.
- `requirements-basic.txt`: optional local file-details-only dependencies.
- `.streamlit/config.toml`: upload limit and server settings.
- `evaluate.py`: descriptive metrics for labeled score CSV files.
- `test_app.py`, `test_evaluate.py`: regression checks.
- `DEPLOYMENT.md`: cloud deployment instructions.

The hosted app uses the SDXL baseline. The alternative ONNX detector was tested separately and has not been integrated. Its experimental code and the Colab notebook have been removed from this working version. Their earlier committed versions remain in GitHub history.

## Interpretation

Camera origin and local editing are separate questions. A camera photo can also be edited. Local edit detection is **not implemented**, and its result remains inconclusive.

The AI score is an uncalibrated model output, not a probability of authenticity. Exploratory bands are <=0.10 camera-photo-like, >=0.90 AI-like, and otherwise inconclusive. Initial tests exposed false positives on original camera photos and low scores on generated images. Do not rely on this baseline for authenticity decisions. Missing metadata never affects inference.

The model is [Organika/sdxl-detector](https://huggingface.co/Organika/sdxl-detector), listed as CC-BY-NC-3.0 for educational/noncommercial use. Attribution: Organika / Colby / umm-maybe as described in the model card.

## Run locally on Windows

For file details and preview only, open PowerShell:

```powershell
cd "C:\Users\Hasini\OneDrive\Desktop\AASL\image-verification"
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-basic.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

This laptop's Windows Application Control blocks NumPy's FFT component needed by the SDXL runtime. Local model inference remains blocked; use the hosted website for pixel analysis. On an approved environment, install `requirements.txt` and run `python detector.py --download` before local inference. On hosted Linux the app downloads weights on its first analysis and caches the model.

## Test and evaluate

Copy an original JPEG phone photo to your PC and upload it. HEIC is unsupported. Camera metadata may be missing; this is not evidence of generation or editing. Try a PNG screenshot and an invalid text file renamed `.jpg` to check graceful handling.

```powershell
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe evaluate.py evaluation-private\reported_scores.csv --output evaluation-private\baseline_metrics.json
```

Evaluation CSV columns are `filename,label,ai_score`, with reference labels `camera` or `generated`. The evaluator does not train a model or select thresholds. Input images remain under `data/`; local results remain under `evaluation-private/`. Both are excluded from Git. `test results.txt` preserves the initial manually recorded scores.

The alternative model matched all six known diagnostic cases, but that small sample does not establish general accuracy. Preserve those results and collect a larger independent dataset before selecting a replacement or tuning thresholds.

## Hosting and privacy

See [DEPLOYMENT.md](DEPLOYMENT.md). The repository is https://github.com/itsWathsa/image-verification and the app is deployed through Streamlit Community Cloud. Hosted uploads are processed on that server. The app does not explicitly save uploaded images or call an external image inference API. Hugging Face supplies model weights. Upload limits are 20 MB and 20 megapixels.

## Later work

Compare broader-generation detectors on independent labeled data; retain an inconclusive outcome. Evaluate TruFor for edit localization and C2PA for signed provenance as separate checks. Neither missing provenance nor missing EXIF proves AI generation or editing.
