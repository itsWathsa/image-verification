# Image Verification

A Python web application for AI-use assessment, edit inspection, and original-image comparison.

The interface uses a warm charcoal theme with soft white text and muted gold accents. Theme colors are configured in .streamlit/config.toml; layout and component styling are in style.css.

## Features

- **AI detection:** a qualified answer about signs of AI generation or AI-assisted editing.
- **Edit detection:** a qualified answer about possible editing, with anomaly and reliability maps.
- **Compare original:** align a supplied original, highlight differences, and measure brightness/contrast changes.
- **File details:** preview, format, stored dimensions, and available camera metadata.
- Persistent session results, CSV/PNG downloads, and a combined JSON report.

Automated assessments may be incorrect. Uncertain cases remain unable to determine. Missing metadata never triggers an AI or edited label. The application does not provide a definitive forensic verdict.

## Windows setup

```powershell
cd "C:\Users\Hasini\OneDrive\Desktop\AASL\image-verification"
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501 if needed. Stop with Ctrl+C. Activation is not required. Use requirements-basic.txt only for preview/file details. Full dependencies are needed for analysis. Supported content: JPEG/PNG, up to 20 MiB and 20 megapixels.

## Result interpretation

AI scores <=0.10 indicate likely not fully AI-generated, but do not exclude AI-assisted editing. Scores >=0.90 indicate likely AI-generated or AI-edited. Intermediate scores cannot determine AI use. Scores are not calibrated probabilities.

For TruFor, <=0.10 is displayed as no strong editing evidence detected; >=0.90 as possible editing detected; intermediate scores cannot determine editing. These are exploratory presentation bands, not independently validated edit/no-edit thresholds. Low scores may miss edits. Maps analyze a maximum 384-pixel edge and may lose small details.

Reference comparison finds differences relative to the supplied original. Marked area is not an editing probability. Lighting, compression, reconstruction, and alignment error can contribute. Weak alignment produces an inconclusive result.

The comparison also reports average brightness difference (0-255 levels) and contrast spread relative to the original. This can reveal small tonal adjustments even when no area exceeds the separate region-marking threshold. These are measurements of pair differences, not proof of what caused them. TruFor alone may miss global brightness, contrast and colour changes.

## Tests and deployment

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

See DEPLOYMENT.md for hosting. Source: https://github.com/itsWathsa/image-verification. The app runs models on the server; the browser needs no local Python or Colab. The latest source requires a cloud rebuild and browser smoke check after changes.

Detectra-OpenFake is the only active AI detector. The goal covers fully generated images and camera images edited using AI. **Analyze image pixels** automatically inspects additional views for borderline initial scores; there is one analysis flow and no mode selector. Closer inspection can find stronger signals but also produced false alarms on conventional edits. See [AI_DETECTION.md](AI_DETECTION.md) for the evaluation labels, measured tradeoffs and dataset evaluation commands. The unused CIFAKE CNN experiment is archived locally outside the active project; website uploads are not training data.

The separate [Columbia adjustment experiment](COLUMBIA.md) used authentic source photographs to generate brightness/contrast/gamma training pairs. Its enhancement candidate performed poorly on held-out data and is not connected to the app. Dataset archives and candidate weights remain in the ignored local cache.

The subsequent [FiveK retouching experiment](FIVEK.md) trained a paired CNN/frequency classifier using real retouching pairs and synthetic tonal changes. It achieved 93% on the held-out expert-retouch task but missed many simple adjustments and falsely flagged 45.9% of external original-camera controls. It failed its release gate and is not an active detector. Full aggregate results are in [FIVEK_RESULTS.json](FIVEK_RESULTS.json). The implemented app improvement is brightness/contrast measurement in original-image comparison.

Inputs and results stay under data/ and evaluation-private/ locally and are excluded from Git. The app does not explicitly save hosted uploads or call an external image-inference API. Model weights download on first use. Review reports before sharing sensitive information.

Methods and licenses: Detectra-OpenFake c84565e (CC-BY-NC-4.0); TruFor ae54475df6f41a491d7615100feb19263dec13f7 (informational/nonprofit license retained under trufor_vendor/). OpenCV performs reference alignment. See TRUFOR.md, REFERENCE_COMPARISON.md and SYSTEM_REPORT.md.
