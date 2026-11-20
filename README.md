# Image verification

A Streamlit application with three sections:

- **AI detection:** evaluates image pixels with the pinned Detectra-OpenFake model.
- **Compare original:** compares an uploaded image with a supplied original/reference.
- **File details:** displays image properties and available camera metadata.

Results persist for the current upload and can be downloaded as JSON. Replacing the main image clears its results. JPEG and PNG uploads are limited to 20 MiB and 20 megapixels.

## Run locally

Use Python 3.11 or newer. On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501. Stop with Ctrl+C. The AI model downloads on first use and is cached in `.models/`.

## Limits and privacy

AI results are model signals, not proof of authenticity. The detector can miss AI-assisted edits and flag camera photographs incorrectly. Original comparison measures differences; it does not authenticate the reference or identify editing software. Camera metadata can be missing or changed.

The app does not explicitly save uploaded images or send them to an external inference API. Keep private photos in `data/`, which is excluded from Git. Review downloaded reports before sharing.

## Development

```powershell
.\.venv\Scripts\python.exe -B -m unittest -v
```

Runtime files are `app.py`, `detector.py`, `conclusions.py`, `reference_compare.py`, `style.css`, and `.streamlit/config.toml`. Evaluation utilities and their tests support the active AI pipeline. See [AI detection](AI_DETECTION.md), [reference comparison](REFERENCE_COMPARISON.md), and [deployment](DEPLOYMENT.md).

The Detectra-OpenFake weights are licensed CC-BY-NC-4.0 for noncommercial use; attribution and the pinned source are documented in AI_DETECTION.md.
