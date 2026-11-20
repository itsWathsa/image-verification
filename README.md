# Image and video verification

A Streamlit application for image review and experimental video-frame screening. The image mode has three sections:

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

## Video screening

Select **Video** in the media selector, upload an MP4, MOV, AVI or WebM and click **Analyze video**. Supported limits: 100 MiB, 10 minutes and 8.85 megapixels per frame (4096 x 2160, in either orientation). Container support depends on the installed decoder; H.264 MP4 is recommended if another file fails.

The app checks up to 12 distinct, evenly spaced frames with the existing frozen image model. It displays the mean sampled-frame score, count scoring at least 0.90, and a timestamped frame table. The highest-scoring sampled frame can be inspected. JSON reports include the media hash, model revision, sampling method and every frame score. A high signal in even one frame is reported as a sampled-frame signal, never a whole-video authenticity verdict. Partial decode failures produce an unavailable result.

This is experimental visual screening, not a trained or validated video detector. Audio, motion, lip synchronisation and unsampled segments are not analysed. Compression and ordinary processing can cause false alarms. A low score does not establish authenticity. No video accuracy claim is available without evaluation on labelled real and generated videos.

Uploads are temporarily written to a system temporary directory for decoding and removed when analysis exits, including handled failures. Decoded images are processed one at a time; only the highest-scoring thumbnail and numeric results remain in session state. No new model or Python dependency is required. Image uploads retain their separate 20 MiB limit.
