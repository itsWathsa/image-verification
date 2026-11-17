# Image verification lab

A beginner-friendly Streamlit app that previews JPEG/PNG files and displays their actual format, stored pixel dimensions, and EXIF camera make/model when available. The preview respects orientation metadata where possible. Uploads are limited to 20 MB and Pillow's image size safety checks remain enabled. Invalid files show a friendly message.

The original file-details milestone is now extended with an **experimental pixel-only AI-generation baseline**. Missing or unreadable metadata shows “Not available” and never affects the model score. Metadata can also be changed, so its presence does not prove authenticity.

This is a local web application: Python runs a Streamlit server on your PC and your browser displays it. It is not automatically published to the internet.

## Host the website without changing Windows security

The app is prepared for **Streamlit Community Cloud**: the website and CPU model run together on a hosted Linux server. See [DEPLOYMENT.md](DEPLOYMENT.md) for GitHub upload and deployment steps. `cloud-upload.zip` contains the minimal deployment files. Hosting is not yet completed or verified; it requires your GitHub/Streamlit accounts.

Cloud hosting uses the full `requirements.txt`. On Linux, the first analysis downloads the pinned model automatically; subsequent analyses reuse the cached model. Uploaded images are processed on the host. On local Windows, model downloads remain an explicit setup step and the existing Windows runtime block remains unresolved. Cloud hosting does not implement local edit detection or improve model accuracy.

## Run pixel analysis in Google Colab (recommended for this laptop)

Windows Application Control currently blocks the local NumPy runtime. The standalone notebook `image_verification_colab.ipynb` runs in Google's hosted runtime without changing Windows security settings. It uses the same experimental detector as the Streamlit app; it does not connect to or repair the local website.

1. Open [Google Colab](https://colab.research.google.com/) and sign in.
2. Choose **File → Upload notebook**, then select `image_verification_colab.ipynb` from this project folder.
3. Connect to a **hosted Python runtime**; CPU is sufficient. Do not select a local runtime.
4. Run the code cells from top to bottom using their play buttons. Wait for package installation and model download to finish.
5. In the upload cell, select an original JPEG phone photo or PNG. It will show a preview and an experimental AI-class score. Try a known generated image separately for comparison. Neither result establishes accuracy.
6. Optionally run the final code cell to download `analysis_results.json` with scores and model/runtime versions.

Images are uploaded to Google's runtime. The notebook does not mount Drive or send images to Hugging Face; Hugging Face supplies model files only. Images, filenames, and results can remain in notebook outputs: clear outputs before sharing and disconnect/delete the runtime when finished. Runtime files are temporary and may need uploading again. See the [Colab FAQ](https://research.google.com/colaboratory/faq.html) and [file upload documentation](https://colab.research.google.com/notebooks/io.ipynb).

The notebook's code cells were syntax-checked, and upload/invalid-file behavior was checked with mocked inference locally. An actual Colab model run has not been verified from this workspace. No detection accuracy is claimed. Edit localization is still explicitly not implemented / inconclusive.

## Pixel-analysis setup

After the base setup below, run these PowerShell commands from the project folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-analysis.txt
.\.venv\Scripts\python.exe detector.py --download
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

The download needs internet access, several hundred MB of disk space, and may take a few minutes. Weights are stored in `.models/` inside this project. Subsequent analysis loads them locally and runs on CPU; uploaded images are not sent to Hugging Face. If Streamlit is already running, stop it with Ctrl+C and restart after installing dependencies.

### Repairing analysis imports

An import error does not always mean a package is missing. On this PC, tracing the `NP_SUPPORTED_MODULES` failure revealed that Windows Application Control blocked NumPy's `_pocketfft_umath` extension. PyTorch swallowed that error before raising the misleading internal import error. The detector now checks `numpy.fft` first to expose the real failure directly. Runtime package replacement alone did not resolve this policy block; real inference and weights setup remain unverified until the runtime is permitted.

Analysis dependencies pin official CPU `torch==2.10.0+cpu` alongside Transformers 4.57.6. This model uses the Pillow-based processor (`use_fast=False`) and does not need torchvision. An installed, failing torchvision can still break Transformers imports, so remove it from this project's environment. The CPU build is provided by [PyTorch's official index](https://pytorch.org/get-started/previous-versions/).

Stop Streamlit before repairing an existing environment, then run:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torchvision
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-analysis.txt
.\.venv\Scripts\python.exe detector.py --download
```

The app's diagnostic expander shows the actual exception and Python executable. If Windows reports Application Control (`WinError 4551`), ask your administrator whether this runtime is permitted or use an approved Python environment; do not disable security controls. A missing local model requires the download command, whereas an internal import failure requires repairing the runtime.

Upload a photo and click **Analyze image pixels**. The app decodes the original image independently of the smaller preview, applies orientation, converts it to RGB, and passes a metadata-free copy through the model's standard image processor. The model resizes/crops its input, so small regions can be lost. Metadata does not enter inference.

### Model and interpretation

Baseline: [Organika/sdxl-detector](https://huggingface.co/Organika/sdxl-detector), pinned revision `b37fede`, a pretrained Swin classifier fine-tuned on Wikimedia/SDXL pairs. Attribution: Organika / Colby / umm-maybe, as described in the linked model card. The model is listed as CC-BY-NC-3.0 and intended for educational/noncommercial use; review its terms before other uses. Loading uses safetensors with remote custom code disabled.

The displayed score is the model's `artificial` softmax output, **not a calibrated probability that an image is fake**. Exploratory bands are ≤0.10 camera-photo-like, ≥0.90 AI-like, and otherwise inconclusive. These bands have not been validated on your data; the overall authenticity conclusion stays inconclusive. A low score cannot prove camera capture, and a high score cannot prove generation. Screenshots, illustrations, unfamiliar generators, edits, and recompression can mislead it. This is a reproducible baseline, not a claim of a best-performing detector.

The second question, local editing, explicitly stays **inconclusive / not implemented**. This classifier does not locate edits. TruFor integration and C2PA checking remain separate planned work.

### Data science evaluation before trusting labels

Collect labeled original camera photos and generated images from several generators, recording source and generator. Define local edits separately, keeping before/after images and pixel masks for later localization. Keep licensing and consent appropriate for your dataset.

Split train/validation/test by source image and near-duplicate family, rather than randomly placing copies or edited versions across splits. Include unseen generators in the test set. Tune thresholds only on validation data, then freeze them before testing. Evaluate clean images and JPEG-recompressed/resized variants, grouped with their originals to prevent leakage.

Report a confusion matrix, AI-class precision/recall/F1, false-positive rate on camera photos, ROC-AUC, and results per generator/source. For abstaining predictions, report coverage (fraction receiving an indication) and accuracy on that fraction, plus the inconclusive rate. Assess calibration before describing scores as probabilities. Running the app on one phone photo is a functionality check, not a scientific accuracy evaluation.

Record the model revision, dependency versions, dataset sources/splits, preprocessing, and thresholds with each experiment. No accuracy is claimed for your dataset yet.

Run the regression checks (these use mocked inference and do not establish detection accuracy):

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

## Set up on Windows

Install Python from python.org if needed (include the Python launcher). Open **PowerShell** and run these commands one line at a time:

```powershell
cd "C:\Users\Hasini\OneDrive\Desktop\AASL\image-verification"
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-basic.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

No virtual environment activation or PowerShell execution-policy change is needed. Streamlit opens a browser; if needed, visit `http://localhost:8501`. Stop the app with **Ctrl+C** in PowerShell.

To run again later:

```powershell
cd "C:\Users\Hasini\OneDrive\Desktop\AASL\image-verification"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Test with a phone photo

1. Take a photo with your phone. Choose JPEG / “Most Compatible” in camera settings if it normally saves HEIC; HEIC is not supported in this milestone.
2. Copy the original JPEG to your PC using USB or a transfer method that preserves the original file. Messaging apps and screenshots may remove camera metadata.
3. Upload it. Check that the preview, JPEG format, and pixel dimensions appear. Camera make/model should appear if the file contains readable EXIF camera fields; “Not available” is also a valid result.
4. Try a PNG screenshot: the preview and dimensions should work even without camera metadata.
5. For an invalid-file check, make a text file containing `not an image`, rename the copy to `invalid.jpg`, and upload it. Expect a friendly error instead of a crash.

The image is processed in memory by this app; the app does not explicitly save uploads or send them to an external detection service. Only the initial model download contacts Hugging Face. When run locally, keep the server local and stop it when finished.

## Later milestones (planned, not implemented)

Keep two separate questions: (1) does the image appear camera-captured or fully AI-generated, and (2) is there evidence that part of it was edited? A camera photo can also be edited. Each assessment must allow **inconclusive** when evidence is insufficient.

- **PyTorch baseline implemented:** evaluate the initial SDXL detector and compare newer models using the same held-out dataset, checking licensing, Windows support, hardware needs, and performance on representative phone, generated, compressed, and resized images.
- Evaluate **TruFor** for locating possible edits. Check installation requirements, model weights/licensing, and localization quality before integration. A suspected region should be presented as evidence with limitations, rather than proof.
- Evaluate **C2PA** for checking signed image history and validating provenance. Report validation results separately from model assessments. Missing credentials do not imply an image is AI-generated or edited, and valid credentials do not establish that the depicted scene is true.

Use representative evaluation data to establish thresholds and an inconclusive outcome before presenting detection claims. Do not infer authenticity from missing metadata.
