# Experimental edit localization

The second app question now has its own **Analyze possible edits** button. It runs TruFor independently of Detectra-OpenFake and displays:

- A whole-image anomaly score, not an authenticity probability.
- A red/blue anomaly overlay using a fixed 0–1 color scale.
- A reliability map: bright is higher predicted reliability; dark needs caution.
- CSV score and plain anomaly/reliability PNG downloads, plus an optional on-screen overlay.

The application displays qualified assessments; no edit/no-edit threshold has been independently validated. A camera photo can be edited, a generated image can produce anomalies, and compression or normal image processing can mislead the model. A low score does not establish absence of edits.

## Runtime and preprocessing

Install `requirements.txt` and run Streamlit as described in README.md. The first edit analysis downloads the approximately 261 MB official weights ZIP and checks its published MD5 (`7bee48f3476c75616c3c5721ab256ff8`). The model is cached in `.models/trufor/`; it is not uploaded to GitHub. You can pre-download and check loading with:

```powershell
.\.venv\Scripts\python.exe edit_detector.py
```

The adapter uses CPU PyTorch 2.10, strict checkpoint loading, and weights-only deserialization with narrowly scoped NumPy scalar/dtype constructors needed by the original checkpoint. It does not use unrestricted checkpoint unpickling. Inference is serialized within the process to limit peak concurrent memory.

To limit memory on the demo host, images are proportionally reduced to a maximum edge of **384 pixels** before analysis. This is an explicit deviation from native-resolution inference and can erase small edit traces or introduce resizing artifacts. Maps apply to the analyzed dimensions shown in the app. Input is RGB float32 divided by **256**, matching the official test dataset, followed by the model's internal ImageNet preprocessing. Anomaly values use softmax class 1; reliability and whole-image outputs use sigmoid. Maps are not dynamically rescaled to make weak signals appear strong.

The inference architecture is vendored under `trufor_vendor/`, from official source revision `ae54475df6f41a491d7615100feb19263dec13f7`. Only imports and inference layer helpers were adapted. See `trufor_vendor/SOURCE.md` and retained license files. The timm training dependency is replaced with evaluation-only DropPath (identity during evaluation), tuple conversion, and PyTorch truncated-normal initialization; no topology or forward attention layers were replaced.

## Result interpretation

The interface uses exploratory presentation bands: <=0.10 no strong editing evidence detected; >=0.90 possible editing detected; otherwise unable to determine. These are not calibrated or independently validated edit/no-edit thresholds. Outputs may be incorrect. A low score must not be presented as confirmed unedited.

Independent evaluation with untouched originals and known edit masks is required. Measure false positives, missed edits, localization precision/recall/IoU and sensitivity to the resolution cap. Do not tune thresholds to already-inspected cases.

## Attribution and license

TruFor: Fabrizio Guillaro, Davide Cozzolino, Avneesh Sud, Nicholas Dufour, Luisa Verdoliva, CVPR 2023. GRIP-UNINA's license permits informational/nonprofit use and requires retained notices and attribution; review `trufor_vendor/LICENSE.txt` and `LICENSE_CMX.txt`. This project is an educational prototype.

Sources: [official repository](https://github.com/grip-unina/TruFor), [official inference instructions](https://github.com/grip-unina/TruFor/blob/main/TruFor_train_test/README.md), [paper](https://arxiv.org/abs/2212.10957).
