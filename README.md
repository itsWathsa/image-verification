# Image verification lab

A Streamlit web app for inspecting image evidence. Users of the hosted website only need a browser. VS Code is for development; Colab is not required.

## Four independent checks

| Tab | What it does | What it cannot establish |
|---|---|---|
| AI evidence | Detectra-OpenFake scores AI-like pixel patterns | Camera origin, fully generated versus AI-edited, or authenticity |
| Possible edits | TruFor displays anomaly and reliability maps | Proof of editing or proof an image is unedited |
| Compare original | Aligns a supplied reference and highlights pixel differences | The reference's authenticity, editing intent, or an exact edited-area percentage |
| Signed history | C2PA checks embedded signature and asset binding; signer trust is separate | Truth of the scene or completeness of recorded history |

File format, stored dimensions, and optional camera EXIF fields are under **File details**. Missing EXIF and missing Content Credentials never produce an AI or edited label. A camera photo can also be edited. Overall authenticity stays inconclusive; there is no single combined fake/real verdict.

Scores use six decimal places to avoid showing a value close to one as 100% certainty. Results survive other checks, panel changes, and downloads. Replacing the main image clears all prior results; replacing the reference clears only its comparison. Results are held for the latest image in the current session, not cached globally per uploaded image. The combined JSON report records completed, unrun, and failed checks separately.

## Run on Windows

Open PowerShell and run one command per line:

```powershell
cd "C:\Users\Hasini\OneDrive\Desktop\AASL\image-verification"
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

No activation or PowerShell execution-policy change is needed. Visit `http://localhost:8501` if the browser does not open. Stop with Ctrl+C. To run later, repeat the `cd` and final command. `requirements-basic.txt` supports preview/file details without the model runtimes; missing optional dependencies produce an unavailable/inconclusive result in the affected panel.

The first generation check downloads about 30 MB; the first edit check downloads approximately 261 MB. Model weights are cached in `.models/`, not committed to Git. TruFor uses CPU PyTorch; the generation check uses ONNX Runtime. Native C2PA is installed with the full requirements. An initial Windows DLL block was observed, but subsequent native verification succeeded without a security-setting change. If a runtime cannot load, that panel reports unavailable while other checks remain usable. Real tests validated the official signed fixture, rejected a tampered copy with a hash mismatch, and handled an unsigned JPEG. The Linux native-provenance workflow independently repeats these checks; inspect its actual result.

## Test with a phone photo

Copy an original JPEG from your phone using USB or another method that preserves the file. HEIC is unsupported. Upload it, preview it, and run each check independently. Camera fields may be unavailable. Try a PNG screenshot and invalid text renamed `.jpg` to check graceful errors.

For the bottle-removal example, upload the Gemini version as the main image and `raw1.jpg` as its reference in **Compare original**. Expect broad differences, not a guaranteed outline of only the removed bottle. See [REFERENCE_COMPARISON.md](REFERENCE_COMPARISON.md).

## Validation and limits

```powershell
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe evaluate.py evaluation-private\reported_scores.csv --output evaluation-private\baseline_metrics.json
```

`evaluate.py` accepts `filename,label,ai_score` with labels `camera` or `generated`; these are reference labels, not detector inputs. Do not put mixed/AI-edited photos into a fully-generated-versus-camera benchmark without defining the target carefully. Record edit presence separately and use masks for localization evaluation. Keep originals and their derivatives together in dataset splits. Thresholds have not been tuned on the six known examples.

The ONNX detector reproduced the six supplied diagnostic labels, but that does not establish general accuracy. TruFor missed the known Gemini bottle-removal edit (edited score 0.090 versus original 0.440), so low anomaly responses must never be called unedited. Reference comparison adds a separate source of evidence; it does not fix that detector failure. New independent labeled images and edited pairs are needed to evaluate errors, inconclusive rate, and localization performance.

Current regression checks cover input handling, score mapping, independent failures, session/result invalidation, reference alignment, and C2PA validation-report semantics. Real ONNX/TruFor inference has run locally; the user also confirmed cloud outputs for both. Native C2PA signed/tampered/unsigned fixture tests also passed locally; cloud execution still requires verification after this update. Regression tests and the small diagnostic sample are not scientific accuracy validation.

## Hosting, data, and licenses

See [DEPLOYMENT.md](DEPLOYMENT.md). Source: https://github.com/itsWathsa/image-verification. Hosted app: https://image-verification-ki8pr7csrq4zmj979arh8h.streamlit.app/.

Uploads are processed on the server running the app. The application does not explicitly save uploaded photos or call an external image inference API. Model files are downloaded from Hugging Face and the official TruFor host. C2PA reads original uploaded bytes and disables remote manifests, online revocation queries, and other network requests. Reports may contain embedded signer statements; treat those as claims rather than facts. Clear downloads/outputs before sharing private information. Upload limits are 20 MB and 20 megapixels.

`data/`, `evaluation-private/`, the original test-results text, virtual environments, and model caches are excluded from Git. Reference comparison uses OpenCV. Detectra-OpenFake weights are CC-BY-NC-4.0 (s-gehring/OpenFake/TensorFold/Community Forensics attribution in the model card). TruFor is informational/nonprofit under its retained upstream licenses. This is an educational prototype.

Implementation details: [TRUFOR.md](TRUFOR.md), [PROVENANCE.md](PROVENANCE.md), [REFERENCE_COMPARISON.md](REFERENCE_COMPARISON.md).
