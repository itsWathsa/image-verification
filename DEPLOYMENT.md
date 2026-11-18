# Deployment on Streamlit Community Cloud

The app and models run together on a Linux host; the user's browser does not need Python, VS Code, or Colab. No paid service is configured by this project.

Repository: https://github.com/itsWathsa/image-verification (branch `main`). Hosted app: https://image-verification-ki8pr7csrq4zmj979arh8h.streamlit.app/.

## Deploy or update

1. Commit the source to the repository. Include `app.py`, `detector.py`, `edit_detector.py`, `reference_compare.py`, `provenance.py`, full `trufor_vendor/` with licenses, `requirements.txt`, and `.streamlit/config.toml`. Never upload `.venv`, `.models`, private photos/results, or secrets.
2. In https://share.streamlit.io/, select the repository, branch `main`, and main file `app.py`. Use Python 3.11 in Advanced settings. No model access token or other secrets are needed for this configuration.
3. Deploy, or let the existing app rebuild when GitHub changes. Watch build logs for dependencies or memory errors. Verify visitor access using Streamlit's Share/access settings; source-repository visibility does not itself determine app access.
4. Confirm tabs **AI evidence**, **Possible edits**, **Compare original**, **Signed history**, and **File details** appear. These checks run only when their buttons are clicked.

Full dependencies include CPU PyTorch (TruFor), ONNX Runtime (generation), OpenCV (reference alignment), and the native C2PA SDK (signed history). Generation weights are about 30 MB; the official TruFor archive is approximately 261 MB. Models download on first use and can download again after a restart/rebuild. TruFor's 384-pixel analysis cap limits memory but may hide small edits. Free-host capacity and concurrency are limited; a process crash or failed download is an unavailable/inconclusive check, not a judgment about the image.

## Verify the update

- Upload an original JPEG and known generated PNG. Run generation and edit checks independently. Verify both results remain after reruns or downloads and that changing the main image clears them.
- Use **Compare original** with a known edited pair. Check alignment diagnostics and gray excluded borders; changing only the reference must clear only that comparison.
- Check **Signed history** using signed, unsigned, and modified signed files. Missing credentials must never imply AI or editing. Native C2PA may be unavailable on this Windows PC due to its DLL policy; the Linux host still needs its own smoke test.
- The repository workflow **Native C2PA verification** tests a checksum-pinned official signed JPEG, a decodable hash-mismatched copy, and an unsigned image on Linux. It does not use uploaded user photos. Check the workflow's actual outcome and report artifact.
- Upload invalid `.jpg` bytes and a PNG without EXIF. Expect friendly errors or unavailable metadata, without authenticity labels.
- If a check fails, inspect its diagnostic details and the Cloud logs. Do not disable Windows security or upgrade to paid hosting just to make a demonstration run without considering alternatives and a budget.

The user confirmed ONNX and TruFor outputs on the prior cloud version. The polished UI, reference mode, and native C2PA need verification after this deployment. Local tests do not establish cloud availability or forensic accuracy. Uploads stay within the hosting/session infrastructure; the app does not explicitly write photos to disk or send them to external model inference services. C2PA network fetching is disabled.

See README.md for interpretation and the separate method/license documentation. The model response is not a calibrated probability, signed claims are not truth, and reference differences are not an exact edit map.
