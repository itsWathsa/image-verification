# Deployment

Install `requirements.txt` and run `streamlit run app.py` with Python 3.11 or newer. Include the runtime files listed in README.md, `.streamlit/config.toml`, and `style.css`. Exclude local photos, virtual environments, and model caches.

The `deployment-repo/` checkout retains the deployment Git history. Its application files mirror the cleaned workspace; review and commit changes there before deploying.

The generation model downloads about 30 MB on first use. Model downloads can recur after a host restart. Runtime performance depends on the host.

After deployment, confirm the AI detection, Compare original and File details tabs. Run both analyses and download a JSON report. Confirm results persist across reruns, replacing the main image clears all results, and replacing the reference clears its comparison.

Video support also requires `video_analysis.py`. The Streamlit upload ceiling is 100 MB; image validation remains 20 MiB. Verify the Video selector, analysis, report download and reset on replacement. OpenCV supplies decoding; codec availability varies by host.
