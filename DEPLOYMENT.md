# Deployment

Host the Streamlit app from GitHub repository itsWathsa/image-verification, branch main, entrypoint app.py, using Python 3.11.

Include app.py, style.css, conclusions.py, detector.py, edit_detector.py, reference_compare.py, requirements.txt, .streamlit/config.toml, and the complete trufor_vendor/ directory with licenses. Exclude private inputs/results, virtual environments, and model caches.

After a push, watch Streamlit Community Cloud build logs and refresh the site. Confirm tabs AI detection, Edit detection, Compare original, and File details. Run each analysis, download a report, then confirm results persist. Replacing the primary image must clear all results; replacing only the reference clears its comparison.

The generation model downloads about 30 MB, and TruFor downloads about 261 MB on first use. CPU time, memory, and first-run delays depend on the host. No latency or uptime guarantee is made. Model downloads can recur after a restart.

The GitHub Actions regression workflow uses Linux Python 3.11. Passing tests do not establish detection accuracy or confirm the live deployment. Verify the interface in a browser after rebuilding.

The application provides automated estimates that may be incorrect. It does not confirm that an image is authentic or unedited. No paid hosting plan is configured.
