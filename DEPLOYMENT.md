# Deploy the web app on Streamlit Community Cloud

The website and CPU model run together on Streamlit's Linux server. Windows Smart App Control stays enabled on your laptop. Colab is not involved in the deployed app.

Streamlit Community Cloud is a free hosting route requiring a GitHub account and a Streamlit Community Cloud account. Check the current hosting terms and limits when creating the app; do not select a paid service for these steps. See [official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app).

## Upload the prepared source

1. Sign in to [GitHub](https://github.com/) and create a repository named `image-verification`. Choose your intended source visibility. A public repository exposes its uploaded code and files, so use only the prepared files below.
2. Open the repository and select **Add file → Upload files**. Upload these root files: `app.py`, `detector.py`, `requirements.txt`, and `README.md`. Commit the upload.
3. Create `.streamlit/config.toml` using **Add file → Create new file**, paste the contents of the local file with that name, and commit it. The folder is created automatically by the slash in its name.

For convenience, `cloud-upload.zip` contains only the deployment files in their correct layout. Extract it on your PC, then upload its contents, **not the ZIP itself**. GitHub's browser upload may omit hidden folders, so confirm `.streamlit/config.toml` exists using step 3.

Do not upload `.venv`, `.models`, photos, tokens, or secrets. The model is downloaded from Hugging Face by the deployed app when first needed. No Hugging Face token is required for this public model.

## Deploy

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub. Grant access to the intended repository.
2. Select **Create app**, then select the repository, branch (usually `main`), and main file **`app.py`**.
3. Under **Advanced settings**, choose **Python 3.11**. No secrets are needed.
4. Deploy and watch the build logs. CPU PyTorch and Transformers are installed from `requirements.txt`; installation can take several minutes.
5. When the app opens, upload a JPEG/PNG and click **Analyze image pixels**. The first click downloads the pinned model weights and loads the model. Wait for it to finish. Later requests reuse the cached model while the process is alive.
6. Use the app's Share/access settings to choose who can visit it. Verify the audience before sharing its URL. A private source repository does not itself establish the app's visitor permissions.

## Check the deployed app

- Upload an original JPEG phone photo and a known generated image. Both should show a preview and a score. No particular score is guaranteed.
- Upload a PNG without EXIF. Its score must not depend on missing camera metadata.
- Upload an invalid `.jpg`; expect a friendly error.
- Confirm local editing still reads **not implemented / inconclusive**.
- If analysis fails, check Cloud logs and the diagnostic expander. A memory-related process restart is different from an invalid image or missing dependency.

This deployment has not yet been built or tested on Community Cloud. Local regression tests mock model inference; the successful Colab run does not establish hosting capacity or detection accuracy. This is a CPU demonstration: concurrent analyses and large images can exhaust free-host resources. Uploads are capped at 20 MB and 20 megapixels. If Cloud cannot accommodate the model, keep using Colab rather than upgrading to a paid plan without deciding on a budget.

Uploads are processed on the hosting provider's server. The application does not explicitly write images to disk, call a remote image inference service, or maintain an image database. Hosting/session infrastructure still handles the uploaded data. Only upload images suitable for that environment. Model files may be downloaded again after the app sleeps, restarts, or rebuilds.

Model: [Organika/sdxl-detector](https://huggingface.co/Organika/sdxl-detector), revision `b37fede`, educational/noncommercial use. Scores are experimental; overall authenticity remains inconclusive until target-data evaluation. Hosting does not improve model accuracy.
