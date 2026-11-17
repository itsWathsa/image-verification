"""Image details and an experimental pixel-based AI-generation baseline."""

import io
import csv
import sys
import warnings

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


def camera_details(image):
    """Return optional EXIF camera fields; unreadable metadata is not evidence."""
    try:
        exif = image.getexif()
        values = []
        for tag in (271, 272):  # EXIF Make and Model
            value = exif.get(tag)
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            values.append(str(value).strip().strip("\x00") if value else "Not available")
        return values[0] or "Not available", values[1] or "Not available"
    except Exception:
        # Some otherwise readable images contain malformed EXIF data.
        return "Not available", "Not available"


def read_image(data, full_resolution=False):
    """Fully decode an allowed image and detach its preview from the input file."""
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("Please choose an image smaller than 20 MB.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("The file contents must be JPEG or PNG.")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Please resize this image to 20 megapixels or smaller before uploading.")
            image.load()
            image_format = image.format
            dimensions = image.size
            make, model = camera_details(image)
            try:
                preview = ImageOps.exif_transpose(image).convert("RGB" if image.mode != "RGBA" else "RGBA")
            except Exception:
                preview = image.convert("RGB")
            if not full_resolution:
                preview.thumbnail((1600, 1600))
            return preview, image_format, dimensions, make, model


@st.cache_resource
def get_detector():
    from detector import load_detector

    return load_detector(download=True)


@st.cache_resource
def get_edit_detector():
    from edit_detector import load_edit_detector

    return load_edit_detector(download=True)


def show_analysis_error(error):
    """Distinguish absent packages from installed packages that cannot import."""
    chain = []
    current = error
    while current is not None and id(current) not in {id(item) for item in chain}:
        chain.append(current)
        current = current.__cause__ or current.__context__
    if any(getattr(item, "winerror", None) == 4551 or "Application Control" in str(item) for item in chain):
        message = "Windows Application Control blocked the model runtime. See README.md for administrator guidance."
    elif isinstance(error, ModuleNotFoundError) and error.name and len(chain) == 1:
        message = f"A required Python module is unavailable: {error.name}. Install requirements.txt using this app's Python."
    elif isinstance(error, ImportError):
        message = "The installed analysis packages could not load. This can be a version compatibility problem; it does not mean packages are missing. See the repair commands in README.md."
    elif isinstance(error, OSError):
        message = "The model or runtime could not load. Complete model setup in README.md and check the diagnostic details below."
    else:
        message = "The model could not complete analysis. Check the diagnostic details below and model setup in README.md."
    st.error(f"Analysis unavailable — result: inconclusive. {message}")
    with st.expander("Analysis diagnostic details"):
        st.code(f"Python: {sys.executable}\n{type(error).__name__}: {error}", language="text")


def main():
    st.set_page_config(page_title="Image verification lab", page_icon="📷")
    st.title("Image verification lab")
    st.write("Preview a JPEG or PNG and explore pixel-based AI-generation evidence.")
    st.caption("Images are processed on the server running this app. On a hosted website, uploads leave your device. The app does not explicitly save images or send them to an external inference API.")
    st.info(
        "Experimental analysis cannot prove an image is legitimate. AI generation "
        "and local editing are separate questions. Missing metadata is not evidence "
        "of either. A camera photo can also be edited."
    )
    uploaded = st.file_uploader("Choose a JPEG or PNG (up to 20 MB)", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return
    try:
        preview, image_format, dimensions, make, model = read_image(uploaded.getvalue())
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        st.error("This image is too large to decode safely. Please upload a smaller image.")
        return
    except ValueError as error:
        st.error(str(error))
        return
    except (UnidentifiedImageError, OSError, SyntaxError):
        st.error("This file could not be read as an image. It may be invalid, damaged, or incomplete. Please try another JPEG or PNG.")
        return
    except Exception:
        st.error("This image could not be processed. Please try another JPEG or PNG.")
        return

    st.image(preview, caption="Preview (rotated using orientation metadata when available)")
    st.subheader("1. Camera-captured or fully AI-generated?")
    st.write("Analyze pixels using the experimental Detectra-OpenFake detector. Camera metadata is not used in the prediction.")
    st.caption("This detector covers multiple generators. Screenshots, illustrations, unfamiliar generators, resizing, and compression can still mislead it. The first analysis downloads about 30 MB of model weights.")
    st.caption("Model: s-gehring/detectra-openfake ? revision c84565e ? educational/noncommercial use (CC-BY-NC-4.0). Single-center-crop inference; no multi-crop or author score scaling.")
    if st.button("Analyze image pixels", type="primary"):
        try:
            from detector import analyze_pixels, MODEL_ID, MODEL_REVISION

            with st.spinner("Analyzing image pixels on CPU..."):
                full_image, *_ = read_image(uploaded.getvalue(), full_resolution=True)
                result = analyze_pixels(full_image, get_detector())
            st.write(f"**Model indication:** {result['signal']}")
            st.metric("AI-class model score", f"{result['ai_score']:.3f}")
            report = io.StringIO(newline="")
            writer = csv.DictWriter(report, fieldnames=["ai_score", "signal", "model", "revision", "authenticity", "local_editing"])
            writer.writeheader()
            writer.writerow({**result, "model": MODEL_ID, "revision": MODEL_REVISION,
                             "authenticity": "inconclusive", "local_editing": "not assessed in generation analysis"})
            st.download_button("Download model score (CSV)", data=report.getvalue(), file_name="image_score.csv", mime="text/csv")
            st.caption("Raw single-crop sigmoid score: low values favor camera-photo-like patterns; high values favor AI-like patterns. This is not a calibrated probability of authenticity. Exploratory bands: ≤0.10 camera-photo-like, ≥0.90 AI-like, otherwise inconclusive.")
            st.warning("Authenticity conclusion: inconclusive. This detector matched six known diagnostic examples, but has not been validated on an independent target dataset. A low score does not establish camera origin or absence of edits.")
        except Exception as error:
            show_analysis_error(error)

    st.subheader("2. Evidence of local editing?")
    st.write("TruFor looks for pixel inconsistencies and produces an anomaly map and a separate reliability map. The AI-generation score is not used here.")
    st.caption("CPU preview analyzes at most 384 pixels on the longest edge. Resizing may erase small edits or create misleading traces. First use downloads about 261 MB of weights; analysis can be slow.")
    st.caption("TruFor by Guillaro et al., GRIP-UNINA (CVPR 2023). Informational/nonprofit use only. A suspicious region is not proof of editing.")
    if st.button("Analyze possible edits", key="analyze_edits"):
        try:
            from edit_detector import analyze_edits, render_maps, SOURCE_REVISION

            with st.spinner("Loading TruFor and examining pixel inconsistencies on CPU..."):
                full_image, *_ = read_image(uploaded.getvalue(), full_resolution=True)
                edit_result = analyze_edits(full_image, get_edit_detector())
            overlay, reliability = render_maps(edit_result)
            st.metric("TruFor whole-image anomaly score", f"{edit_result['score']:.3f}")
            st.image(overlay, caption="Anomaly overlay: red = higher anomaly, blue = lower anomaly. Colors use a fixed 0–1 scale.")
            st.image(reliability, caption="Reliability map: brighter = greater predicted reliability. Dark regions need more caution.")
            st.caption(f"Analyzed size: {edit_result['analyzed_size'][0]} × {edit_result['analyzed_size'][1]} pixels. Maps correspond to this resized image, not original pixel resolution.")
            st.warning("Edit assessment: inconclusive. These are experimental model outputs without validated decision thresholds. Compression, resizing, fully generated images, and ordinary photo processing can cause anomalies. A low anomaly score does not prove absence of edits.")
            report = io.StringIO(newline="")
            writer = csv.DictWriter(report, fieldnames=["model", "source_revision", "anomaly_score", "analyzed_width", "analyzed_height", "edit_assessment"])
            writer.writeheader()
            writer.writerow({"model": "TruFor", "source_revision": SOURCE_REVISION,
                             "anomaly_score": edit_result["score"], "analyzed_width": edit_result["analyzed_size"][0],
                             "analyzed_height": edit_result["analyzed_size"][1], "edit_assessment": "inconclusive"})
            st.download_button("Download edit analysis (CSV)", data=report.getvalue(), file_name="edit_analysis.csv", mime="text/csv")
            image_buffer = io.BytesIO()
            overlay.save(image_buffer, format="PNG")
            st.download_button("Download anomaly overlay", data=image_buffer.getvalue(), file_name="anomaly_overlay.png", mime="image/png")
        except Exception as error:
            st.error("Edit analysis unavailable — result: inconclusive. The generation detector remains separate.")
            with st.expander("Edit analysis diagnostic details"):
                st.code(f"{type(error).__name__}: {error}", language="text")
    st.caption("C2PA signed history remains a separate planned provenance check.")
    st.subheader("File details")
    st.write(f"**Format:** {image_format}")
    st.write(f"**Dimensions:** {dimensions[0]} × {dimensions[1]} pixels (stored image)")
    st.write(f"**Camera make:** {make}")
    st.write(f"**Camera model:** {model}")
    st.caption("Camera fields are read from EXIF metadata when available. Metadata may be absent, removed, or changed; it does not prove authenticity.")


if __name__ == "__main__":
    main()
