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

    # Hosted Linux can fetch weights on first analysis; local Windows remains
    # offline unless the user explicitly runs detector.py --download.
    return load_detector(download=sys.platform == "linux")


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
        message = f"A required Python module is unavailable: {error.name}. Install requirements-analysis.txt using this app's Python."
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
    st.write("Analyze pixels using an educational SDXL detector. Camera metadata is not used in the prediction.")
    st.caption("This baseline was trained on Wikimedia/SDXL pairs. Other generators, screenshots, illustrations, resizing, and compression can produce misleading results. The first analysis downloads model weights and can take several minutes.")
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
                             "authenticity": "inconclusive", "local_editing": "not implemented"})
            st.download_button("Download model score (CSV)", data=report.getvalue(), file_name="image_score.csv", mime="text/csv")
            st.caption("0 = model's human class; 1 = model's artificial class. This is not a calibrated probability of authenticity. Exploratory bands: ≤0.10 camera-photo-like, ≥0.90 AI-like, otherwise inconclusive.")
            st.warning("Authenticity conclusion: inconclusive. Initial checks found both high AI scores on camera photos and low scores on generated images. This baseline is not reliable enough for authenticity decisions. A low score does not establish camera origin or absence of edits.")
        except Exception as error:
            show_analysis_error(error)

    st.subheader("2. Evidence of local editing?")
    st.write("**Inconclusive — edit localization is not yet implemented.** The AI-generation score does not answer this question.")
    st.caption("Planned: evaluate TruFor for suspicious regions and its reliability map. C2PA signed history will be a separate provenance check.")
    st.subheader("File details")
    st.write(f"**Format:** {image_format}")
    st.write(f"**Dimensions:** {dimensions[0]} × {dimensions[1]} pixels (stored image)")
    st.write(f"**Camera make:** {make}")
    st.write(f"**Camera model:** {model}")
    st.caption("Camera fields are read from EXIF metadata when available. Metadata may be absent, removed, or changed; it does not prove authenticity.")


if __name__ == "__main__":
    main()
