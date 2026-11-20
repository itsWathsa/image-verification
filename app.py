"""Independent image evidence checks with per-upload session results."""

import csv
import hashlib
import io
import json
import warnings
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
CHECKS = ("generation", "reference")


def camera_details(image):
    try:
        exif = image.getexif()
        values = []
        for tag in (271, 272):
            value = exif.get(tag)
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            values.append(str(value).strip().strip("\x00") if value else "Not available")
        return values[0] or "Not available", values[1] or "Not available"
    except Exception:
        return "Not available", "Not available"


def read_image(data, full_resolution=False):
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
            image_format, dimensions = image.format, image.size
            make, model = camera_details(image)
            try:
                preview = ImageOps.exif_transpose(image).convert("RGBA" if image.mode == "RGBA" else "RGB")
            except Exception:
                preview = image.convert("RGB")
            if not full_resolution:
                preview.thumbnail((1200, 1200))
            return preview, image_format, dimensions, make, model


@st.cache_resource
def get_detector():
    from detector import load_detector
    return load_detector(download=True)


def reset_for_image(data):
    st.session_state.pop("edits", None)  # Discard results from the removed section.
    digest = hashlib.sha256(data).hexdigest() if data is not None else None
    if st.session_state.get("image_digest") != digest:
        for name in CHECKS:
            st.session_state.pop(name, None)
        st.session_state.pop("reference_digest", None)
        st.session_state["image_digest"] = digest


def reset_for_reference(data):
    digest = hashlib.sha256(data).hexdigest() if data is not None else None
    if st.session_state.get("reference_digest") != digest:
        st.session_state.pop("reference", None)
        st.session_state["reference_digest"] = digest


def error_message(error):
    chain, current = [], error
    while current is not None and id(current) not in {id(item) for item in chain}:
        chain.append(current)
        current = current.__cause__ or current.__context__
    if any(getattr(item, "winerror", None) == 4551 or "Application Control" in str(item) for item in chain):
        return "Windows Application Control blocked this runtime component. See README.md for guidance."
    if isinstance(error, ModuleNotFoundError) and error.name and len(chain) == 1:
        return f"A required Python module is unavailable: {error.name}. Install requirements.txt."
    if isinstance(error, ImportError):
        return "The installed analysis packages could not load. This can be a version compatibility problem. See README.md."
    return str(error) or "The check could not complete."


def run_check(name, operation):
    try:
        st.session_state[name] = {"ok": True, "result": operation()}
    except Exception as error:
        # Keep a small diagnostic, not a traceback retaining image/model memory.
        st.session_state[name] = {"ok": False, "error": error_message(error),
                                  "diagnostic": f"{type(error).__name__}: {error}"}


def saved_result(name):
    state = st.session_state.get(name)
    if not state:
        return None
    if not state["ok"]:
        st.error(f"Analysis unavailable - result: inconclusive. {state['error']}")
        with st.expander(f"{name.capitalize()} diagnostic details"):
            st.code(state["diagnostic"], language="text")
        return None
    return state["result"]


def png_bytes(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def csv_bytes(record):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(record))
    writer.writeheader()
    writer.writerow(record)
    return stream.getvalue()


def generation_panel(data):
    from detector import MODEL_ID, MODEL_REVISION, ANALYSIS_METHOD
    st.subheader("Was AI used to create this image?")
    st.write("Check for AI involvement, including fully generated images and camera photos modified using AI.")
    st.caption("Borderline results automatically receive closer inspection. Automated assessments may be incorrect.")
    if "generation" in st.session_state and st.session_state.get("generation_method") != ANALYSIS_METHOD:
        st.session_state.pop("generation", None)
    if st.button("Analyze image pixels", type="primary", key="analyze_generation"):
        st.session_state["generation_method"] = ANALYSIS_METHOD
        def analyze():
            from detector import analyze_pixels
            image, *_ = read_image(data, full_resolution=True)
            with st.spinner("Checking AI-like pixel patterns..."):
                return analyze_pixels(image, get_detector())
        run_check("generation", analyze)
    result = saved_result("generation")
    if result is not None:
        from conclusions import generation_conclusion
        conclusion = generation_conclusion(result["ai_score"])
        st.info(conclusion)
        with st.expander("Score details"):
            st.metric("AI detection score", f"{result['ai_score']:.6f}")
            st.caption("Model score, not a probability. Low scores do not rule out AI-assisted editing.")
            if len(result.get("crop_scores", [])) > 1:
                st.caption("A closer inspection was applied because the initial score was borderline. The final score is the highest of three center views.")
            if "crop_scores" in result:
                st.json({"method": result["analysis_method"], "initial_score": result["baseline_score"],
                         "views": result["crop_scores"]})
        st.download_button("Download model score (CSV)", csv_bytes({"ai_score": result["ai_score"], "model": MODEL_ID,
                           "revision": MODEL_REVISION, "analysis_method": result.get("analysis_method", "single-center-crop"),
                           "initial_score": result.get("baseline_score", result["ai_score"]),
                           "crop_scores": json.dumps(result.get("crop_scores", [])),
                           "conclusion": conclusion, "notice": "Automated estimate; may be incorrect"}),
                           file_name="image_score.csv", mime="text/csv", key="download_generation")
    with st.expander("Method and limitations"):
        st.write(f"Detectra-OpenFake | {MODEL_ID} | revision {MODEL_REVISION}. Educational/noncommercial CC-BY-NC-4.0.")
        st.write("Analysis starts with a 440-pixel shorter edge and a 384-pixel center crop. Initial raw scores from 0.50 up to 0.95 automatically trigger closer center views at 576 and 768 pixels; the highest score is retained. The same model is used for all views. Metadata is not used.")
        st.caption("Scores are raw sigmoid outputs, without the author's Platt scaling. Exploratory answer bands remain <=0.10 and >=0.90. Closer inspection is experimental and produced false alarms on ordinary brightness and contrast changes during development. An AI-use signal cannot distinguish a fully generated image from an AI-edited photo or locate the edit.")
        st.caption("First use downloads about 30 MB. Small edits, new generators, screenshots, compression and ordinary processing can still cause missed detections or false alarms.")


def reference_panel(data):
    st.subheader("Check brightness, contrast and colour changes")
    st.write("The main upload is the version to examine. Supply its original here to inspect aligned pixel differences.")
    st.caption("An original is needed to measure these adjustments reliably. The comparison measures differences from the supplied reference; it cannot verify that reference's authenticity.")
    if st.checkbox("I have an original/reference image", key="has_reference"):
        reference = st.file_uploader("Upload the original/reference JPEG or PNG", type=["jpg", "jpeg", "png"], key="reference_upload")
        reference_data = reference.getvalue() if reference is not None else None
        reset_for_reference(reference_data)
        if reference_data is not None:
            if st.button("Compare with original", key="compare_reference"):
                def compare():
                    from reference_compare import compare_reference
                    original, *_ = read_image(reference_data, full_resolution=True)
                    candidate, *_ = read_image(data, full_resolution=True)
                    with st.spinner("Aligning shared features and comparing pixels..."):
                        return compare_reference(original, candidate)
                run_check("reference", compare)
            result = saved_result("reference")
            if result is not None:
                tone = result.get("tonal_comparison", {})
                differences = (result["changed_fraction"] > 0 or tone.get("notable_tonal_difference", False)
                               or tone.get("notable_colour_difference", False))
                st.info("Differences detected compared with the supplied original." if differences else "No significant differences detected compared with the supplied original.")
                first, second = st.columns(2)
                first.image(result["aligned_reference"], caption="Aligned reference")
                second.image(result["candidate"], caption="Version being examined")
                st.image(result["difference"], caption="Difference map: brighter = greater pixel difference; gray = excluded borders.")
                with st.expander("View marked differences"):
                    st.image(result["overlay"], caption="Red marks differences under fixed inspection settings, not confirmed edits.")
                st.metric("Marked area within overlap (not an edit probability)", f"{result['changed_fraction']:.1%}")
                if tone:
                    st.write("**Brightness, contrast and colour comparison**")
                    first, second = st.columns(2)
                    first.metric("Average brightness difference", f"{tone['brightness_difference_levels']:+.1f} / 255")
                    ratio = tone["contrast_spread_ratio"]
                    second.metric("Contrast spread relative to original", f"{ratio:.2f}x" if ratio is not None else "Not measurable")
                    st.caption("Positive brightness values mean the examined image is brighter on average. A contrast ratio above 1 means a wider spread of light and dark tones. These measurements describe differences; they do not identify their cause.")
                    if "mean_colour_difference_levels" in tone:
                        st.metric("Average colour difference (brightness removed)", f"{tone['mean_colour_difference_levels']:.1f} levels")
                        st.caption("Colour difference measures changes in RGB balance after removing each pixel's brightness. Values of 2 levels or more contribute to the difference summary, even when the region map is unmarked. Saturation, colour casts, lighting and compression can contribute.")
                st.caption(f"Compared size: {result['analysis_size']}; overlap: {result['overlap_fraction']:.1%}.")
                if result["alignment"].get("spatial_support") == "limited":
                    st.warning("Alignment is supported by a limited part of the image. Extrapolation may contribute to the differences.")
                st.warning("Reference differences are not a calibrated editing estimate. Regeneration, lighting, compression and imperfect alignment may affect broad areas. No marked differences does not prove no edit occurred.")
                with st.expander("Alignment and inspection details"):
                    st.json({"alignment": result["alignment"], "inspection_settings": result.get("inspection_settings", {}),
                             "tonal_comparison": tone})
                st.download_button("Download reference difference overlay", png_bytes(result["overlay"]), file_name="reference_differences.png", mime="image/png")
    else:
        reset_for_reference(None)


def combined_report(image_format, dimensions):
    report = {"image_sha256": st.session_state["image_digest"], "format": image_format,
              "stored_dimensions": dimensions, "notice": "Automated assessments may be incorrect and are not guarantees of authenticity.", "checks": {}}
    for name in CHECKS:
        state = st.session_state.get(name)
        if not state:
            report["checks"][name] = {"status": "not_run"}
        elif not state["ok"]:
            report["checks"][name] = {"status": "unavailable", "diagnostic": state["diagnostic"]}
        else:
            result = dict(state["result"])
            from conclusions import generation_conclusion
            if name == "generation":
                result["conclusion"] = generation_conclusion(result["ai_score"])
                result.pop("signal", None)
            if name == "reference":
                result = {key: result[key] for key in ("changed_fraction", "overlap_fraction", "analysis_size", "alignment", "inspection_settings", "assessment", "tonal_comparison") if key in result}
            status = "completed"
            report["checks"][name] = {"status": status, "result": result}
    return report


def main():
    st.set_page_config(page_title="Image Verification", page_icon=":material/image_search:", layout="wide")
    css = Path(__file__).with_name("style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.markdown('''<section class="workspace-hero">
      <div class="workspace-label">DIGITAL IMAGE REVIEW</div>
      <h1>Image Verification</h1>
      <p>Assess AI involvement and compare source images in one review workspace.</p>
      <div class="hero-meta"><span>01 &nbsp; Upload an image</span><span>02 &nbsp; Review the evidence</span><span>03 &nbsp; Export your results</span></div>
    </section>
    <div class="feature-grid">
      <div class="feature-card"><span class="number">01 / AI DETECTION</span><h3>Assess AI involvement</h3><p>Review signals of AI generation or AI-assisted modification.</p></div>
      <div class="feature-card"><span class="number">02 / ORIGINAL COMPARISON</span><h3>Compare with the source</h3><p>Align an image with its original and inspect visual differences.</p></div>
    </div>''', unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("Upload image for review")
        st.caption("JPEG or PNG · up to 20 MB · up to 20 megapixels")
        uploaded = st.file_uploader("Choose the image to examine", type=["jpg", "jpeg", "png"])
    st.caption("Review guidance: automated assessments support human review. Results do not guarantee image authenticity.")
    if uploaded is None:
        reset_for_image(None)
        st.caption("Once uploaded, your preview and analysis tools will appear here. An original image is optional.")
        return
    data = uploaded.getvalue()
    reset_for_image(data)
    try:
        preview, image_format, dimensions, make, model = read_image(data)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        st.error("This image is too large to decode safely. Please choose a smaller image.")
        return
    except Exception as error:
        st.error(f"Cannot read this JPEG/PNG image: {error}")
        return
    st.image(preview, caption="Preview; orientation metadata is applied when readable.", width=560)
    st.caption("Hosted uploads are processed on the hosting server. The app does not explicitly save uploaded images or send them to an external image inference service. Results are held in this session for the latest image; downloads remain visible after other checks run.")
    tabs = st.tabs(["AI detection", "Compare original", "File details"])
    with tabs[0]: generation_panel(data)
    with tabs[1]: reference_panel(data)
    with tabs[2]:
        st.subheader("File details")
        st.write(f"**Format:** {image_format}")
        st.write(f"**Dimensions:** {dimensions[0]} x {dimensions[1]} pixels (stored image)")
        st.write(f"**Camera make:** {make}")
        st.write(f"**Camera model:** {model}")
        st.caption("EXIF fields can be absent or changed. They are not used by the detectors and do not prove origin.")
    st.divider()
    st.download_button("Download all current results (JSON)", json.dumps(combined_report(image_format, dimensions), indent=2, ensure_ascii=False),
                       file_name="image_evidence.json", mime="application/json", key="download_all")
    st.caption("Your report includes completed results and identifies any checks that were not run or could not finish.")


if __name__ == "__main__":
    main()
