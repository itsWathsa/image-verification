"""Independent image evidence checks with per-upload session results."""

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
        return None
    return state["result"]


def generation_panel(data):
    from detector import ANALYSIS_METHOD
    st.subheader("Was AI used to create this image?")
    st.write("Check for AI involvement, including fully generated images and camera photos modified using AI.")
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
        st.metric("AI detection score", f"{result['ai_score']:.6f}")
        st.caption("Model score, not a probability. AI-assisted edits may be missed.")


def reference_panel(data):
    st.subheader("Check brightness, contrast and colour changes")
    st.write("Upload the original to compare it with your image.")
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
                with st.expander("View marked differences"):
                    st.image(result["overlay"], caption="Highlighted differences; these are not confirmed edits.")
                if tone:
                    first, second, third = st.columns(3)
                    first.metric("Brightness change", f"{tone['brightness_difference_levels']:+.1f} / 255")
                    ratio = tone["contrast_spread_ratio"]
                    second.metric("Contrast relative to original", f"{ratio:.2f}x" if ratio is not None else "Not measurable")
                    if "mean_colour_difference_levels" in tone:
                        third.metric("Colour difference", f"{tone['mean_colour_difference_levels']:.1f} levels")
                if result["alignment"].get("spatial_support") == "limited":
                    st.warning("Limited alignment: some differences may be caused by image positioning.")
                st.caption("Lighting, compression and alignment can affect this comparison. Differences do not confirm editing.")
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


def video_panel():
    from video_analysis import MAX_VIDEO_BYTES, VIDEO_METHOD, analyze_video
    st.subheader("Video AI screening")
    st.caption("MP4, MOV, AVI or WebM | up to 100 MB | 10 minutes | up to 4096 x 2160 (landscape or portrait)")
    st.caption("Experimental: checks up to 12 frames with an image model. Real videos can be falsely flagged. Audio and motion are not analysed; scores do not verify video authenticity.")
    uploaded = st.file_uploader("Choose a video to examine", type=["mp4", "mov", "avi", "webm"], key="video_upload")
    data = uploaded.getvalue() if uploaded is not None else None
    digest = hashlib.sha256(data).hexdigest() if data else None
    identity = (digest, VIDEO_METHOD)
    if st.session_state.get("video_identity") != identity:
        st.session_state.pop("video", None)
        st.session_state["video_identity"] = identity
    if data is None:
        return
    if len(data) > MAX_VIDEO_BYTES:
        st.error("Choose a video no larger than 100 MB.")
        return
    if Path(uploaded.name).suffix.lower() in {".mp4", ".mov", ".webm"}:
        st.video(data, format="video/" + ("quicktime" if uploaded.name.lower().endswith(".mov") else "webm" if uploaded.name.lower().endswith(".webm") else "mp4"))
    if st.button("Analyze video", type="primary", key="analyze_video"):
        bar = st.progress(0., text="Checking sampled frames...")
        try:
            run_check("video", lambda: analyze_video(data, get_detector(), progress=bar.progress))
        finally:
            bar.empty()
    result = saved_result("video")
    if result is not None:
        report = result["report"]
        st.info(report["conclusion"])
        first, second = st.columns(2)
        first.metric("Average sampled-frame score", f"{report['mean_frame_score']:.6f}")
        second.metric("High-signal frames", f"{report['high_signal_frames']} / {report['sample_count']}")
        st.caption("Scores are model signals, not the probability that a video is AI-generated.")
        with st.expander("Review sampled frames"):
            st.image(result["preview"], caption=f"Highest-scoring sampled frame at {result['preview_timestamp']:.2f}s")
            rows = [f"| {frame['timestamp_seconds']:.2f} | {frame['ai_score']:.6f} |" for frame in report["frames"]]
            st.markdown("| Time (s) | AI score |\n| --- | --- |\n" + "\n".join(rows))
        st.download_button("Download video report (JSON)", json.dumps(report, indent=2), file_name="video_evidence.json", mime="application/json", key="download_video")


def main():
    st.set_page_config(page_title="Image & Video Verification", page_icon=":material/image_search:", layout="wide")
    css = Path(__file__).with_name("style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.markdown('''<section class="workspace-hero">
      <div class="workspace-label">DIGITAL MEDIA REVIEW</div>
      <h1>Image &amp; Video Verification</h1>
      <p>Review images, screen video frames for AI-like signals and compare source images.</p>
      <div class="hero-meta"><span>01 &nbsp; Upload media</span><span>02 &nbsp; Review the evidence</span><span>03 &nbsp; Export your results</span></div>
    </section>
    <div class="feature-grid">
      <div class="feature-card"><span class="number">01 / AI DETECTION</span><h3>Assess AI involvement</h3><p>Review signals of AI generation or AI-assisted modification.</p></div>
      <div class="feature-card"><span class="number">02 / ORIGINAL COMPARISON</span><h3>Compare with the source</h3><p>Align an image with its original and inspect visual differences.</p></div>
    </div>''', unsafe_allow_html=True)
    media = st.radio("Media type", ["Image", "Video"], horizontal=True, key="media_type")
    if media == "Video":
        video_panel()
        return
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
    st.image(preview, caption="Image preview", width=560)
    st.caption("Images are processed on the server. Results stay available for the current upload.")
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
    st.download_button("Download report (JSON)", json.dumps(combined_report(image_format, dimensions), indent=2, ensure_ascii=False),
                       file_name="image_evidence.json", mime="application/json", key="download_all")


if __name__ == "__main__":
    main()
