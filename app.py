"""Independent image evidence checks with per-upload session results."""

import csv
import hashlib
import io
import json
import sys
import warnings

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
CHECKS = ("generation", "edits", "reference", "provenance")


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


@st.cache_resource
def get_edit_detector():
    from edit_detector import load_edit_detector
    return load_edit_detector(download=True)


def reset_for_image(data):
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
        prefix = "Edit analysis unavailable" if name == "edits" else "Analysis unavailable"
        st.error(f"{prefix} - result: inconclusive. {state['error']}")
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
    from detector import MODEL_ID, MODEL_REVISION
    st.subheader("AI-generation evidence")
    st.write("Look for AI-like pixel patterns. This cannot reliably distinguish a fully generated image from a camera photo edited with AI.")
    if st.button("Analyze image pixels", type="primary", key="analyze_generation"):
        def analyze():
            from detector import analyze_pixels
            image, *_ = read_image(data, full_resolution=True)
            with st.spinner("Checking AI-like pixel patterns..."):
                return analyze_pixels(image, get_detector())
        run_check("generation", analyze)
    result = saved_result("generation")
    if result is not None:
        st.write(f"**Model indication:** {result['signal']}")
        st.metric("AI-class model score", f"{result['ai_score']:.6f}")
        st.caption("Raw model output, not a probability of authenticity. A low score does not prove camera origin; a high score does not prove the whole image was generated.")
        st.warning("Authenticity conclusion: inconclusive. Independent target-data validation is still needed.")
        st.download_button("Download model score (CSV)", csv_bytes({**result, "model": MODEL_ID,
                           "revision": MODEL_REVISION, "authenticity": "inconclusive"}),
                           file_name="image_score.csv", mime="text/csv", key="download_generation")
    with st.expander("Method and limitations"):
        st.write(f"Detectra-OpenFake | {MODEL_ID} | revision {MODEL_REVISION}. Educational/noncommercial CC-BY-NC-4.0.")
        st.write("Single center crop and raw sigmoid; no author score scaling or multi-crop procedure. Exploratory bands: <=0.10 low AI-like signal, >=0.90 AI-like signal, otherwise inconclusive. Metadata is not used.")
        st.caption("First use downloads about 30 MB. New generators, screenshots, compression and ordinary editing can mislead the model. Six known diagnostic examples are not independent validation.")


def edit_panel(data):
    st.subheader("Possible local edits")
    st.write("TruFor looks for pixel inconsistencies. Its result is independent of the AI-generation score.")
    st.caption("A low anomaly score does not mean unedited: this model missed the known Gemini bottle-removal edit.")
    if st.button("Analyze possible edits", key="analyze_edits"):
        def analyze():
            from edit_detector import analyze_edits
            image, *_ = read_image(data, full_resolution=True)
            with st.spinner("Loading TruFor and examining pixel inconsistencies..."):
                return analyze_edits(image, get_edit_detector())
        run_check("edits", analyze)
    result = saved_result("edits")
    if result is not None:
        import numpy as np
        from edit_detector import render_maps, SOURCE_REVISION
        overlay, reliability = render_maps(result)
        pure_map = Image.fromarray(np.round(result["map"] * 255).astype(np.uint8))
        st.metric("TruFor whole-image anomaly score", f"{result['score']:.6f}")
        first, second = st.columns(2)
        first.image(pure_map, caption="Anomaly map: brighter = higher anomaly. Fixed 0-1 scale.")
        second.image(reliability, caption="Reliability map: brighter = greater model-predicted reliability.")
        with st.expander("View anomaly overlay"):
            st.image(overlay, caption="Red/blue overlay blends with original colors; use the pure map for interpretation.")
        st.caption(f"Analyzed size: {result['analyzed_size'][0]} x {result['analyzed_size'][1]} pixels. Maps apply to this resized version.")
        st.warning("Edit assessment: inconclusive. Maps and scores are experimental evidence, not proof of editing or absence of edits.")
        st.download_button("Download edit analysis (CSV)", csv_bytes({"model": "TruFor", "source_revision": SOURCE_REVISION,
                           "anomaly_score": result["score"], "analyzed_width": result["analyzed_size"][0],
                           "analyzed_height": result["analyzed_size"][1], "edit_assessment": "inconclusive"}),
                           file_name="edit_analysis.csv", mime="text/csv", key="download_edits")
        st.download_button("Download anomaly map", png_bytes(pure_map), file_name="anomaly_map.png", mime="image/png")
        st.download_button("Download reliability map", png_bytes(reliability), file_name="reliability_map.png", mime="image/png")
    with st.expander("TruFor method and limitations"):
        st.write("TruFor by Guillaro et al., GRIP-UNINA, CVPR 2023. Informational/nonprofit use only.")
        st.write("CPU preview is limited to a 384-pixel longest edge. Resizing can erase small edits or introduce artifacts. Fully generated images, recompression and normal photo processing can cause anomalies. No edit/no-edit threshold has been validated.")
        st.caption("First use downloads approximately 261 MB. A failed check is inconclusive; it does not affect other checks.")


def reference_panel(data):
    st.subheader("Compare with an original")
    st.write("The main upload is the version to examine. Supply its original here to inspect aligned pixel differences.")
    st.caption("This is reference comparison, not single-image edit detection. It cannot verify that your reference is authentic.")
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
                st.write(f"**{result.get('comparison_status', 'Reference differences')}**")
                first, second = st.columns(2)
                first.image(result["aligned_reference"], caption="Aligned reference")
                second.image(result["candidate"], caption="Version being examined")
                st.image(result["difference"], caption="Difference map: brighter = greater pixel difference; gray = excluded borders.")
                with st.expander("View marked differences"):
                    st.image(result["overlay"], caption="Red marks differences under fixed inspection settings, not confirmed edits.")
                st.metric("Marked area within overlap (not an edit probability)", f"{result['changed_fraction']:.1%}")
                st.caption(f"Compared size: {result['analysis_size']}; overlap: {result['overlap_fraction']:.1%}.")
                if result["alignment"].get("spatial_support") == "limited":
                    st.warning("Alignment is supported by a limited part of the image. Extrapolation may contribute to the differences.")
                st.warning("Reference differences are not a calibrated editing estimate. Regeneration, lighting, compression and imperfect alignment may affect broad areas. No marked differences does not prove no edit occurred.")
                with st.expander("Alignment and inspection details"):
                    st.json({"alignment": result["alignment"], "inspection_settings": result.get("inspection_settings", {})})
                st.download_button("Download reference difference overlay", png_bytes(result["overlay"]), file_name="reference_differences.png", mime="image/png")
    else:
        reset_for_reference(None)


def provenance_panel(data, image_format):
    st.subheader("Signed image history (C2PA)")
    st.write("Check embedded Content Credentials, their signature and asset binding. Signed statements describe declared history; they do not prove the scene is true.")
    st.caption("This uses the original file bytes. Missing credentials never imply AI generation or editing. External manifests, network trust lists and certificate-status fetching are disabled.")
    if st.button("Check signed history", key="check_provenance"):
        def check():
            from provenance import check_provenance
            with st.spinner("Inspecting embedded Content Credentials..."):
                return check_provenance(data, image_format)
        run_check("provenance", check)
    result = saved_result("provenance")
    if result is not None:
        st.write(f"**{result['summary']}**")
        if result["status"] in ("unavailable", "validation_failed", "inconclusive"):
            st.warning("Provenance assessment is inconclusive. See verification details below.")
        st.write(f"**Signer trust:** {result.get('signer_trust', 'unknown').replace('_', ' ')}")
        if result.get("signer"):
            signer = result["signer"]
            st.text(f"Declared signer: {signer.get('common_name', 'Not stated')}")
            st.caption("A certificate's displayed name is not independently verified identity when signer trust is not established.")
        if result.get("actions"):
            with st.expander("Declared actions (signer statements)"):
                labels = {"c2pa.created": "Created", "c2pa.opened": "Opened existing content",
                          "c2pa.edited": "Edited", "c2pa.resized": "Resized", "c2pa.placed": "Placed content"}
                for action in result["actions"]:
                    code = action.get("action", "Not stated")
                    st.text(labels.get(code, code))
                    if action.get("digitalSourceType"):
                        st.text("Declared source type: " + action["digitalSourceType"].rsplit("/", 1)[-1])
        with st.expander("Verification details and limitations"):
            st.json(result)
        st.download_button("Download signed-history report (JSON)", json.dumps(result, indent=2, ensure_ascii=False),
                           file_name="signed_history.json", mime="application/json")


def combined_report(image_format, dimensions):
    report = {"image_sha256": st.session_state["image_digest"], "format": image_format,
              "stored_dimensions": dimensions, "overall_authenticity": "inconclusive", "checks": {}}
    for name in CHECKS:
        state = st.session_state.get(name)
        if not state:
            report["checks"][name] = {"status": "not_run"}
        elif not state["ok"]:
            report["checks"][name] = {"status": "unavailable", "diagnostic": state["diagnostic"]}
        else:
            result = state["result"]
            if name == "edits":
                result = {key: result[key] for key in ("score", "analyzed_size", "assessment") if key in result}
            elif name == "reference":
                result = {key: result[key] for key in ("changed_fraction", "overlap_fraction", "analysis_size", "alignment", "inspection_settings", "assessment") if key in result}
            status = "completed"
            if name == "provenance" and result.get("status") in {"unavailable", "inconclusive", "validation_failed"}:
                status = result["status"]
            report["checks"][name] = {"status": status, "result": result}
    return report


def main():
    st.set_page_config(page_title="Image verification lab", page_icon="??", layout="wide")
    st.title("Image verification lab")
    st.write("Examine image evidence with four independent checks.")
    st.caption("JPEG or PNG | up to 20 MB and 20 megapixels | educational prototype")
    st.info("A camera photo can also be edited. Model scores, reference differences and signed history answer different questions; no single check proves authenticity.")
    uploaded = st.file_uploader("Choose the image to examine", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        reset_for_image(None)
        st.write("Upload an image to preview it and choose a check. If you have an original, use Compare original to inspect differences.")
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
    tabs = st.tabs(["AI evidence", "Possible edits", "Compare original", "Signed history", "File details"])
    with tabs[0]: generation_panel(data)
    with tabs[1]: edit_panel(data)
    with tabs[2]: reference_panel(data)
    with tabs[3]: provenance_panel(data, image_format)
    with tabs[4]:
        st.subheader("File details")
        st.write(f"**Format:** {image_format}")
        st.write(f"**Dimensions:** {dimensions[0]} x {dimensions[1]} pixels (stored image)")
        st.write(f"**Camera make:** {make}")
        st.write(f"**Camera model:** {model}")
        st.caption("EXIF fields can be absent or changed. They are not used by the detectors and do not prove origin.")
    st.divider()
    st.download_button("Download all current results (JSON)", json.dumps(combined_report(image_format, dimensions), indent=2, ensure_ascii=False),
                       file_name="image_evidence.json", mime="application/json", key="download_all")
    st.caption("Only completed checks are included as results. Unrun and failed checks are recorded explicitly. Overall authenticity remains inconclusive.")


if __name__ == "__main__":
    main()
