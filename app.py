import os
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO
from google import genai

load_dotenv()
ROOT = Path(__file__).parent
MODELS = {
    "eye": ROOT / "models" / "eye_best.pt",
    "chest": ROOT / "models" / "chest_best.pt",
    "bone": ROOT / "models" / "bone_best.pt",
}

st.set_page_config(page_title="MedAssist AI", page_icon="🩺", layout="wide")

@st.cache_resource
def load_yolo(path: str):
    return YOLO(path)

def get_client():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        return None
    return genai.Client(api_key=key)

def route_specialist(choice: str, question: str) -> str:
    if choice != "Auto-detect (experimental)":
        return {"Eye image": "eye", "Chest X-ray / lung image": "chest",
                "Bone X-ray / fracture image": "bone",
                "General medical question": "general"}.get(choice, "unknown")
    q = question.lower()
    if any(x in q for x in ["eye", "retina", "fundus", "vision"]): return "eye"
    if any(x in q for x in ["chest", "lung", "x-ray", "xray", "ct scan"]): return "chest"
    if any(x in q for x in ["bone", "fracture", "wrist", "ankle"]): return "bone"
    return "unknown"

def detect(image: Image.Image, specialist: str):
    path = MODELS.get(specialist)
    if not path or not path.exists():
        return None, f"No trained {specialist} YOLO weights configured. Expected {path.name if path else 'specialist weights'}."
    try:
        model = load_yolo(str(path))
        results = model.predict(image, verbose=False, conf=0.25)
        rows = []
        for result in results:
            for box in result.boxes or []:
                cls = int(box.cls[0].item())
                rows.append({
                    "label": str(result.names.get(cls, cls)),
                    "confidence": round(float(box.conf[0].item()), 4),
                    "box_xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
                })
        return rows, None
    except Exception as exc:
        return None, f"YOLO inference failed: {exc}"

def medical_prompt(question, specialist, detections, language):
    return f"""
You are an educational medical-information assistant, not a clinician. Respond in {language}.
Do not claim to diagnose or rule out disease. Never fabricate image findings or certainty.
Do not prescribe medication or dosage. Explain limitations and recommend assessment by a
qualified healthcare professional. For severe chest pain, difficulty breathing, stroke-like
symptoms, major trauma, or sudden vision loss, advise urgent/emergency care.
Selected specialist: {specialist}
Experimental YOLO output (not a confirmed diagnosis): {detections}
User question: {question or "Explain what this image may show and safe next steps."}
Give: (1) what can/cannot be concluded, (2) possible explanations only if supportable,
(3) safe next steps, (4) when to seek urgent/professional care, (5) a brief disclaimer.
"""

st.markdown("""
<style>
.block-container{max-width:1200px;padding-top:1.5rem}
.hero{padding:1.3rem 1.5rem;border-radius:18px;background:linear-gradient(120deg,#eaf4ff,#f4f0ff);border:1px solid #dce7fa}
</style>
<div class="hero"><h1>🩺 MedAssist AI</h1>
<p>Medical image workflow and health-information chatbot</p>
<p><b>Prototype only:</b> not a medical device or substitute for a clinician.</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Analysis settings")
    choice = st.selectbox("Choose specialist", [
        "Auto-detect (experimental)", "Eye image", "Chest X-ray / lung image",
        "Bone X-ray / fracture image", "General medical question"])
    language = st.selectbox("Response language", ["English", "Telugu", "Hindi", "Kannada", "Tamil"])
    model_name = st.text_input("Gemini model", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    st.divider()
    st.subheader("Optional YOLO model status")
    for name, path in MODELS.items():
        st.write(f"{name.title()}: {'weights found' if path.exists() else 'not configured'}")
    st.caption("Add validated specialist weights in the models folder to enable YOLO inference.")

col1, col2 = st.columns(2, gap="large")
with col1:
    st.subheader("1. Ask or upload")
    question = st.text_area("Your question", placeholder="Ask a general health question or ask about an image...", height=130)
    upload = st.file_uploader("Medical image (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"])
    image = None
    if upload:
        try:
            image = Image.open(upload).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)
        except Exception:
            st.error("Could not open that image.")
    submit = st.button("Analyze / Ask MedAssist", type="primary", use_container_width=True)

with col2:
    st.subheader("2. Response")
    if not submit:
        st.info("Enter a question and/or upload an image, then submit.")
    elif not question.strip() and image is None:
        st.warning("Enter a question or upload an image first.")
    else:
        client = get_client()
        if client is None:
            st.error("Gemini is not configured. Add GEMINI_API_KEY to .env and restart Streamlit.")
        else:
            specialist = route_specialist(choice, question)
            detections, message = None, None
            if image and specialist in MODELS:
                with st.spinner("Running configured specialist detector..."):
                    detections, message = detect(image, specialist)
            elif image and specialist == "unknown":
                message = "Auto-routing cannot reliably identify the image type. Select a specialist in the sidebar."
            if message: st.info(message)
            if detections is not None:
                st.write("Experimental detector output")
                if detections: st.dataframe(detections, use_container_width=True)
                else: st.write("No boxes above the threshold. This does not rule out disease.")
            prompt = medical_prompt(question.strip(), specialist, detections, language)
            try:
                with st.spinner("Generating medical information..."):
                    contents = [prompt]
                    if image: contents.append(image)
                    response = client.models.generate_content(model=model_name.strip(), contents=contents)
                st.markdown(response.text or "The model returned an empty response.")
            except Exception as exc:
                st.error(f"Gemini request failed: {exc}")
                st.caption("Check your API key, model availability, quota, network connection, and terminal logs.")

st.divider()
st.subheader("Accessibility")
st.write("Choose a response language in the sidebar. Speech input/output can be added in a later milestone.")
st.warning("Do not use this prototype for emergency decisions. AI output can be wrong; medical images need professional review.")
