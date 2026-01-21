import streamlit as st
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from diffusers import StableDiffusionPipeline
import librosa
import tempfile

# ------------------------
# Page config FIRST
# ------------------------
st.set_page_config(page_title="Speech to Image Generator", layout="wide")

# ------------------------
# Device
# ------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# CACHED MODEL LOADERS (DO NOT CALL AT STARTUP)
# ------------------------
@st.cache_resource
def load_whisper():
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(DEVICE)
    return processor, model

@st.cache_resource
def load_sd():
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",   # 🔥 MUCH FASTER MODEL
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )
    pipe = pipe.to(DEVICE)
    return pipe

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis")

@st.cache_resource
def get_models():
    processor, whisper_model = load_whisper()
    sd_pipe = load_sd()
    sentiment_pipeline = load_sentiment()
    return processor, whisper_model, sd_pipe, sentiment_pipeline

# ------------------------
# UI
# ------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🎙️ Speech to Image Generator</h1>
    <p style='text-align: center; font-size: 18px; color: gray;'>
    Upload an audio file and generate AI images from it
    </p>
    """,
    unsafe_allow_html=True
)

if DEVICE == "cpu":
    st.warning("⚠️ Running on CPU — first generation may take 2–5 minutes.")
else:
    st.success("🚀 Running on GPU")

# Sidebar
st.sidebar.header("⚙️ Settings")

quality = st.sidebar.selectbox("Image Quality", ["Fast", "Balanced", "High Quality"])

if quality == "Fast":
    steps = 4
elif quality == "Balanced":
    steps = 6
else:
    steps = 8

# Upload audio
uploaded_audio = st.file_uploader("🎤 Upload an audio file", type=["wav", "mp3", "m4a"])

if uploaded_audio is not None:
    st.audio(uploaded_audio)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_audio.read())
        audio_path = tmp.name

    if st.button("🧠 Transcribe & Generate"):

        # ------------------------
        # Load models LAZILY
        # ------------------------
        with st.spinner("🔄 Loading AI models (first time may take 2–5 minutes)..."):
            processor, whisper_model, sd_pipe, sentiment_pipeline = get_models()

        # ------------------------
        # Transcription
        # ------------------------
        st.info("🎧 Transcribing audio...")

        audio_input, _ = librosa.load(audio_path, sr=16000)

        input_features = processor(
            audio_input, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(DEVICE)

        with torch.no_grad():
            predicted_ids = whisper_model.generate(input_features)

        transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)

        st.text_area("📝 Transcription:", transcription)

        # ------------------------
        # Sentiment
        # ------------------------
        sentiment = sentiment_pipeline(transcription)[0]
        st.write(f"**Sentiment:** {sentiment['label']} ({sentiment['score']:.2f})")

        # ------------------------
        # Image generation
        # ------------------------
        st.info("🎨 Generating image... (please wait)")

        with torch.no_grad():
            image = sd_pipe(
                prompt=transcription,
                num_inference_steps=steps,
                guidance_scale=0.0  # required for turbo
            ).images[0]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Generated Image", width=512)

        with col2:
            st.write("### 🧾 Prompt used")
            st.write(transcription)
            st.write("### ⚙️ Settings")
            st.write(f"Steps: {steps}")
            st.write(f"Quality: {quality}")
