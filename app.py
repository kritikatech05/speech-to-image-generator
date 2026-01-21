import streamlit as st
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from diffusers import StableDiffusionPipeline
import librosa
import tempfile

# ------------------------
# Device
# ------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# CACHED MODEL LOADERS
# ------------------------
@st.cache_resource
def load_whisper():
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(DEVICE)
    return processor, model

@st.cache_resource
def load_sd():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )
    pipe = pipe.to(DEVICE)
    return pipe

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis")

# ------------------------
# Load models
# ------------------------
processor, whisper_model = load_whisper()
sd_pipe = load_sd()
sentiment_pipeline = load_sentiment()

# ------------------------
# UI
# ------------------------
st.set_page_config(page_title="Speech to Image Generator", layout="wide")

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
    st.warning("⚠️ Running on CPU — generation may be slow.")
else:
    st.success("🚀 Running on GPU")

# Sidebar
st.sidebar.header("⚙️ Settings")

quality = st.sidebar.selectbox("Image Quality", ["Fast", "Balanced", "High Quality"])

if quality == "Fast":
    steps = 10
elif quality == "Balanced":
    steps = 20
else:
    steps = 35

# Upload audio
uploaded_audio = st.file_uploader("🎤 Upload an audio file", type=["wav", "mp3", "m4a"])

if uploaded_audio is not None:
    st.audio(uploaded_audio)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_audio.read())
        audio_path = tmp.name

    st.info("Transcribing audio...")

    audio_input, _ = librosa.load(audio_path, sr=16000)

    input_features = processor(
        audio_input, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE)

    with torch.no_grad():
        predicted_ids = whisper_model.generate(input_features)

    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)

    st.text_area("📝 Transcription:", transcription)

    sentiment = sentiment_pipeline(transcription)[0]
    st.write(f"**Sentiment:** {sentiment['label']} ({sentiment['score']:.2f})")

    estimated_time = steps * 3
    st.info(f"⏳ Estimated generation time: ~{estimated_time} seconds")

    if st.button("🎨 Generate Image"):
        st.info("Generating image...")

        with torch.no_grad():
            image = sd_pipe(transcription, num_inference_steps=steps).images[0]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Generated Image", width=512)

        with col2:
            st.write("### Prompt used")
            st.write(transcription)
            st.write("### Settings")
            st.write(f"Steps: {steps}")
