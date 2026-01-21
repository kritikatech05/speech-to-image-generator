import streamlit as st
import sounddevice as sd
import torch
import numpy as np
from scipy.io.wavfile import write
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from diffusers import StableDiffusionPipeline
import librosa



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
# Load models ONCE
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
    Generate AI images directly from your voice
    </p>
    """,
    unsafe_allow_html=True
)



if DEVICE == "cpu":
    st.warning("⚠️ Running on CPU — image generation may be slow. For best performance, use a GPU.")
else:
    st.success("🚀 Running on GPU")


# Init session state
if "text_prompt" not in st.session_state:
    st.session_state.text_prompt = ""
if "audio_done" not in st.session_state:
    st.session_state.audio_done = False

# ------------------------
# Controls
# ------------------------
st.sidebar.header("⚙️ Settings")

duration = st.sidebar.slider("Recording duration (seconds)", 3, 10, 5)
quality = st.sidebar.selectbox(
    "Image Quality",
    ["Fast", "Balanced", "High Quality"]
)

if quality == "Fast":
    steps = 10
elif quality == "Balanced":
    steps = 20
else:
    steps = 35


language_mode = st.sidebar.selectbox(
    "Transcription language mode",
    ["Auto Detect", "English", "Hindi"]
)


# ------------------------
# Record Audio
# ------------------------
if st.button("🎤 Record Audio"):
    fs = 16000
    st.info("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    st.success("Recording finished!")

    audio_path = "audio_input.wav"
    write(audio_path, fs, (audio * 32767).astype(np.int16))

    # ------------------------
    # Transcription
    # ------------------------
    st.info("Transcribing...")
    audio_input, _ = librosa.load(audio_path, sr=16000)

    input_features = processor(
        audio_input, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE)

    # Decide language
    generate_kwargs = {}

    if language_mode == "English":
        generate_kwargs["language"] = "en"
    elif language_mode == "Hindi":
        generate_kwargs["language"] = "hi"
    # else Auto Detect -> do nothing

    with torch.no_grad():
        predicted_ids = whisper_model.generate(input_features, **generate_kwargs)

    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)



    st.session_state.text_prompt = transcription
    st.session_state.audio_done = True

# ------------------------
# If audio processed, show text + generate button
# ------------------------
st.caption("💡 Tip: Try adding details like style, lighting, mood, background, etc.")

if st.session_state.audio_done:
    text_prompt = st.text_area(
        "📝 Transcription (edit if needed):",
        st.session_state.text_prompt
    )

    st.session_state.text_prompt = text_prompt

    sentiment = sentiment_pipeline(text_prompt)[0]
    st.write(f"**Sentiment:** {sentiment['label']} ({sentiment['score']:.2f})")

    if st.button("🎨 Generate Image"):

    # Rough estimate (CPU)
        estimated_time = steps * 3  # seconds (very rough estimate)
        st.info(f"⏳ Estimated generation time: ~{estimated_time} seconds")

        st.info("Generating image...")

        with torch.no_grad():
            image = sd_pipe(
                text_prompt,
                num_inference_steps=steps
            ).images[0]

        # Two-column layout
        st.markdown("---")
        st.subheader("🎨 Result")

        st.markdown(
            "<div style='background-color:#fafafa; padding:20px; border-radius:12px;'>",
            unsafe_allow_html=True
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(
                "<style>img { border-radius: 12px; }</style>",
                unsafe_allow_html=True
            )
            st.image(image, caption="Generated Image", width=512)

        with col2:
            st.write("### Prompt used")
            st.write(text_prompt)
            st.write("### Settings")
            st.write(f"Steps: {steps}")

        st.markdown("</div>", unsafe_allow_html=True)
