import streamlit as st
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from diffusers import StableDiffusionPipeline
import librosa

# ------------------------
# Device
# ------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Prompt Enhancement
# ------------------------
def enhance_prompt(text, sentiment, style):
    style_map = {
        "Realistic": "photorealistic, DSLR photo, 50mm lens, ultra realistic, sharp focus, high detail",
        "Anime": "anime style, studio ghibli, vibrant colors, soft shading",
        "Cyberpunk": "cyberpunk neon lighting, futuristic city, glowing lights"
    }

    if sentiment == "POSITIVE":
        mood = "golden hour lighting, vibrant colors"
    elif sentiment == "NEGATIVE":
        mood = "dark, moody, cinematic shadows"
    else:
        mood = "natural lighting"

    return f"{text}, {style_map[style]}, {mood}, 8k, highly detailed"

# ------------------------
# MODEL LOADERS
# ------------------------
@st.cache_resource
def load_whisper():
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(DEVICE)
    return processor, model

@st.cache_resource
def load_sd(mode):
    model_id = "stabilityai/sd-turbo" if mode == "Fast" else "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )
    pipe = pipe.to(DEVICE)
    return pipe

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis")

# ------------------------
# UI
# ------------------------
st.set_page_config(page_title="Speech to Image Generator", layout="wide")
st.markdown("## 🚀 AI Speech-to-Image Generator (Fast + Enhanced)")

if DEVICE == "cpu":
    st.warning("⚠️ Running on CPU — expect slower results")
else:
    st.success("🚀 Running on GPU")

# ------------------------
# Sidebar Controls (DEFINE FIRST)
# ------------------------
st.sidebar.header("⚙️ Settings")

style = st.sidebar.selectbox(
    "Style",
    ["Realistic", "Anime", "Cyberpunk"]
)

mode = st.sidebar.selectbox(
    "Generation Mode",
    ["Fast", "High Quality"]
)

language_mode = st.sidebar.selectbox(
    "Language",
    ["Auto Detect", "English", "Hindi"]
)

# ------------------------
# Load Models (AFTER mode)
# ------------------------
processor, whisper_model = load_whisper()
sd_pipe = load_sd(mode)
sentiment_pipeline = load_sentiment()

# ------------------------
# Session State
# ------------------------
if "text_prompt" not in st.session_state:
    st.session_state.text_prompt = ""
if "audio_done" not in st.session_state:
    st.session_state.audio_done = False

# ------------------------
# Upload Audio
# ------------------------
uploaded_file = st.file_uploader("🎧 Upload audio (wav/mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.success("Audio uploaded!")

    audio_path = "audio_input.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())

    st.info("Transcribing...")
    audio_input, _ = librosa.load(audio_path, sr=16000)

    input_features = processor(
        audio_input, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE)

    generate_kwargs = {}

    if language_mode == "English":
        generate_kwargs["language"] = "en"
    elif language_mode == "Hindi":
        generate_kwargs["language"] = "hi"

    with torch.no_grad():
        predicted_ids = whisper_model.generate(input_features, **generate_kwargs)

    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)

    st.session_state.text_prompt = transcription
    st.session_state.audio_done = True

# ------------------------
# Generate Image
# ------------------------
if st.session_state.audio_done:
    text_prompt = st.text_area("📝 Edit Prompt:", st.session_state.text_prompt)
    st.session_state.text_prompt = text_prompt

    sentiment = sentiment_pipeline(text_prompt)[0]
    st.write(f"**Sentiment:** {sentiment['label']}")

    if st.button("🎨 Generate Image"):

        if len(text_prompt.strip()) < 3:
            st.error("Prompt too short!")
        else:
            st.info("Generating image...")

            enhanced_prompt = enhance_prompt(
                text_prompt,
                sentiment['label'],
                style
            )

            with torch.no_grad():
                image = sd_pipe(
                    enhanced_prompt,
                    negative_prompt="cartoon, anime, painting, blurry, low quality, distorted, unrealistic",
                    num_inference_steps=20 if mode == "High Quality" else 10,
                    guidance_scale=7.5 if mode == "High Quality" else 0.0
                ).images[0]

            st.image(image, caption="Generated Image", width=512)
            st.write("### Prompt Used:")
            st.write(enhanced_prompt)
