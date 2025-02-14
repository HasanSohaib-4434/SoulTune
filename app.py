import streamlit as st
import whisper
import json
import numpy as np
from transformers import pipeline
import tempfile
import os
ffmpeg_path = "/app/.ffmpeg/ffmpeg"
if not os.path.exists(ffmpeg_path):
    st.write("Downloading ffmpeg...")
    subprocess.run([
        "wget", "-q", "-O", "ffmpeg.tar.xz",
        "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    ])
    subprocess.run(["tar", "xf", "ffmpeg.tar.xz"])
    os.makedirs("/app/.ffmpeg", exist_ok=True)
    os.rename("ffmpeg-release-amd64-static", "/app/.ffmpeg")
    st.write("ffmpeg installed!")

os.environ["PATH"] += os.pathsep + os.path.abspath("/app/.ffmpeg")

ffmpeg_installed = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
st.text(ffmpeg_installed.stdout)  


model = whisper.load_model("tiny")

with open('dua.json', 'r', encoding='utf-8') as file:
    dua_data = json.load(file)

with open('verse.json', 'r', encoding='utf-8') as file:
    quranic_verses = json.load(file)

emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def analyze_emotion(text):
    result = emotion_analyzer(text)
    emotion_scores = {score['label']: score['score'] for score in result[0]}
    emotion = max(emotion_scores, key=emotion_scores.get)
    return emotion

def get_duas(emotion):
    return dua_data.get(emotion, [])

def get_quranic_verses(emotion):
    return quranic_verses.get(emotion, [])

st.title("Emotion-based Dua and Quranic Verse Suggestion")

st.markdown("### Upload your audio file (MP3 or WAV):")

uploaded_file = st.file_uploader("Upload your audio file", type=["mp3", "wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name

    result = model.transcribe(temp_file_path)
    text = result["text"]
    st.markdown(f"#### Transcribed Text:")
    st.text_area("", text, height=150)

    emotion = analyze_emotion(text)
    st.markdown(f"<h3 style='color: #4CAF50;'>Detected Emotion: {emotion.capitalize()}</h3>", unsafe_allow_html=True)

    duas = get_duas(emotion)
    if duas:
        st.markdown("<h3 style='color: #FF7043;'>Suggested Duas:</h3>", unsafe_allow_html=True)
        for dua in duas:
            st.markdown(
                f"<div style='padding: 15px; border-radius: 8px; background-color: #2E3B4E; color: white; margin-bottom: 10px;'>"
                f"<strong>Arabic:</strong> {dua['dua']}<br>"
                f"<strong>Urdu:</strong> {dua['urdu']}<br>"
                f"<strong>English:</strong> {dua['english']}</div>",
                unsafe_allow_html=True
            )
    else:
        st.write("No duas available for this emotion.")

    verses = get_quranic_verses(emotion)
    if verses:
        st.markdown("<h3 style='color: #FF7043;'>Suggested Quranic Verses:</h3>", unsafe_allow_html=True)
        for verse in verses:
            st.markdown(
                f"<div style='padding: 15px; border-radius: 8px; background-color: #2E3B4E; color: white; margin-bottom: 10px;'>"
                f"<strong>Arabic:</strong> {verse['verse']}<br>"
                f"<strong>Urdu:</strong> {verse['urdu']}<br>"
                f"<strong>English:</strong> {verse['english']}</div>",
                unsafe_allow_html=True
            )
    else:
        st.write("No Quranic verses available for this emotion.")

    # Cleanup: Delete the temporary file after use
    os.remove(temp_file_path)
