import os
import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import sounddevice as sd
import wavio
from tensorflow.keras.models import load_model

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'  # Suppress CPU warnings
tf.get_logger().setLevel("ERROR")  # Suppress general TensorFlow logs

# ✅ Disable oneDNN warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ✅ Suppress TensorFlow deprecation warnings
tf.get_logger().setLevel("ERROR")

# ✅ Set Page Config - ONLY ONCE at the start!
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ✅ Use a common directory for both uploaded and recorded audio
AUDIO_SAVE_PATH = "audio_files"
if not os.path.exists(AUDIO_SAVE_PATH):
    os.makedirs(AUDIO_SAVE_PATH)

# ✅ Load Model
model_path = r"C:\Users\PRIYA GUPTA\OneDrive\Desktop\OneDrive\Desktop\SAP\TESS Toronto emotional speech set data\speech_Emotion_Recognition.keras"

try:
    model = load_model(model_path, compile=False)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy")  # Recompile with default optimizer
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    model = None  # Prevents errors later


# ✅ Feature Extraction Function
def extract_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=22050)

        # Ensure audio is not empty
        if len(y) == 0:
            st.error("❌ Extracted audio is empty. Try recording again.")
            return None

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # Ensure features are extracted
        if mfccs.shape[1] == 0:
            st.error("❌ No features extracted from audio!")
            return None

        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        st.error(f"❌ Error processing audio file: {e}")
        return None


# ✅ Prediction Function
def predict_emotion(audio_file, model):
    features = extract_features(audio_file)
    if features is None:
        return None
    features = np.expand_dims(features, axis=0)
    try:
        prediction = model.predict(features)
        emotions = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
        return emotions[np.argmax(prediction)]
    except Exception as e:
        st.error(f"❌ Model prediction failed: {e}")
        return None


# ✅ Audio Recording Function
def record_audio(filename, duration=3, fs=22050):
    st.info("🎙️ Recording... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Ensure recording completes

    if np.any(recording):  # Check if recording is not silent
        wavio.write(filename, recording, fs, sampwidth=2)
        st.success(f"✅ Recording saved as {filename}!")
        st.audio(filename, format='audio/wav')  # Playback recorded audio
    else:
        st.error("❌ Recording failed! No sound detected.")


# 🎤 **UI Components**
st.title("🎤 Speech Emotion Recognition")
st.write("Upload or record an audio file to predict the emotion!")

# ✅ Choose input method
option = st.radio("Choose an option:", ["📂 Upload Audio", "🎙️ Record Audio"])
audio_path = None

# ✅ Handle file upload
if option == "📂 Upload Audio":
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav", "mp3"])
    if uploaded_file:
        audio_path = os.path.join(AUDIO_SAVE_PATH, "temp_upload.wav")
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ Audio uploaded successfully!")
        st.audio(audio_path, format='audio/wav')

# ✅ Handle recording
if option == "🎙️ Record Audio":
    if st.button("🎤 Start Recording"):
        audio_path = os.path.join(AUDIO_SAVE_PATH, "recorded_audio.wav")
        record_audio(audio_path)
        st.audio(audio_path, format='audio/wav')

# ✅ Predict Emotion
if model is not None and audio_path:
    if st.button("🔍 Predict Emotion"):
        try:
            emotion = predict_emotion(audio_path, model)
            if emotion:
                st.success(f"**Predicted Emotion: {emotion} 🎭**")
            else:
                st.error("❌ Could not predict emotion. Try a different audio file.")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

        # ✅ Remove audio after prediction to avoid clutter
        os.remove(audio_path)
