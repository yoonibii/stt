import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import torch
import os

# --------- Settings ---------
LANGUAGE_TO_TRANSLATE = 'ko'  # Target language (Korean)
SAMPLE_RATE = 16000           # 16kHz sampling rate
CHUNK_DURATION = 2            # seconds to record before processing
MODEL_SIZE = "tiny"           # Model size ("tiny" for lightweight)

# --------- Initialize ---------
print("Loading Whisper model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")  # CPU + int8 for M2 Mac efficiency
translator = GoogleTranslator(source='auto', target=LANGUAGE_TO_TRANSLATE)

# --------- Functions ---------

def record_audio_chunk(duration=CHUNK_DURATION, samplerate=SAMPLE_RATE):
    print("Recording...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Block until recording is done
    return recording.flatten()

def transcribe(audio_np):
    segments, _ = model.transcribe(audio_np, language ="en")
    full_text = ""
    for segment in segments:
        full_text += segment.text + " "
    return full_text.strip()

def translate(text):
    if text.strip() == "":
        return ""
    translated = translator.translate(text)
    return translated

# --------- Main Loop ---------

print("Starting real-time captioning...\n")

try:
    while True:
        audio_chunk = record_audio_chunk()
        recognized_text = transcribe(audio_chunk)
        translated_text = translate(recognized_text)

        if recognized_text.strip() != "":
            print(f"\n[Recognized] {recognized_text}")
            print(f"[Translated] {translated_text}\n")

except KeyboardInterrupt:
    print("\nStopped by user.")

