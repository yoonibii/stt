import sounddevice as sd
import numpy as np
from queue import Queue
from threading import Thread
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import torch
import time

# --------- Settings ---------
LANGUAGE_TO_TRANSLATE = 'ko'  # Target language (Korean)
SAMPLE_RATE = 16000           # 16kHz sampling rate
CHUNK_DURATION = 0.5          # seconds per chunk (small chunk for low latency)
MODEL_SIZE = "tiny"            # Whisper model size

# --------- Initialize ---------
print("Loading Whisper model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
translator = GoogleTranslator(source='auto', target=LANGUAGE_TO_TRANSLATE)

# --------- Find BlackHole Input Device ---------
print("Finding BlackHole device...")
blackhole_device_index = None
device_list = sd.query_devices()
for idx, dev in enumerate(device_list):
    if 'BlackHole' in dev['name']:
        blackhole_device_index = idx
        break
if blackhole_device_index is None:
    raise RuntimeError("BlackHole device not found. Make sure BlackHole 2ch is installed and enabled.")
print(f"Using BlackHole device index: {blackhole_device_index}")

# --------- Queues ---------
audio_queue = Queue()

# --------- Functions ---------

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_chunk = indata.copy().flatten()
    audio_queue.put(audio_chunk)

def recognize_stream():
    buffer_audio = np.array([], dtype=np.float32)
    while True:
        chunk = audio_queue.get()
        buffer_audio = np.concatenate((buffer_audio, chunk))

        # Process every 1.5 seconds worth of audio (sliding window)
        if len(buffer_audio) >= int(SAMPLE_RATE * 1.5):
            segment_audio = buffer_audio[:int(SAMPLE_RATE * 1.5)]
            buffer_audio = buffer_audio[int(SAMPLE_RATE * 0.5):]  # Shift window by 0.5s (overlap a bit)

            # Transcribe
            segments, _ = model.transcribe(segment_audio, language="en")
            full_text = " ".join([segment.text for segment in segments]).strip()

            if full_text:
                # Translate
                try:
                    translated_text = translator.translate(full_text)
                    print(f"\n[Original]: {full_text}\n[Translated]: {translated_text}\n")
                except Exception as e:
                    print(f"Translation Error: {e}")

# --------- Start Streaming ---------

print("Starting real-time Zoom captioning...\n")

stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='float32',
    callback=audio_callback,
    device=blackhole_device_index
)

# Start recognition thread
recognition_thread = Thread(target=recognize_stream)
recognition_thread.daemon = True
recognition_thread.start()

# Start audio stream
with stream:
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting...")

