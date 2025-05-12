
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import os
import logging
from pathlib import Path
from typing import Optional
from openai import OpenAI
import torch

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
meeting_transcript = "" 

try:
    while True:
        audio_chunk = record_audio_chunk()
        recognized_text = transcribe(audio_chunk)
        translated_text = translate(recognized_text)

        if recognized_text.strip() != "":
            print(f"\n[Recognized] {recognized_text}")
            print(f"[Translated] {translated_text}\n")
            meeting_transcript += translated_text + " " 

except KeyboardInterrupt:
    print("\n회의 종료됨. 번역 저장 중...")
    with open("meeting_transcript.txt", "w", encoding="utf-8") as f:
        f.write(meeting_transcript)
        print("번역 내용이 'meeting_transcript.txt'에 저장되었습니다.")

## --------- summarization ---------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', ''))

def build_prompt(meeting_transcript: str) -> str:
    try:
        template = {
            'system_role': "You are a summarization AI that categorizes and summarizes the entire meeting content.",
            'output_format': [
                "Next meeting date and time: OOO",
                "Agenda items: OOO",
                "Things to do: OOO",
                "Team members' opinions: OOO",
                "Mentor's feedback: OOO"
            ]
        }

        prompt = f"""
{template['system_role']}
Summarize the entire meeting content below using the following format:

[Example Output]
{chr(10).join(template['output_format'])}

[Full Meeting Transcript]
{meeting_transcript}

[Summarized Result]
"""
        return prompt

    except Exception as e:
        logging.error(f"Failed to build prompt: {str(e)}")
        raise

def summarize_meeting(transcript: str):
    prompt = build_prompt(transcript)


    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a summarization AI that categorizes and summarizes the entire meeting content."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )


    return response.choices[0].message.content.strip()

# --------- Main Loop ---------
if __name__ == "__main__":
    if not os.path.exists("meeting_transcript.txt"):
        print("meeting_transcript.txt doesn't exist.")
        exit()

    with open("meeting_transcript.txt", "r", encoding="utf-8") as f:
        meeting_transcript = f.read()

    try:
        summary = summarize_meeting(meeting_transcript)
        print("\n Meeting Summary Results:\n")
        print(summary)
    except Exception as e:
        logging.error(f"Summary Failure: {str(e)}")