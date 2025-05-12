import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import tkinter as tk
import threading
import sounddevice as sd
import numpy as np
import queue
import whisper
from googletrans import Translator
import datetime
import platform

# --- SETTINGS  ---
SAMPLE_RATE = 16000
BLOCK_DURATION = 3  # 초
MODEL_SIZE = "tiny"
# --- Queue ---
audio_queue = queue.Queue()

# --- WHISPER ---
model = whisper.load_model(MODEL_SIZE)
translator = Translator()

# --- TRANSLATOR ---
translation_direction = {"src": "en", "dest": "ko"}

# --- SAVING FILE ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_file = open(f"subtitle_record_{timestamp}.txt", "a", encoding="utf-8")

# --- AUDIO CAPTURE  ---
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def audio_processing():
    buffer = np.empty((0, 1), dtype=np.float32)

    while True:
        indata = audio_queue.get()
        buffer = np.append(buffer, indata, axis=0)

        if len(buffer) >= SAMPLE_RATE * BLOCK_DURATION:
            process_block = buffer[:SAMPLE_RATE * BLOCK_DURATION]
            buffer = buffer[SAMPLE_RATE * BLOCK_DURATION:]

            try:
                result = model.transcribe(process_block.flatten(), fp16=False, language=translation_direction["src"])
                text = result['text'].strip()

                if text:
                    translated = translator.translate(text, src=translation_direction["src"], dest=translation_direction["dest"]).text
                    update_gui(translated)
                    save_text(translated)
            except Exception as e:
                print(f"ERROR!: {e}")

# --- GUI UPDATE ---
def update_gui(text):
    transcript_box.configure(state='normal')
    transcript_box.insert(tk.END, f"{text}\n\n")
    transcript_box.configure(state='disabled')
    transcript_box.see(tk.END)

# --- CAPTIONS  ---
def save_text(text):
    now = datetime.datetime.now().strftime("[%H:%M:%S]")
    save_file.write(f"{now} {text}\n")
    save_file.flush()

# --- TRANSLATION SETTINGS  ---
def switch_translation():
    if translation_direction["src"] == "en":
        translation_direction["src"] = "ko"
        translation_direction["dest"] = "en"
        switch_button.config(text=" Korean ➔English")
    else:
        translation_direction["src"] = "en"
        translation_direction["dest"] = "ko"
        switch_button.config(text="English -> Korean")

# --- Ending Program ---
def on_closing():
    save_file.close()
    root.destroy()

# --- BlackHole ---
def find_blackhole_device():
    devices = sd.query_devices()
    os_name = platform.system()
    target_name = "BlackHole" if os_name == "Darwin" else "VB-Audio"

    for i, dev in enumerate(devices):
        if target_name in dev['name']:
            return i
    return None 

# --- Start Audio Streaming ---
def start_audio_stream():
    device_index = find_blackhole_device()
    print(f"Audio Index Used: {device_index}")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, device=device_index, callback=audio_callback):
        audio_processing()

# --- Tkinter GUI Settings ---
root = tk.Tk()
root.title("📝 Live Translation Script")
root.geometry("800x600")
root.configure(bg="#f0f0f0")

transcript_box = tk.Text(root, height=25, font=("Arial", 16), bg="white", wrap="word", state='disabled')
transcript_box.pack(padx=20, pady=20, fill="both", expand=True)

switch_button = tk.Button(root, text="English -> Korean Mode", command=switch_translation,
                           font=("Arial", 14), bg="#4CAF50", fg="black")
switch_button.pack(pady=10)

root.protocol("WM_DELETE_WINDOW", on_closing)

# --- AUDIO Streaming ---
thread = threading.Thread(target=start_audio_stream)
thread.daemon = True
thread.start()

# --- GUI Mainloop ---
root.mainloop()


