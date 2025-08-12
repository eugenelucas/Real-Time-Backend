import base64
import asyncio
import io
import os
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from openai import AzureOpenAI
import soundfile as sf
import numpy as np
from scipy.signal import resample
from dotenv import load_dotenv  # ✅ Import dotenv

# Load environment variables
load_dotenv()
os.makedirs("saved_audio_chunks", exist_ok=True)

app = FastAPI()

# CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


api_key = os.getenv("AZURE_API_KEY")
api_version = os.getenv("AZURE_API_VERSION", "2024-06-01")
azure_endpoint = os.getenv("AZURE_ENDPOINT")
chunk_interval = float(os.getenv("CHUNK_INTERVAL", 2.0))  # default to 2.0s if missing

# Azure client
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)
model_name = "whisper"  # Azure deployment name

def process_wav_bytes(wav_bytes: bytes) -> bytes:
    """Ensure mono 16kHz WAV for Whisper."""
    data, samplerate = sf.read(io.BytesIO(wav_bytes))

    if len(data.shape) == 2:
        data = np.mean(data, axis=1)

    if samplerate != 16000:
        duration = len(data) / samplerate
        new_len = int(duration * 16000)
        data = resample(data, new_len)

    data_int16 = np.int16(data * 32767)
    buffer = io.BytesIO()
    sf.write(buffer, data_int16, 16000, format="WAV")
    buffer.seek(0)
    return buffer.read()

pcm_buffer = bytearray()
from scipy.io import wavfile

@app.websocket("/ws/translate")
async def websocket_translate(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection open")

    pcm_buffer = bytearray()
    chunk_count = 0 
    last_send_time = time.time()

    try:
        while True:
            pcm_chunk = await websocket.receive_bytes()
            pcm_buffer.extend(pcm_chunk)

            now = time.time()
            if now - last_send_time >= chunk_interval:
                chunk_count += 1
                print(f"[#{chunk_count}] Sending {len(pcm_buffer)} bytes to Whisper")

                # Convert to WAV
                wav_io = io.BytesIO()
                wavfile.write(wav_io, 16000, np.frombuffer(pcm_buffer, dtype=np.int16))
                wav_io.seek(0)
                wav_io.name = f"audio_{chunk_count}.wav"

                whisper_start = time.time()
                result = client.audio.translations.create(
                    file=wav_io,
                    model=model_name
                )
                whisper_end = time.time()

                await websocket.send_text(result.text.strip())

                print(
                    f"[#{chunk_count}] Done — whisper_time={whisper_end - whisper_start:.2f}s, "
                    f"gap={now - last_send_time:.2f}s"
                )

                pcm_buffer.clear()
                last_send_time = now

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        print("WebSocket connection closed")