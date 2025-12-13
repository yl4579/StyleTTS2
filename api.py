# StyleTTS 2 HTTP Streaming API by @fakerybakery - Copyright (c) 2023 mrfakename. All rights reserved.
# Docs: API_DOCS.md
# To-Do:
# * Support voice cloning
# * Implement authentication, user "credits" system w/ SQLite3
import io
import markdown
from tortoise.utils.text import split_and_recombine_text
from flask import Flask, Response, request, jsonify
from scipy.io.wavfile import write
import phonemizer
import numpy as np
import ljinference
import msinference
import torch
from flask_cors import CORS

# Download required NLTK data
import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def genHeader(sampleRate, bitsPerSample, channels):
    datasize = 2000 * 10**6
    o = bytes("RIFF", "ascii")
    o += (datasize + 36).to_bytes(4, "little")
    o += bytes("WAVE", "ascii")
    o += bytes("fmt ", "ascii")
    o += (16).to_bytes(4, "little")
    o += (1).to_bytes(2, "little")
    o += (channels).to_bytes(2, "little")
    o += (sampleRate).to_bytes(4, "little")
    o += (sampleRate * channels * bitsPerSample // 8).to_bytes(4, "little")
    o += (channels * bitsPerSample // 8).to_bytes(2, "little")
    o += (bitsPerSample).to_bytes(2, "little")
    o += bytes("data", "ascii")
    o += (datasize).to_bytes(4, "little")
    return o


voicelist = [
    "f-us-1",
    "f-us-2",
    "f-us-3",
    "f-us-4",
    "m-us-1",
    "m-us-2",
    "m-us-3",
    "m-us-4",
]
voices = {}

global_phonemizer = phonemizer.backend.EspeakBackend(
    language="en-us", preserve_punctuation=True, with_stress=True
)
print("Computing voices")
for v in voicelist:
    voices[v] = msinference.compute_style(f"voices/{v}.wav")
print("Starting Flask app")

app = Flask(__name__)
cors = CORS(app)


@app.route("/")
def index():
    with open("API_DOCS.md", "r") as f:
        return markdown.markdown(f.read())


def synthesize(text, voice, steps):
    v = voice.lower()
    return msinference.inference(
        text, voices[v], alpha=0.3, beta=0.7, diffusion_steps=steps, embedding_scale=1
    )


def ljsynthesize(text, steps):
    return ljinference.inference(
        text,
        torch.randn(1, 1, 256).to("cuda" if torch.cuda.is_available() else "cpu"),
        diffusion_steps=steps,
        embedding_scale=1,
    )


@app.route("/api/v1/stream", methods=["POST"])
def serve_wav_stream():
    if "text" not in request.form or "voice" not in request.form:
        error_response = {
            "error": 'Missing required fields. Please include "text" and "voice" in your request.'
        }
        return jsonify(error_response), 400
    text = request.form["text"].strip()
    voice = request.form["voice"].strip().lower()

    # Get diffusion steps from request or use default
    steps = int(request.form.get("steps", 7))

    if voice not in voices:
        error_response = {
            "error": "Invalid voice selected. Available voices: " + ", ".join(voicelist)
        }
        return jsonify(error_response), 400

    v = voices[voice]
    texts = split_and_recombine_text(text)

    def generate():
        is_first_chunk = True
        for t in texts:
            # Generate audio using the pre-computed voice style
            wav = msinference.inference(
                t, v, alpha=0.3, beta=0.7, diffusion_steps=steps, embedding_scale=1
            )
            output_buffer = io.BytesIO()
            write(output_buffer, 24000, wav)

            # Seek to start and skip WAV header for chunks
            output_buffer.seek(0)
            if is_first_chunk:
                # For first chunk, include WAV header
                data = output_buffer.read()
                is_first_chunk = False
            else:
                # For subsequent chunks, skip the 44-byte WAV header
                output_buffer.seek(44)
                data = output_buffer.read()
            yield data

    return Response(generate(), mimetype="audio/x-wav")


@app.route("/api/v1/static", methods=["POST"])
def serve_wav():
    if "text" not in request.form or "voice" not in request.form:
        error_response = {
            "error": 'Missing required fields. Please include "text" and "voice" in your request.'
        }
        return jsonify(error_response), 400
    text = request.form["text"].strip()
    voice = request.form["voice"].strip().lower()
    if voice not in voices:
        error_response = {"error": "Invalid voice selected"}
        return jsonify(error_response), 400
    texts = split_and_recombine_text(text)
    audios = []
    for t in texts:
        audios.append(
            msinference.inference(
                t,
                voices[voice],
                alpha=0.3,
                beta=0.7,
                diffusion_steps=7,
                embedding_scale=1,
            )
        )
    output_buffer = io.BytesIO()
    write(output_buffer, 24000, np.concatenate(audios))
    response = Response(output_buffer.getvalue())
    response.headers["Content-Type"] = "audio/wav"
    return response


if __name__ == "__main__":
    app.run("0.0.0.0")