# StyleTTS2 HTTP Streaming API Documentation

## Overview

The HTTP Streaming API provides text-to-speech synthesis with real-time audio streaming. The server uses Flask and returns WAV audio data.

## Base URL

```
http://localhost:5000
```

## Endpoints

### GET /

Returns API documentation in HTML format.

---

### POST /api/v1/stream

Synthesizes speech from text with streaming audio response.

**Request Body (form-data):**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to synthesize |
| `voice` | string | Yes | Voice ID (see available voices below) |
| `steps` | integer | No | Diffusion steps (default: 7, higher = better quality) |

**Response:**
- Content-Type: `audio/x-wav`
- Streams WAV audio data in chunks

**Example with curl:**

```bash
curl -X POST http://localhost:5000/api/v1/stream \
  -d "text=Hello, this is a test of the streaming API." \
  -d "voice=f-us-1" \
  -d "steps=7" \
  --output output.wav
```

**Example with Python:**

```python
import requests

response = requests.post(
    "http://localhost:5000/api/v1/stream",
    data={
        "text": "Hello, this is a test.",
        "voice": "f-us-1",
        "steps": 7
    },
    stream=True
)

with open("output.wav", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
```

---

### POST /api/v1/static

Synthesizes speech from text and returns complete audio file.

**Request Body (form-data):**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to synthesize |
| `voice` | string | Yes | Voice ID |

**Response:**
- Content-Type: `audio/wav`
- Returns complete WAV file

**Example:**

```bash
curl -X POST http://localhost:5000/api/v1/static \
  -d "text=Hello world" \
  -d "voice=m-us-1" \
  --output output.wav
```

---

## Available Voices

| Voice ID | Description |
|----------|-------------|
| `f-us-1` | Female US English #1 |
| `f-us-2` | Female US English #2 |
| `f-us-3` | Female US English #3 |
| `f-us-4` | Female US English #4 |
| `m-us-1` | Male US English #1 |
| `m-us-2` | Male US English #2 |
| `m-us-3` | Male US English #3 |
| `m-us-4` | Male US English #4 |

---

## Error Responses

All errors return JSON with an `error` field:

```json
{
  "error": "Missing required fields. Please include \"text\" and \"voice\" in your request."
}
```

**Common errors:**
- `400`: Missing required fields or invalid voice selection

---

## Testing

Use the provided test client:

```bash
# List available voices
python test_api_client.py --list-voices

# Check server status
python test_api_client.py --check-server

# Synthesize speech
python test_api_client.py -t "Hello world" -v f-us-1 -o output.wav

# With custom diffusion steps
python test_api_client.py -t "Hello world" -v m-us-2 -o output.wav -s 10
```

---

## Starting the Server

```bash
python api.py
```

The server starts on `http://0.0.0.0:5000` by default.