# StyleTTS2 WebSocket API Documentation

## Overview

The WebSocket API provides real-time text-to-speech streaming with bidirectional communication. It supports chunked text input and returns base64-encoded MP3 audio.

## Quick Start

```bash
# Start WebSocket server
python ws_server.py

# Test with client
python test_ws_client.py --text "Hello world" --voice f-us-1
```

## Endpoints

### WebSocket: `/ws/tts`

```
ws://localhost:8765/ws/tts
```

### REST: `/health`

Health check endpoint.

```bash
curl http://localhost:8765/health
```

Response:
```json
{
  "status": "healthy",
  "ready": true,
  "stats": {
    "queue_size": 0,
    "total_requests": 10,
    "failed_requests": 0,
    "success_rate": 1.0
  }
}
```

### REST: `/voices`

List available voices.

```bash
curl http://localhost:8765/voices
```

Response:
```json
{
  "voices": ["f-us-1", "f-us-2", "f-us-3", "f-us-4", "m-us-1", "m-us-2", "m-us-3", "m-us-4"]
}
```

## WebSocket Message Format

### Input (Client → Server)

```json
{
  "text": "Text to synthesize",
  "voice": "f-us-1",
  "flush": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Text chunk to synthesize |
| `voice` | string | No | Voice ID (default: f-us-1) |
| `flush` | boolean | No | Set `true` to generate audio from buffered text |

### Output (Server → Client)

**Audio response:**
```json
{
  "audio": "base64_encoded_mp3_data",
  "isFinal": false
}
```

**Welcome message (on connect):**
```json
{
  "status": "connected",
  "available_voices": ["f-us-1", ...],
  "message": "Send text chunks with 'flush: true' to generate audio"
}
```

**Error response:**
```json
{
  "error": "Error message",
  "isFinal": true
}
```

## Available Voices

| Voice ID | Description |
|----------|-------------|
| `f-us-1` to `f-us-4` | Female US English |
| `m-us-1` to `m-us-4` | Male US English |

## Usage Patterns

### Single Text (Simple)

Send all text at once with `flush: true`:

```python
await websocket.send(json.dumps({
    "text": "Hello, this is a complete sentence.",
    "voice": "f-us-1",
    "flush": True
}))
```

### Chunked Streaming (Real-time)

Buffer text chunks, then flush to generate audio:

```python
# Send chunks without flushing
await websocket.send(json.dumps({"text": "Hello, ", "voice": "f-us-1", "flush": False}))
await websocket.send(json.dumps({"text": "this is ", "flush": False}))
await websocket.send(json.dumps({"text": "streaming.", "flush": True}))  # Flush to generate
```

## Code Examples

### Python

```python
import asyncio
import websockets
import json
import base64

async def synthesize(text, voice="f-us-1"):
    uri = "ws://localhost:8765/ws/tts"
    async with websockets.connect(uri) as ws:
        # Receive welcome
        await ws.recv()
        
        # Send text
        await ws.send(json.dumps({
            "text": text,
            "voice": voice,
            "flush": True
        }))
        
        # Receive audio
        audio_chunks = []
        while True:
            response = json.loads(await ws.recv())
            if "audio" in response and response["audio"]:
                audio_chunks.append(base64.b64decode(response["audio"]))
            if response.get("isFinal"):
                break
        
        # Save audio
        with open("output.mp3", "wb") as f:
            f.write(b"".join(audio_chunks))

asyncio.run(synthesize("Hello world"))
```

### JavaScript

```javascript
const ws = new WebSocket('ws://localhost:8765/ws/tts');

ws.onopen = () => {
    ws.send(JSON.stringify({
        text: 'Hello world',
        voice: 'f-us-1',
        flush: true
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.audio) {
        // Decode base64 and play audio
        const audioBytes = atob(data.audio);
        // ... play audio
    }
    if (data.isFinal) {
        ws.close();
    }
};
```

## Configuration

Default settings in `ws_server.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| Port | 8765 | WebSocket server port |
| Idle timeout | 120s | Connection idle timeout |
| Max queue | 100 | Maximum pending requests |
| Ping interval | 20s | WebSocket keepalive |

## Testing

```bash
# Simple test
python test_ws_client.py --text "Hello" --voice f-us-1 --output test.mp3

# Chunked streaming test
python test_ws_client.py --text "Long text here" --chunked

# Custom server
python test_ws_client.py --uri ws://192.168.1.100:8765/ws/tts --text "Hello"
```

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| Invalid voice | Voice ID not recognized | Use voice from `/voices` endpoint |
| Queue full | Too many pending requests | Retry after delay |
| Connection timeout | Idle for 2+ minutes | Reconnect |
| Invalid JSON | Malformed message | Check JSON syntax |
