"""
Production-grade WebSocket TTS server with FastAPI.
Handles chunked text input and streams base64 audio output.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tortoise.utils.text import split_and_recombine_text
import uvicorn

from inference_manager import InferenceManager
from audio_streamer import AudioStreamer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
inference_manager = None
audio_streamer = AudioStreamer(sample_rate=24000)

# Available voices
AVAILABLE_VOICES = [
    "f-us-1",
    "f-us-2",
    "f-us-3",
    "f-us-4",
    "m-us-1",
    "m-us-2",
    "m-us-3",
    "m-us-4",
]

# Server state
server_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    global server_ready, inference_manager
    logger.info("Starting StyleTTS2 WebSocket server...")

    # Download NLTK data if needed
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        logger.info("Downloading NLTK punkt_tab...")
        nltk.download("punkt_tab", quiet=True)

    # Initialize inference manager with precomputed voices
    logger.info("Precomputing voice styles...")
    inference_manager = InferenceManager(
        max_queue_size=100, voice_list=AVAILABLE_VOICES
    )
    logger.info("Models loaded and voices precomputed")

    server_ready = True
    logger.info("Server ready to accept WebSocket connections")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down server...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="StyleTTS2 WebSocket API",
    description="Production TTS streaming service with WebSocket support",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not server_ready:
        raise HTTPException(status_code=503, detail="Server initializing")

    stats = inference_manager.get_stats()
    return {"status": "healthy", "ready": server_ready, "stats": stats}


@app.get("/voices")
async def list_voices():
    """List available voices."""
    return {"voices": AVAILABLE_VOICES}


class ConnectionManager:
    """Manages WebSocket connection state and text buffering."""

    def __init__(self, websocket: WebSocket, connection_id: str):
        self.websocket = websocket
        self.connection_id = connection_id
        self.text_buffer = []
        self.voice = "f-us-1"  # Default voice
        self.last_activity = time.time()
        self.active = True

    async def send_json(self, data: dict):
        """Send JSON message to client."""
        if not self.active:
            return
        try:
            await self.websocket.send_json(data)
            self.last_activity = time.time()
        except Exception as e:
            logger.debug(f"Failed to send message (connection likely closed): {e}")
            self.active = False

    async def send_error(self, error_message: str):
        """Send error message to client."""
        if self.active:
            await self.send_json({"error": error_message, "isFinal": True})

    def add_text(self, text: str):
        """Add text chunk to buffer."""
        if text:
            self.text_buffer.append(text)
            self.last_activity = time.time()

    def get_buffered_text(self) -> str:
        """Get and clear text buffer."""
        text = " ".join(self.text_buffer)
        self.text_buffer = []
        return text

    def is_idle(self, timeout_seconds: int = 120) -> bool:
        """Check if connection has been idle for too long."""
        return (time.time() - self.last_activity) > timeout_seconds


@app.websocket("/ws/tts")
async def websocket_tts_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for TTS streaming.

    Expected input JSON:
    {
        "text": "chunk of text",
        "voice": "f-us-1",  # optional, defaults to f-us-1
        "flush": true/false  # if true, generate audio from buffered text
    }

    Output JSON:
    {
        "audio": "base64_encoded_audio_bytes",
        "isFinal": true/false
    }
    """
    await websocket.accept()

    connection_id = f"{websocket.client.host}:{websocket.client.port}"
    conn_manager = ConnectionManager(websocket, connection_id)

    logger.info(f"WebSocket connected: {connection_id}")

    # Send welcome message
    await conn_manager.send_json(
        {
            "status": "connected",
            "available_voices": AVAILABLE_VOICES,
            "message": "Send text chunks with 'flush: true' to generate audio",
        }
    )

    try:
        # Start idle timeout monitor
        timeout_task = asyncio.create_task(monitor_idle_timeout(conn_manager))

        while conn_manager.active:
            try:
                # Receive message with timeout
                data = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)

                # Parse message
                text = data.get("text", "").strip()
                voice = data.get("voice", conn_manager.voice).lower()
                flush = data.get("flush", False)

                # Validate voice
                if voice not in AVAILABLE_VOICES:
                    await conn_manager.send_error(
                        f"Invalid voice '{voice}'. Available: {AVAILABLE_VOICES}"
                    )
                    continue

                conn_manager.voice = voice

                # Add text to buffer
                if text:
                    conn_manager.add_text(text)

                # Generate audio if flush is requested
                if flush:
                    buffered_text = conn_manager.get_buffered_text()

                    if not buffered_text:
                        await conn_manager.send_json({"audio": "", "isFinal": True})
                        continue

                    # Split long text into manageable chunks
                    text_chunks = split_and_recombine_text(buffered_text)

                    logger.info(
                        f"Processing {len(text_chunks)} text chunks for {connection_id}, "
                        f"voice={voice}"
                    )

                    # Process each text chunk
                    for chunk_idx, text_chunk in enumerate(text_chunks):
                        is_last_text_chunk = chunk_idx == len(text_chunks) - 1

                        # Generate audio
                        audio_wav = await inference_manager.generate_audio(
                            text=text_chunk,
                            voice=voice,
                            alpha=0.3,
                            beta=0.7,
                            diffusion_steps=7,
                            embedding_scale=1.0,
                        )

                        if audio_wav is None:
                            await conn_manager.send_error(
                                "Audio generation failed or queue full"
                            )
                            break

                        # Send complete audio as single base64-encoded WAV
                        base64_audio = audio_streamer.encode_full_audio(audio_wav)
                        await conn_manager.send_json(
                            {
                                "audio": base64_audio,
                                "isFinal": is_last_text_chunk,
                            }
                        )

                        logger.info(
                            f"Sent complete audio for chunk {chunk_idx + 1}/{len(text_chunks)}"
                        )

            except asyncio.TimeoutError:
                # No message received, continue loop
                continue

            except WebSocketDisconnect as e:
                # Client disconnected gracefully
                logger.info(f"Client disconnected: {connection_id} (code: {e.code})")
                break

            except json.JSONDecodeError:
                await conn_manager.send_error("Invalid JSON format")

            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                await conn_manager.send_error(f"Processing error: {str(e)}")

        # Cancel timeout monitor
        timeout_task.cancel()

    except WebSocketDisconnect as e:
        logger.info(f"WebSocket disconnected: {connection_id} (code: {e.code})")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        if conn_manager.active:
            conn_manager.active = False
        logger.info(f"Connection closed: {connection_id}")


async def monitor_idle_timeout(
    conn_manager: ConnectionManager, timeout_seconds: int = 120
):
    """Monitor connection for idle timeout (default 2 minutes)."""
    try:
        while conn_manager.active:
            await asyncio.sleep(30)  # Check every 30 seconds

            if conn_manager.is_idle(timeout_seconds):
                logger.warning(
                    f"Connection {conn_manager.connection_id} idle for {timeout_seconds}s, closing"
                )
                await conn_manager.send_json(
                    {
                        "status": "timeout",
                        "message": f"Connection idle for {timeout_seconds} seconds",
                        "isFinal": True,
                    }
                )
                conn_manager.active = False
                await conn_manager.websocket.close()
                break
    except asyncio.CancelledError:
        pass


def main():
    """Run the WebSocket server."""
    import sys

    # Parse arguments
    host = "0.0.0.0"
    port = 8765

    if len(sys.argv) > 1:
        port = int(sys.argv[1])

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )


if __name__ == "__main__":
    main()
