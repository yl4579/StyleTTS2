#!/usr/bin/env python3
"""
WebSocket TTS client for testing StyleTTS2 WebSocket API.
"""

import asyncio
import json
import base64
import sys
import argparse
import websockets
from pathlib import Path


async def test_websocket_tts(
    uri: str, text: str, voice: str = "m-us-1", output_file: str = "output.mp3"
):
    """
    Test WebSocket TTS endpoint.

    Args:
        uri: WebSocket URI (e.g., ws://localhost:8765/ws/tts)
        text: Text to synthesize
        voice: Voice ID
        output_file: Output MP3 file path
    """
    print(f"Connecting to {uri}...")

    audio_chunks = []

    async with websockets.connect(uri) as websocket:
        # Receive welcome message
        welcome = await websocket.recv()
        print(f"Server: {welcome}")

        # Send text with flush
        message = {"text": text, "voice": voice, "flush": True}

        print(f"\nSending: {text[:50]}...")
        print(f"Voice: {voice}")

        await websocket.send(json.dumps(message))

        # Receive audio chunks
        chunk_count = 0
        while True:
            response = await websocket.recv()
            data = json.loads(response)

            if "error" in data:
                print(f"Error: {data['error']}")
                break

            if "audio" in data and data["audio"]:
                audio_chunks.append(data["audio"])
                chunk_count += 1
                print(
                    f"Received chunk {chunk_count} (isFinal: {data.get('isFinal', False)})"
                )

            if data.get("isFinal"):
                print(f"\nReceived {chunk_count} audio chunks")
                break

        # Combine and save audio
        if audio_chunks:
            print(f"Saving audio to {output_file}...")

            # Decode all chunks
            audio_bytes = b"".join(base64.b64decode(chunk) for chunk in audio_chunks)

            # Write to file
            with open(output_file, "wb") as f:
                f.write(audio_bytes)

            print(f"✓ Saved {len(audio_bytes)} bytes to {output_file}")
        else:
            print("No audio received")


async def test_chunked_streaming(
    uri: str, text_chunks: list, voice: str = "m-us-1", output_file: str = "output.mp3"
):
    """
    Test chunked text streaming (simulates real-time text input).

    Args:
        uri: WebSocket URI
        text_chunks: List of text chunks to send
        voice: Voice ID
        output_file: Output MP3 file path
    """
    print(f"Connecting to {uri}...")

    audio_chunks = []

    async with websockets.connect(uri) as websocket:
        # Receive welcome message
        welcome = await websocket.recv()
        print(f"Server: {welcome}")

        # Send text chunks incrementally
        for i, chunk in enumerate(text_chunks):
            is_last = i == len(text_chunks) - 1

            message = {
                "text": chunk,
                "voice": voice,
                "flush": is_last,  # Only flush on last chunk
            }

            print(f"\nSending chunk {i + 1}/{len(text_chunks)}: {chunk[:30]}...")
            await websocket.send(json.dumps(message))

            # Small delay to simulate real-time input
            if not is_last:
                await asyncio.sleep(0.1)

        # Receive audio
        chunk_count = 0
        while True:
            response = await websocket.recv()
            data = json.loads(response)

            if "error" in data:
                print(f"Error: {data['error']}")
                break

            if "audio" in data and data["audio"]:
                audio_chunks.append(data["audio"])
                chunk_count += 1
                print(f"Received audio chunk {chunk_count}")

            if data.get("isFinal"):
                print(f"\nReceived {chunk_count} total audio chunks")
                break

        # Save audio
        if audio_chunks:
            audio_bytes = b"".join(base64.b64decode(chunk) for chunk in audio_chunks)
            with open(output_file, "wb") as f:
                f.write(audio_bytes)
            print(f"✓ Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="WebSocket TTS Client")
    parser.add_argument(
        "--uri",
        default="ws://localhost:8765/ws/tts",
        help="WebSocket URI (default: ws://localhost:8765/ws/tts)",
    )
    parser.add_argument(
        "--text",
        default="Hello, this is a test of the WebSocket TTS streaming service.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--voice",
        default="m-us-1",
        choices=[
            "f-us-1",
            "f-us-2",
            "f-us-3",
            "f-us-4",
            "m-us-1",
            "m-us-2",
            "m-us-3",
            "m-us-4",
        ],
        help="Voice ID",
    )
    parser.add_argument("--output", default="output.mp3", help="Output MP3 file")
    parser.add_argument(
        "--chunked", action="store_true", help="Test chunked streaming mode"
    )

    args = parser.parse_args()

    try:
        if args.chunked:
            # Split text into chunks for testing
            words = args.text.split()
            chunks = [" ".join(words[i : i + 3]) for i in range(0, len(words), 3)]
            print(f"Testing chunked mode with {len(chunks)} chunks")
            asyncio.run(
                test_chunked_streaming(args.uri, chunks, args.voice, args.output)
            )
        else:
            asyncio.run(
                test_websocket_tts(args.uri, args.text, args.voice, args.output)
            )
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
