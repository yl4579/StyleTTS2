"""
Audio streaming utilities for WebSocket TTS.
Handles MP3 encoding and base64 chunking for incremental audio delivery.
"""

import io
import base64
import numpy as np
from pydub import AudioSegment
import logging

logger = logging.getLogger(__name__)


class AudioStreamer:
    """Streams audio as base64-encoded MP3 chunks."""

    def __init__(self, sample_rate: int = 24000):
        """
        Initialize audio streamer.

        Args:
            sample_rate: Audio sample rate in Hz (default 24000)
        """
        self.sample_rate = sample_rate

    def create_mp3_bytes(self, audio_data: np.ndarray) -> bytes:
        """
        Convert numpy audio array to MP3 bytes.

        Args:
            audio_data: Audio waveform as numpy array

        Returns:
            MP3 file bytes
        """
        # Normalize audio data to 16-bit range
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)

        # Create AudioSegment from numpy array
        audio_segment = AudioSegment(
            audio_data.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=2,  # 16-bit
            channels=1,  # Mono
        )

        # Export to MP3
        buffer = io.BytesIO()
        audio_segment.export(buffer, format="mp3", bitrate="128k")
        buffer.seek(0)
        return buffer.read()

    def encode_audio_chunk(self, audio_data: np.ndarray) -> str:
        """
        Encode audio data to base64 MP3 string.

        Args:
            audio_data: Audio waveform as numpy array

        Returns:
            Base64-encoded MP3 audio string
        """
        mp3_bytes = self.create_mp3_bytes(audio_data)
        return base64.b64encode(mp3_bytes).decode("utf-8")

    def stream_audio_frames(self, audio_data: np.ndarray, chunk_duration_ms: int = 300):
        """
        Generator that yields audio in fixed-duration base64 MP3 chunks.

        Args:
            audio_data: Complete audio waveform as numpy array
            chunk_duration_ms: Duration of each chunk in milliseconds

        Yields:
            Tuples of (base64_audio, is_first_chunk, is_final_chunk)
        """
        chunk_samples = int(self.sample_rate * chunk_duration_ms / 1000)
        total_samples = len(audio_data)

        if total_samples == 0:
            logger.warning("Empty audio data received")
            return

        num_chunks = (total_samples + chunk_samples - 1) // chunk_samples

        for i in range(num_chunks):
            start_idx = i * chunk_samples
            end_idx = min(start_idx + chunk_samples, total_samples)

            chunk = audio_data[start_idx:end_idx]
            is_first = i == 0
            is_final = i == num_chunks - 1

            # Encode chunk as MP3
            base64_audio = self.encode_audio_chunk(chunk)

            yield base64_audio, is_first, is_final

    def encode_full_audio(self, audio_data: np.ndarray) -> str:
        """
        Encode complete audio as MP3 to base64.

        Args:
            audio_data: Audio waveform as numpy array

        Returns:
            Base64-encoded complete MP3 file
        """
        return self.encode_audio_chunk(audio_data)
