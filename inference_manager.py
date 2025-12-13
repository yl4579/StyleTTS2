"""
GPU-safe inference manager with request queuing and semaphore control.
Handles concurrent TTS requests with low latency and prevents GPU contention.
"""

import asyncio
import time
import logging
from typing import Optional, List
import numpy as np
import msinference

logger = logging.getLogger(__name__)


class InferenceManager:
    """Manages GPU inference with concurrency control and queuing."""

    def __init__(
        self, max_queue_size: int = 100, voice_list: Optional[List[str]] = None
    ):
        """
        Initialize inference manager.

        Args:
            max_queue_size: Maximum number of pending inference requests
            voice_list: List of voice IDs to precompute styles for
        """
        self.gpu_semaphore = asyncio.Semaphore(1)  # Single GPU access at a time
        self.max_queue_size = max_queue_size
        self.queue_size = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.voices = {}

        # Precompute voice styles
        if voice_list:
            logger.info("Precomputing %d voice styles...", len(voice_list))
            for voice_id in voice_list:
                voice_path = f"voices/{voice_id}.wav"
                try:
                    self.voices[voice_id] = msinference.compute_style(voice_path)
                    logger.info("  ✓ Loaded voice: %s", voice_id)
                except Exception as e:
                    logger.error("  ✗ Failed to load voice %s: %s", voice_id, str(e))

        logger.info(
            "InferenceManager initialized with max_queue_size=%d, %d voices loaded",
            max_queue_size,
            len(self.voices),
        )

    async def generate_audio(
        self,
        text: str,
        voice: str,
        alpha: float = 0.3,
        beta: float = 0.7,
        diffusion_steps: int = 7,
        embedding_scale: float = 1.0,
    ) -> Optional[np.ndarray]:
        """
        Generate audio from text using specified voice.

        Args:
            text: Input text to synthesize
            voice: Voice ID (e.g., 'f-us-1', 'm-us-2')
            alpha: Style weight (default 0.3)
            beta: Prosody weight (default 0.7)
            diffusion_steps: Number of diffusion steps (default 7)
            embedding_scale: Style embedding scale (default 1.0)

        Returns:
            Audio waveform as numpy array (24kHz sample rate) or None on failure
        """
        # Check queue capacity
        if self.queue_size >= self.max_queue_size:
            logger.warning(
                "Queue full, rejecting request (queue_size=%d)", self.queue_size
            )
            self.failed_requests += 1
            return None

        self.queue_size += 1
        self.total_requests += 1

        try:
            # Acquire GPU semaphore
            async with self.gpu_semaphore:
                start_time = time.time()

                # Run inference in thread pool to avoid blocking event loop
                loop = asyncio.get_event_loop()
                wav = await loop.run_in_executor(
                    None,
                    self._inference_sync,
                    text,
                    voice,
                    alpha,
                    beta,
                    diffusion_steps,
                    embedding_scale,
                )

                elapsed = time.time() - start_time
                logger.info(
                    "Generated audio for text='%s...' voice=%s in %.2fs",
                    text[:30],
                    voice,
                    elapsed,
                )

                return wav

        except Exception as e:
            logger.error("Inference failed: %s", str(e), exc_info=True)
            self.failed_requests += 1
            return None
        finally:
            self.queue_size -= 1

    def _inference_sync(
        self,
        text: str,
        voice: str,
        alpha: float,
        beta: float,
        diffusion_steps: int,
        embedding_scale: float,
    ) -> np.ndarray:
        """Synchronous inference call wrapped by async executor."""
        # Get precomputed voice style
        if voice not in self.voices:
            raise ValueError(f"Voice '{voice}' not found in precomputed voices")

        voice_style = self.voices[voice]

        # Use msinference with precomputed style tensor
        wav = msinference.inference(
            text,
            voice_style,
            alpha=alpha,
            beta=beta,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale,
        )
        return wav

    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            "queue_size": self.queue_size,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (
                (self.total_requests - self.failed_requests) / self.total_requests
                if self.total_requests > 0
                else 1.0
            ),
        }
