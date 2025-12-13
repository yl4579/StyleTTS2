#!/usr/bin/env python3
"""
StyleTTS2 Streaming API Client
This script demonstrates how to use the StyleTTS2 streaming API endpoint.
"""

import requests
import sys
from pathlib import Path
import argparse


class StyleTTS2Client:
    """Client for interacting with StyleTTS2 Streaming API"""

    def __init__(self, base_url="http://localhost:5000"):
        """
        Initialize the client

        Args:
            base_url (str): Base URL of the API server
        """
        self.base_url = base_url.rstrip("/")
        self.stream_endpoint = f"{self.base_url}/api/v1/stream"

        # Available voices
        self.available_voices = [
            "f-us-1",
            "f-us-2",
            "f-us-3",
            "f-us-4",
            "m-us-1",
            "m-us-2",
            "m-us-3",
            "m-us-4",
        ]

    def check_server(self):
        """Check if the API server is running"""
        try:
            response = requests.get(self.base_url, timeout=5)
            if response.status_code == 200:
                print(f"✓ API server is running at {self.base_url}")
                return True
            else:
                print(f"✗ API server returned status code: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"✗ Cannot connect to API server at {self.base_url}")
            print("  Make sure the server is running with: python api.py")
            return False
        except Exception as e:
            print(f"✗ Error checking server: {e}")
            return False

    def synthesize_stream(self, text, voice, output_file, steps=7):
        """
        Synthesize speech using the streaming endpoint

        Args:
            text (str): Text to synthesize
            voice (str): Voice ID to use
            output_file (str): Path to save the output WAV file
            steps (int): Number of diffusion steps (default: 7, higher=better quality but slower)

        Returns:
            bool: True if successful, False otherwise
        """
        if voice not in self.available_voices:
            print(f"✗ Invalid voice '{voice}'")
            print(f"  Available voices: {', '.join(self.available_voices)}")
            return False

        print(f"\n{'=' * 60}")
        print("Synthesizing with streaming API")
        print(f"{'=' * 60}")
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"Voice: {voice}")
        print(f"Diffusion steps: {steps}")
        print(f"Output: {output_file}")
        print(f"{'=' * 60}\n")

        # Prepare the request
        data = {"text": text, "voice": voice, "steps": str(steps)}

        try:
            # Make streaming request
            print("Sending request to API...")
            response = requests.post(
                self.stream_endpoint,
                data=data,
                stream=True,
                timeout=300,  # 5 minutes timeout
            )

            if response.status_code != 200:
                print(f"✗ API returned error status: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"  Error: {error_data.get('error', 'Unknown error')}")
                except Exception:
                    print(f"  Response: {response.text[:200]}")
                return False

            # Create output directory if it doesn't exist
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Stream the audio data to file
            print("Receiving audio stream...")
            chunk_count = 0
            total_bytes = 0

            with open(output_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        chunk_count += 1
                        total_bytes += len(chunk)

                        # Print progress every 10 chunks
                        if chunk_count % 10 == 0:
                            print(
                                f"  Received {chunk_count} chunks ({total_bytes:,} bytes)...",
                                end="\r",
                            )

            print("\n\n✓ Audio saved successfully!")
            print(f"  File: {output_file}")
            print(f"  Size: {total_bytes:,} bytes ({total_bytes / 1024:.2f} KB)")
            print(f"  Chunks received: {chunk_count}")

            return True

        except requests.exceptions.Timeout:
            print(
                "✗ Request timed out. The text might be too long or the server is slow."
            )
            return False
        except requests.exceptions.ConnectionError:
            print("✗ Connection error. Make sure the API server is running.")
            return False
        except Exception as e:
            print(f"✗ Error during synthesis: {e}")
            import traceback

            traceback.print_exc()
            return False

    def list_voices(self):
        """List all available voices"""
        print("\nAvailable Voices:")
        print("=" * 40)
        for i, voice in enumerate(self.available_voices, 1):
            voice_type = "Female" if voice.startswith("f-") else "Male"
            print(f"{i}. {voice:10s} - {voice_type} US English")
        print("=" * 40)


def main():
    parser = argparse.ArgumentParser(
        description="StyleTTS2 Streaming API Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple synthesis
  python test_api_client.py -t "Hello world" -v f-us-1 -o output.wav
  
  # With custom diffusion steps for better quality
  python test_api_client.py -t "Hello world" -v m-us-2 -o output.wav -s 10
  
  # Long text synthesis
  python test_api_client.py -t "This is a longer text..." -v f-us-3 -o long.wav
  
  # List available voices
  python test_api_client.py --list-voices
  
  # Custom server URL
  python test_api_client.py -t "Hello" -v f-us-1 -o out.wav --url http://192.168.1.100:5000
        """,
    )

    parser.add_argument("-t", "--text", type=str, help="Text to synthesize")
    parser.add_argument(
        "-v", "--voice", type=str, help="Voice ID (e.g., f-us-1, m-us-2)"
    )
    parser.add_argument("-o", "--output", type=str, help="Output WAV file path")
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=7,
        help="Diffusion steps (default: 7, higher=better quality)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:5000",
        help="API server URL (default: http://localhost:5000)",
    )
    parser.add_argument(
        "--list-voices", action="store_true", help="List available voices and exit"
    )
    parser.add_argument(
        "--check-server",
        action="store_true",
        help="Check if server is running and exit",
    )

    args = parser.parse_args()

    # Initialize client
    client = StyleTTS2Client(base_url=args.url)

    # Handle list voices
    if args.list_voices:
        client.list_voices()
        return 0

    # Handle check server
    if args.check_server:
        if client.check_server():
            return 0
        else:
            return 1

    # Validate required arguments
    if not args.text or not args.voice or not args.output:
        parser.print_help()
        print("\n✗ Error: --text, --voice, and --output are required for synthesis")
        print("\nQuick start:")
        print('  python test_api_client.py -t "Hello world" -v f-us-1 -o output.wav')
        return 1

    # Check server before synthesis
    if not client.check_server():
        return 1

    # Perform synthesis
    success = client.synthesize_stream(
        text=args.text, voice=args.voice, output_file=args.output, steps=args.steps
    )

    if success:
        print("\n✓ Synthesis completed successfully!")
        print("\nYou can play the audio with:")
        print(f"  ffplay {args.output}")
        print("  or")
        print(f"  aplay {args.output}")
        return 0
    else:
        print("\n✗ Synthesis failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
