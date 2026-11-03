"""Text-to-speech for Iris: turns text into MP3 bytes.

Two voice providers, picked by TTS_PROVIDER in Cue/.env:
  elevenlabs (default)  ElevenLabs API, needs ELEVENLABS_API_KEY
  google                Google Cloud TTS, needs GOOGLE_TTS_API_KEY or a service account

Both return MP3, so the rest of the app (api.py, the virtual mic, Electron)
doesn't care which one is used.
"""
import os
from pathlib import Path

import requests

# Cue/.env (gitignored) holds local settings and keys; never commit it.
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class TTSConfigError(Exception):
    """The provider isn't set up: missing or rejected key. Fix Cue/.env and restart."""


class TTSProviderError(Exception):
    """The provider couldn't make the audio right now (outage, quota, bad voice ID)."""


def load_env_file(path: Path = ENV_FILE) -> None:
    """Read KEY=VALUE lines from Cue/.env into the environment.

    Variables already set in the shell win, so a terminal `export` always
    overrides the file. Blank lines and lines starting with # are skipped.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file()


def provider() -> str:
    """Which voice service to use: "elevenlabs" (default) or "google"."""
    return os.environ.get("TTS_PROVIDER", "elevenlabs").strip().lower()


def synthesize(text: str) -> bytes:
    """Convert text into MP3 audio with the configured provider."""
    if provider() == "google":
        return synthesize_google(text)
    return synthesize_elevenlabs(text)


# --- ElevenLabs --------------------------------------------------------------

ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
# "Sarah", a built-in (premade) voice that free plans can use; override with
# ELEVENLABS_VOICE_ID. Voice Library voices need a paid plan (HTTP 402).
DEFAULT_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
# Flash v2.5 is ElevenLabs' fastest model: good for live conversation.
DEFAULT_MODEL = "eleven_flash_v2_5"


def synthesize_elevenlabs(text: str) -> bytes:
    """Convert text into MP3 audio using the ElevenLabs API.

    The key is only read from the environment and sent in a header; it is
    never logged or included in error messages.
    """
    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        raise TTSConfigError("ELEVENLABS_API_KEY is not set.")
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "").strip() or DEFAULT_VOICE_ID

    try:
        response = requests.post(
            ELEVENLABS_URL.format(voice_id=voice_id),
            params={"output_format": "mp3_44100_128"},
            headers={"xi-api-key": api_key, "Accept": "audio/mpeg"},
            json={
                "text": text,
                "model_id": os.environ.get("ELEVENLABS_MODEL", "").strip() or DEFAULT_MODEL,
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        raise TTSProviderError("Could not reach ElevenLabs.") from exc

    if response.status_code == 401:
        raise TTSConfigError("ElevenLabs rejected the API key.")
    if response.status_code == 402:
        # Usually a Voice Library voice on a free plan: pick a premade voice instead.
        raise TTSConfigError("This ElevenLabs voice needs a paid plan. Set ELEVENLABS_VOICE_ID to a premade voice.")
    if not response.ok:
        # Status only: ElevenLabs' error body can echo request details.
        raise TTSProviderError(f"ElevenLabs returned HTTP {response.status_code}.")
    return response.content


# --- Google Cloud (fallback) -------------------------------------------------

def synthesize_google(text: str) -> bytes:
    """Convert text into MP3 audio using Google's Text-to-Speech API.

    Auth: GOOGLE_TTS_API_KEY if set, otherwise a service-account JSON file
    named by GOOGLE_APPLICATION_CREDENTIALS.
    """
    # Imported here so the Google package is only needed when Google is used.
    from google.api_core.exceptions import GoogleAPICallError, RetryError
    from google.auth.exceptions import GoogleAuthError
    from google.cloud import texttospeech

    api_key = os.environ.get("GOOGLE_TTS_API_KEY", "").strip()
    try:
        client = (texttospeech.TextToSpeechClient(client_options={"api_key": api_key})
                  if api_key else texttospeech.TextToSpeechClient())
    except GoogleAuthError as exc:
        raise TTSConfigError("Google credentials are unavailable.") from exc

    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Your selected US English voice.
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Standard-G",
    )

    # Generate MP3 audio at normal speed and unchanged pitch.
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.0,
        pitch=0.0,
    )

    try:
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
    except GoogleAuthError as exc:
        raise TTSConfigError("Google credentials are unavailable.") from exc
    except (GoogleAPICallError, RetryError) as exc:
        if "API_KEY_INVALID" in str(exc) or "API key not valid" in str(exc):
            raise TTSConfigError("Google rejected the API key.") from exc
        raise TTSProviderError("Google could not generate speech.") from exc

    return response.audio_content


if __name__ == "__main__":
    # This test runs only when you execute this file directly.
    audio = synthesize("Hello! This is Iris speaking.")

    # Save the test recording outside the Git repository.
    output = Path.home() / "Downloads" / "iris-test.mp3"
    output.write_bytes(audio)

    print(f"Audio saved to: {output}")
