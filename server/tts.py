from pathlib import Path
from google.cloud import texttospeech


def synthesize(text):
    """Convert text into MP3 audio using Google's Text-to-Speech API."""
    # Reads the credentials from GOOGLE_APPLICATION_CREDENTIALS.
    client = texttospeech.TextToSpeechClient()

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

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )

    return response.audio_content


if __name__ == "__main__":
    # This test runs only when you execute this file directly.
    audio = synthesize("Hello! This is Iris speaking.")

    # Save the test recording outside the Git repository.
    output = Path.home() / "Downloads" / "iris-test.mp3"
    output.write_bytes(audio)

    print(f"Audio saved to: {output}")