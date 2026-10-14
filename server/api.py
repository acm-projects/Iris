"""Iris HTTP API.

Run from the project root:
    python -m uvicorn server.api:app --reload --host 127.0.0.1 --port 8000
Open http://127.0.0.1:8000/docs to try the endpoints in Swagger UI.
"""

from io import BytesIO
from threading import Lock

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from google.api_core.exceptions import GoogleAPICallError, RetryError
from google.auth.exceptions import GoogleAuthError
from pydantic import BaseModel, Field, field_validator

from server.tts import synthesize
from backend.engine.engine import play


# FastAPI uses this metadata to label the automatically generated API docs.
app = FastAPI(
    title="Iris Backend API",
    description="Python API for Iris: health, text-to-speech, and virtual microphone playback.",
    version="0.1.0",
)


@app.get("/health", tags=["Health"], summary="Check that the API is running")
def health() -> dict[str, str]:
    """Check the HTTP server only; does not contact Google or open devices."""
    # FastAPI converts this Python dictionary into a JSON response.
    return {"status": "ok", "service": "iris-api"}


class SpeechRequest(BaseModel):
    """The JSON body accepted by POST /tts; also describes it in Swagger UI."""

    text: str = Field(
        min_length=1,
        max_length=5000,
        description="Text to speak. Must not be blank or exceed 5,000 UTF-8 bytes.",
        examples=["Hello! This is Iris speaking."],
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("Text must not be blank.")
        # Google's limit is bytes, not characters; some characters use several bytes.
        if len(text.encode("utf-8")) > 5000:
            raise ValueError("Text must not exceed 5,000 UTF-8 bytes.")
        return text


@app.post(
    "/tts",
    tags=["Text-to-Speech"],
    summary="Generate MP3 speech from text",
    response_class=Response,
    responses={
        200: {
            "description": "MP3 audio using en-US-Standard-G.",
            "content": {"audio/mpeg": {"schema": {"type": "string", "format": "binary"}}},
        },
        502: {"description": "Google could not complete the speech request."},
        503: {"description": "Google credentials are missing or invalid."},
    },
)
def text_to_speech(request: SpeechRequest) -> Response:
    """Send text to Google and return audio. Does not play it through a device."""
    # A regular def lets FastAPI run this blocking Google call in a worker thread.
    try:
        audio = synthesize(request.text)
    except GoogleAuthError as exc:
        raise HTTPException(
            status_code=503,
            detail="Google credentials are unavailable. Check GOOGLE_APPLICATION_CREDENTIALS.",
        ) from exc
    except (GoogleAPICallError, RetryError) as exc:
        # Keep provider error details out of the public response.
        raise HTTPException(
            status_code=502,
            detail="Speech generation failed. Check Google API access, billing, and quota.",
        ) from exc

    # Send bytes directly instead of JSON; do not create a file on the server.
    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={"Content-Disposition": 'attachment; filename="iris-speech.mp3"'},
    )


# Keep individual uploads and decoded audio bounded for this local speech API.
MAX_MP3_BYTES = 10 * 1024 * 1024
MAX_AUDIO_SECONDS = 600
# This lock covers API requests in this process, not Electron's separate engine.
_microphone_lock = Lock()


class MicrophoneResult(BaseModel):
    """Returned only after playback has finished."""

    status: str
    device: str
    durationMs: int


def decode_mp3(data: bytes):
    """Convert MP3 bytes into float32 samples and their original sample rate."""
    import soundfile as sf

    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    try:
        with sf.SoundFile(BytesIO(data)) as recording:
            # Inspect the actual content rather than trusting the filename or MIME type.
            if recording.format != "MP3":
                raise HTTPException(status_code=415, detail="Upload an MP3 audio file.")
            if recording.channels not in (1, 2):
                raise HTTPException(status_code=400, detail="Audio must be mono or stereo.")
            rate = recording.samplerate
            max_frames = rate * MAX_AUDIO_SECONDS
            if recording.frames > max_frames:
                raise HTTPException(status_code=413, detail="Audio must be 10 minutes or shorter.")
            audio = recording.read(frames=max_frames + 1, dtype="float32")
    except (sf.SoundFileError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="The MP3 could not be decoded.") from exc

    if len(audio) == 0:
        raise HTTPException(status_code=400, detail="The MP3 contains no audio samples.")
    if len(audio) > max_frames:
        raise HTTPException(status_code=413, detail="Audio must be 10 minutes or shorter.")
    return audio, rate


@app.post(
    "/microphone",
    tags=["Virtual Microphone"],
    summary="Play an MP3 through the virtual microphone",
    response_model=MicrophoneResult,
    responses={
        400: {"description": "Empty or invalid audio."},
        409: {"description": "Another API playback request is in progress."},
        413: {"description": "File exceeds 10 MiB or audio exceeds 10 minutes."},
        415: {"description": "The uploaded audio is not MP3."},
        503: {"description": "Virtual microphone or audio dependencies unavailable."},
    },
)
def push_audio_to_microphone(
    file: UploadFile = File(description="MP3 returned by /tts, or another MP3 speech clip."),
) -> MicrophoneResult:
    """Upload MP3 audio and wait for virtual-mic playback to finish.

    Requires VB-CABLE on Windows or BlackHole on macOS. Select the matching
    microphone in your meeting app. Run one API worker, and do not use the
    Electron engine's separate speak/test-tone controls during this playback.
    This endpoint does not call Google or start the camera.
    """
    # Reject overlapping API playback instead of silently interrupting speech.
    if not _microphone_lock.acquire(blocking=False):
        file.file.close()
        raise HTTPException(status_code=409, detail="Virtual microphone is busy. Try again after playback.")
    try:
        data = file.file.read(MAX_MP3_BYTES + 1)
        if len(data) > MAX_MP3_BYTES:
            raise HTTPException(status_code=413, detail="MP3 file must not exceed 10 MiB.")
        try:
            audio, rate = decode_mp3(data)
            import sounddevice as sd

            # Reuse the teammate's device discovery, resampling, and playback.
            # None selects a virtual loopback device, not the default speakers.
            try:
                result = play(audio, rate, None)
            except sd.PortAudioError as exc:
                raise HTTPException(
                    status_code=503, detail="The virtual microphone could not play this audio."
                ) from exc
        except (ImportError, OSError, RuntimeError) as exc:
            raise HTTPException(
                status_code=503,
                detail="Audio playback unavailable. Check Python audio dependencies and the virtual microphone driver.",
            ) from exc
        return MicrophoneResult(status="played", **result)
    finally:
        file.file.close()
        _microphone_lock.release()
