import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import whisper
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
model = None

# All ways a caller might refer to Bangla/Bengali → Whisper's canonical code
_BANGLA_ALIASES = {
    "bangla", "bengali", "bengali (bangladesh)", "bd",
    "bangladesh", "বাংলা", "বাংলাদেশ",
}


def _resolve_language(lang: Optional[str]) -> Optional[str]:
    """Normalise language input; map Bangla aliases → 'bn'."""
    if lang is None:
        return None
    normalised = lang.strip().lower()
    if normalised in _BANGLA_ALIASES or normalised == "bn":
        return "bn"
    # Validate against Whisper's known language table
    if normalised not in whisper.tokenizer.LANGUAGES:
        raise ValueError(
            f"Unknown language '{lang}'. "
            "Use an ISO 639-1 code (e.g. 'bn', 'en', 'hi') or leave blank for auto-detect."
        )
    return normalised


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print(f"Loading Whisper model: {WHISPER_MODEL_NAME}")
    model = whisper.load_model(WHISPER_MODEL_NAME)
    print("Model loaded.")
    yield
    model = None


app = FastAPI(
    title="Whisper STT API",
    description="Speech-to-Text API powered by OpenAI Whisper",
    version="1.0.0",
    lifespan=lifespan,
)

SUPPORTED_EXTENSIONS = {
    ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a",
    ".wav", ".webm", ".ogg", ".flac",
}


@app.get("/health")
def health():
    return {"status": "ok", "model": WHISPER_MODEL_NAME}


@app.get("/languages")
def languages():
    """List all language codes supported by the loaded Whisper model."""
    return {
        "languages": {
            code: name for code, name in sorted(whisper.tokenizer.LANGUAGES.items())
        }
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    task: str = Form("transcribe"),
    temperature: float = Form(0.0),
    word_timestamps: bool = Form(False),
):
    """
    Transcribe or translate audio.
    - **file**: audio file (mp3, wav, m4a, flac, ogg, webm, etc.)
    - **language**: ISO 639-1 code, or **bangla / বাংলা / bengali** for Bangla (auto-detect if omitted)
    - **task**: `transcribe` or `translate` (translate → English)
    - **temperature**: sampling temperature (0.0 = greedy)
    - **word_timestamps**: include per-word timestamps
    """
    suffix = Path(file.filename).suffix.lower() if file.filename else ".tmp"
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    if task not in ("transcribe", "translate"):
        raise HTTPException(status_code=400, detail="task must be 'transcribe' or 'translate'")

    try:
        resolved_lang = _resolve_language(language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        start = time.perf_counter()
        result = model.transcribe(
            tmp_path,
            language=resolved_lang,
            task=task,
            temperature=temperature,
            word_timestamps=word_timestamps,
        )
        elapsed = round(time.perf_counter() - start, 3)
    finally:
        os.unlink(tmp_path)

    segments = [
        {
            "id": s["id"],
            "start": round(s["start"], 3),
            "end": round(s["end"], 3),
            "text": s["text"].strip(),
            **({"words": s.get("words", [])} if word_timestamps else {}),
        }
        for s in result["segments"]
    ]

    return JSONResponse({
        "text": result["text"].strip(),
        "language": result["language"],
        "task": task,
        "duration_seconds": elapsed,
        "segments": segments,
    })
