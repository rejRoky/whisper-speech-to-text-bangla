# Whisper Speech-to-Text API (Bangla / বাংলা)

A Dockerized REST API for speech-to-text transcription powered by [OpenAI Whisper](https://github.com/openai/whisper), with first-class support for **Bangla (Bengali)**.

## Features

- Transcribe audio in 99+ languages with auto-detection
- Native Bangla aliases — pass `bangla`, `bengali`, `বাংলা`, or `bn`
- Translate any language to English
- Per-word timestamps
- GPU-accelerated (CUDA 12.1 + cuDNN 8)
- Whisper model pre-downloaded at build time for instant startup

## Requirements

- Docker & Docker Compose
- NVIDIA GPU with CUDA 12.1 support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Quick Start

```bash
# 1. Clone the repo
git clone git@github.com:rejRoky/whisper-speech-to-text-bangla.git
cd whisper-speech-to-text-bangla

# 2. (Optional) Set model size — default is medium
cp .env.example .env
# edit .env: WHISPER_MODEL=large-v3

# 3. Build and run
docker compose up --build
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### `GET /health`
Returns service status and loaded model name.

```json
{ "status": "ok", "model": "medium" }
```

### `GET /languages`
Lists all language codes supported by the loaded Whisper model.

### `POST /transcribe`
Transcribe or translate an audio file.

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file (mp3, wav, m4a, flac, ogg, webm, etc.) |
| `language` | string | auto-detect | ISO 639-1 code, or `bangla` / `বাংলা` / `bengali` |
| `task` | string | `transcribe` | `transcribe` or `translate` (→ English) |
| `temperature` | float | `0.0` | Sampling temperature |
| `word_timestamps` | bool | `false` | Include per-word timestamps |

**Example — Bangla transcription:**
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.mp3" \
  -F "language=bangla"
```

**Example — Auto-detect + word timestamps:**
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "word_timestamps=true"
```

**Response:**
```json
{
  "text": "আমার সোনার বাংলা আমি তোমায় ভালোবাসি",
  "language": "bn",
  "task": "transcribe",
  "duration_seconds": 1.23,
  "segments": [
    { "id": 0, "start": 0.0, "end": 3.5, "text": "আমার সোনার বাংলা আমি তোমায় ভালোবাসি" }
  ]
}
```

## Whisper Model Sizes

| Model | Parameters | Speed | Accuracy |
|---|---|---|---|
| `tiny` | 39M | fastest | lowest |
| `base` | 74M | fast | low |
| `small` | 244M | moderate | moderate |
| `medium` | 769M | slow | good |
| `large-v3` | 1550M | slowest | best |

Set via `WHISPER_MODEL` in `.env` or as a build arg:
```bash
WHISPER_MODEL=large-v3 docker compose up --build
```

## Testing

```bash
# Health check only
./test_api.sh

# Transcribe a file
./test_api.sh path/to/audio.mp3
```

## License

MIT
