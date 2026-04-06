"""STT backend implementations: OpenAI Whisper API, Google Cloud STT, Wyoming proxy."""

import asyncio
import io
import json
import logging
import struct
import wave
from abc import ABC, abstractmethod

import numpy as np

_LOGGER = logging.getLogger(__name__)


class STTBackend(ABC):
    """Base class for STT backends."""

    @abstractmethod
    async def transcribe(
        self, audio_bytes: bytes, sample_rate: int, sample_width: int, channels: int, language: str
    ) -> str:
        """Transcribe audio bytes to text."""


def _pcm_to_wav(audio_bytes: bytes, sample_rate: int, sample_width: int, channels: int) -> bytes:
    """Convert raw PCM audio bytes to WAV format."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# OpenAI Whisper API
# ---------------------------------------------------------------------------

class OpenAISTT(STTBackend):
    """Transcribe using OpenAI Whisper API."""

    def __init__(self, api_key: str, model: str = "whisper-1"):
        self.api_key = api_key
        self.model = model

    async def transcribe(
        self, audio_bytes: bytes, sample_rate: int, sample_width: int, channels: int, language: str
    ) -> str:
        import aiohttp

        wav_data = _pcm_to_wav(audio_bytes, sample_rate, sample_width, channels)

        form = aiohttp.FormData()
        form.add_field("file", wav_data, filename="audio.wav", content_type="audio/wav")
        form.add_field("model", self.model)
        form.add_field("language", language)
        form.add_field("response_format", "json")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    data=form,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        _LOGGER.error("OpenAI API error %d: %s", resp.status, error_text[:500])
                        return ""
                    result = await resp.json()
                    text = result.get("text", "").strip()
                    _LOGGER.debug("OpenAI transcription: '%s'", text)
                    return text
        except asyncio.TimeoutError:
            _LOGGER.error("OpenAI API timeout")
            return ""
        except Exception as e:
            _LOGGER.error("OpenAI API error: %s", e)
            return ""


# ---------------------------------------------------------------------------
# Google Cloud Speech-to-Text v2
# ---------------------------------------------------------------------------

class GoogleSTT(STTBackend):
    """Transcribe using Google Cloud Speech-to-Text REST API."""

    def __init__(self, api_key: str, model: str = "latest_long"):
        self.api_key = api_key
        self.model = model

    async def transcribe(
        self, audio_bytes: bytes, sample_rate: int, sample_width: int, channels: int, language: str
    ) -> str:
        import aiohttp
        import base64

        # Map language codes for Google (de -> de-DE)
        google_lang = language
        if len(language) == 2:
            lang_map = {
                "de": "de-DE", "en": "en-US", "fr": "fr-FR", "es": "es-ES",
                "it": "it-IT", "nl": "nl-NL", "pt": "pt-BR", "pl": "pl-PL",
                "ru": "ru-RU", "ja": "ja-JP", "ko": "ko-KR", "zh": "zh-CN",
            }
            google_lang = lang_map.get(language, f"{language}-{language.upper()}")

        # Encode audio as base64
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        # Build request for v1 REST API
        payload = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": sample_rate,
                "languageCode": google_lang,
                "model": self.model,
                "audioChannelCount": channels,
            },
            "audio": {
                "content": audio_b64,
            },
        }

        url = "https://speech.googleapis.com/v1/speech:recognize"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers={"x-goog-api-key": self.api_key},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        _LOGGER.error("Google STT API error %d: %s", resp.status, error_text[:500])
                        return ""
                    result = await resp.json()
                    results = result.get("results", [])
                    if not results:
                        _LOGGER.debug("Google STT: no results")
                        return ""
                    text = results[0].get("alternatives", [{}])[0].get("transcript", "").strip()
                    _LOGGER.debug("Google transcription: '%s'", text)
                    return text
        except asyncio.TimeoutError:
            _LOGGER.error("Google STT API timeout")
            return ""
        except Exception as e:
            _LOGGER.error("Google STT API error: %s", e)
            return ""


# ---------------------------------------------------------------------------
# Wyoming upstream proxy (for local Whisper etc.)
# ---------------------------------------------------------------------------

class WyomingSTT(STTBackend):
    """Transcribe by forwarding to an upstream Wyoming STT service."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    async def transcribe(
        self, audio_bytes: bytes, sample_rate: int, sample_width: int, channels: int, language: str
    ) -> str:
        from wyoming.asr import Transcript
        from wyoming.audio import AudioChunk, AudioStart, AudioStop
        from wyoming.event import Event

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=10.0,
            )
        except (ConnectionError, asyncio.TimeoutError) as e:
            _LOGGER.error("Cannot connect to upstream STT %s:%d: %s", self.host, self.port, e)
            return ""

        try:
            start = AudioStart(rate=sample_rate, width=sample_width, channels=channels)
            writer.write(start.event().to_bytes())

            chunk_size = sample_rate * sample_width  # 1 second chunks
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = AudioChunk(
                    audio=audio_bytes[i : i + chunk_size],
                    rate=sample_rate,
                    width=sample_width,
                    channels=channels,
                )
                writer.write(chunk.event().to_bytes())

            writer.write(AudioStop().event().to_bytes())
            await writer.drain()

            while True:
                raw_event = await asyncio.wait_for(Event.from_reader(reader), timeout=30.0)
                if raw_event is None:
                    break
                if Transcript.is_type(raw_event.type):
                    transcript = Transcript.from_event(raw_event)
                    _LOGGER.debug("Wyoming transcription: '%s'", transcript.text)
                    return transcript.text

            return ""
        except asyncio.TimeoutError:
            _LOGGER.error("Upstream Wyoming STT timed out")
            return ""
        except Exception as e:
            _LOGGER.error("Upstream Wyoming STT error: %s", e)
            return ""
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass


def create_stt_backend(
    provider: str,
    openai_api_key: str = "",
    openai_model: str = "whisper-1",
    google_api_key: str = "",
    google_model: str = "latest_long",
    wyoming_host: str = "core-whisper",
    wyoming_port: int = 10300,
) -> STTBackend:
    """Factory function to create the appropriate STT backend."""

    if provider == "openai":
        if not openai_api_key:
            raise ValueError("OpenAI API key is required when using 'openai' provider")
        _LOGGER.info("Using OpenAI Whisper API (model: %s)", openai_model)
        return OpenAISTT(api_key=openai_api_key, model=openai_model)

    elif provider == "google":
        if not google_api_key:
            raise ValueError("Google API key is required when using 'google' provider")
        _LOGGER.info("Using Google Cloud STT (model: %s)", google_model)
        return GoogleSTT(api_key=google_api_key, model=google_model)

    elif provider == "wyoming":
        _LOGGER.info("Using upstream Wyoming STT at %s:%d", wyoming_host, wyoming_port)
        return WyomingSTT(host=wyoming_host, port=wyoming_port)

    else:
        raise ValueError(f"Unknown STT provider: {provider}")
