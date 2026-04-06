"""Wyoming STT proxy handler with speaker identification."""

import asyncio
import logging

import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

from .speaker_db import SpeakerDatabase
from .stt_backends import STTBackend

_LOGGER = logging.getLogger(__name__)


class SpeakerIdHandler(AsyncEventHandler):
    """Handles Wyoming events: collects audio, identifies speaker, runs STT."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        speaker_db: SpeakerDatabase,
        stt_backend: STTBackend,
        language: str,
    ):
        super().__init__(reader, writer)
        self.speaker_db = speaker_db
        self.stt_backend = stt_backend
        self.language = language

        self._audio_buffer: bytearray = bytearray()
        self._audio_rate: int = 16000
        self._audio_width: int = 2
        self._audio_channels: int = 1
        self._is_recording: bool = False

    async def handle_event(self, event: Event) -> bool:
        """Process Wyoming events."""

        if Describe.is_type(event.type):
            await self._send_info()
            return True

        if AudioStart.is_type(event.type):
            start = AudioStart.from_event(event)
            self._audio_rate = start.rate
            self._audio_width = start.width
            self._audio_channels = start.channels
            self._audio_buffer.clear()
            self._is_recording = True
            _LOGGER.debug(
                "Audio start: rate=%d, width=%d, channels=%d",
                start.rate, start.width, start.channels,
            )
            return True

        if AudioChunk.is_type(event.type):
            if self._is_recording:
                chunk = AudioChunk.from_event(event)
                self._audio_buffer.extend(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            self._is_recording = False
            duration = len(self._audio_buffer) / (
                self._audio_rate * self._audio_width * self._audio_channels
            )
            _LOGGER.debug("Audio stop: %.2fs collected", duration)

            audio_bytes = bytes(self._audio_buffer)
            self._audio_buffer.clear()

            # Run speaker identification and STT in parallel
            speaker_task = asyncio.get_running_loop().run_in_executor(
                None, self._identify_speaker, audio_bytes
            )
            stt_task = self.stt_backend.transcribe(
                audio_bytes,
                self._audio_rate,
                self._audio_width,
                self._audio_channels,
                self.language,
            )

            speaker_result, transcript_text = await asyncio.gather(
                speaker_task, stt_task
            )

            speaker_name, user_id, confidence = speaker_result

            # Build enriched transcript
            if transcript_text:
                enriched = f"[Speaker:{speaker_name}] {transcript_text}"
            else:
                enriched = ""

            _LOGGER.info(
                "Result: speaker=%s (%.2f), text='%s'",
                speaker_name, confidence, transcript_text,
            )

            transcript = Transcript(text=enriched)
            await self.write_event(transcript.event())
            return False  # Close connection after response

        if Transcribe.is_type(event.type):
            return True

        _LOGGER.debug("Unhandled event type: %s", event.type)
        return True

    def _identify_speaker(self, audio_bytes: bytes):
        """Run speaker identification (runs in executor)."""
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            return self.speaker_db.identify_speaker(
                audio_array, sample_rate=self._audio_rate
            )
        except Exception as e:
            _LOGGER.error("Speaker identification failed: %s", e)
            return self.speaker_db.unknown_label, None, 0.0

    async def _send_info(self):
        """Send service info in response to Describe."""
        info = Info(
            asr=[
                AsrProgram(
                    name="wyoming-speaker-id",
                    description="Speaker-identifying STT proxy",
                    version="0.2.0",
                    attribution=Attribution(
                        name="Wyoming Speaker ID",
                        url="",
                    ),
                    installed=True,
                    models=[
                        AsrModel(
                            name="speaker-id-proxy",
                            description=f"STT with speaker identification ({self.language})",
                            version="0.2.0",
                            attribution=Attribution(
                                name="Resemblyzer + STT backend",
                                url="https://github.com/resemble-ai/Resemblyzer",
                            ),
                            installed=True,
                            languages=[self.language],
                        )
                    ],
                )
            ]
        )
        await self.write_event(info.event())


def create_server(
    host: str,
    port: int,
    speaker_db: SpeakerDatabase,
    stt_backend: STTBackend,
    language: str,
) -> tuple:
    """Create the Wyoming server and handler factory."""

    def handler_factory(reader, writer):
        return SpeakerIdHandler(
            reader=reader,
            writer=writer,
            speaker_db=speaker_db,
            stt_backend=stt_backend,
            language=language,
        )

    server = AsyncServer.from_uri(f"tcp://{host}:{port}")
    return server, handler_factory
