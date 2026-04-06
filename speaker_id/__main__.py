"""Wyoming Speaker ID — Main entry point."""

import argparse
import asyncio
import logging
import sys

from aiohttp import web

from .handler import create_server
from .speaker_db import SpeakerDatabase
from .stt_backends import create_stt_backend
from .web_ui import create_web_app

_LOGGER = logging.getLogger("speaker_id")


def main():
    parser = argparse.ArgumentParser(description="Wyoming Speaker ID proxy")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10310)
    parser.add_argument("--web-port", type=int, default=8756)

    # STT provider
    parser.add_argument("--stt-provider", default="openai", choices=["openai", "google", "wyoming"])
    parser.add_argument("--openai-api-key", default="")
    parser.add_argument("--openai-model", default="whisper-1")
    parser.add_argument("--google-api-key", default="")
    parser.add_argument("--google-model", default="latest_long")
    parser.add_argument("--upstream-host", default="core-whisper")
    parser.add_argument("--upstream-port", type=int, default=10300)

    # Speaker ID
    parser.add_argument("--profiles-dir", default="/data/profiles")
    parser.add_argument("--similarity-threshold", type=float, default=0.75)
    parser.add_argument("--unknown-label", default="Unbekannt")
    parser.add_argument("--language", default="de")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    _LOGGER.info("=== Wyoming Speaker ID v0.2.0 ===")
    _LOGGER.info("STT provider: %s", args.stt_provider)
    _LOGGER.info("Language: %s", args.language)
    _LOGGER.info("Similarity threshold: %.2f", args.similarity_threshold)

    # Initialize speaker database
    speaker_db = SpeakerDatabase(
        profiles_dir=args.profiles_dir,
        similarity_threshold=args.similarity_threshold,
        unknown_label=args.unknown_label,
    )

    # Pre-load the speaker encoder model into the module-level cache
    _LOGGER.info("Pre-loading speaker encoder model...")
    from .speaker_db import _get_encoder
    _get_encoder()
    _LOGGER.info("Speaker encoder model ready")

    # Create STT backend
    stt_backend = create_stt_backend(
        provider=args.stt_provider,
        openai_api_key=args.openai_api_key,
        openai_model=args.openai_model,
        google_api_key=args.google_api_key,
        google_model=args.google_model,
        wyoming_host=args.upstream_host,
        wyoming_port=args.upstream_port,
    )

    asyncio.run(_run(args, speaker_db, stt_backend))


async def _run(args, speaker_db: SpeakerDatabase, stt_backend):
    """Run Wyoming server and web UI concurrently."""

    wyoming_server, handler_factory = create_server(
        host=args.host,
        port=args.port,
        speaker_db=speaker_db,
        stt_backend=stt_backend,
        language=args.language,
    )

    web_app = create_web_app(speaker_db)
    web_runner = web.AppRunner(web_app)
    await web_runner.setup()
    web_site = web.TCPSite(web_runner, args.host, args.web_port)

    await web_site.start()
    _LOGGER.info("Web UI: http://%s:%d", args.host, args.web_port)
    _LOGGER.info("Wyoming STT: tcp://%s:%d", args.host, args.port)

    await wyoming_server.run(handler_factory)


if __name__ == "__main__":
    main()
