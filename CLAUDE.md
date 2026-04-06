# Wyoming Speaker ID

Home Assistant add-on that acts as a Wyoming STT proxy, adding speaker recognition to the voice pipeline. Identifies who is speaking via Resemblyzer (GE2E embeddings) and enriches STT transcripts with `[Speaker:Name]` tags.

## Architecture

```
Satellite → HA → Wyoming Speaker ID (tcp :10310)
                    ├─ Resemblyzer (who?)     ─┐
                    └─ STT backend  (what?)    ─┤ asyncio.gather()
                                                └─→ "[Speaker:Name] transcript"
```

Two servers run concurrently:
- **Wyoming server** (port 10310) — receives audio from HA, returns enriched transcripts
- **Web UI** (port 8756) — aiohttp app for speaker enrollment/management

Speaker ID and STT execute in parallel via `asyncio.gather()` — zero additional latency. STT is called on **every** voice command (not just enrollment).

## Project Structure

```
speaker_id/
  __main__.py      # Entry point: arg parsing, init, runs Wyoming server + web UI
  handler.py       # SpeakerIdHandler(AsyncEventHandler): collects audio, parallel ID+STT, returns enriched Transcript
  speaker_db.py    # SpeakerDatabase: enroll/identify/delete speakers, Resemblyzer embeddings, cosine similarity
  stt_backends.py  # STTBackend ABC → OpenAISTT, GoogleSTT, WyomingSTT implementations
  web_ui.py        # aiohttp routes + embedded HTML/JS enrollment UI (German, dark theme)
```

Supporting files:
- `config.yaml` — HA add-on manifest (options, ports, arch). Must have `init: false` for s6-overlay.
- `build.yaml` — multi-arch build config (aarch64, amd64), base image `ghcr.io/home-assistant/{arch}-base-debian:bookworm`
- `Dockerfile` — Debian-based, installs resemblyzer + deps, pre-downloads encoder model
- `rootfs/etc/s6-overlay/s6-rc.d/speaker-id/` — s6 service config (type, run, dependencies.d/base)
- `rootfs/etc/s6-overlay/s6-rc.d/user/contents.d/speaker-id` — registers service with s6

## Critical Build/Deploy Knowledge

### Base Image: Debian, NOT Alpine
Alpine (`ghcr.io/hassio-addons/base`) fails with exit code 79/99 on `apk add` due to dependency resolution issues with the large package set. Use Debian (`ghcr.io/home-assistant/{arch}-base-debian:bookworm`) — same as official HA add-ons (Whisper, Piper).

### PyTorch: CPU-only, installed separately
PyTorch must be installed in its own `RUN` step with `--index-url https://download.pytorch.org/whl/cpu`. The `--index-url` flag does NOT work as a per-requirement option in requirements.txt — it's a global pip flag only. Without CPU-only, pip pulls ~2GB of CUDA/nvidia packages that fill up HA storage.

### s6-overlay
- `init: false` is **required** in `config.yaml` — without it, HA supervisor injects Docker's tini as PID 1, and s6-overlay crashes with "can only run as pid 1"
- `ENTRYPOINT ["/init"]` in Dockerfile ensures s6-overlay starts correctly
- s6 config files (`type`, `dependencies.d/base`, `contents.d/speaker-id`) must be in `rootfs/`, not created via `RUN` commands
- All rootfs files need CRLF→LF conversion (Windows dev environment): `find /etc/s6-overlay -type f -exec sed -i 's/\r$//' {} +`
- Run script must be `chmod +x`

### Wyoming 1.5.4 API
- `AsyncServer.from_uri(uri)` takes ONLY the URI — no `handler_factory` parameter
- Handler factory is passed to `server.run(handler_factory)`, not `from_uri()`
- `AsrProgram` and `AsrModel` both require a `version` argument (positional dataclass field)
- `create_server()` returns `(server, handler_factory)` tuple

## Key Patterns

- **Factory functions**: `create_server()`, `create_stt_backend()`, `create_web_app()`
- **ABC for backends**: `STTBackend` with `async transcribe()` method
- **Lazy encoder loading**: global `_encoder` + `_get_encoder()` in `speaker_db.py`. Pre-loaded via `_get_encoder()` in `__main__.py` to populate the cache (don't create a standalone VoiceEncoder).
- **Executor offload**: CPU-bound work (embedding, audio conversion) runs via `asyncio.get_running_loop().run_in_executor()` (NOT deprecated `get_event_loop()`)
- **Graceful degradation**: failures return empty strings or unknown_label, never crash the service
- **Embedded UI**: HTML/CSS/JS is a single `"""triple-quoted"""` string `HTML_PAGE` in `web_ui.py`

## Code Style

- Python 3.11 in container (Debian bookworm)
- Type hints throughout
- snake_case functions/variables, PascalCase classes
- Module-level `_LOGGER = logging.getLogger(__name__)`
- German language in UI strings and user-facing labels (unknown_label default: "Unbekannt")

## Web UI Gotchas

### Python string escaping in HTML_PAGE
The HTML/JS is a Python `"""triple-double-quoted"""` string. Python still processes escape sequences inside it:
- `\'` becomes `'` — **breaks JS string quoting**. Use `&quot;` HTML entities for quotes in onclick handlers.
- `\/` is not a recognized escape but triggers DeprecationWarning. Use `.endsWith('/')` instead of regex with `\/`.
- Template literals with backticks work fine in `"""` strings.

### HTTPS required for microphone
`navigator.mediaDevices.getUserMedia` requires a secure context (HTTPS or localhost). The UI auto-detects this and falls back to file upload over HTTP. For mic recording, proxy through NPM with HTTPS.

### Reverse proxy support
- Routes registered at both `/` and `/speaker-id/` prefixes
- All `fetch()` calls use `basePath` (auto-detected from `window.location.pathname`)
- NPM custom location: `/speaker-id` → `http://192.168.2.28:8756`

### Security
- `esc()` helper sanitizes all user data before innerHTML injection (XSS prevention)
- `user_id` validated as alphanumeric + hyphens/underscores (path traversal prevention)
- ffmpeg uses `create_subprocess_exec` with stdin pipe (no command injection)
- Google API key sent via `x-goog-api-key` header, not URL query string

## Audio Processing

- Web UI records webm/opus via MediaRecorder → converted to 16kHz mono PCM via ffmpeg subprocess
- `_convert_webm_to_numpy()` returns **float32 normalized to [-1, 1]** (not int16) — required by Resemblyzer
- Handler receives int16 PCM from Wyoming protocol and passes raw bytes to speaker_db

## STT Backends

| Backend | API | Key Config |
|---------|-----|------------|
| `openai` | `POST /v1/audio/transcriptions` (Whisper) | `--openai-api-key`, `--openai-model whisper-1` |
| `google` | `POST /v1/speech:recognize` (Cloud STT) | `--google-api-key`, `--google-model latest_long` |
| `wyoming` | Upstream Wyoming TCP protocol | `--upstream-host core-whisper`, `--upstream-port 10300` |

OpenAI/Google are billed per API call (~$0.006/min). Use `wyoming` backend with local Whisper add-on for free local processing.

## Speaker Recognition

- **Library**: Resemblyzer 0.1.3 (GE2E, 256-dim embeddings)
- **Enrollment**: 3 audio samples → averaged + normalized embedding → saved as JSON in `/data/profiles/`
- **Identification**: cosine similarity (dot product of unit vectors) against all profiles
- **Threshold**: 0.75 default (`--similarity-threshold`)
- **Audio constraints**: enrollment samples must be >=0.1s, identification requires >=0.5s
- **Sample rate**: 16kHz mono expected throughout

## Dependencies

- `wyoming==1.5.4` — Wyoming protocol
- `resemblyzer==0.1.3` — speaker embeddings
- `torch` (CPU-only) — required by resemblyzer, installed separately
- `aiohttp==3.10.11` — async HTTP
- `numpy>=1.24.0`, `librosa==0.10.2`, `soundfile==0.12.1` — audio processing
- `ffmpeg` — system dependency for webm→PCM conversion

## Configuration

All CLI args mirror `config.yaml` options. The s6 run script reads HA config via bashio and maps to CLI args. Key defaults:
- Wyoming port: 10310, Web UI port: 8756
- Profiles dir: `/data/profiles`
- Language: `de`
- Debug: off

## Deployment

- **Target**: Home Assistant OS (hassio) on x86_64 or aarch64
- **Current hardware**: HA at 192.168.2.28, NPM at ha.home (HTTPS)
- **Current STT provider**: OpenAI Whisper
- Persistent data in `/data/` survives rebuilds (profiles, enrollment audio)
- Pre-downloads Resemblyzer encoder model at Docker build time
- The `[Speaker:Name]` tag in transcript text is a workaround — HA has no native speech-processor API yet
- HA integration: Settings → Devices & Services → Wyoming Protocol → add host/port of the add-on container
