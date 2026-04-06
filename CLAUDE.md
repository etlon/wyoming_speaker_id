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

## Integration with hassio_assist_memory

The `[Speaker:Name]` tag is the contract between this project and [hassio_assist_memory](../hassio_assist_memory/). The flow:

1. **Speaker ID** identifies voice → `[Speaker:Cedric] Mach das Licht aus`
2. **Assist Memory** parses the tag via regex `^\[Speaker:(\w+)\]\s*(.+)$` in `event_listener.py`
3. Memories get scoped: `personal` (per speaker) or `household` (shared)
4. Conversation agent prompt gets injected with speaker-grouped memories
5. LLM responds with personalized context

No code changes needed on either side — the tag format is already implemented in both projects.

## Speaker Profiles (Folder-Based)

```
/share/wyoming-speaker-id/profiles/
├── cedric/
│   ├── sample1.webm
│   ├── pipeline_1717234567.wav    ← captured via learn mode
│   ├── recording_from_phone.mp3
│   └── .embedding.json            ← auto-computed cache
├── lovis/
│   └── ...
└── _unknown/                      ← unrecognized voices saved here
    ├── unknown_1717234600.wav
    └── unknown_1717234650.wav
```

- Each speaker = a folder. Name = folder name.
- Audio files in any format (wav, mp3, webm, ogg, flac, m4a, opus) are the voice samples.
- `.embedding.json` is a cache, auto-regenerated when samples change.
- `_unknown/` holds unrecognized audio for later review and assignment.
- Profiles stored in `/share/` — survives add-on reinstalls.
- You can add/remove files directly via filesystem (Samba/SSH) — restart or retrain to pick up changes.

## Project Structure

```
speaker_id/
  __main__.py      # Entry point: arg parsing, init, runs Wyoming server + web UI
  handler.py       # SpeakerIdHandler(AsyncEventHandler): audio collection, parallel ID+STT, learn mode, unknown capture
  speaker_db.py    # Folder-based profiles: scan, compute embeddings, cache, CRUD, move, rename
  stt_backends.py  # STTBackend ABC → OpenAISTT, GoogleSTT, WyomingSTT
  web_ui.py        # aiohttp routes + embedded HTML/JS UI (Syne/JetBrains Mono, warm studio theme)
```

Supporting files:
- `config.yaml` — HA add-on manifest. Must have `init: false` for s6-overlay.
- `build.yaml` — multi-arch build, base image `ghcr.io/home-assistant/{arch}-base-debian:bookworm`
- `Dockerfile` — Debian-based, CPU-only PyTorch in separate RUN step
- `rootfs/etc/s6-overlay/s6-rc.d/speaker-id/` — s6 service config

## Critical Build/Deploy Knowledge

### Base Image: Debian, NOT Alpine
Alpine (`ghcr.io/hassio-addons/base`) fails with exit code 79/99 on `apk add`. Use Debian (`ghcr.io/home-assistant/{arch}-base-debian:bookworm`) — same as official HA add-ons.

### PyTorch: CPU-only, installed separately
Must be its own `RUN` step with `--index-url https://download.pytorch.org/whl/cpu`. The `--index-url` flag does NOT work as a per-requirement option in requirements.txt. Without CPU-only, pip pulls ~2GB of CUDA packages.

### s6-overlay
- `init: false` **required** in `config.yaml` — without it, s6-overlay crashes "can only run as pid 1"
- `ENTRYPOINT ["/init"]` in Dockerfile
- s6 config files must be in `rootfs/`, not created via `RUN` commands
- CRLF→LF conversion needed (Windows dev): `find /etc/s6-overlay -type f -exec sed -i 's/\r$//' {} +`
- Run script must be `chmod +x`

### Wyoming 1.5.4 API
- `AsyncServer.from_uri(uri)` — NO `handler_factory` parameter
- Handler factory passed to `server.run(handler_factory)`
- `AsrProgram` and `AsrModel` both require `version` argument
- `create_server()` returns `(server, handler_factory)` tuple

## Key Patterns

- **Factory functions**: `create_server()`, `create_stt_backend()`, `create_web_app()`
- **ABC for backends**: `STTBackend` with `async transcribe()` method
- **Lazy encoder loading**: `_get_encoder()` in `speaker_db.py`, pre-loaded in `__main__.py`
- **Executor offload**: CPU-bound work via `asyncio.get_running_loop().run_in_executor()`
- **Module-level state**: `_learn_mode_speaker`, `_save_unknown` in `handler.py` (shared across handler instances)
- **Embedded UI**: HTML/CSS/JS as `"""triple-quoted"""` string in `web_ui.py`

## Web UI Features

- Speaker cards with audio playback, inline rename (click filename), delete per sample
- Learn mode — captures pipeline audio from satellite as training samples
- Unknown voice section — listen, assign to speaker via dropdown, or delete
- File upload (multiple files) + mic recording
- Separate save (just store files) and train (compute embedding) actions
- Settings: threshold slider, unknown label, toggle unknown capture
- Backup download (.zip) and import
- Reverse proxy support (`/speaker-id/` prefix, `basePath` auto-detection)

### Web UI Gotchas
- Python `"""` strings still process escapes: `\'` → `'` breaks JS. Use `&quot;` HTML entities.
- HTTPS required for mic (getUserMedia). Falls back to file upload over HTTP.
- NPM custom location: `/speaker-id` → `http://192.168.2.28:8756`

## Audio Processing

- Web UI: webm/opus → ffmpeg → 16kHz mono float32 [-1, 1]
- speaker_db: any audio format → ffmpeg subprocess → 16kHz mono float32
- Handler: int16 PCM from Wyoming → raw bytes to speaker_db (converts internally)
- Pipeline capture (learn mode / unknown): raw PCM wrapped in WAV header

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/speakers` | GET | List speakers with samples |
| `/api/speakers/{name}` | DELETE | Delete speaker + all samples |
| `/api/speakers/{name}/retrain` | POST | Recompute embedding |
| `/api/speakers/{name}/samples/{file}` | DELETE | Delete single sample |
| `/api/speakers/{name}/samples/{file}/rename` | POST | Rename sample (JSON: `{new_name}`) |
| `/api/enroll` | POST | Upload samples (multipart: name + files) |
| `/api/identify` | POST | Test identification (multipart: audio) |
| `/api/audio/{name}/{file}` | GET | Serve audio file for playback |
| `/api/move` | POST | Move sample between speakers (JSON: `{from, filename, to}`) |
| `/api/learn` | GET/POST | Get/set learn mode (JSON: `{speaker}` or null) |
| `/api/settings` | GET/POST | Get/set threshold, unknown_label, save_unknown |
| `/api/backup` | GET | Download all profiles as .zip |
| `/api/backup` | POST | Import .zip backup (multipart) |

## Deployment

- **Target**: Home Assistant OS on x86_64 or aarch64
- **Current hardware**: HA at 192.168.2.28, NPM at ha.home (HTTPS)
- **Current STT provider**: OpenAI Whisper
- **Profiles dir**: `/share/wyoming-speaker-id/profiles/` (persistent across reinstalls)
- **Deploy method**: Copy files to `\\192.168.2.28\addons\wyoming_speaker_id\`, reload add-on store, install/rebuild
- **Git**: `git@github.com:etlon/wyoming_speaker_id.git`
- Pre-downloads Resemblyzer encoder model at Docker build time
- `[Speaker:Name]` tag in transcript is a workaround — HA has no native speech-processor API yet

## Similar Projects

- [VoiceBM](https://github.com/cybericebyte/VoiceBM) — MQTT-based, uses Sherpa-ONNX (faster)
- [speaker-recognition](https://github.com/EuleMitKeule/speaker-recognition) — HA custom integration, also Resemblyzer
- [wyoming_speaker_recognition](https://github.com/mitrokun/wyoming_speaker_recognition) — Wyoming proxy approach
