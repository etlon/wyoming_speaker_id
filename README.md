# Wyoming Speaker ID

> **Disclaimer**
> This add-on was built in a single session with Claude (AI) at maximum speed. The code works, but it's AI slop — expect rough edges, minimal tests, and creative engineering decisions. Contributions welcome: fork it, open PRs, file issues. If something breaks, it probably does. Even this disclaimer was made with ai lol

## What is this?

A Home Assistant add-on that identifies *who* is speaking to your voice assistant. It sits between your microphone (e.g. a Satellite1) and your STT engine (e.g. OpenAI Whisper), runs speaker recognition in parallel, and tags every transcript with the speaker's name: `[Speaker:Cedric] Turn on the lights`. Your LLM conversation agent can then personalize responses based on who's talking.

Key features:
- Folder-based voice profiles — just audio files, no black-box databases
- Learn mode — train speaker profiles directly from your satellite's microphone
- Unknown voice capture — unrecognized speakers get saved for later review and assignment
- Web UI for managing speakers, samples, and settings
- Backup/restore as zip
- Works with OpenAI Whisper, Google Cloud STT, or local Wyoming/Whisper

## Architecture

```
Satellite1 ──audio──▶ Home Assistant ──audio──▶ ┌─────────────────────────┐
                                                │   Wyoming Speaker ID    │
                                                │                         │
                                                │  ┌───────────────────┐  │
                                                │  │ Resemblyzer       │──│──▶ Who is speaking?
                                                │  │ (Speaker Embeddings)│ │
                                                │  └───────────────────┘  │
                                                │         parallel        │
                                                │  ┌───────────────────┐  │
                                                │  │ OpenAI / Google / │──│──▶ What was said?
                                                │  │ Whisper STT       │  │
                                                │  └───────────────────┘  │
                                                │                         │
                                                │  ▶ "[Speaker:Max] Turn  │
                                                │     on the lights"      │
                                                └────────────┬────────────┘
                                                             │
                                                             ▼
                                                    LLM Conversation Agent
```

## Installation

### 1. Copy add-on files

Copy the `wyoming-speaker-id/` folder to `/addons/wyoming-speaker-id/` on your HA system (e.g. via Samba share).

### 2. Install the add-on

**Settings** → **Add-ons** → **Add-on Store** → ⋮ → **Check for updates** → Install the local add-on.

### 3. Configuration

#### OpenAI API (recommended):

```yaml
stt_provider: openai
openai_api_key: "sk-..."
openai_model: "whisper-1"
language: "de"
similarity_threshold: 0.75
unknown_speaker_label: "Unknown"
```

#### Google Cloud STT:

```yaml
stt_provider: google
google_api_key: "AIza..."
google_model: "latest_long"
language: "de"
similarity_threshold: 0.75
unknown_speaker_label: "Unknown"
```

#### Local Whisper (Wyoming):

```yaml
stt_provider: wyoming
upstream_stt_host: "core-whisper"
upstream_stt_port: 10300
language: "de"
similarity_threshold: 0.75
unknown_speaker_label: "Unknown"
```

### 4. Set up the voice pipeline

**Settings** → **Voice assistants** → Edit your pipeline:
- **Speech-to-text**: Select `Wyoming Speaker ID` (instead of Whisper or Cloud STT directly)
- The rest of the pipeline (wake word, LLM, TTS) stays unchanged

### 5. Enroll speakers

Open the Web UI: `http://<your-ha-ip>:8756`

Speaker profiles are folder-based. Each speaker is a directory containing audio samples:

```
profiles/
├── cedric/
│   ├── sample1.webm
│   ├── sample2.wav
│   └── sample3.mp3
└── lovis/
    ├── recording1.webm
    └── recording2.wav
```

You can add samples in three ways:
1. **Record** via the Web UI (requires HTTPS for microphone access)
2. **Upload** audio files via the Web UI file picker
3. **Learn mode** — enable it for a speaker, then talk to your satellite. Each voice command gets saved as a training sample automatically.

After adding samples, click **Train** on the speaker's card to compute the voice embedding.

Unrecognized voices are automatically saved to an `_unknown/` folder. You can listen to them in the Web UI and assign them to the right speaker.

### 6. Customize the LLM prompt

Add to your conversation agent's prompt:

```
Voice requests start with [Speaker:Name].
Use the speaker's name for personalized responses.
"Notify me" refers to the identified speaker.
For [Speaker:Unknown], optionally ask for their name.
```

## Files

```
wyoming-speaker-id/
├── config.yaml              # HA add-on configuration
├── build.yaml               # Multi-arch build config
├── Dockerfile               # Container build
├── requirements.txt         # Python dependencies
├── speaker_id/
│   ├── __main__.py          # Entry point
│   ├── handler.py           # Wyoming STT proxy + speaker ID
│   ├── speaker_db.py        # Folder-based speaker profiles with Resemblyzer
│   ├── stt_backends.py      # OpenAI / Google / Wyoming STT backends
│   └── web_ui.py            # Web UI for enrollment and management
└── rootfs/                  # s6-overlay service config
```

## Performance

| Component | Duration | Note |
|---|---|---|
| Speaker embedding | 200–500ms | Runs in parallel with STT |
| OpenAI Whisper API | 500–1500ms | Depends on audio length |
| Google Cloud STT | 500–1500ms | Depends on audio length |
| Local Whisper | 2000–8000ms | Depends on hardware |
| **Total overhead** | **~0ms** | **Speaker ID runs in parallel!** |

The add-on needs ~300MB RAM for the Resemblyzer model. Speaker profiles are a few KB each.

## Tips

- **Satellite1**: The XMOS chip provides pre-processed audio (echo cancellation, noise suppression) — great for speaker recognition
- **Threshold**: Default 0.75 works well for 2–4 speakers. Increase to 0.80+ if you get false matches
- **Enrollment**: Use natural speech at normal volume, different sentences. Don't whisper or shout.
- **Learn mode**: Best way to train — captures audio exactly as the pipeline receives it from your satellite
- **Multiple satellites**: Recognition works across all satellites using the same pipeline

## Limitations

- No official HA speech processor API — the speaker tag is embedded in the transcript text as a workaround
- Unreliable when multiple people speak simultaneously
- Children's voices change over time — update profiles regularly
- API costs apply for cloud STT (Whisper: ~$0.006/min, Google: similar)
- Profiles are stored in `/share/wyoming-speaker-id/profiles/` — survives add-on reinstalls but not HA OS reinstalls (use backup/restore)

## License

MIT — see below.

**This software is provided "as is", without warranty of any kind. The author is not responsible for any damages, data loss, API costs, or other issues arising from the use of this add-on. Use at your own risk.**
