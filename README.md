# Wyoming Speaker ID

🎙️ **Sprechererkennung für Home Assistant Voice Pipeline**

Ein Home Assistant Add-on, das erkennt *wer* gerade spricht und das Transkript entsprechend anreichert. Unterstützt **OpenAI Whisper API**, **Google Cloud STT** und lokales **Wyoming/Whisper** als STT-Backend.

## Architektur

```
Satellite1 ──audio──▶ Home Assistant ──audio──▶ ┌─────────────────────────┐
                                                │   Wyoming Speaker ID    │
                                                │                         │
                                                │  ┌───────────────────┐  │
                                                │  │ Resemblyzer       │──│──▶ Wer spricht?
                                                │  │ (Speaker Embeddings)│ │
                                                │  └───────────────────┘  │
                                                │         parallel        │
                                                │  ┌───────────────────┐  │
                                                │  │ OpenAI / Google / │──│──▶ Was wurde gesagt?
                                                │  │ Whisper STT       │  │
                                                │  └───────────────────┘  │
                                                │                         │
                                                │  ▶ "[Speaker:Max] Mach  │
                                                │     das Licht an"       │
                                                └────────────┬────────────┘
                                                             │
                                                             ▼
                                                    LLM Conversation Agent
```

## Installation

### 1. Add-on Dateien kopieren

Kopiere den `wyoming-speaker-id/` Ordner nach `/addons/wyoming-speaker-id/` auf deinem HA-System.

### 2. Add-on installieren

**Einstellungen** → **Add-ons** → **Add-on Store** → ⋮ → **Auf Updates prüfen** → Lokales Add-on installieren.

### 3. Konfiguration

#### Mit OpenAI API (empfohlen):

```yaml
stt_provider: openai
openai_api_key: "sk-..."
openai_model: "whisper-1"
language: "de"
similarity_threshold: 0.75
unknown_speaker_label: "Unbekannt"
```

#### Mit Google Cloud STT:

```yaml
stt_provider: google
google_api_key: "AIza..."
google_model: "latest_long"
language: "de"
similarity_threshold: 0.75
unknown_speaker_label: "Unbekannt"
```

#### Mit lokalem Whisper (Wyoming):

```yaml
stt_provider: wyoming
upstream_stt_host: "core-whisper"
upstream_stt_port: 10300
language: "de"
similarity_threshold: 0.75
unknown_speaker_label: "Unbekannt"
```

### 4. Voice Pipeline umstellen

**Einstellungen** → **Sprachassistenten** → Pipeline bearbeiten:
- **Speech-to-Text**: `Wyoming Speaker ID` auswählen (statt Whisper oder Cloud STT)
- Rest der Pipeline (Wake Word, LLM, TTS) bleibt unverändert

### 5. Sprecher registrieren

Web-UI öffnen: `http://<deine-ha-ip>:8756`

1. Name eingeben (z.B. "Max")
2. Optional: Home Assistant User-ID eintragen
3. 3 Sprachproben aufnehmen (je 3-8 Sek., verschiedene Sätze sprechen)
4. "Sprecher registrieren" klicken
5. Mit "Test-Aufnahme" die Erkennung prüfen

### 6. LLM-Prompt anpassen

Füge dem Conversation Agent Prompt hinzu:

```
Sprachanfragen beginnen mit [Speaker:Name].
Nutze den Namen für personalisierte Antworten.
"Benachrichtige mich" bezieht sich auf den erkannten Sprecher.
Bei [Speaker:Unbekannt] frage optional nach dem Namen.
```

## Dateien

```
wyoming-speaker-id/
├── config.yaml              # HA Add-on Konfiguration
├── build.yaml               # Multi-Arch Build
├── Dockerfile               # Container-Build
├── requirements.txt         # Python Dependencies
├── speaker_id/
│   ├── __main__.py          # Entry Point
│   ├── handler.py           # Wyoming STT Proxy + Speaker ID
│   ├── speaker_db.py        # Resemblyzer Speaker Profiles
│   ├── stt_backends.py      # OpenAI / Google / Wyoming STT
│   └── web_ui.py            # Enrollment Web UI
└── rootfs/                  # s6-overlay Service
```

## Performance

| Komponente | Dauer | Hinweis |
|---|---|---|
| Speaker Embedding | 200–500ms | Läuft parallel zum STT |
| OpenAI Whisper API | 500–1500ms | Abhängig von Audiolänge |
| Google Cloud STT | 500–1500ms | Abhängig von Audiolänge |
| Lokales Whisper | 2000–8000ms | Abhängig von Hardware |
| **Gesamt-Overhead** | **~0ms** | **Speaker ID läuft parallel!** |

Das Add-on braucht ca. 300MB RAM für das Resemblyzer-Modell. Speaker-Profile sind wenige KB groß.

## Tipps

- **Satellite1**: Der XMOS-Chip liefert vorverarbeitetes Audio (Echo Cancellation, Noise Suppression) — ideal für Speaker Recognition
- **Threshold**: Standard 0.75 funktioniert gut für 2-4 Sprecher. Bei Verwechslungen auf 0.80+ erhöhen
- **Enrollment**: Verschiedene Sätze in normaler Sprechlautstärke, nicht flüstern oder schreien
- **Mehrere Satellites**: Erkennung funktioniert satellite-übergreifend

## Einschränkungen

- Kein offizieller HA Speech-Processor-Support — Speaker-Tag wird in den Transkript-Text eingebettet
- Bei gleichzeitigem Sprechen mehrerer Personen unzuverlässig
- Kinderstimmen ändern sich — Profile regelmäßig aktualisieren
- OpenAI/Google API-Kosten beachten (Whisper: ~$0.006/min, Google: ähnlich)

## Lizenz

MIT
