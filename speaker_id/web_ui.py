"""Web UI for speaker enrollment and management."""

import asyncio
import io
import json
import logging
import uuid
from pathlib import Path

import numpy as np
from aiohttp import web

from .speaker_db import SpeakerDatabase

_LOGGER = logging.getLogger(__name__)

ENROLLMENT_AUDIO_DIR = Path("/data/enrollment_audio")
ENROLLMENT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

HTML_PAGE = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Speaker ID — Sprecher verwalten</title>
<style>
  :root { --bg: #1a1a2e; --card: #16213e; --accent: #0f3460; --highlight: #e94560;
          --text: #eee; --muted: #888; --ok: #4ecca3; --border: #2a2a4a; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: var(--bg); color: var(--text); padding: 1.5rem; max-width: 800px; margin: 0 auto; }
  h1 { margin-bottom: 1.5rem; font-size: 1.5rem; }
  h2 { font-size: 1.1rem; margin-bottom: 1rem; color: var(--muted); }

  .card { background: var(--card); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
          border: 1px solid var(--border); }

  label { display: block; margin-bottom: .4rem; font-size: .9rem; color: var(--muted); }
  input[type=text] { width: 100%; padding: .7rem; border-radius: 8px; border: 1px solid var(--border);
                     background: var(--bg); color: var(--text); font-size: 1rem; margin-bottom: 1rem; }

  button { padding: .7rem 1.4rem; border-radius: 8px; border: none; font-size: .95rem;
           cursor: pointer; transition: all .15s; font-weight: 600; }
  .btn-primary { background: var(--highlight); color: #fff; }
  .btn-primary:hover { filter: brightness(1.15); }
  .btn-primary:disabled { opacity: .5; cursor: not-allowed; }
  .btn-danger { background: #c0392b; color: #fff; font-size: .8rem; padding: .5rem .9rem; }
  .btn-danger:hover { filter: brightness(1.15); }
  .btn-secondary { background: var(--accent); color: var(--text); }

  .recording { animation: pulse 1s infinite; }
  @keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(233,69,96,.5); }
                      50% { box-shadow: 0 0 0 12px rgba(233,69,96,0); } }

  .speaker-list { list-style: none; }
  .speaker-item { display: flex; justify-content: space-between; align-items: center;
                  padding: .8rem; border-bottom: 1px solid var(--border); }
  .speaker-item:last-child { border-bottom: none; }
  .speaker-name { font-weight: 600; }
  .speaker-id { font-size: .8rem; color: var(--muted); }

  .status { padding: .6rem 1rem; border-radius: 8px; margin-top: 1rem; font-size: .9rem; }
  .status-ok { background: rgba(78,204,163,.15); color: var(--ok); }
  .status-err { background: rgba(192,57,43,.15); color: #e74c3c; }
  .status-info { background: rgba(15,52,96,.3); color: var(--muted); }

  .samples-info { font-size: .85rem; color: var(--muted); margin: .5rem 0; }
  .progress-dots { display: flex; gap: .5rem; margin: .8rem 0; }
  .dot { width: 14px; height: 14px; border-radius: 50%; background: var(--border); }
  .dot.filled { background: var(--ok); }
  .dot.active { background: var(--highlight); }

  .hidden { display: none; }
</style>
</head>
<body>

<h1>🎙️ Speaker ID — Sprecherverwaltung</h1>

<!-- Enrolled Speakers -->
<div class="card">
  <h2>Registrierte Sprecher</h2>
  <ul class="speaker-list" id="speakerList">
    <li class="speaker-item" style="color: var(--muted)">Lade...</li>
  </ul>
</div>

<!-- Enrollment -->
<div class="card">
  <h2>Neuen Sprecher registrieren</h2>

  <label for="speakerName">Name</label>
  <input type="text" id="speakerName" placeholder="z.B. Max">

  <label for="speakerUserId">Home Assistant User-ID (optional)</label>
  <input type="text" id="speakerUserId" placeholder="wird automatisch generiert">

  <p class="samples-info">
    Nimm 3 Sprachproben auf (je 3-8 Sekunden). Sprich natürlich — z.B. verschiedene Sätze.
  </p>

  <!-- Secure context: mic recording -->
  <div id="micEnroll">
    <div class="progress-dots" id="progressDots">
      <div class="dot" id="dot0"></div>
      <div class="dot" id="dot1"></div>
      <div class="dot" id="dot2"></div>
    </div>
    <button class="btn-primary" id="recordBtn" onclick="toggleRecording()">
      Aufnahme starten
    </button>
  </div>

  <!-- Insecure context fallback: file upload -->
  <div id="fileEnroll" class="hidden">
    <div class="status status-info" style="margin-bottom:1rem">
      Mikrofon-Zugriff nur über HTTPS möglich. Bitte lade stattdessen Audiodateien hoch (WAV, MP3, OGG, WEBM).
    </div>
    <div class="progress-dots" id="fileProgressDots">
      <div class="dot" id="fdot0"></div>
      <div class="dot" id="fdot1"></div>
      <div class="dot" id="fdot2"></div>
    </div>
    <label for="fileInput" style="display:inline" id="fileLabel">Probe 1/3 auswählen:</label>
    <input type="file" id="fileInput" accept="audio/*" onchange="handleFileUpload(this)" style="margin:.5rem 0">
  </div>

  <button class="btn-secondary hidden" id="enrollBtn" onclick="enrollSpeaker()">
    Sprecher registrieren
  </button>

  <div id="statusBox" class="hidden"></div>
</div>

<!-- Test -->
<div class="card">
  <h2>Sprecher testen</h2>
  <p class="samples-info">Nimm eine kurze Probe auf oder lade eine Audiodatei hoch.</p>
  <div id="micTest">
    <button class="btn-primary" id="testRecordBtn" onclick="toggleTestRecording()">
      Test-Aufnahme
    </button>
  </div>
  <div id="fileTest" class="hidden">
    <input type="file" id="testFileInput" accept="audio/*" onchange="handleTestFileUpload(this)">
  </div>
  <div id="testResult" class="hidden"></div>
</div>

<script>
let mediaRecorder = null;
let audioChunks = [];
let samples = [];
let currentSample = 0;
let isRecording = false;
let hasMic = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
let fileCurrentSample = 0;
// Auto-detect base path from current URL (works behind reverse proxy)
let basePath = window.location.pathname.endsWith('/') ? window.location.pathname.slice(0, -1) : window.location.pathname;

// --- HTML escape helper (XSS prevention) ---
function esc(s) {
  var d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// --- Detect secure context and toggle UI ---
function initUI() {
  if (hasMic) {
    document.getElementById('micEnroll').classList.remove('hidden');
    document.getElementById('fileEnroll').classList.add('hidden');
    document.getElementById('micTest').classList.remove('hidden');
    document.getElementById('fileTest').classList.add('hidden');
  } else {
    document.getElementById('micEnroll').classList.add('hidden');
    document.getElementById('fileEnroll').classList.remove('hidden');
    document.getElementById('micTest').classList.add('hidden');
    document.getElementById('fileTest').classList.remove('hidden');
    updateFileLabel();
  }
}

// --- File Upload Fallback ---
function updateFileLabel() {
  const label = document.querySelector('label[for="fileInput"]');
  if (label) label.textContent = 'Probe ' + (fileCurrentSample + 1) + '/3 auswählen:';
}

function handleFileUpload(input) {
  if (!input.files[0]) return;
  samples.push(input.files[0]);
  document.getElementById('fdot' + fileCurrentSample).classList.add('filled');
  fileCurrentSample++;
  input.value = '';
  if (fileCurrentSample >= 3) {
    document.getElementById('enrollBtn').classList.remove('hidden');
    document.getElementById('fileInput').disabled = true;
    showStatus('Alle 3 Proben geladen. Klicke auf "Sprecher registrieren".', 'ok');
  } else {
    showStatus('Probe ' + fileCurrentSample + '/3 geladen. Nächste Datei auswählen.', 'info');
    updateFileLabel();
  }
}

function handleTestFileUpload(input) {
  if (!input.files[0]) return;
  const file = input.files[0];
  const formData = new FormData();
  formData.append('audio', file, file.name);
  showTestResult('Analysiere...', 'info');

  fetch(basePath + '/api/identify', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
      if (data.speaker) {
        showTestResult(
          'Erkannt: <strong>' + esc(data.speaker) + '</strong> (Confidence: ' + (data.confidence * 100).toFixed(1) + '%)',
          data.confidence >= 0.75 ? 'ok' : 'info'
        );
      } else {
        showTestResult('Kein Sprecher erkannt.', 'err');
      }
    })
    .catch(e => showTestResult('Fehler: ' + e.message, 'err'));
  input.value = '';
}

// --- Mic Enrollment Recording ---
function toggleRecording() {
  if (isRecording) { stopRecording(); }
  else { startRecording(); }
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true }
    });
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
    audioChunks = [];
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = () => {
      const blob = new Blob(audioChunks, { type: 'audio/webm' });
      samples.push(blob);
      stream.getTracks().forEach(t => t.stop());
      document.getElementById('dot' + currentSample).classList.remove('active');
      document.getElementById('dot' + currentSample).classList.add('filled');
      currentSample++;
      if (currentSample >= 3) {
        document.getElementById('enrollBtn').classList.remove('hidden');
        showStatus('Alle 3 Proben aufgenommen. Klicke auf "Sprecher registrieren".', 'ok');
      } else {
        showStatus('Probe ' + currentSample + '/3 aufgenommen. Nächste Probe aufnehmen.', 'info');
      }
      isRecording = false;
      document.getElementById('recordBtn').textContent = 'Aufnahme starten';
      document.getElementById('recordBtn').classList.remove('recording');
    };
    mediaRecorder.start();
    isRecording = true;
    document.getElementById('recordBtn').textContent = 'Aufnahme stoppen';
    document.getElementById('recordBtn').classList.add('recording');
    document.getElementById('dot' + currentSample).classList.add('active');
    showStatus('Sprich jetzt... (Probe ' + (currentSample + 1) + '/3)', 'info');
  } catch (e) {
    showStatus('Mikrofon-Fehler: ' + e.message, 'err');
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
  }
}

async function enrollSpeaker() {
  const name = document.getElementById('speakerName').value.trim();
  if (!name) { showStatus('Bitte einen Namen eingeben.', 'err'); return; }

  let userId = document.getElementById('speakerUserId').value.trim();
  if (!userId) userId = crypto.randomUUID ? crypto.randomUUID() : Date.now().toString();

  const formData = new FormData();
  formData.append('name', name);
  formData.append('user_id', userId);
  samples.forEach((s, i) => formData.append('sample_' + i, s, 'sample_' + i + '.webm'));

  document.getElementById('enrollBtn').disabled = true;
  showStatus('Verarbeite Sprachproben... (kann einige Sekunden dauern)', 'info');

  try {
    const res = await fetch(basePath + '/api/enroll', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.success) {
      showStatus('"' + name + '" erfolgreich registriert!', 'ok');
      resetEnrollment();
      loadSpeakers();
    } else {
      showStatus('Fehler: ' + data.error, 'err');
    }
  } catch (e) {
    showStatus('Netzwerkfehler: ' + e.message, 'err');
  }
  document.getElementById('enrollBtn').disabled = false;
}

function resetEnrollment() {
  samples = [];
  currentSample = 0;
  fileCurrentSample = 0;
  document.querySelectorAll('.dot').forEach(d => { d.classList.remove('filled', 'active'); });
  document.getElementById('enrollBtn').classList.add('hidden');
  document.getElementById('speakerName').value = '';
  document.getElementById('speakerUserId').value = '';
  if (!hasMic) {
    document.getElementById('fileInput').disabled = false;
    updateFileLabel();
  }
}

// --- Test Recording ---
let testRecording = false;
function toggleTestRecording() {
  if (testRecording) { stopTestRecording(); }
  else { startTestRecording(); }
}

async function startTestRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true }
    });
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
    audioChunks = [];
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = async () => {
      const blob = new Blob(audioChunks, { type: 'audio/webm' });
      stream.getTracks().forEach(t => t.stop());
      testRecording = false;
      document.getElementById('testRecordBtn').textContent = 'Test-Aufnahme';
      document.getElementById('testRecordBtn').classList.remove('recording');

      const formData = new FormData();
      formData.append('audio', blob, 'test.webm');
      showTestResult('Analysiere...', 'info');

      try {
        const res = await fetch(basePath + '/api/identify', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.speaker) {
          showTestResult(
            'Erkannt: <strong>' + esc(data.speaker) + '</strong> (Confidence: ' + (data.confidence * 100).toFixed(1) + '%)',
            data.confidence >= 0.75 ? 'ok' : 'info'
          );
        } else {
          showTestResult('Kein Sprecher erkannt.', 'err');
        }
      } catch (e) {
        showTestResult('Fehler: ' + e.message, 'err');
      }
    };
    mediaRecorder.start();
    testRecording = true;
    document.getElementById('testRecordBtn').textContent = 'Stoppen';
    document.getElementById('testRecordBtn').classList.add('recording');
    showTestResult('Sprich jetzt...', 'info');
  } catch (e) {
    showTestResult('Mikrofon-Fehler: ' + e.message, 'err');
  }
}

function stopTestRecording() {
  if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop();
}

// --- UI Helpers ---
function showStatus(msg, type) {
  const el = document.getElementById('statusBox');
  el.className = 'status status-' + type;
  el.innerHTML = msg;
  el.classList.remove('hidden');
}

function showTestResult(msg, type) {
  const el = document.getElementById('testResult');
  el.className = 'status status-' + type;
  el.innerHTML = msg;
  el.classList.remove('hidden');
}

async function loadSpeakers() {
  try {
    const res = await fetch(basePath + '/api/speakers');
    const data = await res.json();
    const list = document.getElementById('speakerList');
    if (!data.speakers || data.speakers.length === 0) {
      list.innerHTML = '<li class="speaker-item" style="color:var(--muted)">Noch keine Sprecher registriert.</li>';
      return;
    }
    list.innerHTML = data.speakers.map(s =>
      '<li class="speaker-item"><div>' +
      '<span class="speaker-name">' + esc(s.name) + '</span><br>' +
      '<span class="speaker-id">' + esc(s.user_id) + '</span>' +
      '</div><button class="btn-danger" onclick="deleteSpeaker(&quot;' + esc(s.user_id) + '&quot;)">Löschen</button></li>'
    ).join('');
  } catch (e) {
    document.getElementById('speakerList').innerHTML =
      '<li class="speaker-item" style="color:#e74c3c">Fehler beim Laden</li>';
  }
}

async function deleteSpeaker(userId) {
  if (!confirm('Sprecher wirklich löschen?')) return;
  await fetch(basePath + '/api/speakers/' + userId, { method: 'DELETE' });
  loadSpeakers();
}

loadSpeakers();
initUI();
</script>
</body>
</html>"""


def create_web_app(speaker_db: SpeakerDatabase) -> web.Application:
    """Create the aiohttp web application for enrollment."""

    app = web.Application(client_max_size=50 * 1024 * 1024)  # 50MB max upload

    async def index(request):
        return web.Response(text=HTML_PAGE, content_type="text/html")

    async def list_speakers(request):
        speakers = speaker_db.list_speakers()
        return web.json_response({"speakers": speakers})

    async def delete_speaker(request):
        user_id = request.match_info["user_id"]
        success = speaker_db.delete_speaker(user_id)
        return web.json_response({"success": success})

    async def enroll(request):
        """Handle speaker enrollment with audio samples."""
        try:
            reader = await request.multipart()
            name = None
            user_id = None
            audio_blobs = []

            async for part in reader:
                if part.name == "name":
                    name = (await part.read()).decode("utf-8")
                elif part.name == "user_id":
                    user_id = (await part.read()).decode("utf-8")
                elif part.name and part.name.startswith("sample_"):
                    data = await part.read()
                    audio_blobs.append(data)

            if not name:
                return web.json_response({"success": False, "error": "Name fehlt"})
            if user_id and not user_id.replace("-", "").replace("_", "").isalnum():
                return web.json_response({"success": False, "error": "Ungültige User-ID"})
            if not audio_blobs:
                return web.json_response({"success": False, "error": "Keine Audiodaten"})

            # Convert webm audio to numpy arrays using ffmpeg
            audio_arrays = []
            for blob_data in audio_blobs:
                audio_array = await _convert_webm_to_numpy(blob_data)
                if audio_array is not None and len(audio_array) > 0:
                    audio_arrays.append(audio_array)

            if not audio_arrays:
                return web.json_response(
                    {"success": False, "error": "Audiodaten konnten nicht verarbeitet werden"}
                )

            # Run enrollment in executor to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                speaker_db.enroll_speaker,
                name,
                user_id,
                audio_arrays,
            )

            return web.json_response({"success": True})

        except Exception as e:
            _LOGGER.exception("Enrollment failed")
            return web.json_response({"success": False, "error": str(e)})

    async def identify(request):
        """Handle test identification."""
        try:
            reader = await request.multipart()
            audio_data = None

            async for part in reader:
                if part.name == "audio":
                    audio_data = await part.read()

            if not audio_data:
                return web.json_response({"speaker": None, "confidence": 0})

            audio_array = await _convert_webm_to_numpy(audio_data)
            if audio_array is None:
                return web.json_response({"speaker": None, "confidence": 0})

            loop = asyncio.get_running_loop()
            name, user_id, confidence = await loop.run_in_executor(
                None,
                speaker_db.identify_speaker,
                audio_array,
                16000,
            )

            return web.json_response({
                "speaker": name,
                "user_id": user_id,
                "confidence": round(confidence, 4),
            })

        except Exception as e:
            _LOGGER.exception("Identification test failed")
            return web.json_response({"speaker": None, "confidence": 0, "error": str(e)})

    # Register routes at root and under /speaker-id for reverse proxy support
    for prefix in ["", "/speaker-id"]:
        app.router.add_get(prefix + "/", index)
        app.router.add_get(prefix + "/api/speakers", list_speakers)
        app.router.add_delete(prefix + "/api/speakers/{user_id}", delete_speaker)
        app.router.add_post(prefix + "/api/enroll", enroll)
        app.router.add_post(prefix + "/api/identify", identify)
    # Redirect /speaker-id (no trailing slash) to /speaker-id/
    app.router.add_get("/speaker-id", lambda r: web.HTTPFound("/speaker-id/"))

    return app


async def _convert_webm_to_numpy(webm_data: bytes) -> np.ndarray | None:
    """Convert webm/opus audio to 16kHz mono int16 numpy array using ffmpeg."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", "pipe:0",
            "-ar", "16000", "-ac", "1", "-f", "s16le",
            "-acodec", "pcm_s16le", "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=webm_data)

        if proc.returncode != 0:
            _LOGGER.error("ffmpeg error: %s", stderr.decode(errors="replace")[-500:])
            return None

        raw = np.frombuffer(stdout, dtype=np.int16)
        return raw.astype(np.float32) / 32768.0

    except Exception as e:
        _LOGGER.error("Audio conversion failed: %s", e)
        return None
