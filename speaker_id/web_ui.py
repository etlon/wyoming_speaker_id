"""Web UI for speaker enrollment and management."""

import asyncio
import io
import json
import logging
import uuid
from pathlib import Path

import numpy as np
from aiohttp import web

from .handler import set_learn_mode, get_learn_mode, set_save_unknown, get_save_unknown
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

<!-- Speakers + Samples -->
<div id="speakerList"></div>

<!-- Add Speaker / Add Samples -->
<div class="card">
  <h2 id="enrollTitle">Neuen Sprecher hinzufügen</h2>

  <label for="speakerName">Name (= Ordnername)</label>
  <input type="text" id="speakerName" placeholder="z.B. Cedric">

  <p class="samples-info">
    Sprachproben aufnehmen oder hochladen (je 3-8 Sekunden, mind. 2-3 Proben empfohlen).
  </p>

  <!-- Secure context: mic recording -->
  <div id="micEnroll">
    <button class="btn-primary" id="recordBtn" onclick="toggleRecording()">
      Aufnahme starten
    </button>
    <span id="sampleCount" class="samples-info"></span>
  </div>

  <!-- Insecure context fallback: file upload -->
  <div id="fileEnroll" class="hidden">
    <div class="status status-info" style="margin-bottom:1rem">
      Mikrofon-Zugriff nur über HTTPS möglich. Bitte lade Audiodateien hoch.
    </div>
    <input type="file" id="fileInput" accept="audio/*" multiple onchange="handleFileUpload(this)" style="margin:.5rem 0">
  </div>

  <button class="btn-secondary hidden" id="enrollBtn" onclick="saveSamples()">
    Proben speichern
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

<!-- Settings -->
<div class="card">
  <h2>Einstellungen</h2>
  <label for="settingThreshold">Erkennungsschwelle (0.0 - 1.0)</label>
  <div style="display:flex;gap:.5rem;align-items:center;margin-bottom:1rem">
    <input type="range" id="settingThresholdRange" min="0.3" max="0.95" step="0.01" style="flex:1" oninput="document.getElementById('settingThreshold').value=this.value">
    <input type="number" id="settingThreshold" min="0.3" max="0.95" step="0.01" style="width:5rem;padding:.4rem;border-radius:6px;border:1px solid var(--border);background:var(--bg);color:var(--text)" oninput="document.getElementById('settingThresholdRange').value=this.value">
  </div>
  <p class="samples-info" style="margin-bottom:1rem">Niedriger = mehr Treffer (mehr falsch-positiv). Höher = strenger (mehr unerkannt). Standard: 0.75</p>

  <label for="settingUnknown">Label für unbekannte Sprecher</label>
  <input type="text" id="settingUnknown" placeholder="Unbekannt" style="margin-bottom:1rem">

  <label>Nicht erkannte Aufnahmen speichern</label>
  <div style="margin-bottom:1rem">
    <label style="display:inline;cursor:pointer"><input type="checkbox" id="settingSaveUnknown" style="margin-right:.3rem">Pipeline-Audio bei unerkannten Sprechern speichern</label>
  </div>

  <button class="btn-primary" onclick="saveSettings()">Einstellungen speichern</button>
  <div id="settingsStatus" class="hidden"></div>
</div>

<script>
var mediaRecorder = null;
var audioChunks = [];
var samples = [];
var isRecording = false;
var hasMic = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
var basePath = window.location.pathname.endsWith('/') ? window.location.pathname.slice(0, -1) : window.location.pathname;

function esc(s) { var d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

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
  }
}

function toggleRecording() { if (isRecording) { stopRecording(); } else { startRecording(); } }

async function startRecording() {
  try {
    var stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true } });
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
    audioChunks = [];
    mediaRecorder.ondataavailable = function(e) { audioChunks.push(e.data); };
    mediaRecorder.onstop = function() {
      var blob = new Blob(audioChunks, { type: 'audio/webm' });
      samples.push(blob);
      stream.getTracks().forEach(function(t) { t.stop(); });
      isRecording = false;
      document.getElementById('recordBtn').textContent = 'Aufnahme starten';
      document.getElementById('recordBtn').classList.remove('recording');
      document.getElementById('sampleCount').textContent = samples.length + ' Probe(n) aufgenommen';
      document.getElementById('enrollBtn').classList.remove('hidden');
      showStatus('Probe aufgenommen. Weitere aufnehmen oder speichern.', 'info');
    };
    mediaRecorder.start();
    isRecording = true;
    document.getElementById('recordBtn').textContent = 'Aufnahme stoppen';
    document.getElementById('recordBtn').classList.add('recording');
    showStatus('Sprich jetzt...', 'info');
  } catch (e) { showStatus('Mikrofon-Fehler: ' + e.message, 'err'); }
}

function stopRecording() { if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop(); }

function handleFileUpload(input) {
  if (!input.files.length) return;
  for (var i = 0; i < input.files.length; i++) { samples.push(input.files[i]); }
  document.getElementById('enrollBtn').classList.remove('hidden');
  showStatus(samples.length + ' Probe(n) ausgewählt.', 'info');
  input.value = '';
}

async function saveSamples() {
  var name = document.getElementById('speakerName').value.trim();
  if (!name) { showStatus('Bitte einen Namen eingeben.', 'err'); return; }
  if (samples.length === 0) { showStatus('Bitte mindestens eine Probe aufnehmen.', 'err'); return; }
  var formData = new FormData();
  formData.append('name', name);
  samples.forEach(function(s, i) {
    var ext = (s.name && s.name.includes('.')) ? s.name.split('.').pop() : 'webm';
    formData.append('sample_' + i, s, 'sample_' + Date.now() + '_' + i + '.' + ext);
  });
  document.getElementById('enrollBtn').disabled = true;
  showStatus('Speichere Proben...', 'info');
  try {
    var res = await fetch(basePath + '/api/enroll', { method: 'POST', body: formData });
    var data = await res.json();
    if (data.success) {
      showStatus(esc(name) + ': Profil gespeichert! (' + data.samples + ' Proben insgesamt)', 'ok');
      samples = [];
      document.getElementById('sampleCount').textContent = '';
      document.getElementById('enrollBtn').classList.add('hidden');
      document.getElementById('speakerName').value = '';
      loadSpeakers();
    } else { showStatus('Fehler: ' + esc(data.error), 'err'); }
  } catch (e) { showStatus('Netzwerkfehler: ' + e.message, 'err'); }
  document.getElementById('enrollBtn').disabled = false;
}

var testRecording = false;
function toggleTestRecording() { if (testRecording) { stopTestRecording(); } else { startTestRecording(); } }
async function startTestRecording() {
  try {
    var stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true } });
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
    audioChunks = [];
    mediaRecorder.ondataavailable = function(e) { audioChunks.push(e.data); };
    mediaRecorder.onstop = async function() {
      var blob = new Blob(audioChunks, { type: 'audio/webm' });
      stream.getTracks().forEach(function(t) { t.stop(); });
      testRecording = false;
      document.getElementById('testRecordBtn').textContent = 'Test-Aufnahme';
      document.getElementById('testRecordBtn').classList.remove('recording');
      var formData = new FormData();
      formData.append('audio', blob, 'test.webm');
      showTestResult('Analysiere...', 'info');
      try {
        var res = await fetch(basePath + '/api/identify', { method: 'POST', body: formData });
        var data = await res.json();
        if (data.speaker) { showTestResult('Erkannt: <strong>' + esc(data.speaker) + '</strong> (Confidence: ' + (data.confidence * 100).toFixed(1) + '%)', data.confidence >= 0.75 ? 'ok' : 'info'); }
        else { showTestResult('Kein Sprecher erkannt.', 'err'); }
      } catch (e) { showTestResult('Fehler: ' + e.message, 'err'); }
    };
    mediaRecorder.start(); testRecording = true;
    document.getElementById('testRecordBtn').textContent = 'Stoppen';
    document.getElementById('testRecordBtn').classList.add('recording');
    showTestResult('Sprich jetzt...', 'info');
  } catch (e) { showTestResult('Mikrofon-Fehler: ' + e.message, 'err'); }
}
function stopTestRecording() { if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop(); }
function handleTestFileUpload(input) {
  if (!input.files[0]) return;
  var formData = new FormData();
  formData.append('audio', input.files[0], input.files[0].name);
  showTestResult('Analysiere...', 'info');
  fetch(basePath + '/api/identify', { method: 'POST', body: formData }).then(function(r) { return r.json(); }).then(function(data) {
    if (data.speaker) { showTestResult('Erkannt: <strong>' + esc(data.speaker) + '</strong> (Confidence: ' + (data.confidence * 100).toFixed(1) + '%)', data.confidence >= 0.75 ? 'ok' : 'info'); }
    else { showTestResult('Kein Sprecher erkannt.', 'err'); }
  }).catch(function(e) { showTestResult('Fehler: ' + e.message, 'err'); });
  input.value = '';
}

function showStatus(msg, type) { var el = document.getElementById('statusBox'); el.className = 'status status-' + type; el.innerHTML = msg; el.classList.remove('hidden'); }
function showTestResult(msg, type) { var el = document.getElementById('testResult'); el.className = 'status status-' + type; el.innerHTML = msg; el.classList.remove('hidden'); }

async function loadSpeakers() {
  try {
    var res = await fetch(basePath + '/api/speakers');
    var data = await res.json();
    var container = document.getElementById('speakerList');
    if (!data.speakers || data.speakers.length === 0) {
      container.innerHTML = '<div class="card"><h2>Registrierte Sprecher</h2><p class="samples-info">Noch keine Sprecher registriert.</p></div>';
      return;
    }
    // Separate _unknown from real speakers
    var speakers = data.speakers.filter(function(s) { return s.name !== '_unknown'; });
    var unknown = data.speakers.find(function(s) { return s.name === '_unknown'; });
    var speakerNames = speakers.map(function(s) { return s.name; });

    var html = speakers.map(function(s) {
      var sampleList = s.samples.map(function(f) {
        return '<li style="display:flex;justify-content:space-between;align-items:center;padding:.3rem 0;gap:.5rem">' +
          '<audio controls preload="none" style="height:28px;flex-shrink:0" src="' + basePath + '/api/audio/' + encodeURIComponent(s.name) + '/' + encodeURIComponent(f) + '"></audio>' +
          '<span style="font-size:.85rem;color:var(--muted);flex:1;overflow:hidden;text-overflow:ellipsis">' + esc(f) + '</span>' +
          '<button class="btn-danger" style="font-size:.7rem;padding:.2rem .5rem" onclick="deleteSample(&quot;' + esc(s.name) + '&quot;, &quot;' + esc(f) + '&quot;)">X</button></li>';
      }).join('');
      var learnActive = (s.name === currentLearnSpeaker);
      return '<div class="card">' +
        '<div style="display:flex;justify-content:space-between;align-items:center">' +
        '<h2>' + esc(s.name) + (s.enrolled ? ' <span style="color:var(--ok);font-size:.8rem">aktiv</span>' : ' <span style="color:var(--highlight);font-size:.8rem">nicht trainiert</span>') + '</h2>' +
        '<button class="btn-danger" onclick="deleteSpeaker(&quot;' + esc(s.name) + '&quot;)">Sprecher löschen</button></div>' +
        '<ul style="list-style:none;margin:.5rem 0">' + sampleList + '</ul>' +
        '<div style="display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.5rem">' +
        '<p class="samples-info" style="flex:1">' + s.samples.length + ' Probe(n)</p>' +
        '<button class="btn-primary" onclick="retrainSpeaker(&quot;' + esc(s.name) + '&quot;)">Profil trainieren</button>' +
        '<button class="' + (learnActive ? 'btn-danger recording' : 'btn-secondary') + '" onclick="toggleLearnMode(&quot;' + esc(s.name) + '&quot;)">' + (learnActive ? 'Lernmodus stoppen' : 'Lernmodus (Satellite)') + '</button>' +
        '</div></div>';
    }).join('');

    // Unknown samples section
    if (unknown && unknown.samples.length > 0) {
      var assignOptions = '<option value="">Zuweisen an...</option>' + speakerNames.map(function(n) {
        return '<option value="' + esc(n) + '">' + esc(n) + '</option>';
      }).join('');
      var unknownList = unknown.samples.map(function(f) {
        return '<li style="display:flex;justify-content:space-between;align-items:center;padding:.4rem 0;gap:.5rem">' +
          '<audio controls preload="none" style="height:28px;flex-shrink:0" src="' + basePath + '/api/audio/_unknown/' + encodeURIComponent(f) + '"></audio>' +
          '<span style="font-size:.85rem;color:var(--muted);flex:1;overflow:hidden;text-overflow:ellipsis">' + esc(f) + '</span>' +
          '<select onchange="assignSample(&quot;' + esc(f) + '&quot;, this.value)" style="padding:.3rem;border-radius:4px;background:var(--bg);color:var(--text);border:1px solid var(--border)">' + assignOptions + '</select>' +
          '<button class="btn-danger" style="font-size:.7rem;padding:.2rem .5rem" onclick="deleteSample(&quot;_unknown&quot;, &quot;' + esc(f) + '&quot;)">X</button></li>';
      }).join('');
      html += '<div class="card" style="border-color:var(--highlight)">' +
        '<h2>Nicht erkannte Aufnahmen <span style="color:var(--highlight);font-size:.8rem">' + unknown.samples.length + ' Probe(n)</span></h2>' +
        '<p class="samples-info">Anhören und einem Sprecher zuweisen, oder löschen.</p>' +
        '<ul style="list-style:none;margin:.5rem 0">' + unknownList + '</ul></div>';
    }

    container.innerHTML = html;
  } catch (e) {
    document.getElementById('speakerList').innerHTML = '<div class="card" style="color:#e74c3c">Fehler beim Laden</div>';
  }
}

async function assignSample(filename, toSpeaker) {
  if (!toSpeaker) return;
  try {
    var res = await fetch(basePath + '/api/move', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({from: '_unknown', filename: filename, to: toSpeaker})
    });
    var data = await res.json();
    if (data.success) { loadSpeakers(); showStatus('Probe verschoben nach ' + esc(toSpeaker), 'ok'); }
    else { showStatus('Fehler: ' + esc(data.error), 'err'); }
  } catch (e) { showStatus('Fehler: ' + e.message, 'err'); }
}

var currentLearnSpeaker = null;

async function retrainSpeaker(name) {
  showStatus('Trainiere Profil...', 'info');
  try {
    var res = await fetch(basePath + '/api/speakers/' + encodeURIComponent(name) + '/retrain', { method: 'POST' });
    var data = await res.json();
    if (data.success) { showStatus(esc(name) + ': Profil neu berechnet (' + data.samples + ' Proben)', 'ok'); loadSpeakers(); }
    else { showStatus('Fehler: ' + esc(data.error), 'err'); }
  } catch (e) { showStatus('Fehler: ' + e.message, 'err'); }
}

async function toggleLearnMode(name) {
  var newSpeaker = (currentLearnSpeaker === name) ? null : name;
  try {
    var res = await fetch(basePath + '/api/learn', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({speaker: newSpeaker}) });
    var data = await res.json();
    if (data.success) {
      currentLearnSpeaker = newSpeaker;
      loadSpeakers();
      if (newSpeaker) { showStatus('Lernmodus aktiv: Sprich zum Satellite. Jede Spracheingabe wird als Probe fuer "' + esc(newSpeaker) + '" gespeichert.', 'ok'); }
      else { showStatus('Lernmodus deaktiviert.', 'info'); }
    }
  } catch (e) { showStatus('Fehler: ' + e.message, 'err'); }
}

async function deleteSample(speaker, filename) {
  if (!confirm('Probe löschen?')) return;
  await fetch(basePath + '/api/speakers/' + encodeURIComponent(speaker) + '/samples/' + encodeURIComponent(filename), { method: 'DELETE' });
  loadSpeakers();
}

async function deleteSpeaker(name) {
  if (!confirm('Sprecher und alle Proben löschen?')) return;
  await fetch(basePath + '/api/speakers/' + encodeURIComponent(name), { method: 'DELETE' });
  loadSpeakers();
}

// --- Settings ---
async function loadSettings() {
  try {
    var res = await fetch(basePath + '/api/settings');
    var data = await res.json();
    document.getElementById('settingThreshold').value = data.similarity_threshold;
    document.getElementById('settingThresholdRange').value = data.similarity_threshold;
    document.getElementById('settingUnknown').value = data.unknown_label;
    document.getElementById('settingSaveUnknown').checked = data.save_unknown;
  } catch (e) {}
}
async function saveSettings() {
  var el = document.getElementById('settingsStatus');
  try {
    var res = await fetch(basePath + '/api/settings', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        similarity_threshold: parseFloat(document.getElementById('settingThreshold').value),
        unknown_label: document.getElementById('settingUnknown').value.trim(),
        save_unknown: document.getElementById('settingSaveUnknown').checked
      })
    });
    var data = await res.json();
    if (data.success) { el.className = 'status status-ok'; el.innerHTML = 'Gespeichert!'; el.classList.remove('hidden'); }
    else { el.className = 'status status-err'; el.innerHTML = 'Fehler: ' + esc(data.error); el.classList.remove('hidden'); }
  } catch (e) { el.className = 'status status-err'; el.innerHTML = 'Fehler: ' + e.message; el.classList.remove('hidden'); }
}

// Load learn mode status, then speakers
fetch(basePath + '/api/learn').then(function(r) { return r.json(); }).then(function(data) {
  currentLearnSpeaker = data.speaker || null;
  loadSpeakers();
}).catch(function() { loadSpeakers(); });
loadSettings();
initUI();
</script>
</body>
</html>"""


def create_web_app(speaker_db: SpeakerDatabase) -> web.Application:
    """Create the aiohttp web application for enrollment."""

    app = web.Application(client_max_size=50 * 1024 * 1024)

    async def index(request):
        return web.Response(text=HTML_PAGE, content_type="text/html")

    async def list_speakers(request):
        speakers = speaker_db.list_speakers()
        return web.json_response({"speakers": speakers})

    async def delete_speaker(request):
        name = request.match_info["name"]
        success = speaker_db.delete_speaker(name)
        return web.json_response({"success": success})

    async def delete_sample(request):
        name = request.match_info["name"]
        filename = request.match_info["filename"]
        success = speaker_db.delete_sample(name, filename)
        return web.json_response({"success": success})

    async def enroll(request):
        """Save audio samples and compute speaker embedding."""
        try:
            reader = await request.multipart()
            name = None
            saved_files = []

            async for part in reader:
                if part.name == "name":
                    name = (await part.read()).decode("utf-8").strip()
                elif part.name and part.name.startswith("sample_"):
                    data = await part.read()
                    filename = part.filename or (part.name + ".webm")
                    if data and name:
                        speaker_db.save_sample(name, data, filename)
                        saved_files.append(filename)

            if not name:
                return web.json_response({"success": False, "error": "Name fehlt"})
            if not saved_files:
                return web.json_response({"success": False, "error": "Keine Audiodaten"})

            total = len(speaker_db._get_sample_files(speaker_db.profiles_dir / name))
            return web.json_response({"success": True, "samples": total})

        except Exception as e:
            _LOGGER.exception("Enrollment failed")
            return web.json_response({"success": False, "error": str(e)})

    async def retrain(request):
        """Recompute embedding for a speaker from their current samples."""
        name = request.match_info["name"]
        try:
            loop = asyncio.get_running_loop()
            ok = await loop.run_in_executor(None, speaker_db.recompute_speaker, name)
            if not ok:
                return web.json_response({"success": False, "error": "Keine gültigen Proben"})
            total = len(speaker_db._get_sample_files(speaker_db.profiles_dir / name))
            return web.json_response({"success": True, "samples": total})
        except Exception as e:
            _LOGGER.exception("Retrain failed")
            return web.json_response({"success": False, "error": str(e)})

    async def learn_mode(request):
        """Toggle learn mode — save pipeline audio as training samples."""
        try:
            data = await request.json()
            speaker = data.get("speaker")
            # Create folder if enabling for new speaker
            if speaker:
                (speaker_db.profiles_dir / speaker).mkdir(parents=True, exist_ok=True)
            set_learn_mode(speaker)
            return web.json_response({"success": True, "speaker": speaker})
        except Exception as e:
            _LOGGER.exception("Learn mode toggle failed")
            return web.json_response({"success": False, "error": str(e)})

    async def learn_status(request):
        """Get current learn mode status."""
        return web.json_response({"speaker": get_learn_mode()})

    async def move_sample(request):
        """Move a sample from one speaker to another."""
        try:
            data = await request.json()
            from_speaker = data.get("from")
            filename = data.get("filename")
            to_speaker = data.get("to")
            if not all([from_speaker, filename, to_speaker]):
                return web.json_response({"success": False, "error": "Missing fields"})
            ok = speaker_db.move_sample(from_speaker, filename, to_speaker)
            return web.json_response({"success": ok})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)})

    async def serve_audio(request):
        """Serve an audio file for playback."""
        name = request.match_info["name"]
        filename = request.match_info["filename"]
        filepath = speaker_db.profiles_dir / name / filename
        if not filepath.exists():
            return web.Response(status=404)
        return web.FileResponse(filepath)

    async def get_settings(request):
        return web.json_response({
            "similarity_threshold": speaker_db.similarity_threshold,
            "unknown_label": speaker_db.unknown_label,
            "save_unknown": get_save_unknown(),
        })

    async def post_settings(request):
        try:
            data = await request.json()
            if "similarity_threshold" in data:
                val = float(data["similarity_threshold"])
                if 0.1 <= val <= 1.0:
                    speaker_db.similarity_threshold = val
            if "unknown_label" in data:
                speaker_db.unknown_label = data["unknown_label"] or "Unbekannt"
            if "save_unknown" in data:
                set_save_unknown(bool(data["save_unknown"]))
            _LOGGER.info("Settings updated: threshold=%.2f, unknown=%s, save_unknown=%s",
                         speaker_db.similarity_threshold, speaker_db.unknown_label, get_save_unknown())
            return web.json_response({"success": True})
        except Exception as e:
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

            audio_array = await _convert_audio_to_numpy(audio_data)
            if audio_array is None:
                return web.json_response({"speaker": None, "confidence": 0})

            loop = asyncio.get_running_loop()
            name, user_id, confidence = await loop.run_in_executor(
                None, speaker_db.identify_speaker, audio_array, 16000,
            )
            return web.json_response({
                "speaker": name, "user_id": user_id,
                "confidence": round(confidence, 4),
            })
        except Exception as e:
            _LOGGER.exception("Identification test failed")
            return web.json_response({"speaker": None, "confidence": 0, "error": str(e)})

    for prefix in ["", "/speaker-id"]:
        app.router.add_get(prefix + "/", index)
        app.router.add_get(prefix + "/api/speakers", list_speakers)
        app.router.add_delete(prefix + "/api/speakers/{name}", delete_speaker)
        app.router.add_delete(prefix + "/api/speakers/{name}/samples/{filename}", delete_sample)
        app.router.add_post(prefix + "/api/speakers/{name}/retrain", retrain)
        app.router.add_post(prefix + "/api/enroll", enroll)
        app.router.add_post(prefix + "/api/identify", identify)
        app.router.add_post(prefix + "/api/learn", learn_mode)
        app.router.add_get(prefix + "/api/learn", learn_status)
        app.router.add_post(prefix + "/api/move", move_sample)
        app.router.add_get(prefix + "/api/audio/{name}/{filename}", serve_audio)
        app.router.add_get(prefix + "/api/settings", get_settings)
        app.router.add_post(prefix + "/api/settings", post_settings)
    app.router.add_get("/speaker-id", lambda r: web.HTTPFound("/speaker-id/"))

    return app


async def _convert_audio_to_numpy(audio_data: bytes) -> np.ndarray | None:
    """Convert audio bytes to 16kHz mono float32 numpy array via ffmpeg."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", "pipe:0",
            "-ar", "16000", "-ac", "1", "-f", "s16le",
            "-acodec", "pcm_s16le", "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=audio_data)
        if proc.returncode != 0:
            _LOGGER.error("ffmpeg error: %s", stderr.decode(errors="replace")[-500:])
            return None
        raw = np.frombuffer(stdout, dtype=np.int16)
        return raw.astype(np.float32) / 32768.0
    except Exception as e:
        _LOGGER.error("Audio conversion failed: %s", e)
        return None
