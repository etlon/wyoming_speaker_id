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
<title>Speaker ID</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0c0b0e; --surface: #16151a; --surface2: #1e1d23;
    --amber: #c8956c; --amber-dim: rgba(200,149,108,.12); --amber-glow: rgba(200,149,108,.25);
    --red: #bf4a3c; --red-dim: rgba(191,74,60,.12);
    --green: #6ab87a; --green-dim: rgba(106,184,122,.12);
    --text: #d8d2c8; --text-dim: #7a756d; --text-bright: #f0ece4;
    --border: #2a2830; --border-light: #3a3840;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'DM Sans', sans-serif; background: var(--bg); color: var(--text);
    padding: 2rem 1.5rem; max-width: 860px; margin: 0 auto;
    background-image: url("data:image/svg+xml,%3Csvg width='40' height='40' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence baseFrequency='.65' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.03'/%3E%3C/svg%3E");
  }

  /* Typography */
  h1 { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.6rem; letter-spacing: -.02em;
       color: var(--text-bright); margin-bottom: .3rem; }
  h1 span { color: var(--amber); }
  .subtitle { font-size: .8rem; color: var(--text-dim); letter-spacing: .08em; text-transform: uppercase;
              font-family: 'JetBrains Mono', monospace; margin-bottom: 2rem; }
  h2 { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.05rem; color: var(--text-bright);
       letter-spacing: -.01em; }
  h3 { font-family: 'Syne', sans-serif; font-weight: 600; font-size: .9rem; color: var(--text-dim);
       text-transform: uppercase; letter-spacing: .06em; margin-bottom: .8rem; }

  /* Cards */
  .card {
    background: var(--surface); border-radius: 10px; padding: 1.4rem; margin-bottom: 1rem;
    border: 1px solid var(--border); position: relative; overflow: hidden;
  }
  .card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-light) 30%, var(--border-light) 70%, transparent);
  }
  .card-unknown { border-color: var(--red); }
  .card-unknown::before { background: linear-gradient(90deg, transparent, var(--red) 30%, var(--red) 70%, transparent); }

  /* Forms */
  label { display: block; margin-bottom: .35rem; font-size: .78rem; color: var(--text-dim);
          font-family: 'JetBrains Mono', monospace; letter-spacing: .03em; text-transform: uppercase; }
  input[type=text], input[type=number] {
    width: 100%; padding: .6rem .8rem; border-radius: 6px; border: 1px solid var(--border);
    background: var(--bg); color: var(--text); font-size: .9rem; font-family: 'DM Sans', sans-serif;
    margin-bottom: .8rem; transition: border-color .2s;
  }
  input[type=text]:focus, input[type=number]:focus { border-color: var(--amber); outline: none; box-shadow: 0 0 0 2px var(--amber-dim); }
  input[type=range] { accent-color: var(--amber); }

  /* Buttons */
  button {
    padding: .55rem 1.1rem; border-radius: 6px; border: 1px solid var(--border); font-size: .82rem;
    cursor: pointer; transition: all .15s ease; font-weight: 500; font-family: 'DM Sans', sans-serif;
    background: var(--surface2); color: var(--text);
  }
  button:hover { border-color: var(--border-light); background: var(--border); }
  .btn-amber { background: var(--amber-dim); color: var(--amber); border-color: rgba(200,149,108,.25); }
  .btn-amber:hover { background: rgba(200,149,108,.2); border-color: var(--amber); }
  .btn-red { background: var(--red-dim); color: var(--red); border-color: rgba(191,74,60,.2); }
  .btn-red:hover { background: rgba(191,74,60,.2); border-color: var(--red); }
  .btn-green { background: var(--green-dim); color: var(--green); border-color: rgba(106,184,122,.2); }
  .btn-green:hover { background: rgba(106,184,122,.2); border-color: var(--green); }
  button:disabled { opacity: .4; cursor: not-allowed; }

  .recording { animation: rec-pulse 1.2s ease infinite; }
  @keyframes rec-pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(191,74,60,.4); } 50% { box-shadow: 0 0 0 8px rgba(191,74,60,0); } }

  /* Tags */
  .tag { display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: .68rem; padding: .15rem .45rem;
         border-radius: 3px; letter-spacing: .02em; }
  .tag-active { background: var(--green-dim); color: var(--green); }
  .tag-inactive { background: var(--amber-dim); color: var(--amber); }

  /* Status */
  .status { padding: .5rem .8rem; border-radius: 6px; margin-top: .8rem; font-size: .82rem; }
  .status-ok { background: var(--green-dim); color: var(--green); }
  .status-err { background: var(--red-dim); color: var(--red); }
  .status-info { background: var(--amber-dim); color: var(--amber); }

  /* Sample rows */
  .sample-row {
    display: flex; align-items: center; gap: .5rem; padding: .35rem 0;
    border-bottom: 1px solid var(--border);
  }
  .sample-row:last-child { border-bottom: none; }
  .sample-row audio { height: 26px; flex-shrink: 0; filter: sepia(.3) saturate(.5) brightness(.85); }
  .sample-name {
    font-family: 'JetBrains Mono', monospace; font-size: .75rem; color: var(--text-dim);
    flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: pointer;
    padding: .2rem .4rem; border-radius: 3px; border: 1px solid transparent; transition: all .15s;
  }
  .sample-name:hover { border-color: var(--border-light); color: var(--text); }
  .sample-name-editing {
    background: var(--bg); border-color: var(--amber) !important; color: var(--text-bright);
    outline: none; cursor: text; font-family: 'JetBrains Mono', monospace; font-size: .75rem;
    padding: .2rem .4rem; border-radius: 3px; flex: 1;
  }

  select {
    padding: .3rem .5rem; border-radius: 4px; background: var(--bg); color: var(--text);
    border: 1px solid var(--border); font-size: .78rem; font-family: 'DM Sans', sans-serif;
  }
  select:focus { border-color: var(--amber); outline: none; }

  .meta { font-family: 'JetBrains Mono', monospace; font-size: .72rem; color: var(--text-dim); }
  .flex { display: flex; }
  .gap-sm { gap: .4rem; }
  .gap-md { gap: .6rem; }
  .wrap { flex-wrap: wrap; }
  .between { justify-content: space-between; }
  .center { align-items: center; }
  .hidden { display: none; }
  .mt { margin-top: .6rem; }
  .mb { margin-bottom: .8rem; }

  /* Checkbox */
  input[type=checkbox] { accent-color: var(--amber); margin-right: .4rem; }

  /* Divider */
  hr { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
</style>
</head>
<body>

<h1>Speaker <span>ID</span></h1>
<div class="subtitle">Voice profile management</div>

<!-- Speakers + Samples -->
<div id="speakerList"></div>

<!-- Add Samples -->
<div class="card">
  <h3>Proben hinzufügen</h3>
  <label for="speakerName">Sprecher</label>
  <input type="text" id="speakerName" placeholder="Name eingeben...">

  <div id="micEnroll">
    <div class="flex gap-sm center">
      <button class="btn-amber" id="recordBtn" onclick="toggleRecording()">Aufnahme</button>
      <span id="sampleCount" class="meta"></span>
    </div>
  </div>
  <div id="fileEnroll" class="hidden">
    <div class="status status-info mb">Kein Mikrofon (HTTPS erforderlich). Dateien hochladen:</div>
  </div>
  <input type="file" id="fileInput" accept="audio/*" multiple onchange="handleFileUpload(this)" style="margin:.4rem 0;font-size:.8rem">

  <div class="mt">
    <button class="hidden btn-green" id="enrollBtn" onclick="saveSamples()">Proben speichern</button>
  </div>
  <div id="statusBox" class="hidden"></div>
</div>

<!-- Test -->
<div class="card">
  <h3>Erkennung testen</h3>
  <div class="flex gap-sm">
    <div id="micTest"><button class="btn-amber" id="testRecordBtn" onclick="toggleTestRecording()">Aufnahme</button></div>
    <div id="fileTest" class="hidden"><input type="file" id="testFileInput" accept="audio/*" onchange="handleTestFileUpload(this)" style="font-size:.8rem"></div>
  </div>
  <div id="testResult" class="hidden"></div>
</div>

<hr>

<!-- Settings -->
<div class="card">
  <h3>Einstellungen</h3>
  <label>Schwelle</label>
  <div class="flex gap-md center mb">
    <input type="range" id="settingThresholdRange" min="0.3" max="0.95" step="0.01" style="flex:1" oninput="document.getElementById('settingThreshold').value=this.value">
    <input type="number" id="settingThreshold" min="0.3" max="0.95" step="0.01" style="width:4.5rem">
  </div>
  <div class="meta mb">Niedriger = mehr Treffer. Standard: 0.75</div>

  <label>Unknown Label</label>
  <input type="text" id="settingUnknown" placeholder="Unbekannt">

  <label style="display:inline;cursor:pointer;text-transform:none;font-family:'DM Sans',sans-serif;font-size:.85rem">
    <input type="checkbox" id="settingSaveUnknown"> Unerkannte Aufnahmen speichern
  </label>

  <div class="mt"><button class="btn-amber" onclick="saveSettings()">Speichern</button></div>
  <div id="settingsStatus" class="hidden"></div>

  <hr style="margin:1.2rem 0;border-color:var(--border)">
  <h3>Backup</h3>
  <div class="flex gap-sm wrap">
    <a id="backupLink" class="btn-green" style="text-decoration:none;display:inline-block" onclick="this.href=B+'/api/backup'">Backup herunterladen (.zip)</a>
    <label class="btn-amber" style="cursor:pointer;display:inline-block">
      Backup importieren
      <input type="file" accept=".zip" onchange="importBackup(this)" style="display:none">
    </label>
  </div>
  <div id="backupStatus" class="hidden"></div>
</div>

<script>
var MR=null,chunks=[],samples=[],rec=false,testRec=false,mic=!!(navigator.mediaDevices&&navigator.mediaDevices.getUserMedia);
var B=window.location.pathname.endsWith('/')?window.location.pathname.slice(0,-1):window.location.pathname;
var learnSpeaker=null;
function E(s){var d=document.createElement('div');d.textContent=s;return d.innerHTML}
function $(id){return document.getElementById(id)}
function initUI(){if(mic){$('micEnroll').classList.remove('hidden');$('fileEnroll').classList.add('hidden');$('micTest').classList.remove('hidden');$('fileTest').classList.add('hidden')}else{$('micEnroll').classList.add('hidden');$('fileEnroll').classList.remove('hidden');$('micTest').classList.add('hidden');$('fileTest').classList.remove('hidden')}}
function msg(id,m,t){var el=$(id);el.className='status status-'+t;el.innerHTML=m;el.classList.remove('hidden')}
function showStatus(m,t){msg('statusBox',m,t)}
function showTestResult(m,t){msg('testResult',m,t)}

// Recording
function toggleRecording(){rec?stopRec():startRec()}
async function startRec(){try{var s=await navigator.mediaDevices.getUserMedia({audio:{sampleRate:16000,channelCount:1,echoCancellation:true}});MR=new MediaRecorder(s,{mimeType:'audio/webm;codecs=opus'});chunks=[];MR.ondataavailable=function(e){chunks.push(e.data)};MR.onstop=function(){samples.push(new Blob(chunks,{type:'audio/webm'}));s.getTracks().forEach(function(t){t.stop()});rec=false;$('recordBtn').textContent='Aufnahme';$('recordBtn').classList.remove('recording');$('sampleCount').textContent=samples.length+' aufgenommen';$('enrollBtn').classList.remove('hidden')};MR.start();rec=true;$('recordBtn').textContent='Stoppen';$('recordBtn').classList.add('recording');showStatus('Sprich jetzt...','info')}catch(e){showStatus('Mikrofon: '+e.message,'err')}}
function stopRec(){if(MR&&MR.state==='recording')MR.stop()}
function handleFileUpload(inp){if(!inp.files.length)return;for(var i=0;i<inp.files.length;i++)samples.push(inp.files[i]);$('enrollBtn').classList.remove('hidden');showStatus(samples.length+' Datei(en)','info');inp.value=''}

async function saveSamples(){var n=$('speakerName').value.trim();if(!n){showStatus('Name eingeben','err');return}if(!samples.length){showStatus('Proben fehlen','err');return}var fd=new FormData();fd.append('name',n);samples.forEach(function(s,i){var x=(s.name&&s.name.includes('.'))?s.name.split('.').pop():'webm';fd.append('sample_'+i,s,'sample_'+Date.now()+'_'+i+'.'+x)});$('enrollBtn').disabled=true;showStatus('Speichere...','info');try{var r=await(await fetch(B+'/api/enroll',{method:'POST',body:fd})).json();if(r.success){showStatus(E(n)+': '+r.samples+' Proben gespeichert','ok');samples=[];$('sampleCount').textContent='';$('enrollBtn').classList.add('hidden');$('speakerName').value='';load()}else showStatus(E(r.error),'err')}catch(e){showStatus(e.message,'err')}$('enrollBtn').disabled=false}

// Test
function toggleTestRecording(){testRec?stopTestRec():startTestRec()}
async function startTestRec(){try{var s=await navigator.mediaDevices.getUserMedia({audio:{sampleRate:16000,channelCount:1,echoCancellation:true}});MR=new MediaRecorder(s,{mimeType:'audio/webm;codecs=opus'});chunks=[];MR.ondataavailable=function(e){chunks.push(e.data)};MR.onstop=async function(){var b=new Blob(chunks,{type:'audio/webm'});s.getTracks().forEach(function(t){t.stop()});testRec=false;$('testRecordBtn').textContent='Aufnahme';$('testRecordBtn').classList.remove('recording');var fd=new FormData();fd.append('audio',b,'test.webm');showTestResult('Analysiere...','info');try{var d=await(await fetch(B+'/api/identify',{method:'POST',body:fd})).json();if(d.speaker)showTestResult('<strong>'+E(d.speaker)+'</strong> &mdash; '+(d.confidence*100).toFixed(1)+'%',d.confidence>=0.75?'ok':'info');else showTestResult('Nicht erkannt','err')}catch(e){showTestResult(e.message,'err')}};MR.start();testRec=true;$('testRecordBtn').textContent='Stoppen';$('testRecordBtn').classList.add('recording');showTestResult('Sprich...','info')}catch(e){showTestResult(e.message,'err')}}
function stopTestRec(){if(MR&&MR.state==='recording')MR.stop()}
function handleTestFileUpload(inp){if(!inp.files[0])return;var fd=new FormData();fd.append('audio',inp.files[0],inp.files[0].name);showTestResult('Analysiere...','info');fetch(B+'/api/identify',{method:'POST',body:fd}).then(function(r){return r.json()}).then(function(d){if(d.speaker)showTestResult('<strong>'+E(d.speaker)+'</strong> &mdash; '+(d.confidence*100).toFixed(1)+'%',d.confidence>=0.75?'ok':'info');else showTestResult('Nicht erkannt','err')}).catch(function(e){showTestResult(e.message,'err')});inp.value=''}

// Rename (click-to-edit)
function startRename(speaker,filename,el){var inp=document.createElement('input');inp.type='text';inp.className='sample-name-editing';inp.value=filename;el.replaceWith(inp);inp.focus();inp.select();function finish(){var nv=inp.value.trim();if(!nv||nv===filename){inp.replaceWith(el);return}fetch(B+'/api/speakers/'+encodeURIComponent(speaker)+'/samples/'+encodeURIComponent(filename)+'/rename',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({new_name:nv})}).then(function(r){return r.json()}).then(function(d){if(d.success)load();else{inp.replaceWith(el);showStatus(E(d.error||'Fehler'),'err')}}).catch(function(){inp.replaceWith(el)})}inp.onblur=finish;inp.onkeydown=function(e){if(e.key==='Enter')finish();if(e.key==='Escape'){inp.replaceWith(el)}}}

// Speaker list
async function load(){try{var d=await(await fetch(B+'/api/speakers')).json();var c=$('speakerList');if(!d.speakers||!d.speakers.length){c.innerHTML='<div class="card"><h3>Sprecher</h3><div class="meta">Keine Sprecher vorhanden</div></div>';return}
var sp=d.speakers.filter(function(s){return s.name!=='_unknown'});var unk=d.speakers.find(function(s){return s.name==='_unknown'});var names=sp.map(function(s){return s.name});
var h=sp.map(function(s){var sl=s.samples.map(function(f){return'<div class="sample-row"><audio controls preload="none" src="'+B+'/api/audio/'+encodeURIComponent(s.name)+'/'+encodeURIComponent(f)+'"></audio><span class="sample-name" onclick="startRename(&quot;'+E(s.name)+'&quot;,&quot;'+E(f)+'&quot;,this)">'+E(f)+'</span><button class="btn-red" style="padding:.15rem .4rem;font-size:.72rem" onclick="delSample(&quot;'+E(s.name)+'&quot;,&quot;'+E(f)+'&quot;)">&times;</button></div>'}).join('');
var la=(s.name===learnSpeaker);return'<div class="card"><div class="flex between center mb"><div class="flex center gap-sm"><h2>'+E(s.name)+'</h2>'+(s.enrolled?'<span class="tag tag-active">aktiv</span>':'<span class="tag tag-inactive">nicht trainiert</span>')+'</div><button class="btn-red" onclick="delSpeaker(&quot;'+E(s.name)+'&quot;)">&times; Löschen</button></div>'+sl+'<div class="flex gap-sm wrap mt"><span class="meta" style="flex:1;align-self:center">'+s.samples.length+' Proben</span><button class="btn-green" onclick="retrain(&quot;'+E(s.name)+'&quot;)">Trainieren</button><button class="'+(la?'btn-red recording':'btn-amber')+'" onclick="toggleLearn(&quot;'+E(s.name)+'&quot;)">'+(la?'Lernen stoppen':'Lernen (Satellite)')+'</button></div></div>'}).join('');

if(unk&&unk.samples.length){var opts='<option value="">Zuweisen...</option>'+names.map(function(n){return'<option value="'+E(n)+'">'+E(n)+'</option>'}).join('');
var ul=unk.samples.map(function(f){return'<div class="sample-row"><audio controls preload="none" src="'+B+'/api/audio/_unknown/'+encodeURIComponent(f)+'"></audio><span class="sample-name" onclick="startRename(&quot;_unknown&quot;,&quot;'+E(f)+'&quot;,this)">'+E(f)+'</span><select onchange="assign(&quot;'+E(f)+'&quot;,this.value)">'+opts+'</select><button class="btn-red" style="padding:.15rem .4rem;font-size:.72rem" onclick="delSample(&quot;_unknown&quot;,&quot;'+E(f)+'&quot;)">&times;</button></div>'}).join('');
h+='<div class="card card-unknown"><div class="flex between center mb"><h2>Nicht erkannt</h2><span class="meta">'+unk.samples.length+' Aufnahmen</span></div>'+ul+'</div>'}
c.innerHTML=h}catch(e){$('speakerList').innerHTML='<div class="card" style="color:var(--red)">Fehler</div>'}}

async function assign(f,to){if(!to)return;try{var d=await(await fetch(B+'/api/move',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({from:'_unknown',filename:f,to:to})})).json();if(d.success){load();showStatus('Verschoben nach '+E(to),'ok')}}catch(e){showStatus(e.message,'err')}}
async function retrain(n){showStatus('Trainiere...','info');try{var d=await(await fetch(B+'/api/speakers/'+encodeURIComponent(n)+'/retrain',{method:'POST'})).json();if(d.success){showStatus(E(n)+': '+d.samples+' Proben trainiert','ok');load()}else showStatus(E(d.error),'err')}catch(e){showStatus(e.message,'err')}}
async function toggleLearn(n){var ns=(learnSpeaker===n)?null:n;try{var d=await(await fetch(B+'/api/learn',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({speaker:ns})})).json();if(d.success){learnSpeaker=ns;load();if(ns)showStatus('Lernmodus: Sprich zum Satellite &rarr; '+E(ns),'ok');else showStatus('Lernmodus aus','info')}}catch(e){showStatus(e.message,'err')}}
async function delSample(s,f){if(!confirm('Probe löschen?'))return;await fetch(B+'/api/speakers/'+encodeURIComponent(s)+'/samples/'+encodeURIComponent(f),{method:'DELETE'});load()}
async function delSpeaker(n){if(!confirm('Sprecher und alle Proben löschen?'))return;await fetch(B+'/api/speakers/'+encodeURIComponent(n),{method:'DELETE'});load()}

// Settings
async function loadSettings(){try{var d=await(await fetch(B+'/api/settings')).json();$('settingThreshold').value=d.similarity_threshold;$('settingThresholdRange').value=d.similarity_threshold;$('settingUnknown').value=d.unknown_label;$('settingSaveUnknown').checked=d.save_unknown}catch(e){}}
async function saveSettings(){var el=$('settingsStatus');try{var d=await(await fetch(B+'/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({similarity_threshold:parseFloat($('settingThreshold').value),unknown_label:$('settingUnknown').value.trim(),save_unknown:$('settingSaveUnknown').checked})})).json();msg('settingsStatus',d.success?'Gespeichert':E(d.error),d.success?'ok':'err')}catch(e){msg('settingsStatus',e.message,'err')}}

async function importBackup(inp){if(!inp.files[0])return;var fd=new FormData();fd.append('backup',inp.files[0]);msg('backupStatus','Importiere...','info');try{var d=await(await fetch(B+'/api/backup',{method:'POST',body:fd})).json();if(d.success){msg('backupStatus','Import: '+d.speakers+' Sprecher, '+d.samples+' Proben','ok');load()}else msg('backupStatus',E(d.error),'err')}catch(e){msg('backupStatus',e.message,'err')}inp.value=''}

fetch(B+'/api/learn').then(function(r){return r.json()}).then(function(d){learnSpeaker=d.speaker||null;load()}).catch(function(){load()});
loadSettings();initUI();
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

    async def rename_sample(request):
        """Rename an audio sample file."""
        name = request.match_info["name"]
        filename = request.match_info["filename"]
        try:
            data = await request.json()
            new_name = data.get("new_name", "").strip()
            if not new_name:
                return web.json_response({"success": False, "error": "Name fehlt"})
            ok = speaker_db.rename_sample(name, filename, new_name)
            return web.json_response({"success": ok})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)})

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

    async def backup_download(request):
        """Download all profiles as a zip file."""
        import zipfile
        import io
        import time
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for speaker_dir in sorted(speaker_db.profiles_dir.iterdir()):
                if not speaker_dir.is_dir() or speaker_dir.name.startswith("."):
                    continue
                for f in sorted(speaker_dir.iterdir()):
                    if f.is_file() and not f.name.startswith("."):
                        zf.write(f, f"{speaker_dir.name}/{f.name}")
        buf.seek(0)
        ts = time.strftime("%Y%m%d_%H%M%S")
        return web.Response(
            body=buf.read(),
            content_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="speaker_id_backup_{ts}.zip"'},
        )

    async def backup_import(request):
        """Import a zip backup, merging with existing profiles."""
        import zipfile
        import io
        try:
            reader = await request.multipart()
            zip_data = None
            async for part in reader:
                if part.name == "backup":
                    zip_data = await part.read()
            if not zip_data:
                return web.json_response({"success": False, "error": "Keine Datei"})
            buf = io.BytesIO(zip_data)
            speakers_added = set()
            samples_count = 0
            with zipfile.ZipFile(buf, 'r') as zf:
                for info in zf.infolist():
                    if info.is_dir() or info.filename.startswith("."):
                        continue
                    parts = info.filename.split("/")
                    if len(parts) != 2:
                        continue
                    speaker_name, filename = parts
                    if filename.startswith("."):
                        continue
                    dest_dir = speaker_db.profiles_dir / speaker_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest = dest_dir / filename
                    if not dest.exists():
                        dest.write_bytes(zf.read(info.filename))
                        samples_count += 1
                    speakers_added.add(speaker_name)
            return web.json_response({"success": True, "speakers": len(speakers_added), "samples": samples_count})
        except Exception as e:
            _LOGGER.exception("Backup import failed")
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
        app.router.add_post(prefix + "/api/speakers/{name}/samples/{filename}/rename", rename_sample)
        app.router.add_post(prefix + "/api/enroll", enroll)
        app.router.add_post(prefix + "/api/identify", identify)
        app.router.add_post(prefix + "/api/learn", learn_mode)
        app.router.add_get(prefix + "/api/learn", learn_status)
        app.router.add_post(prefix + "/api/move", move_sample)
        app.router.add_get(prefix + "/api/audio/{name}/{filename}", serve_audio)
        app.router.add_get(prefix + "/api/settings", get_settings)
        app.router.add_post(prefix + "/api/settings", post_settings)
        app.router.add_get(prefix + "/api/backup", backup_download)
        app.router.add_post(prefix + "/api/backup", backup_import)
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
