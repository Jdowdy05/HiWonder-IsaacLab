# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Loopback inspection controls and bounded, asynchronous lossless frame export.

This module never accesses the environment, USD, or CUDA. The simulator thread
drains validated commands and submits owned CPU snapshots at its export cadence.
"""

import io
import json
import queue
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from isaaclab_hiwonder.lighting import validate_preview_controls

_PAGE = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HiWonder physical lighting inspection</title>
<style>
body{background:#111921;color:#e4edf5;font:15px system-ui;margin:22px;max-width:1380px}
h1{font-size:23px}button,select{background:#263746;color:white;padding:9px;border:1px solid #536576;
border-radius:5px;margin:4px;cursor:pointer}.views{display:flex;gap:16px;flex-wrap:wrap}
figure{margin:10px 0}img{width:640px;height:480px;max-width:100%;object-fit:contain;background:black}
figcaption{margin:6px 0}.controls{display:flex;gap:20px;flex-wrap:wrap}
label{display:block;width:245px;margin:12px 0}input{width:185px;vertical-align:middle}
output{display:inline-block;min-width:45px}pre{white-space:pre-wrap;color:#b8cfdf;font-size:12px}
#error{color:#ffb09d}a{color:#a5d9ff}
</style><h1>HiWonder wrist camera and physical RGB spots</h1>
<p>Zero-action hold · held-out uniform rooms · no optimizer. Sliders change physical lights in all rooms.
While running, the policy image is the actual transformed observation from the same environment step.
While paused, raw RGB is a render-only preview and policy RGB remains the last simulation observation.</p>
<div><select id="room" aria-label="Room"></select>
<button onclick="send({action:'reset'})">Reset episodes</button>
<button id="pause" onclick="send({action:'pause',paused:!paused})">Pause</button>
<button onclick="send({action:'clear'})">Use randomized profile / clear overrides</button>
<button onclick="exportProfile()">Export partial lighting profile</button>
<button onclick="send({action:'stop'})">Stop inspection</button></div>
<div class="controls" id="controls"></div><p id="summary">Waiting for simulator...</p><p id="error"></p>
<div class="views"><figure><figcaption id="raw-caption">Raw wrist RGB · full configured resolution</figcaption>
<img id="raw" alt="Raw wrist camera"></figure><figure>
<figcaption id="policy-caption">Actual policy RGB · ISP, smudge and image randomization</figcaption>
<img id="policy" alt="Policy observation"></figure></div>
<details><summary>Applied controls and live configuration</summary><pre id="status"></pre></details>
<script>
const token=__TOKEN__, count=__COUNT__;
let paused=false, live={}, busy=false, sequence=-1;
const specs=[['saturation','RGB saturation',0,1,.01,.7],['power','Colored light power ×',0,3,.01,1],
 ['radius_m','Spot radius (m)',.01,.2,.001,.06],['speed','Motion/hue speed ×',0,3,.01,1],
 ['fill_intensity','White fill intensity',0,2000,1,120]];
const controls=document.getElementById('controls');
for(const [key,label,min,max,step,value] of specs){
 const node=document.createElement('label');node.textContent=label+' ';
 const input=document.createElement('input');Object.assign(input,{type:'range',min,max,step,value,id:key});
 const output=document.createElement('output');output.textContent=value;
 input.oninput=()=>output.textContent=input.value;
 input.onchange=()=>send({action:'lighting',controls:{[key]:Number(input.value)}});
 node.append(input,output);controls.append(node);
}
const room=document.getElementById('room');
for(let i=0;i<count;i++){
 const option=document.createElement('option');option.value=i;option.textContent='Room '+i;room.append(option)}
room.onchange=()=>send({action:'room',index:Number(room.value)});
async function send(command){try{
 const response=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json',
 'X-Inspection-Token':token},body:JSON.stringify(command)});
 const result=await response.json();if(!response.ok)throw new Error(result.error);
 document.getElementById('error').textContent='';
}catch(error){document.getElementById('error').textContent=String(error)}}
function exportProfile(){
 const lighting=live.lighting;
 if(!lighting?.profile){document.getElementById('error').textContent='Waiting for applied lighting settings';return}
 const profile=structuredClone(lighting.profile), o=lighting.overrides||{};
 if('saturation' in o)profile.color_saturation=[o.saturation,o.saturation];
 if('power' in o)profile.intensity=profile.intensity.map(v=>v*o.power);
 if('radius_m' in o)profile.spot_radius_m=[o.radius_m,o.radius_m];
 if('speed' in o)profile.angular_speed=profile.angular_speed.map(v=>v*o.speed);
 if('fill_intensity' in o)profile.white_fill_intensity=[o.fill_intensity,o.fill_intensity];
 const blob=new Blob([JSON.stringify({randomization:{lighting:profile}},null,2)+'\\n'],{type:'application/json'});
 const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='lighting-preview.json';
 link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);
}
async function poll(){if(busy)return;busy=true;try{
 const response=await fetch('/api/status',{cache:'no-store'});live=await response.json();
 paused=!!live.paused;document.getElementById('pause').textContent=paused?'Resume':'Pause';
 document.getElementById('raw-caption').textContent=paused?
 'Raw wrist RGB · render-only preview; physics paused':'Raw wrist RGB · full configured resolution';
 document.getElementById('policy-caption').textContent=paused?
 'Policy RGB · last simulation observation; held while paused':
 'Actual policy RGB · ISP, smudge and image randomization';
 if(document.activeElement!==room)room.value=live.room??0;
 document.getElementById('status').textContent=JSON.stringify(live,null,2);
 document.getElementById('summary').textContent=
 `Room ${live.room??0} · ${live.mode??'starting'} · step ${live.step??0} · `+
 `native display ${Number(live.display_hz??0).toFixed(1)} Hz · `+
 `configured capture ${live.camera?.fps??'?'} Hz · browser export ${Number(live.export?.wall_hz??0).toFixed(1)} Hz`;
 const profile=live.lighting?.profile;
 const mean=range=>range?.length===2?(range[0]+range[1])/2:undefined;
 const defaults={saturation:mean(profile?.color_saturation),power:1,radius_m:mean(profile?.spot_radius_m),
 speed:1,fill_intensity:mean(profile?.white_fill_intensity)};
 for(const [key] of specs){
 const input=document.getElementById(key), value=live.lighting?.overrides?.[key]??defaults[key];
 if(value!==undefined&&document.activeElement!==input){input.value=value;input.nextElementSibling.textContent=value}}
 const next=live.export?.sequence??0;
 if(next>sequence&&next>0){
 const labels=['raw','policy'];
 const frames=await Promise.all(labels.map(label=>fetch('/wrist_'+label+'.png?sequence='+next,{cache:'no-store'})));
 if(frames.some(frame=>frame.status===410)){sequence=-1;return}
 if(frames.some(frame=>!frame.ok))throw new Error('Frame pair unavailable');
 const blobs=await Promise.all(frames.map(frame=>frame.blob()));
 labels.forEach((label,i)=>{
 const url=URL.createObjectURL(blobs[i]), image=document.getElementById(label), old=image.src;
 image.src=url;if(old.startsWith('blob:'))URL.revokeObjectURL(old);
 });sequence=next}
}catch(error){document.getElementById('error').textContent='Inspection unavailable: '+error}finally{busy=false}}
setInterval(poll,200);poll();
</script></html>"""


def _validate_command(value, num_envs):
    if not isinstance(value, dict):
        raise ValueError("Command must be a JSON object")
    action = value.get("action")
    if action == "lighting" and set(value) == {"action", "controls"}:
        controls = value["controls"]
        if not isinstance(controls, dict) or not controls:
            raise ValueError("Expected known lighting controls")
        value["controls"] = validate_preview_controls(controls)
    elif action == "room" and set(value) == {"action", "index"}:
        if type(value["index"]) is not int or not 0 <= value["index"] < num_envs:
            raise ValueError("Room index is outside this cohort")
    elif action == "pause" and set(value) == {"action", "paused"}:
        if type(value["paused"]) is not bool:
            raise ValueError("paused must be a boolean")
    elif action not in ("reset", "clear", "stop") or set(value) != {"action"}:
        raise ValueError("Unknown action or unexpected fields")
    return value


class InspectionServer:
    """Own a loopback server, command queue and one latest-only PNG encoding worker.

    Submitted CPU arrays are copied before returning. HTTP and PNG/file work never
    call simulator code. Intermediate pending frames are replaced, never queued.
    """

    def __init__(self, output_directory: Path, num_envs: int, port: int = 8769):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self._commands = queue.Queue(maxsize=64)
        self._condition = threading.Condition()
        self._pending = None
        self._closing = False
        self._status = {}
        self._frame_history = {}
        self._export = {"sequence": 0, "wall_hz": 0.0, "error": None}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(2.0)

            def log_message(self, *_args):
                pass

            def respond(self, code, body, content_type="application/json"):
                if isinstance(body, dict):
                    body = json.dumps(body, allow_nan=False).encode()
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Cross-Origin-Resource-Policy", "same-origin")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = urlsplit(self.path).path
                if path == "/":
                    page = _PAGE.replace("__TOKEN__", json.dumps(owner.token)).replace("__COUNT__", str(num_envs))
                    self.respond(200, page.encode(), "text/html; charset=utf-8")
                elif path == "/api/status":
                    with owner._condition:
                        status = {**owner._status, "export": dict(owner._export)}
                    self.respond(200, status)
                elif path in ("/wrist_raw.png", "/wrist_policy.png"):
                    try:
                        query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                        if set(query) - {"sequence"} or ("sequence" in query and len(query["sequence"]) != 1):
                            raise ValueError("Expected one sequence parameter")
                        sequence = int(query["sequence"][0]) if "sequence" in query else None
                        if sequence is not None and sequence < 1:
                            raise ValueError("sequence must be positive")
                    except ValueError as error:
                        self.respond(400, {"error": str(error)})
                        return
                    with owner._condition:
                        latest = owner._export["sequence"]
                        selected = latest if sequence is None else sequence
                        frame = owner._frame_history.get(selected, {}).get(path)
                    if frame is not None:
                        self.respond(200, frame, "image/png")
                    elif selected <= latest and selected > 0:
                        self.respond(410, {"error": "Frame pair expired; request a fresh status sequence"})
                    else:
                        self.respond(503, {"error": "Frame pair has not been published"})
                else:
                    self.respond(404, {"error": "Unknown endpoint"})

            def do_POST(self):
                if self.path != "/api/control":
                    self.respond(404, {"error": "Unknown endpoint"})
                    return
                origin = self.headers.get("Origin")
                if self.headers.get("X-Inspection-Token") != owner.token or (
                    origin is not None and origin != owner.url.rstrip("/")
                ):
                    self.respond(403, {"error": "Invalid local control token or origin"})
                    return
                try:
                    size = int(self.headers.get("Content-Length", "-1"))
                    if size > 4096:
                        self.respond(413, {"error": "Control body exceeds 4 KiB"})
                        return
                    if size < 0 or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                        raise ValueError("Expected JSON with Content-Length")
                    if self.headers.get("Transfer-Encoding"):
                        raise ValueError("Chunked control bodies are not supported")
                    body = self.rfile.read(size)
                    command = _validate_command(json.loads(body), num_envs)
                    owner._commands.put_nowait(command)
                except (ValueError, UnicodeDecodeError, OverflowError, RecursionError) as error:
                    self.respond(400, {"error": str(error)})
                    return
                except queue.Full:
                    self.respond(429, {"error": "Control queue is full; wait for simulator"})
                    return
                self.respond(202, {"queued": True})

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_port}/"
        self._http_thread = threading.Thread(target=self._server.serve_forever, name="hiwonder-http", daemon=True)
        self._worker = threading.Thread(target=self._encode_frames, name="hiwonder-png", daemon=True)

    def __enter__(self):
        self._worker.start()
        self._http_thread.start()
        return self

    def __exit__(self, *_args):
        self._server.shutdown()
        self._server.server_close()
        self._http_thread.join()
        with self._condition:
            self._closing = True
            self._condition.notify()
        self._worker.join()

    def drain_commands(self) -> list[dict]:
        """Take validated browser commands for application by the simulator thread."""
        commands = []
        while True:
            try:
                commands.append(self._commands.get_nowait())
            except queue.Empty:
                return commands

    def publish_status(self, status: dict):
        """Publish an owned JSON-safe status snapshot without simulator references."""
        status = json.loads(json.dumps(status, allow_nan=False))
        with self._condition:
            self._status = status

    def submit_frames(self, raw, policy):
        """Submit full-resolution HWC uint8 RGB arrays; replace a pending export."""
        import numpy as np

        images = []
        for image in (raw, policy):
            image = np.asarray(image)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Expected HWC uint8 RGB frames")
            images.append(np.array(image, copy=True, order="C"))
        with self._condition:
            if self._closing:
                raise RuntimeError("Inspection export is closed")
            self._pending = (images, dict(self._status))
            self._condition.notify()

    def _encode_frames(self):
        from PIL import Image

        last = time.perf_counter()
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._closing)
                if self._pending is None:
                    status = dict(self._status)
                    export = dict(self._export)
                    break
                images, status = self._pending
                self._pending = None
            try:
                frames = {}
                for label, image in zip(("raw", "policy"), images):
                    buffer = io.BytesIO()
                    Image.fromarray(image).save(buffer, format="PNG", compress_level=1)
                    data = buffer.getvalue()
                    name = f"wrist_{label}.png"
                    temporary = self.output_directory / (name + ".tmp")
                    temporary.write_bytes(data)
                    temporary.replace(self.output_directory / name)
                    frames["/" + name] = data
                now = time.perf_counter()
                with self._condition:
                    self._export = {
                        "sequence": self._export["sequence"] + 1,
                        "wall_hz": 1 / (now - last),
                        "error": None,
                    }
                    self._frame_history[self._export["sequence"]] = frames
                    if len(self._frame_history) > 3:
                        del self._frame_history[next(iter(self._frame_history))]
                    export = dict(self._export)
                last = now
                self._write_status(status, export)
            except Exception as error:
                with self._condition:
                    self._export["error"] = str(error)
        self._write_status(status, export)

    def _write_status(self, status, export):
        temporary = self.output_directory / "live_status.json.tmp"
        temporary.write_text(json.dumps({**status, "export": export}, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.output_directory / "live_status.json")
