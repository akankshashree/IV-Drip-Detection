"""
monitor_iv.py - Robust IV Fluid Monitoring System & Real-Time Dashboard.
Key Features & Fixes:
  1. Resolution-independent ROI management (handles webcams, HD phone streams, and folder images).
  2. Dual-mode fluid estimation: MobileNetV2 deep learning classifier + computer vision meniscus detection.
  3. Continuous fluid percentage (0-100%) and 4-tier discrete status (full, normal, low, empty).
  4. Depletion rate (%/min, mL/hr) and Estimated Time-to-Empty (TTE) calculation.
  5. Live visual annotations: Bottle ROI bounding box and detected fluid boundary line.
  6. Resilient camera streaming with auto-reconnect for phone/IP cameras.
  7. Safe folder replay mode (avoids out-of-bounds crops on pre-cropped test frames).
  8. Interactive web dashboard on http://localhost:5000 with real-time SVG gauge, history chart, and CSV export.
"""
import os, csv, json, time, logging, threading, glob
from collections import deque, Counter
from datetime import datetime
import cv2, numpy as np, requests
import tensorflow as tf
from flask import Flask, Response, jsonify, send_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# This file lives in <project>\src, so resolve assets from the project root.
ROOT = (os.path.dirname(SCRIPT_DIR)
        if os.path.basename(SCRIPT_DIR).lower() == "src"
        else SCRIPT_DIR)
def project_path(value):
    """Resolve relative paths from the project root, not the current terminal folder."""
    return value if os.path.isabs(value) else os.path.join(ROOT, value)

# ==============================================================================
# SETTINGS & CONFIGURATION
# ==============================================================================
SOURCE = os.environ.get("SOURCE", "webcam")             # "webcam" or "folder"
CAMERA = int(os.environ.get("CAMERA", 0)) if os.environ.get("CAMERA", "0").isdigit() else os.environ.get("CAMERA", 0)
TEST_FOLDER = project_path(os.environ.get("TEST_FOLDER", "test_images"))  # Used in folder mode
LOOP_FOLDER = os.environ.get("LOOP_FOLDER", "0") == "1"      # Replay folder images continuously
INTERVAL = float(os.environ.get("INTERVAL", 2.0))            # Seconds between inference cycles
CONF_MIN = 0.55               # Confidence threshold below which reading is marked "uncertain"
MODEL_PATH = project_path(os.environ.get("MODEL_PATH", "models/mobilenetv2_iv.keras"))
CLASSES_PATH = project_path(os.environ.get("CLASSES_PATH", "models/mobilenetv2_labels.json"))
ROI_FILE = project_path(os.environ.get("ROI_FILE", "roi.json"))  # Optional bottle ROI config
CAMERA_TIMEOUT = 15           # Seconds without camera frames before raising alert
LOG_FILE = project_path(os.environ.get("LOG_FILE", "outputs/readings_log.csv"))
BOTTLE_CAPACITY_ML = 500      # Standard IV bag capacity in mL for flow calculation
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")
ALERT_STATES = ("low", "critical", "empty")

# Dataset labels -> canonical dashboard statuses and nominal percentage levels.
# Supports this project's sal_data_* labels and the original full/normal/low/empty labels.
LABEL_TO_STATUS = {
    "sal_data_100": "full", "100": "full", "100%": "full", "full": "full",
    "sal_data_80": "normal", "80": "normal", "80%": "normal", "normal": "normal",
    "sal_data_50": "low", "50": "low", "50%": "low", "low": "low",
    "sal_data_empty": "empty", "empty": "empty", "0": "empty", "0%": "empty",
}
LEVEL_WEIGHTS = {"empty": 0.0, "low": 50.0, "normal": 80.0, "full": 100.0}

def canonical_status(label):
    key = str(label).strip().lower()
    if key not in LABEL_TO_STATUS:
        raise ValueError(
            f"Unknown model class label {label!r}. Update LABEL_TO_STATUS in this script "
            "to match the labels JSON exactly."
        )
    return LABEL_TO_STATUS[key]

# Load the project's Keras model and its matching class-label JSON.
if not os.path.isfile(CLASSES_PATH):
    raise FileNotFoundError(f"Missing labels file: {CLASSES_PATH}")
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    classes = json.load(f)
if not isinstance(classes, list) or not classes:
    raise ValueError("The labels JSON must contain a non-empty list of class names.")
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Missing Keras model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
if len(classes) != int(model.output_shape[-1]):
    raise ValueError(
        f"Label/model mismatch: {len(classes)} labels but model outputs "
        f"{model.output_shape[-1]} values. Choose the matching labels JSON."
    )
if tuple(model.input_shape[1:]) != (224, 224, 3):
    raise ValueError(f"Expected model input (224, 224, 3), got {model.input_shape[1:]}")
# Validate label mapping before the camera loop starts.
for class_name in classes:
    canonical_status(class_name)
print(f"Loaded Keras model: {MODEL_PATH}")
print(f"Loaded classes: {classes}")

# Load ROI configuration if available
ROI_CONFIG = None
if os.path.exists(ROI_FILE):
    try:
        ROI_CONFIG = json.load(open(ROI_FILE))
        print(f"Loaded ROI configuration from {ROI_FILE}: {ROI_CONFIG}")
    except Exception as e:
        print(f"Warning: could not parse {ROI_FILE}: {e}")

def parse_roi(roi_cfg, frame_shape, source_mode):
    """
    Parses ROI config safely for any frame resolution.
    If source is folder and image is pre-cropped (<=300x300), skips ROI.
    """
    fh, fw = frame_shape[:2]
    if source_mode == "folder" and fw <= 300 and fh <= 300:
        return 0, 0, fw, fh

    if roi_cfg is None:
        # Default fallback: center 50% width, 80% height
        cw, ch = int(fw * 0.50), int(fh * 0.80)
        cx, cy = (fw - cw) // 2, (fh - ch) // 2
        return cx, cy, cw, ch

    if isinstance(roi_cfg, dict):
        if "norm" in roi_cfg:
            ymin, xmin, ymax, xmax = roi_cfg["norm"]
            return int(xmin * fw), int(ymin * fh), int((xmax - xmin) * fw), int((ymax - ymin) * fh)
        x = roi_cfg.get("x", 0)
        y = roi_cfg.get("y", 0)
        w = roi_cfg.get("w", fw)
        h = roi_cfg.get("h", fh)
        ref_w = roi_cfg.get("frame_w", 640)
        ref_h = roi_cfg.get("frame_h", 480)
    elif isinstance(roi_cfg, (list, tuple)) and len(roi_cfg) == 4:
        x, y, w, h = roi_cfg
        if max(x, y, w, h) <= 1.0:
            return int(x * fw), int(y * fh), int(w * fw), int(h * fh)
        ref_w = 640
        ref_h = 480
    else:
        return 0, 0, fw, fh

    scale_x = fw / float(ref_w)
    scale_y = fh / float(ref_h)
    rx = max(0, min(fw - 1, int(x * scale_x)))
    ry = max(0, min(fh - 1, int(y * scale_y)))
    rw = max(10, min(fw - rx, int(w * scale_x)))
    rh = max(10, min(fh - ry, int(h * scale_y)))
    return rx, ry, rw, rh

def detect_fluid_meniscus(crop_bgr):
    """
    Detects the horizontal liquid-air boundary (meniscus) within the bottle ROI
    using horizontal Sobel-Y gradient analysis.
    Returns (meniscus_y_rel, estimated_pct).
    """
    h, w = crop_bgr.shape[:2]
    if h < 30 or w < 30:
        return None, None

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel = np.abs(sobel_y)

    # Search central 60% width (avoid sidewalls)
    x1, x2 = int(w * 0.20), int(w * 0.80)
    # Search usable vertical region (10% to 90% height)
    y1, y2 = int(h * 0.10), int(h * 0.90)   
    usable_h = y2 - y1
    if usable_h < 10 or x2 <= x1:
        return None, None

    strip = abs_sobel[y1:y2, x1:x2].mean(axis=1)
    peak_y_rel = int(np.argmax(strip))
    meniscus_y = y1 + peak_y_rel

    # Inverted IV bottle: liquid empties downward from y1 to y2
    fill_pct = max(0.0, min(100.0, ((y2 - meniscus_y) / float(usable_h)) * 100.0))
    return meniscus_y, fill_pct

def predict_crop(crop_bgr):
    """Predict with the existing 224x224 RGB Keras model and its matching preprocessing."""
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("Bottle ROI is empty. Check roi.json or camera framing.")
    img = cv2.cvtColor(cv2.resize(crop_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 127.5 - 1.0
    probs = np.asarray(model.predict(img[None, ...], verbose=0)[0], dtype=np.float32)
    if not np.all(np.isfinite(probs)):
        raise ValueError("Model returned non-finite prediction scores.")
    pred_idx = int(np.argmax(probs))
    raw_label = classes[pred_idx]
    label = canonical_status(raw_label)
    conf = float(probs[pred_idx])
    expected_pct = sum(LEVEL_WEIGHTS[canonical_status(c)] * float(probs[i])
                       for i, c in enumerate(classes))
    return label, conf, float(expected_pct), probs

# ==============================================================================
# STATE & ALERT MANAGEMENT
# ==============================================================================
lock = threading.Lock()
state = {
    "status": "unknown",
    "continuous_level": 0.0,
    "confidence": 0.0,
    "uncertain": False,
    "time": "-",
    "tte": "--",
    "depletion_rate": 0.0,
    "source_status": "Online"
}
history = deque(maxlen=60)
alerts = deque(maxlen=20)
latest_frame = {"jpg": None}
time_series = deque(maxlen=30)  # (timestamp, continuous_pct) for flow rate & TTE

def send_alert(msg):
    now_str = datetime.now().strftime("%H:%M:%S")
    print(f"[{now_str}] ALERT: {msg}")
    with lock:
        alerts.appendleft({"time": now_str, "msg": msg})
    if BOT_TOKEN and CHAT_ID:
        try:
            requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                         params={"chat_id": CHAT_ID, "text": f"IV MONITOR: {msg}"}, timeout=8)
        except Exception as e:
            print("Telegram dispatch error:", e)

# ==============================================================================
# FLASK WEB DASHBOARD
# ==============================================================================
PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>IV Fluid Level Monitor</title>
<style>
:root{--bg:#f1f5f9;--surface:#ffffff;--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--ok:#059669;--warn:#d97706;--bad:#dc2626;--accent:#2563eb;--unsure:#94a3b8}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;--c:var(--unsure)}
body[data-s=full],body[data-s=normal]{--c:var(--ok)}
body[data-s=low]{--c:var(--warn)}
body[data-s=empty],body[data-s=critical]{--c:var(--bad)}
main{max-width:1100px;margin:0 auto;padding:24px 20px 48px;display:grid;gap:18px;grid-template-columns:1.2fr 1fr}
section{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,0.03)}
.wide{grid-column:1/-1}
header.wide{display:flex;justify-content:space-between;align-items:center;background:none;border:0;padding:0;box-shadow:none}
h1{font-size:22px;font-weight:700;margin:0;letter-spacing:-0.5px}
h2{font-size:14px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;color:var(--muted);margin:0 0 14px}
.badge{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:20px;font-size:13px;font-weight:500;background:#e2e8f0;color:var(--ink)}
.badge.live{background:#dcfce7;color:#166534}
.badge.offline{background:#fee2e2;color:#991b1b}
.cam{position:relative;aspect-ratio:4/3;background:#0f172a;border-radius:10px;overflow:hidden;display:grid;place-items:center;color:#94a3b8}
.cam img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
.level-card{display:flex;gap:24px;align-items:center}
.pct-display{font-size:48px;font-weight:800;line-height:1;color:var(--c);letter-spacing:-1px}
.status-name{font-size:18px;font-weight:600;color:var(--ink);margin:4px 0 2px}
.note{color:var(--muted);font-size:13px;margin:0 0 12px}
.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px}
.metric-box{background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.metric-val{font-size:16px;font-weight:700;color:var(--ink)}
.metric-lbl{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin-top:4px}
.bar i{display:block;height:100%;width:0;background:var(--c);transition:width .4s}
#fill{fill:var(--c);transition:y .6s,height .6s}
ul{list-style:none;margin:0;padding:0}
li{padding:10px 0;border-bottom:1px solid var(--line);font-size:14px}
li:last-child{border:0}
.mute{color:var(--muted)}
a{color:var(--accent);text-decoration:none;font-weight:500}
a:hover{text-decoration:underline}
@media(max-width:768px){main{grid-template-columns:1fr;padding:16px}}
</style></head><body data-s="unknown"><main>
<header class="wide">
  <div><h1>IV Fluid Monitor</h1><div class="mute" style="font-size:13px">Computer Vision & Deep Learning Infusion Safety</div></div>
  <span class="badge live" id="live-badge">Connecting...</span>
</header>

<section>
  <h2>Camera View & Bottle Tracking</h2>
  <div class="cam"><span id="cam-hint">Waiting for camera stream...</span><img id="cam" alt="Live Camera Preview"></div>
</section>

<section>
  <h2>Fluid Level Estimation</h2>
  <div class="level-card">
    <svg viewBox="0 0 150 220" width="120" role="img" aria-label="IV Bottle Gauge">
      <defs><clipPath id="cp"><rect x="25" y="40" width="70" height="165" rx="14"/></clipPath></defs>
      <rect id="fill" x="25" y="205" width="70" height="0" clip-path="url(#cp)"/>
      <rect x="25" y="40" width="70" height="165" rx="14" fill="none" stroke="#0f172a" stroke-width="3"/>
      <rect x="45" y="20" width="30" height="20" rx="3" fill="none" stroke="#0f172a" stroke-width="3"/>
      <rect x="40" y="8" width="40" height="12" rx="3" fill="#0f172a"/>
      <g stroke="#94a3b8" stroke-width="1.5">
        <line x1="95" x2="105" y1="45" y2="45"/><line x1="95" x2="105" y1="78" y2="78"/>
        <line x1="95" x2="105" y1="123" y2="123"/><line x1="95" x2="105" y1="200" y2="200"/>
      </g>
      <g font-size="10" font-weight="600" fill="#64748b">
        <text x="110" y="48">100%</text><text x="110" y="81">80%</text>
        <text x="110" y="126">50%</text><text x="110" y="203">0%</text>
      </g>
    </svg>
    <div style="flex:1">
      <div class="pct-display" id="pct-val">--%</div>
      <div class="status-name" id="status-name">Waiting for data...</div>
      <p class="note" id="note-text">Initializing sensor inputs</p>
      <div style="font-size:12px;color:var(--muted)">Model Confidence: <b id="conf-val">0%</b></div>
      <div class="bar"><i id="conf-bar"></i></div>
      <div class="metric-grid">
        <div class="metric-box"><div class="metric-lbl">Est. Time to Empty</div><div class="metric-val" id="tte-val">--</div></div>
        <div class="metric-box"><div class="metric-lbl">Depletion Rate</div><div class="metric-val" id="flow-val">--</div></div>
      </div>
    </div>
  </div>
</section>

<section class="wide">
  <h2>Continuous Fill Level Trend (Last 60 Readings)</h2>
  <svg id="chart" viewBox="0 0 700 140" width="100%" role="img" aria-label="Level Trend Chart"></svg>
</section>

<section class="wide">
  <h2>Safety Alerts & Log</h2>
  <ul id="alerts"></ul>
  <p style="margin-top:14px"><a href="/log.csv">Download Complete Audit Log (CSV)</a></p>
</section>
</main>

<script>
const $=id=>document.getElementById(id);
const NAMES={full:"Full (100%)",normal:"Normal (80%)",low:"Low (50%)",empty:"Empty (0%)",critical:"Critical (<10%)",unknown:"Initializing"};
const COL={full:"#059669",normal:"#059669",low:"#d97706",empty:"#dc2626",critical:"#dc2626"};

function renderChart(hist){
  const W=700,H=140,L=40,R=10,T=10,B=20,N=60,y=v=>T+(H-T-B)*(1-v/100);
  let svg="";
  [0,25,50,75,100].forEach(v=>{
    svg+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="#e2e8f0" stroke-dasharray="2"/>
          <text x="${L-8}" y="${y(v)+4}" text-anchor="end" font-size="10" fill="#94a3b8">${v}%</text>`;
  });
  const valid=hist.map((p,i)=>({x:L+(W-L-R)*i/(N-1),p})).filter(o=>o.p.level!==null&&o.p.level!==undefined);
  if(valid.length>1){
    const pts=valid.map(o=>`${o.x.toFixed(1)},${y(o.p.level).toFixed(1)}`).join(" ");
    svg+=`<polyline points="${pts}" fill="none" stroke="#0284c7" stroke-width="2.5"/>`;
    valid.forEach(o=>{
      const c=COL[o.p.status]||"#64748b";
      svg+=`<circle cx="${o.x.toFixed(1)}" cy="${y(o.p.level).toFixed(1)}" r="3" fill="${c}"/>`;
    });
  }
  $("chart").innerHTML=svg;
}

function renderState(d){
  document.body.dataset.s=d.uncertain?"unsure":d.status;
  const isOk = d.source_status === "Online";
  $("live-badge").className = "badge " + (isOk ? "live" : "offline");
  $("live-badge").textContent = isOk ? "Live • " + d.time : d.source_status;

  const lvl = Math.round(d.continuous_level);
  $("pct-val").textContent = lvl + "%";
  $("status-name").textContent = NAMES[d.status] || d.status;
  $("note-text").textContent = d.uncertain ? "Low confidence reading. Verifying..." : "Updated at " + d.time;

  // Bottle fill height (max 165px)
  const fillH = 165 * Math.max(0, Math.min(100, d.continuous_level)) / 100;
  $("fill").setAttribute("y", 205 - fillH);
  $("fill").setAttribute("height", fillH);

  const confPct = Math.round(d.confidence * 100);
  $("conf-val").textContent = confPct + "%";
  $("conf-bar").style.width = confPct + "%";

  $("tte-val").textContent = d.tte || "--";
  $("flow-val").textContent = d.depletion_rate ? d.depletion_rate.toFixed(1) + " %/min" : "Stable";

  renderChart(d.history || []);

  const ul = $("alerts");
  ul.replaceChildren();
  if(!d.alerts || !d.alerts.length){
    const li = document.createElement("li");
    li.className = "mute";
    li.textContent = "No alerts triggered. System operating normally.";
    ul.append(li);
  } else {
    d.alerts.forEach(a=>{
      const li = document.createElement("li");
      li.innerHTML = `<b>${a.time}</b> &nbsp; ${a.msg}`;
      ul.append(li);
    });
  }
}

async function tick(){
  try {
    const r = await fetch("/api/state");
    if(r.ok) renderState(await r.json());
  } catch(e){}
  const im = new Image();
  im.onload = () => {
    $("cam").src = im.src;
    $("cam").style.display = "block";
    $("cam-hint").style.display = "none";
  };
  im.src = "/frame.jpg?" + Date.now();
}

tick();
setInterval(tick, 1500);
</script></body></html>"""

app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)

@app.route("/")
def home():
    return PAGE

@app.route("/api/state")
def api_state():
    with lock:
        return jsonify(**state, history=list(history), alerts=list(alerts))

@app.route("/frame.jpg")
def frame_jpg():
    if latest_frame["jpg"] is None:
        return Response(status=404)
    return Response(latest_frame["jpg"], mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

@app.route("/log.csv")
def log_csv():
    if not os.path.exists(LOG_FILE):
        return Response("No readings logged yet", status=404)
    return send_file(os.path.abspath(LOG_FILE), mimetype="text/csv",
                     as_attachment=True, download_name="readings_log.csv")

threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, threaded=True), daemon=True).start()

# ==============================================================================
# FRAME CAPTURE WORKER & RESILIENT STREAMING
# ==============================================================================
newest_cam_data = {"frame": None, "t": time.time(), "reconnect_needed": False}

if SOURCE == "webcam":
    def camera_worker():
        global newest_cam_data
        cam = None
        consecutive_fails = 0
        while True:
            if cam is None or newest_cam_data["reconnect_needed"]:
                if cam is not None:
                    cam.release()
                cam = cv2.VideoCapture(CAMERA, cv2.CAP_DSHOW) if (os.name == "nt" and isinstance(CAMERA, int)) else cv2.VideoCapture(CAMERA)
                cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                newest_cam_data["reconnect_needed"] = False
                consecutive_fails = 0
                time.sleep(0.5)

            ok, f = cam.read()
            if ok and f is not None:
                newest_cam_data["frame"] = f
                newest_cam_data["t"] = time.time()
                consecutive_fails = 0
            else:
                consecutive_fails += 1
                if consecutive_fails > 10:
                    newest_cam_data["reconnect_needed"] = True
                time.sleep(0.2)
    threading.Thread(target=camera_worker, daemon=True).start()

    def get_next_frame():
        return newest_cam_data["frame"]
else:
    # Folder test mode
    folder_files = sorted(glob.glob(os.path.join(TEST_FOLDER, "*.*")))
    f_idx = {"i": 0}
    def get_next_frame():
        if not folder_files:
            return None
        if f_idx["i"] >= len(folder_files):
            if not LOOP_FOLDER:
                return None
            f_idx["i"] = 0
        f = cv2.imread(folder_files[f_idx["i"]])
        f_idx["i"] += 1
        return f

# ==============================================================================
# MAIN PROCESSING & MONITORING LOOP
# ==============================================================================
os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)
is_new_log = not os.path.exists(LOG_FILE)
log_handle = open(LOG_FILE, "a", newline="")
csv_writer = csv.writer(log_handle)
if is_new_log:
    csv_writer.writerow(["timestamp", "discrete_status", "continuous_level", "confidence", "tte_min", "depletion_rate", "alert_state"])

smoothed_level = None
recent_votes = deque(maxlen=3)
last_alert_sent = None
cam_offline_alerted = False

print("\n" + "=" * 70)
print("IV FLUID MONITOR RUNNING - DASHBOARD AT http://localhost:5000")
print(f"Source: {SOURCE} | Capacity: {BOTTLE_CAPACITY_ML} mL | Interval: {INTERVAL}s")
print("=" * 70)

MAX_FRAMES = int(os.environ.get("MAX_FRAMES", 0))
processed_count = 0

while True:
    frame = get_next_frame()

    # Camera offline check
    if SOURCE == "webcam":
        time_since_last_frame = time.time() - newest_cam_data["t"]
        if frame is None or time_since_last_frame > CAMERA_TIMEOUT:
            if not cam_offline_alerted:
                send_alert("CAMERA OFFLINE: Video input lost. Check patient infusion manually.")
                cam_offline_alerted = True
            with lock:
                state["source_status"] = "Camera Offline"
            print("Waiting for camera feed...")
            time.sleep(1.0)
            continue
        cam_offline_alerted = False
    elif frame is None:
        print("End of test folder reached. Dashboard remaining active (press Ctrl+C to stop).")
        with lock:
            state["source_status"] = "Folder End"
        while True:
            time.sleep(1.0)

    # 1. Parse ROI safely
    rx, ry, rw, rh = parse_roi(ROI_CONFIG, frame.shape, SOURCE)
    crop = frame[ry:ry+rh, rx:rx+rw]

    # 2. Dual-mode fluid estimation
    # A) MobileNetV2 Softmax expectation & class
    cls_label, conf, softmax_level, probs = predict_crop(crop)

    # B) Computer vision meniscus boundary detection
    meniscus_rel_y, meniscus_level = detect_fluid_meniscus(crop)

    # C) Combined continuous level estimation
    # If classifier confirms low or normal (where meniscus is visible in body), fuse with edge detection
    if meniscus_level is not None and cls_label in ("normal", "low") and abs(meniscus_level - softmax_level) < 25.0:
        raw_level = 0.65 * softmax_level + 0.35 * meniscus_level
    else:
        raw_level = softmax_level

    # D) Temporal exponential smoothing (alpha=0.35)
    if smoothed_level is None:
        smoothed_level = raw_level
    else:
        smoothed_level = 0.35 * raw_level + 0.65 * smoothed_level

    # E) Categorical classification update
    uncertain = (conf < CONF_MIN)
    if not uncertain:
        recent_votes.append(cls_label)
        if len(recent_votes) == recent_votes.maxlen:
            stable_status = Counter(recent_votes).most_common(1)[0][0]
        else:
            stable_status = cls_label
    else:
        stable_status = state["status"] if state["status"] != "unknown" else cls_label

    # Override status to critical if continuous level is under 10%
    if smoothed_level <= 10.0 and stable_status != "empty":
        current_status = "critical"
    else:
        current_status = stable_status

    # 3. Flow rate and Time-to-Empty (TTE) calculation
    now_t = time.time()
    time_series.append((now_t, smoothed_level))
    tte_str = "--"
    depletion_rate = 0.0

    if len(time_series) >= 5:
        dt_min = (time_series[-1][0] - time_series[0][0]) / 60.0
        d_lvl = time_series[0][1] - time_series[-1][1]  # positive if depleting
        if dt_min > 0.1 and d_lvl > 0.2:
            depletion_rate = d_lvl / dt_min  # % per minute
            if depletion_rate > 0.05:
                tte_minutes = smoothed_level / depletion_rate
                if tte_minutes < 60:
                    tte_str = f"{int(tte_minutes)} min"
                else:
                    hrs = int(tte_minutes // 60)
                    mins = int(tte_minutes % 60)
                    tte_str = f"{hrs}h {mins}m"

    # 4. Annotate camera display frame
    annotated = frame.copy()
    # Draw bottle ROI box
    cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (0, 220, 0), 2)
    cv2.putText(annotated, f"IV Bottle ROI", (rx, max(20, ry - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

    # Draw detected meniscus line if present
    if meniscus_rel_y is not None:
        abs_my = ry + meniscus_rel_y
        cv2.line(annotated, (rx, abs_my), (rx + rw, abs_my), (255, 200, 0), 2)
        cv2.putText(annotated, f"Meniscus: {meniscus_level:.0f}%", (rx + rw + 5, abs_my + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

    # Top overlay info banner
    now_str = datetime.now().strftime("%H:%M:%S")
    banner_txt = f"Level: {smoothed_level:.1f}% ({current_status.upper()}) | Conf: {conf:.0%} | TTE: {tte_str}"
    cv2.putText(annotated, banner_txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
    cv2.putText(annotated, banner_txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Encode JPEG for dashboard
    disp_h = int(annotated.shape[0] * 520 / annotated.shape[1])
    small_disp = cv2.resize(annotated, (520, disp_h))
    jpg_bytes = cv2.imencode(".jpg", small_disp, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    # 5. Update shared thread state
    with lock:
        latest_frame["jpg"] = jpg_bytes
        state.update({
            "status": current_status,
            "continuous_level": float(smoothed_level),
            "confidence": float(conf),
            "uncertain": bool(uncertain),
            "time": now_str,
            "tte": tte_str,
            "depletion_rate": float(depletion_rate),
            "source_status": "Online"
        })
        history.append({
            "level": round(float(smoothed_level), 1),
            "status": current_status,
            "conf": round(float(conf), 2)
        })

    # Log to CSV
    csv_writer.writerow([now_str, current_status, f"{smoothed_level:.1f}", f"{conf:.3f}", tte_str, f"{depletion_rate:.2f}", last_alert_sent or "none"])
    log_handle.flush()

    # Print terminal status line
    probs_str = " ".join(f"{c}:{probs[i]:.2f}" for i, c in enumerate(classes))
    print(f"[{now_str}] Level: {smoothed_level:5.1f}% | Status: {current_status:8s} | Conf: {conf:.2f} | TTE: {tte_str:>7s} | Probs: {probs_str}", flush=True)

    # 6. Safety Alert Logic
    if not uncertain:
        if current_status in ALERT_STATES and current_status != last_alert_sent:
            send_alert(f"IV Status {current_status.upper()} ({smoothed_level:.0f}% remaining). Check patient bedside immediately!")
            last_alert_sent = current_status
        elif current_status not in ALERT_STATES:
            last_alert_sent = None

    processed_count += 1
    if MAX_FRAMES > 0 and processed_count >= MAX_FRAMES:
        print(f"Test run completed: {processed_count} frames processed successfully.", flush=True)
        break

    time.sleep(INTERVAL)
