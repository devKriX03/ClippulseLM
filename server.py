import os, re, glob, subprocess, tempfile
import numpy as np
from flask import Flask, request, send_file, jsonify

app = Flask(__name__)
TOKEN = os.environ.get("TOKEN", "")

@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"] = "*"
    r.headers["Access-Control-Allow-Headers"] = "*"
    return r

@app.route("/")
def home():
    return "ClipPulse server is running"

@app.route("/clip", methods=["GET", "OPTIONS"])
def clip():
    if request.method == "OPTIONS":
        return "", 204
    if TOKEN and request.args.get("token") != TOKEN:
        return jsonify(error="Wrong token"), 401
    vid = request.args.get("v", "")
    if not re.fullmatch(r"[\w-]{11}", vid):
        return jsonify(error="Bad video id"), 400
    try:
        a = max(0, int(request.args["start"]))
        b = int(request.args["end"])
    except Exception:
        return jsonify(error="Bad times"), 400
    if not 0 < b - a <= 180:
        return jsonify(error="Clip must be 1 to 180 seconds"), 400

    d = tempfile.mkdtemp()
    cmd = ["yt-dlp", "--no-playlist", "-f", "bv*[height<=1080]+ba/b",
           "--download-sections", f"*{a}-{b}", "--force-keyframes-at-cuts",
           "--merge-output-format", "mp4", "-o", d + "/clip.%(ext)s"]
    if request.args.get("live") == "1":
        cmd.append("--live-from-start")
    cmd.append("https://www.youtube.com/watch?v=" + vid)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=280)
    except subprocess.TimeoutExpired:
        return jsonify(error="Took too long. Try a shorter clip."), 504
    files = glob.glob(d + "/clip*.mp4")
    if r.returncode != 0 or not files:
        msg = (r.stderr.strip().splitlines() or ["Could not cut this clip"])[-1]
        return jsonify(error=msg[:200]), 500
    return send_file(files[0], mimetype="video/mp4", as_attachment=True, download_name="clip.mp4")


def zs(x):
    m = np.median(x)
    return (x - m) / (np.median(np.abs(x - m)) * 1.4826 + 1e-6)

def pipe(cmd):
    return subprocess.run(cmd, capture_output=True, timeout=250).stdout

@app.route("/analyze")
def analyze():
    if TOKEN and request.args.get("token") != TOKEN:
        return jsonify(error="Wrong token"), 401
    vid = request.args.get("v", "")
    if not re.fullmatch(r"[\w-]{11}", vid):
        return jsonify(error="Bad video id"), 400
    try:
        a = max(0, int(request.args["start"]))
        dur = min(int(request.args["dur"]), 300)
    except Exception:
        return jsonify(error="Bad times"), 400
    d = tempfile.mkdtemp()
    cmd = ["yt-dlp", "--no-playlist", "-f", "bv*[height<=360]+ba/b[height<=360]/b",
           "--download-sections", f"*{a}-{a + dur}", "--merge-output-format", "mp4",
           "-o", d + "/s.%(ext)s"]
    if request.args.get("live") == "1":
        cmd.append("--live-from-start")
    cmd.append("https://www.youtube.com/watch?v=" + vid)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=250)
        files = glob.glob(d + "/s*.mp4")
        if r.returncode != 0 or not files:
            msg = (r.stderr.strip().splitlines() or ["Could not read this stream"])[-1]
            return jsonify(error=msg[:200]), 500
        f = files[0]
        au = np.frombuffer(pipe(["ffmpeg", "-v", "error", "-i", f, "-vn", "-ac", "1",
                                 "-ar", "8000", "-f", "s16le", "-"]), dtype=np.int16).astype(np.float32) / 32768
        n = len(au) // 8000
        if n < 20:
            return jsonify(error="Not enough footage to scan"), 400
        loud = np.zeros(n); voice = np.zeros(n)
        fr = np.fft.rfftfreq(8000, 1 / 8000)
        band = (fr >= 300) & (fr <= 3400)
        for i in range(n):
            w = au[i * 8000:(i + 1) * 8000]
            loud[i] = 20 * np.log10(np.sqrt(np.mean(w ** 2)) + 1e-6)
            p = np.abs(np.fft.rfft(w)) ** 2
            voice[i] = p[band].sum() / (p.sum() + 1e-9)
        raw = np.frombuffer(pipe(["ffmpeg", "-v", "error", "-i", f, "-an", "-vf",
                                  "fps=2,scale=160:90,format=gray", "-f", "rawvideo", "-"]), dtype=np.uint8)
        k = 160 * 90
        nf = len(raw) // k
        motion = np.zeros(n)
        if nf > 2:
            fm = raw[:nf * k].reshape(nf, k).astype(np.float32)
            df = np.abs(np.diff(fm, axis=0)).mean(axis=1)
            for i in range(min(n, len(df) // 2)):
                motion[i] = df[i * 2:i * 2 + 2].mean()
        score = 0.6 * (zs(loud) + 0.5 * zs(voice)) + 0.4 * zs(motion)
        score = np.convolve(score, np.ones(3) / 3, "same")
        picks = []
        for i in np.argsort(-score):
            if score[i] < 1.5 or len(picks) >= 6:
                break
            if all(abs(int(i) - p) >= 30 for p in picks):
                picks.append(int(i))
        return jsonify(moments=[dict(t=a + p, score=round(float(score[p]), 2),
                                     loud=round(float(loud[p]), 1), motion=round(float(motion[p]), 1))
                                for p in sorted(picks)])
    except subprocess.TimeoutExpired:
        return jsonify(error="Took too long. Scan fewer minutes."), 504
