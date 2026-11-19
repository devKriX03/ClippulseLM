import os, re, glob, time, shutil, threading, queue, subprocess, tempfile
from datetime import datetime
import numpy as np
import requests
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)
TOKEN = os.environ.get("TOKEN", "")
YT_KEY = os.environ.get("YT_API_KEY", "")
DATA = os.environ.get("DATA_DIR", "/tmp/clippulse")
os.makedirs(DATA, exist_ok=True)
BIN = 5000
SCAN_EVERY = 180
MAX_VOD = 10800
VOD_BEFORE, VOD_AFTER = 12, 15
COLORS = ["#F97316", "#38BDF8", "#A78BFA", "#22C55E", "#F43F5E", "#FACC15"]
LOCK = threading.RLock()
streams, vods, clips, keys = {}, {}, {}, {}
jobs = queue.Queue()
HYPE = re.compile(r"\b(?:lol|lmao|omg|pog|poggers|wow|clip|insane|wtf|gg|rip|crazy|haha+)\b|let'?s go|no way|[\U0001F602\U0001F923\U0001F62E\U0001F525]", re.I)


@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"] = "*"
    r.headers["Access-Control-Allow-Headers"] = "*"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return r


@app.before_request
def gate():
    if request.method == "OPTIONS":
        return "", 204
    if request.path in ("/", "/health"):
        return None
    body = request.get_json(silent=True) or {}
    if TOKEN and request.args.get("token") != TOKEN and body.get("token") != TOKEN:
        return jsonify(error="Wrong password"), 401
    return None


def bad(msg, code=400):
    return jsonify(error=msg), code


def vid_from(text):
    text = (text or "").strip()
    m = re.search(r"(?:v=|youtu\.be/|/live/|/shorts/|/embed/)([\w-]{11})", text) or re.fullmatch(r"([\w-]{11})", text)
    return m.group(1) if m else ""


def yt(path, params):
    if not YT_KEY:
        raise RuntimeError("Server is missing YT_API_KEY. Add it in Render Environment Variables.")
    r = requests.get("https://www.googleapis.com/youtube/v3/" + path, params={**params, "key": YT_KEY}, timeout=25)
    j = r.json() if r.content else {}
    if r.status_code >= 400:
        raise RuntimeError(j.get("error", {}).get("message", "YouTube request failed (%d)" % r.status_code))
    return j


def dur_s(iso):
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    if not m:
        return 0
    h, mi, s = (int(x or 0) for x in m.groups())
    return h * 3600 + mi * 60 + s


def fetch_part(vid, a, b, live, height):
    """Download only seconds a..b. Returns a file path; caller must cleanup(path)."""
    d = tempfile.mkdtemp(dir=DATA)
    cmd = ["yt-dlp", "--no-playlist", "-f", f"bv*[height<={height}]+ba/b[height<={height}]/b",
           "--download-sections", f"*{int(a)}-{int(b)}", "--force-keyframes-at-cuts",
           "--merge-output-format", "mp4", "-o", d + "/s.%(ext)s"]
    if live:
        cmd.append("--live-from-start")
    cmd.append("https://www.youtube.com/watch?v=" + vid)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=280)
        files = glob.glob(d + "/s*.mp4")
        if r.returncode != 0 or not files:
            lines = (r.stderr or "").strip().splitlines()
            raise RuntimeError((lines[-1] if lines else "Could not download this part")[:200])
        return files[0]
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise


def cleanup(path):
    if path:
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)


def zs(x):
    m = np.median(x)
    return (x - m) / (np.median(np.abs(x - m)) * 1.4826 + 1e-6)


def analyze(path):
    au = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
                        capture_output=True, timeout=120).stdout
    au = np.frombuffer(au, dtype=np.int16).astype(np.float32) / 32768
    n = len(au) // 8000
    if n < 20:
        raise RuntimeError("Not enough footage to scan")
    fr = np.fft.rfftfreq(8000, 1 / 8000)
    band = (fr >= 300) & (fr <= 3400)
    loud = np.zeros(n)
    voice = np.zeros(n)
    for i in range(n):
        w = au[i * 8000:(i + 1) * 8000]
        loud[i] = 20 * np.log10(np.sqrt(np.mean(w ** 2)) + 1e-6)
        p = np.abs(np.fft.rfft(w)) ** 2
        voice[i] = p[band].sum() / (p.sum() + 1e-9)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-an", "-vf", "fps=2,scale=160:90,format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, timeout=120).stdout
    raw = np.frombuffer(raw, dtype=np.uint8)
    k = 160 * 90
    nf = len(raw) // k
    motion = np.zeros(n)
    if nf > 2:
        fm = raw[:nf * k].reshape(nf, k).astype(np.float32)
        df = np.abs(np.diff(fm, axis=0)).mean(axis=1)
        for i in range(min(n, len(df) // 2)):
            motion[i] = df[i * 2:i * 2 + 2].mean()
    score = 0.6 * (zs(loud) + 0.5 * zs(voice)) + 0.4 * zs(motion)
    return np.convolve(score, np.ones(3) / 3, "same")


def pick(score, limit=6, thr=1.5):
    out = []
    for i in np.argsort(-score):
        if score[i] < thr or len(out) >= limit:
            break
        if all(abs(int(i) - p) >= 30 for p, _ in out):
            out.append((int(i), float(score[i])))
    return sorted(out)


def add_clip(vid, start, end, src, score=0.0):
    start, end = max(0, int(start)), int(end)
    if end - start < 3:
        return None
    key = (vid, start // 10, end // 10)
    with LOCK:
        cid = keys.get(key)
        if cid:
            c = clips[cid]
            if c["status"] != "failed":
                return cid
            c.update(status="queued", error="")
        else:
            st = streams.get(vid) or vods.get(vid) or {}
            cid = "%d%04d" % (int(time.time() * 1000), len(clips))
            clips[cid] = {"id": cid, "vid": vid, "creator": st.get("creator", "Video"), "title": st.get("title", ""),
                          "start": start, "end": end, "src": src, "score": round(float(score), 2),
                          "status": "queued", "error": "", "file": "", "created": time.time(), "live": vid in streams}
            keys[key] = cid
    jobs.put({"type": "cut", "id": cid})
    return cid


def do_cut(cid):
    with LOCK:
        c = clips[cid]
        c["status"] = "making"
    p = None
    try:
        p = fetch_part(c["vid"], c["start"], c["end"], c["live"], 720)
        dst = os.path.join(DATA, f"clip_{cid}.mp4")
        shutil.move(p, dst)
        with LOCK:
            c.update(file=dst, status="ready")
    except Exception as e:
        with LOCK:
            c.update(status="failed", error=str(e)[:200])
    finally:
        cleanup(p)


def do_scan(vid):
    s = streams.get(vid)
    if not s or not s["run"]:
        return
    el = time.time() - s["t0"]
    a, b = max(0, int(el - 150)), int(el - 20)
    if b - a < 60:
        with LOCK:
            s["note"] = "Not enough live footage yet for a scan"
        return
    p = None
    try:
        p = fetch_part(vid, a, b, True, 360)
        found = pick(analyze(p))
        for i, sc in found:
            t = a + i
            add_clip(vid, t - 12, t + s["after"], "ai", sc)
        with LOCK:
            s["note"] = f"Scan done: {len(found)} moment(s) found"
    except Exception as e:
        with LOCK:
            s["note"] = "Scan failed: " + str(e)[:150]
    finally:
        cleanup(p)


def do_vod(vid):
    v = vods[vid]
    total, step = v["duration"], 300
    with LOCK:
        v["status"] = "scanning"
    for a in range(0, total, step):
        if v["stop"]:
            break
        b = min(total, a + step)
        p = None
        try:
            p = fetch_part(vid, a, b, False, 360)
            for i, sc in pick(analyze(p)):
                add_clip(vid, a + i - VOD_BEFORE, a + i + VOD_AFTER, "ai", sc)
        except Exception as e:
            with LOCK:
                v["err"] = str(e)[:200]
        finally:
            cleanup(p)
        with LOCK:
            v["done"] = b
    with LOCK:
        v["status"] = "stopped" if v["stop"] else "finished"


def worker():
    while True:
        j = jobs.get()
        try:
            if j["type"] == "cut":
                do_cut(j["id"])
            elif j["type"] == "scan":
                do_scan(j["vid"])
            elif j["type"] == "vod":
                do_vod(j["vid"])
        except Exception as e:
            print("job error", e, flush=True)
        finally:
            jobs.task_done()


threading.Thread(target=worker, daemon=True).start()


def base_at(s, b):
    if s["minB"] is None:
        return None
    tot, c = 0, 0
    for i in range(b - 14, b - 2):
        if i < s["minB"]:
            continue
        tot += s["bins"].get(i, 0)
        c += 1
    if c < 5:
        return None
    return max(tot / c / (BIN / 1000), 0.15)


def detect(s):
    if s["minB"] is None:
        return
    for b in range(max(s["ev"] + 1, s["minB"] + 2), s["maxB"] - 1):
        s["ev"] = b
        base = base_at(s, b)
        if base is None:
            continue
        hy = s["hy"].get(b, 0) + s["hy"].get(b - 1, 0)
        cur = (s["bins"].get(b, 0) + s["bins"].get(b - 1, 0) + 0.5 * hy) / (2 * BIN / 1000)
        score = cur / base
        if score < s["sens"] or cur < 0.5:
            continue
        peak = b * BIN / 1000
        last = s["spikes"][-1] if s["spikes"] else None
        if last and not last["done"] and peak - last["peak"] < 30:
            if score > last["score"]:
                last.update(peak=peak, score=score)
        else:
            s["spikes"].append({"peak": peak, "score": score, "done": False})
    for sp in s["spikes"]:
        if not sp["done"] and sp["peak"] < (s["maxB"] - 3) * BIN / 1000:
            sp["done"] = True
            off = sp["peak"] - s["t0"]
            add_clip(s["vid"], off - s["before"], off + s["after"], "chat", sp["score"])


def ingest(s, items):
    for it in items:
        if it["id"] in s["seen"]:
            continue
        s["seen"].add(it["id"])
        sn = it.get("snippet", {})
        if sn.get("type") not in (None, "textMessageEvent", "superChatEvent", "superStickerEvent"):
            continue
        ts = sn.get("publishedAt")
        if not ts:
            continue
        b = int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000) // BIN
        s["bins"][b] = s["bins"].get(b, 0) + 1
        if HYPE.search(sn.get("displayMessage", "") or ""):
            s["hy"][b] = s["hy"].get(b, 0) + 1
        s["minB"] = b if s["minB"] is None else min(s["minB"], b)
        s["maxB"] = max(s["maxB"], b)
    detect(s)


def chat_loop(vid):
    s = streams[vid]
    while s["run"]:
        if s["pause"]:
            time.sleep(2)
            continue
        wait = 30
        try:
            p = {"liveChatId": s["chat"], "part": "snippet", "maxResults": 200}
            if s["page"]:
                p["pageToken"] = s["page"]
            j = yt("liveChat/messages", p)
            with LOCK:
                s["polls"] += 1
                s["page"] = j.get("nextPageToken", s["page"])
                ingest(s, j.get("items", []))
                s["status"], s["err"] = "live", ""
            wait = max(j.get("pollingIntervalMillis", 30000) / 1000, 30)
        except Exception as e:
            m = str(e)
            if re.search(r"quota|forbidden|ended|disabled|not found|not live|no longer", m, re.I):
                with LOCK:
                    s.update(run=False, status="stopped", err=m[:200])
                break
            with LOCK:
                s.update(status="retrying", err=m[:200])
            wait = 60
        now = time.time()
        if s["run"] and not s["pause"] and now - s["last_scan"] >= SCAN_EVERY and jobs.qsize() < 3:
            s["last_scan"] = now
            jobs.put({"type": "scan", "vid": vid})
        time.sleep(wait)


def get_stream(body):
    vid = vid_from(body.get("url") or body.get("vid") or "")
    return vid, streams.get(vid)


@app.route("/")
def home():
    return "ClipPulse server is running"


@app.route("/health")
def health():
    return "ok"


@app.route("/watch", methods=["POST"])
def watch():
    body = request.get_json(silent=True) or {}
    vid = vid_from(body.get("url", ""))
    if not vid:
        return bad("That is not a YouTube link.")
    with LOCK:
        if vid in streams and streams[vid]["run"]:
            return jsonify(ok=True, note="Already watching this creator")
        color = COLORS[len(streams) % len(COLORS)]
    try:
        j = yt("videos", {"part": "liveStreamingDetails,snippet", "id": vid})
        v = (j.get("items") or [None])[0]
        if not v:
            raise RuntimeError("Video not found. Check the link.")
        d = v.get("liveStreamingDetails", {})
        if d.get("actualEndTime"):
            raise RuntimeError("This stream has ended. Only live streams can be watched.")
        if not d.get("activeLiveChatId") or not d.get("actualStartTime"):
            raise RuntimeError("Not live right now, or its chat is turned off.")
        t0 = datetime.fromisoformat(d["actualStartTime"].replace("Z", "+00:00")).timestamp()
    except Exception as e:
        return bad(str(e))
    s = {"vid": vid, "creator": v["snippet"].get("channelTitle", "Creator"), "title": v["snippet"].get("title", ""),
         "chat": d["activeLiveChatId"], "t0": t0, "page": "", "bins": {}, "hy": {}, "minB": None, "maxB": 0,
         "ev": -1, "seen": set(), "spikes": [], "polls": 1, "run": True, "pause": False, "status": "live",
         "err": "", "note": "", "last_scan": 0, "color": color,
         "before": int(body.get("before", 25)), "after": int(body.get("after", 10)),
         "sens": float(body.get("sens", 2.2)), "live": True}
    with LOCK:
        streams[vid] = s
    threading.Thread(target=chat_loop, args=(vid,), daemon=True).start()
    return jsonify(ok=True)


@app.route("/pause", methods=["POST"])
def pause():
    vid, s = get_stream(request.get_json(silent=True) or {})
    if not s:
        return bad("Not watching this creator.", 404)
    with LOCK:
        s["pause"] = not s["pause"]
        s["status"] = "paused" if s["pause"] else "live"
    return jsonify(paused=s["pause"])


@app.route("/stop", methods=["POST"])
def stop():
    vid, s = get_stream(request.get_json(silent=True) or {})
    if not s:
        return bad("Not watching this creator.", 404)
    with LOCK:
        s.update(run=False, status="stopped")
    return jsonify(ok=True)


@app.route("/scan", methods=["POST"])
def scan():
    vid, s = get_stream(request.get_json(silent=True) or {})
    if not s or not s["run"]:
        return bad("Start watching this creator first.")
    jobs.put({"type": "scan", "vid": vid})
    with LOCK:
        s["note"] = "Scan queued"
    return jsonify(ok=True)


@app.route("/cut", methods=["POST"])
def cut():
    body = request.get_json(silent=True) or {}
    vid = vid_from(body.get("url") or body.get("vid") or "")
    if not vid:
        return bad("That is not a YouTube link.")
    try:
        a, b = int(float(body.get("start", 0))), int(float(body.get("end", 0)))
    except Exception:
        return bad("Start and end must be numbers of seconds.")
    if not 3 <= b - a <= 180:
        return bad("A clip must be 3 to 180 seconds long.")
    return jsonify(ok=True, id=add_clip(vid, a, b, "manual"))


@app.route("/vod", methods=["POST"])
def vod():
    body = request.get_json(silent=True) or {}
    vid = vid_from(body.get("url", ""))
    if not vid:
        return bad("That is not a YouTube link.")
    try:
        j = yt("videos", {"part": "contentDetails,snippet", "id": vid})
        v = (j.get("items") or [None])[0]
        if not v:
            raise RuntimeError("Video not found. Check the link.")
        total = min(dur_s(v["contentDetails"].get("duration")), MAX_VOD)
        if total < 30:
            raise RuntimeError("This video is too short to scan.")
    except Exception as e:
        return bad(str(e))
    with LOCK:
        if vid in vods and vods[vid]["status"] in ("queued", "scanning"):
            return jsonify(ok=True, note="Already scanning this video")
        vods[vid] = {"vid": vid, "creator": v["snippet"].get("channelTitle", "Video"),
                     "title": v["snippet"].get("title", ""), "duration": total, "done": 0,
                     "status": "queued", "err": "", "stop": False}
    jobs.put({"type": "vod", "vid": vid})
    return jsonify(ok=True, duration=total)


@app.route("/status")
def status():
    with LOCK:
        out = []
        for vid, s in streams.items():
            series, line, base = [], None, None
            if s["maxB"]:
                series = [s["bins"].get(b, 0) for b in range(s["maxB"] - 59, s["maxB"] + 1)]
                base = base_at(s, s["maxB"] - 1)
                if base:
                    line = base * s["sens"] * (BIN / 1000)
            out.append({"vid": vid, "creator": s["creator"], "title": s["title"], "status": s["status"],
                        "err": s["err"], "note": s["note"], "paused": s["pause"], "polls": s["polls"],
                        "series": series, "line": line, "base": round(base, 1) if base else None,
                        "color": s["color"]})
        vs = [{k: v[k] for k in ("vid", "creator", "title", "duration", "done", "status", "err")} for v in vods.values()]
        quota = sum(x["polls"] for x in streams.values()) * 5
        return jsonify(streams=out, vods=vs, queue=jobs.qsize(), quota=quota)


@app.route("/clips")
def list_clips():
    with LOCK:
        items = [dict(c) for c in clips.values()]
    items.sort(key=lambda c: c["created"], reverse=True)
    keep = ("id", "vid", "creator", "title", "start", "end", "src", "score", "status", "error", "created")
    return jsonify(clips=[{k: c[k] for k in keep} for c in items])


@app.route("/file/<cid>")
def get_file(cid):
    with LOCK:
        c = dict(clips.get(cid, {}))
    if not c or c.get("status") != "ready" or not os.path.exists(c.get("file", "")):
        return bad("This clip is not ready yet.", 404)
    safe = re.sub(r"[^\w-]+", "_", c["creator"])[:40] or "clip"
    return send_file(c["file"], mimetype="video/mp4", as_attachment=True,
                     download_name=f"{safe}_{c['start']}s.mp4")
