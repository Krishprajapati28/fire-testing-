"""
Fire & Smoke Detection — RTSP Stream
=====================================
Model  : best.onnx  (YOLOv8/v9 DFL, input 640x640, classes: fire / smoke)
Stream : rtsp://admin12345:admin12345@192.168.1.102:554/stream1

Run:
    python fire_smoke_detection.py

Optional flags:
    --model   best.onnx          override model path
    --rtsp    rtsp://...         override stream URL
    --conf    0.35               confidence threshold
    --iou     0.45               NMS IoU threshold
    --save                       record annotated video to disk
    --no-show                    headless / server mode (no window)
"""

import argparse
import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ─────────────────────────── LOGGING ────────────────────────────────────── #

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("FireSmoke")

# ─────────────────────────── CONFIGURATION ──────────────────────────────── #

DEFAULT_MODEL  = "best.onnx"
DEFAULT_RTSP   = "rtsp://admin12345:admin12345@192.168.1.102:554/stream1"

CLASS_NAMES    = ["fire", "smoke"]
INPUT_SIZE     = 640
CONF_THRESH    = 0.35
IOU_THRESH     = 0.45
REG_MAX        = 16           # DFL bins per coordinate (64 / 4)

# BGR colours for bounding boxes
CLASS_COLORS = {
    "fire":  (0,   50,  255),   # vivid red-orange
    "smoke": (180, 180, 180),   # grey
}

# RTSP reconnection settings
RECONNECT_DELAY = 5    # seconds between retries
MAX_RETRIES     = 0    # 0 = retry forever

# ──────────────────────────── MODEL LOADING ─────────────────────────────── #

def load_session(model_path: str):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess      = ort.InferenceSession(model_path, providers=providers)
    inp_name  = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    log.info("Model loaded : %s", model_path)
    log.info("Provider     : %s", sess.get_providers()[0])
    log.info("Input        : %s  shape=%s", inp_name, sess.get_inputs()[0].shape)
    log.info("Classes      : %s", CLASS_NAMES)
    return sess, inp_name, out_names

# ─────────────────────────── PREPROCESSING ──────────────────────────────── #

def preprocess(frame: np.ndarray, size: int = INPUT_SIZE):
    """
    Letterbox-resize frame to (size x size), normalize to [0,1].
    Returns:
        blob       : float32 NCHW array  [1, 3, size, size]
        orig_shape : (H, W) of the original frame
    """
    h, w    = frame.shape[:2]
    scale   = min(size / h, size / w)
    nh, nw  = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas  = np.full((size, size, 3), 114, dtype=np.uint8)
    top     = (size - nh) // 2
    left    = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized

    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0   # BGR->RGB, /255
    blob = blob.transpose(2, 0, 1)[np.newaxis]              # HWC->NCHW
    return blob, (h, w)

# ────────────────────────── DFL POSTPROCESSING ──────────────────────────── #

def _make_anchors(strides, size):
    anchors, stride_out = [], []
    for stride in strides:
        g  = size // stride
        xs = np.arange(g, dtype=np.float32) + 0.5
        ys = np.arange(g, dtype=np.float32) + 0.5
        sy, sx = np.meshgrid(ys, xs, indexing="ij")
        anchors.append(np.stack([sx, sy], axis=-1).reshape(-1, 2))
        stride_out.append(np.full((g * g, 1), stride, dtype=np.float32))
    return np.concatenate(anchors), np.concatenate(stride_out)


def _softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _dfl_to_boxes(box_raw, anchors, strides):
    box_raw = box_raw.reshape(-1, 4, REG_MAX)
    box_raw = _softmax(box_raw)
    proj    = np.arange(REG_MAX, dtype=np.float32)
    ltrb    = (box_raw * proj).sum(axis=-1)
    x1 = (anchors[:, 0] - ltrb[:, 0]) * strides[:, 0]
    y1 = (anchors[:, 1] - ltrb[:, 1]) * strides[:, 0]
    x2 = (anchors[:, 0] + ltrb[:, 2]) * strides[:, 0]
    y2 = (anchors[:, 1] + ltrb[:, 3]) * strides[:, 0]
    return np.stack([x1, y1, x2, y2], axis=1)


def postprocess(raw_outputs, orig_shape,
                conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH):
    """
    Convert raw model outputs to list of detection dicts.
    Each dict: { box:(x1,y1,x2,y2), conf:float, class_id:int, label:str }
    """
    strides = [8, 16, 32]
    anchors, stride_tensors = _make_anchors(strides, INPUT_SIZE)

    # Flatten 3 heads -> (8400, 66)
    preds = []
    for out in raw_outputs:
        out = out[0]
        c, h, w = out.shape
        preds.append(out.reshape(c, h * w).T)
    pred = np.concatenate(preds, axis=0)

    # Sigmoid on class logits, confidence filter
    cls_prob  = 1 / (1 + np.exp(-pred[:, 64:]))
    class_ids = cls_prob.argmax(axis=1)
    confs     = cls_prob[np.arange(len(cls_prob)), class_ids]
    mask      = confs >= conf_thresh
    if not mask.any():
        return []

    boxes = _dfl_to_boxes(pred[mask, :64], anchors[mask], stride_tensors[mask])

    # Scale to original image
    oh, ow  = orig_shape
    sx, sy  = ow / INPUT_SIZE, oh / INPUT_SIZE
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] * sx).clip(0, ow)
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] * sy).clip(0, oh)

    confs_f = confs[mask]
    class_f = class_ids[mask]

    # Per-class NMS
    results = []
    for cid in np.unique(class_f):
        idx  = class_f == cid
        keep = cv2.dnn.NMSBoxes(
            boxes[idx].tolist(), confs_f[idx].tolist(), conf_thresh, iou_thresh
        )
        if len(keep) == 0:
            continue
        b = boxes[idx]
        s = confs_f[idx]
        for k in keep.flatten():
            x1, y1, x2, y2 = b[k].astype(int)
            label = CLASS_NAMES[int(cid)] if int(cid) < len(CLASS_NAMES) else str(cid)
            results.append({
                "box":      (x1, y1, x2, y2),
                "conf":     float(s[k]),
                "class_id": int(cid),
                "label":    label,
            })
    return results

# ─────────────────────────── DRAWING ────────────────────────────────────── #

def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label  = det["label"]
        conf   = det["conf"]
        color  = CLASS_COLORS.get(label, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label.upper()}  {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def draw_overlay(frame, fps, connected, rtsp_url, cam_id=0):
    """Draw FPS, live status, stream URL, and timestamp on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 105), (20, 20, 20), -1)
    cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # LIVE / RECONNECTING
    dot_color = (0, 220, 0) if connected else (0, 0, 220)
    status    = "LIVE" if connected else "RECONNECTING..."
    cv2.putText(frame, f"  {status}", (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, dot_color, 2)

    # Stream URL (hide credentials)
    safe_url = rtsp_url.split("@")[-1] if "@" in rtsp_url else rtsp_url
    cv2.putText(frame, f"CAM {cam_id}  {safe_url}", (10, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Timestamp bottom-right
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, ts, (w - tw - 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return frame

# ─────────────────────────── RTSP READER ────────────────────────────────── #

class RTSPReader:
    """
    Background-threaded RTSP reader with auto-reconnect.
    Always returns the latest frame — zero accumulation lag.
    """

    def __init__(self, url: str, cam_id: int = 0):
        self.url       = url
        self.cam_id    = cam_id
        self.connected = False
        self._q        = queue.Queue(maxsize=1)
        self._stop     = threading.Event()
        self._thread   = threading.Thread(
            target=self._loop, daemon=True, name=f"RTSP-{cam_id}"
        )

    def start(self):
        self._thread.start()
        log.info("[Cam %d] Reader thread started.", self.cam_id)
        return self

    def read(self):
        """Returns (True, frame) or (False, None) if no new frame yet."""
        try:
            return True, self._q.get_nowait()
        except queue.Empty:
            return False, None

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=6)
        log.info("[Cam %d] Reader stopped.", self.cam_id)

    def _open(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self):
        retries = 0
        while not self._stop.is_set():
            cap = self._open()

            if not cap.isOpened():
                self.connected = False
                retries += 1
                log.warning("[Cam %d] Connection failed (attempt %d/%s). Retry in %ds...",
                            self.cam_id, retries,
                            MAX_RETRIES if MAX_RETRIES else "inf",
                            RECONNECT_DELAY)
                if MAX_RETRIES and retries > MAX_RETRIES:
                    log.error("[Cam %d] Max retries reached. Stopping.", self.cam_id)
                    break
                time.sleep(RECONNECT_DELAY)
                continue

            log.info("[Cam %d] Connected to stream successfully.", self.cam_id)
            self.connected = True
            retries        = 0

            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    log.warning("[Cam %d] Stream lost. Reconnecting...", self.cam_id)
                    self.connected = False
                    break
                # Keep queue at size 1 — always the freshest frame
                if self._q.full():
                    try:
                        self._q.get_nowait()
                    except queue.Empty:
                        pass
                self._q.put(frame)

            cap.release()
            if not self._stop.is_set():
                time.sleep(RECONNECT_DELAY)

# ─────────────────────── MAIN DETECTION LOOP ────────────────────────────── #

def run_rtsp(sess, inp_name, out_names,
             rtsp_url = DEFAULT_RTSP,
             conf     = CONF_THRESH,
             iou      = IOU_THRESH,
             show     = True,
             save     = False,
             cam_id   = 0):
    """
    Full detection loop on a single RTSP stream.

    Keyboard shortcuts (when show=True):
        Q  — quit
        S  — save a snapshot of the current frame
    """

    reader = RTSPReader(rtsp_url, cam_id=cam_id).start()

    # Wait for first frame (up to 20 s)
    log.info("[Cam %d] Waiting for first frame...", cam_id)
    deadline, last_frame = time.time() + 20, None
    while time.time() < deadline:
        ok, f = reader.read()
        if ok:
            last_frame = f
            break
        time.sleep(0.05)

    if last_frame is None:
        reader.stop()
        raise RuntimeError(
            f"\n[Cam {cam_id}] No frame received within 20 s.\n"
            f"Stream URL : {rtsp_url}\n"
            "Check      : camera power, IP/port reachable, credentials correct, port 554 open."
        )

    h, w = last_frame.shape[:2]
    log.info("[Cam %d] Stream resolution: %dx%d", cam_id, w, h)

    # Optional video recorder
    writer = None
    if save:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = Path(f"rtsp_cam{cam_id}_{ts}.mp4")
        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
        writer  = cv2.VideoWriter(str(outfile), fourcc, 20, (w, h))
        log.info("[Cam %d] Recording -> %s", cam_id, outfile)

    frame_count = 0
    total_inf   = 0.0
    win_name    = f"Fire & Smoke Detection  |  Cam {cam_id}"
    log.info("Running. Press Q to quit, S to snapshot.")

    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                frame = last_frame          # hold last frame while buffering
            else:
                last_frame = frame

            # Inference
            blob, orig_shape = preprocess(frame)
            t0   = time.time()
            raw  = sess.run(out_names, {inp_name: blob})
            dt   = time.time() - t0
            total_inf   += dt
            frame_count += 1

            detections = postprocess(raw, orig_shape, conf, iou)

            # Render
            vis     = frame.copy()
            vis     = draw_detections(vis, detections)
            avg_fps = frame_count / total_inf if total_inf > 0 else 0
            vis     = draw_overlay(vis, avg_fps, reader.connected, rtsp_url, cam_id)

            # Console alert
            if detections:
                labels = [f"{d['label']}({d['conf']:.2f})" for d in detections]
                log.warning("[Cam %d] ALERT: %s", cam_id, "  ".join(labels))

            if writer:
                writer.write(vis)

            if show:
                cv2.imshow(win_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    log.info("Quit by user.")
                    break
                elif key == ord("s"):
                    snap = Path(f"snapshot_cam{cam_id}_{datetime.now().strftime('%H%M%S')}.jpg")
                    cv2.imwrite(str(snap), vis)
                    log.info("Snapshot saved -> %s", snap)

    except KeyboardInterrupt:
        log.info("Interrupted (Ctrl-C).")
    finally:
        reader.stop()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        if frame_count:
            log.info("Done. Frames: %d | Avg inference: %.1f ms/frame",
                     frame_count, total_inf / frame_count * 1000)

# ─────────────────────────────── CLI ────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser(
        description="Fire & Smoke ONNX Detector  —  RTSP stream",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",   default=DEFAULT_MODEL, help="Path to best.onnx")
    p.add_argument("--rtsp",    default=DEFAULT_RTSP,  help="RTSP stream URL")
    p.add_argument("--conf",    type=float, default=CONF_THRESH, help="Confidence threshold")
    p.add_argument("--iou",     type=float, default=IOU_THRESH,  help="NMS IoU threshold")
    p.add_argument("--save",    action="store_true", help="Record annotated video to disk")
    p.add_argument("--no-show", action="store_true", help="Headless mode (no display window)")
    p.add_argument("--cam-id",  type=int, default=0,  help="Camera ID label (overlay + filenames)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sess, inp_name, out_names = load_session(args.model)
    run_rtsp(
        sess, inp_name, out_names,
        rtsp_url = args.rtsp,
        conf     = args.conf,
        iou      = args.iou,
        show     = not args.no_show,
        save     = args.save,
        cam_id   = args.cam_id,
    )
