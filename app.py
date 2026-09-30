"""
WiFi CSI People-Counting — Dataset Replay backend.

Loads your REAL trained models (random_forest.joblib, svm.joblib, xgboost.json)
and your REAL held-out test set (test_features.npz — the exact 2,800-window
evaluation data saved by the training notebook), then replays it window by
window through the real models, exactly as they were actually evaluated.

Nothing in this file invents CSI, labels, or confidence values. Where a value
genuinely isn't available (SVM confidence), the API says so explicitly rather
than fabricating one.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000 in a browser.

"""

import os
import threading
import time

import joblib
import numpy as np
import xgboost as xgb
from flask import Flask, jsonify, request, render_template

# ---------------------------------------------------------------------------
# Load real artifacts
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
TEST_FEATURES_PATH = os.path.join(PROJECT_DIR, "data", "test_features.npz")
RF_PATH = os.path.join(MODEL_DIR, "random_forest.joblib")
SVM_PATH = os.path.join(MODEL_DIR, "svm.joblib")
XGB_PATH = os.path.join(MODEL_DIR, "xgboost.json")

for path, label in [(TEST_FEATURES_PATH, "test_features.npz"), (RF_PATH, "random_forest.joblib"),
                     (SVM_PATH, "svm.joblib"), (XGB_PATH, "xgboost.json")]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {label} at {path}. Set PEOPLE_COUNTING_MODEL_DIR "
            f"to your actual models folder if it's not at the default path."
        )

_test_data = np.load(TEST_FEATURES_PATH, allow_pickle=True)
X_TEST = _test_data["X"]                # (2800, 768) - real, already-normalized summary features
Y_TEST = _test_data["y"]                # (2800,) - real ground-truth people counts
SOURCE_IDS = _test_data["source_ids"]   # (2800,) - real original EHUNAM filenames
N_TOTAL = len(Y_TEST)
N_SUBCARRIERS = X_TEST.shape[1] // 3    # 768 = mean(256) + std(256) + max(256)
CLASSES = sorted(int(c) for c in np.unique(Y_TEST))

rf_model = joblib.load(RF_PATH)
svm_model = joblib.load(SVM_PATH)
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_PATH)

MODELS = {
    "svm": {"name": "SVM", "obj": svm_model, "primary": True},
    "rf": {"name": "Random Forest", "obj": rf_model, "primary": False},
    "xgb": {"name": "XGBoost", "obj": xgb_model, "primary": False},
}

# Real, previously-measured evaluation results (from the training notebook's
# own printed output) - shown as static reference metrics, never recomputed
# live from the replay, and always labeled as such in the UI.
EVAL_METRICS = {
    "svm": {"accuracy": 0.967, "macro_f1": 0.901, "mae": 0.040},
    "xgb": {"accuracy": 0.951, "macro_f1": 0.871, "mae": 0.059},
    "rf":  {"accuracy": 0.938, "macro_f1": 0.866, "mae": 0.070},
}

DATA_SOURCE_NOTE = (
    f"{N_TOTAL} held-out windows from EHUNAM, produced from 7 readable processed "
    f"batches (batch_0003, 0004, 0006, 0012, 0014, 0016, 0017 - 11 of the original "
    f"18 batch files were corrupted and unreadable when this test set was built). "
    f"This is a real stratified random split, not a held-out-environment split."
)

# ---------------------------------------------------------------------------
# Simulation state
# ---------------------------------------------------------------------------
app = Flask(__name__)

state_lock = threading.Lock()
state = {
    "index": -1,            # -1 = not started yet
    "running": False,
    "speed": 1.0,
    "history": [],          # list of step dicts, most recent last
    "events": [],           # list of event dicts
    "alerts": [],           # list of alert dicts
    "expected_occupancy": 3,
    "primary_model": "svm",
}
BASE_INTERVAL = 2.0  # seconds per step at 1x speed
_thread = None
_stop_flag = threading.Event()


def get_confidence(model_key, x_row):
    model = MODELS[model_key]["obj"]
    try:
        proba = model.predict_proba(x_row.reshape(1, -1))[0]
        idx = int(np.argmax(proba))
        classes = getattr(model, "classes_", CLASSES)
        return float(proba[idx]), int(classes[idx]) if len(classes) == len(proba) else None, True, None
    except Exception:
        return None, None, False, f"{MODELS[model_key]['name']} was not trained with probability estimates (no confidence available)."


def run_step(index):
    x_row = X_TEST[index]
    truth = int(Y_TEST[index])
    source = str(SOURCE_IDS[index])

    preds = {}
    for key, m in MODELS.items():
        model = m["obj"]
        predicted = int(model.predict(x_row.reshape(1, -1))[0])
        confidence, conf_class, available, reason = get_confidence(key, x_row)
        preds[key] = {
            "predicted": predicted,
            "correct": predicted == truth,
            "confidence": confidence,
            "confidence_available": available,
            "confidence_note": reason,
        }

    mean_feat = x_row[:N_SUBCARRIERS].tolist()
    std_feat = x_row[N_SUBCARRIERS:2 * N_SUBCARRIERS].tolist()
    max_feat = x_row[2 * N_SUBCARRIERS:3 * N_SUBCARRIERS].tolist()

    with state_lock:
        primary_key = state.get("primary_model", "svm")
        primary_pred = preds[primary_key]["predicted"]
        expected = state["expected_occupancy"]
        alert = primary_pred > expected

        step = {
            "index": index,
            "total": N_TOTAL,
            "ground_truth": truth,
            "source_file": source,
            "predictions": preds,
            "csi_summary": {"mean": mean_feat, "std": std_feat, "max": max_feat, "n_subcarriers": N_SUBCARRIERS},
            "security": {
                "expected_occupancy": expected,
                "detected_occupancy": primary_pred,
                "alert": alert,
                "primary_model": primary_key,
            },
            "timestamp": time.strftime("%H:%M:%S"),
        }
        state["history"].append(step)
        if len(state["history"]) > 300:
            state["history"] = state["history"][-300:]

        prev = state["history"][-2] if len(state["history"]) > 1 else None
        prev_pred = prev["predictions"][primary_key]["predicted"] if prev else None
        prev_alert = (prev_pred > expected) if prev_pred is not None else False

        # Simple, human-friendly events and alerts
        if alert:
            if not prev_alert or (prev_pred is not None and primary_pred > prev_pred):
                state["alerts"].append({
                    "id": f"alert-{index}-{int(time.time()*1000)}",
                    "time": step["timestamp"],
                    "type": "unexpected_occupancy",
                    "title": "Unexpected occupancy",
                    "message": f"{primary_pred} people detected at home. Expected maximum: {expected} people.",
                    "detected": primary_pred,
                    "expected": expected,
                    "level": "danger",
                    "acknowledged": False
                })
            state["events"].append({
                "time": step["timestamp"],
                "text": f"{primary_pred} people detected",
                "status": "Unexpected occupancy",
                "level": "danger",
                "type": "alert",
                "count": primary_pred
            })
        elif prev_alert and not alert:
            state["alerts"].append({
                "id": f"norm-{index}-{int(time.time()*1000)}",
                "time": step["timestamp"],
                "type": "occupancy_normal",
                "title": "Occupancy returned to normal",
                "message": f"{primary_pred} people detected (normal).",
                "detected": primary_pred,
                "expected": expected,
                "level": "success",
                "acknowledged": True
            })
            state["events"].append({
                "time": step["timestamp"],
                "text": f"{primary_pred} people detected",
                "status": "Normal",
                "level": "info",
                "type": "normal",
                "count": primary_pred
            })
        elif prev_pred is not None and prev_pred > 0 and primary_pred == 0:
            state["events"].append({
                "time": step["timestamp"],
                "text": "No people detected",
                "status": "Normal",
                "level": "info",
                "type": "normal",
                "count": 0
            })
        elif prev_pred is not None and primary_pred != prev_pred:
            state["events"].append({
                "time": step["timestamp"],
                "text": f"{primary_pred} people detected",
                "status": "Normal",
                "level": "info",
                "type": "normal",
                "count": primary_pred
            })
        elif not prev:
            state["events"].append({
                "time": step["timestamp"],
                "text": f"{primary_pred} people detected",
                "status": "Normal",
                "level": "info",
                "type": "normal",
                "count": primary_pred
            })

        if len(state["events"]) > 100:
            state["events"] = state["events"][-100:]
        if len(state.get("alerts", [])) > 50:
            state["alerts"] = state["alerts"][-50:]


def replay_loop():
    while not _stop_flag.is_set():
        with state_lock:
            if state["index"] + 1 >= N_TOTAL:
                state["running"] = False
                break
            state["index"] += 1
            idx = state["index"]
            speed = state["speed"]
        run_step(idx)
        if _stop_flag.wait(timeout=max(0.05, BASE_INTERVAL / speed)):
            break


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/simulation/info")
def info():
    return jsonify({
        "dataset": "EHUNAM",
        "data_type": "WiFi CSI",
        "representation": "Amplitude (mean/std/max per subcarrier)",
        "window_size": "128 x 256",
        "n_subcarriers": N_SUBCARRIERS,
        "n_features": X_TEST.shape[1],
        "classes": CLASSES,
        "models": [m["name"] for m in MODELS.values()],
        "replay_mode": "Held-out testing data (test_features.npz)",
        "hardware": "ESP32 not connected",
        "total_windows": N_TOTAL,
        "data_source_note": DATA_SOURCE_NOTE,
    })


@app.route("/api/simulation/metrics")
def metrics():
    return jsonify({k: {**v, "name": MODELS[k]["name"]} for k, v in EVAL_METRICS.items()})


@app.route("/api/simulation/status")
def status():
    with state_lock:
        curr_alert = False
        detected = 0
        if state["history"]:
            curr = state["history"][-1]
            curr_alert = curr["security"]["alert"]
            detected = curr["security"]["detected_occupancy"]
        return jsonify({
            "running": state["running"],
            "index": state["index"],
            "total": N_TOTAL,
            "speed": state["speed"],
            "expected_occupancy": state["expected_occupancy"],
            "primary_model": state.get("primary_model", "svm"),
            "alert": curr_alert,
            "detected_occupancy": detected,
        })


@app.route("/api/simulation/start", methods=["POST"])
def start():
    global _thread, _stop_flag
    with state_lock:
        if state["index"] + 1 >= N_TOTAL:
            return jsonify({"error": "Replay already complete. Reset to replay again."}), 400
        if state["running"]:
            return jsonify(status().json)
        state["running"] = True
        state["events"].append({
            "time": time.strftime("%H:%M:%S"),
            "text": "Night monitoring started",
            "status": "Active",
            "level": "info",
            "type": "system"
        })
    _stop_flag.clear()
    _thread = threading.Thread(target=replay_loop, daemon=True)
    _thread.start()
    return jsonify(status().json)


@app.route("/api/simulation/pause", methods=["POST"])
def pause():
    global _thread
    _stop_flag.set()
    if _thread and _thread.is_alive():
        _thread.join(timeout=0.2)
    with state_lock:
        state["running"] = False
        state["events"].append({
            "time": time.strftime("%H:%M:%S"),
            "text": "Monitoring paused",
            "status": "Paused",
            "level": "info",
            "type": "system"
        })
    return jsonify(status().json)


@app.route("/api/simulation/reset", methods=["POST"])
def reset():
    global _thread
    _stop_flag.set()
    if _thread and _thread.is_alive():
        _thread.join(timeout=0.2)
    with state_lock:
        state["index"] = -1
        state["running"] = False
        state["history"] = []
        state["events"] = []
        state["alerts"] = []
    return jsonify(status().json)


@app.route("/api/simulation/step", methods=["POST"])
def step():
    with state_lock:
        if state["index"] + 1 >= N_TOTAL:
            return jsonify({"error": "Replay already complete. Reset to replay again."}), 400
        state["index"] += 1
        idx = state["index"]
    run_step(idx)
    return jsonify(status().json)


@app.route("/api/simulation/speed", methods=["POST"])
def set_speed():
    speed = float(request.json.get("speed", 1.0))
    with state_lock:
        state["speed"] = speed
    return jsonify(status().json)


@app.route("/api/simulation/expected_occupancy", methods=["POST"])
def set_expected():
    val = int(request.json.get("value", 0))
    with state_lock:
        state["expected_occupancy"] = val
    return jsonify(status().json)


@app.route("/api/simulation/primary_model", methods=["POST"])
def set_primary_model():
    data = request.get_json(silent=True) or {}
    m = data.get("model", "svm")
    if m in MODELS:
        with state_lock:
            state["primary_model"] = m
    return jsonify({"primary_model": state.get("primary_model", "svm")})


@app.route("/api/simulation/alerts/acknowledge", methods=["POST"])
def acknowledge_alert():
    data = request.get_json(silent=True) or {}
    alert_id = data.get("id")
    with state_lock:
        for a in state.get("alerts", []):
            if alert_id is None or a["id"] == alert_id:
                a["acknowledged"] = True
    return jsonify({"success": True})


@app.route("/api/simulation/current")
def current():
    with state_lock:
        if not state["history"]:
            return jsonify(None)
        return jsonify(state["history"][-1])


@app.route("/api/simulation/history")
def history():
    with state_lock:
        return jsonify({
            "history": state["history"],
            "events": list(reversed(state["events"])),
            "alerts": list(reversed(state.get("alerts", [])))
        })


if __name__ == "__main__":
    print(f"Loaded {N_TOTAL} real held-out test windows, classes present: {CLASSES}")
    app.run(debug=True, port=5000)
