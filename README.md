# Sentinel — Ambient Wi-Fi Home Security & People Counting

A smart-home security application that uses non-invasive ambient Wi-Fi sensing to monitor indoor occupancy and detect unexpected presence without cameras, microphones, or wearable tags.

---

## 1. Product Concept

Traditional home security relies on cameras that compromise personal privacy, or infrared motion sensors that only detect coarse movement rather than true occupancy count.

**Sentinel** monitors ambient Wi-Fi channel state disturbances (CSI) caused by human bodies moving and breathing in the space. The machine learning intelligence operates quietly in the background:
- **Zero Cameras or Microphones:** 100% privacy-preserving; works in total darkness.
- **Accurate Occupancy Counting:** Knows how many people are present, not just whether motion occurred.
- **Night Guard & Perimeter Rules:** Set expected household members (e.g. 3 people tonight). If detected occupancy exceeds this threshold, an instant security alert is triggered.
- **Consumer-First UX:** The main interface uses natural smart-home language (*People at Home*, *Night Monitoring*, *Expected Count*, *Security Alerts*) rather than ML jargon or dataset labels.

---

## 2. Application Architecture

```
Monitored Living Area
  (Wi-Fi Hub 📡 ⟷ 📶 Receiver)
             │
             ▼
Channel State Information (CSI)
  (256 OFDM Subcarriers, 128×256 packet windows)
             │
             ▼
Normalized Subcarrier Statistics (768 Dimensions)
  (Mean, Std Dev, Max amplitude per subcarrier)
             │
             ▼
Pre-Trained Machine Learning Models
  ├── SVM (Primary Classifier — 96.7% benchmark accuracy)
  ├── XGBoost (95.1% benchmark accuracy)
  └── Random Forest (93.8% benchmark accuracy)
             │
             ▼
Ambient Security Decision Engine
  ├── If detected_occupancy > expected_occupancy → 🚨 Unexpected Occupancy Alert
  └── If detected_occupancy <= expected_occupancy → ✓ Normal Home Status
             │
             ▼
Consumer Smart-Home Dashboard (Flask + Web UI)
```

---

## 3. Key Navigation & Views

1. **Home (Dashboard):**
   - Quick "Set up your home" onboarding: configure expected occupants tonight (`[-] 3 [+]`) and schedule (`22:00 → 07:00`).
   - Prominent **Home Status** hero card: instant confirmation of safety (`✓ Everything looks normal` or `🚨 Security Alert`).
   - Core metrics: *People at Home*, *Night Monitoring Active*, and *Security Status*.
   - Room occupancy glance and recent activity feed.

2. **Monitoring (Live Sensing):**
   - Architectural 2D floor plan of the Living Room and entryway.
   - Wi-Fi Hub (TX) and Sensing Sensor (RX) layout with sensing zone overlay.
   - Occupancy visualization displaying detected people as avatar markers in real-time.
   - Play, pause, step (+1 window), reset, and speed controls (0.5×–5×).

3. **Activity (Event Timeline):**
   - Comprehensive chronological event log in realistic human language (`22:01 — Night monitoring started`, `22:14 — 3 people detected`, `23:14 — 4 people detected 🚨`).
   - Filters: *All Activity*, *Alerts Only*, *Normal Activity*.

4. **Alerts (Security Incidents Center):**
   - Real-time alert banner with unread badge counter.
   - Detailed incident breakdown modal: timestamp, detected count, expected threshold, confidence, and recommended security actions.
   - Incident acknowledgment and resolution tracking.

5. **Settings (Home Settings):**
   - Household members baseline configuration.
   - Night monitoring schedule start/end hours.
   - Alert trigger sensitivity rules.
   - Camera-free privacy shield status.

6. **System (Technical Evaluation):**
   - Dedicated for researchers and technical evaluators.
   - Model benchmark cards: SVM (96.7% accuracy, 0.040 MAE), XGBoost (95.1%), Random Forest (93.8%).
   - Live stream chart comparing Ground Truth vs. SVM, XGBoost, and RF predictions.
   - Wi-Fi CSI subcarrier amplitude chart (Mean, Std, Max across 256 subcarriers).
   - Active classifier toggle: switch between SVM, XGBoost, and Random Forest on the fly.

---

## 4. How to Run Locally

### Requirements
- Python 3.10+
- Dependencies listed in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

### Start the Application
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

---

## 5. Demonstration Mode & Real Data Notes

Because physical ESP32 hardware is not currently connected to this host machine, the application operates in **Demonstration Monitoring Mode**:
- **Real:** The 2,800 held-out CSI evaluation windows (`data/test_features.npz`), the ground-truth people counts (classes 0–7), the three pre-trained models (`models/svm.joblib`, `models/xgboost.json`, `models/random_forest.joblib`), and all live inference predictions.
- **Simulated:** The playback timing (replayed sequentially to simulate a live 2-second sampling interval) and the schematic room visualization.
- **Transparent:** Clicking the **Demo Monitoring** badge in the app header opens an explanation modal clarifying that the stream uses recorded held-out sensing data while real-time physical hardware is in development.
