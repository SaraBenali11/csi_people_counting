# Wi-Fi Home Security & People Counting

A simple smart-home security website that uses Wi-Fi signals to count how many people are in the house.

It works in the background without cameras or microphones.

---

## What it does

- **Counts people at home:** Knows how many people are present using Wi-Fi signals.
- **Night monitoring:** Set how many people should be at home (for example, 3 people).
- **Security alerts:** If more people are detected than expected, it sends an alert immediately.
- **Simple design:** Uses clear everyday language (Home Status, People at Home, Night Monitoring, Alerts).
- **No cameras:** Completely private. Works in the dark.

---

## How it works

1. Wi-Fi signals travel across the room between your router and a receiver.
2. People moving or breathing change the Wi-Fi signals.
3. Trained AI models (SVM, Random Forest, XGBoost) estimate the number of people.
4. If people detected > expected people, an alert is created.

---

## Pages in the app

1. **Home:**
   - "Set up your home": Choose expected people tonight `[-] 3 [+]`.
   - "Start Night Monitoring" button.
   - Home Status card: Shows `✓ Normal` or `🚨 Alert`.
   - 3 simple cards: People at home, Night monitoring, Security status.
   - Living room view: Shows simple icons for detected people.
   - Recent activity list.
   - Play / Pause / Reset controls.

2. **Activity:**
   - Timeline of recent events (e.g., `22:14 — 3 people detected (Normal)`).
   - Filter by: All, Alerts, Normal.

3. **Alerts:**
   - Shows active security warnings when occupancy exceeds the expected limit.
   - Button to acknowledge alerts.

4. **Settings:**
   - Change expected people count.
   - Change monitoring start and end times.

5. **System (Technical Details):**
   - Separate section showing AI model accuracy (SVM: 96.7%, XGBoost: 95.1%, Random Forest: 93.8%).
   - For developers and researchers only.

---

## How to run it

```bash
cd csi_replay_app
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** in your web browser.

---

## Demo note

The application runs in **demo mode** using 2,800 recorded Wi-Fi test samples through the trained AI models, because physical sensor hardware is not connected. All counting predictions and alert logic are real.
