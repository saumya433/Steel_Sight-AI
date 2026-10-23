---
title: SteelSight AI
emoji: 🔩
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# SteelSight AI - live demo

Two-stage surface defect inspection for steel strip: an anomaly screen that flags
anything abnormal (even unseen defect types), a YOLOv8 detector that names and
localises six known classes, and a per-class confidence gate that routes frames to
PASS / REVIEW / AUTO_FLAG / UNKNOWN.

Upload a frame on the **Inspect** tab or replay a coil on the **Live line** tab.

Code: https://github.com/<you>/steelsight · Team MNIT Matrix, Stainless Spark 2026.

*Copy this file to the root of the Space as `README.md` (the front-matter is what tells the Space to run `app.py`).
Keep the GitHub repo's own README for GitHub.*
