# SteelSight AI - full pipeline on Colab (T4 GPU).  Paste each "# %%" block into its own cell.
# Runtime -> Change runtime type -> T4 GPU first.

# %% 1. get the code + deps  (replace <you> with your GitHub user once the repo is pushed; or upload the zip and unzip it)
!git clone https://github.com/<you>/steelsight.git
%cd steelsight
!pip install -q -r requirements.txt
!nvidia-smi | head -12

# %% 2. data: upload archive (2).zip (NEU-DET from Kaggle) to /content, then
!unzip -q "/content/archive (2).zip" -d /content
!python scripts/build_dataset.py --src /content/NEU-DET --dst /content/YOLO
# config/data.yaml now points at /content/YOLO

# %% 3. mount Drive so nothing is lost when the runtime dies
from google.colab import drive; drive.mount("/content/drive")
!mkdir -p /content/drive/MyDrive/SteelSight

# %% 4a. v1 baseline (12-15 min) - OR skip and copy your existing best.pt in 4b
!python scripts/train.py --model yolov8n.pt --epochs 30 --name v1_n30 --aug default
!cp models/v1_n30.pt models/best_v1.pt

# %% 4b. (alternative) reuse the best.pt you already trained
# !cp /content/drive/MyDrive/SteelSight/best_v1.pt models/best_v1.pt

# %% 5. Stage 1 anomaly screen (~3-5 min on T4): bank + calibrated threshold + AUROC report
!python scripts/train_anomaly.py --data /content/YOLO --out models/anomaly_bank.pt --percentile 99.5
!cat reports/anomaly.json

# %% 6. evaluate like production: mAP + per-class thresholds -> config/thresholds.json, reports/
!python scripts/evaluate.py --weights models/best_v1.pt --data config/data.yaml
!cat config/thresholds.json

# %% 7. replay the validation set as a coil (logs to data/steelsight.db for the dashboard)
!python scripts/simulate_coil.py --images /content/YOLO/val/images --coil COIL-DEMO-001 --limit 120
# to SHOW the roll-diagnosis mechanism on synthetic positions (clearly labelled synthetic):
# !python scripts/simulate_coil.py --images /content/YOLO/val/images --limit 60 --demo-periodic 1.41

# %% 8. v2 (optional, ~1 h): bigger model, longer, strong augmentation
# !python scripts/train.py --model yolov8s.pt --epochs 100 --name v2_s100 --aug strong
# !python scripts/evaluate.py --weights models/v2_s100.pt      # only adopt v2 if it beats v1 here

# %% 9. export + speed numbers for the deck
!python scripts/export.py --weights models/best_v1.pt --format onnx --bench
# TensorRT FP16 on the T4 (needs tensorrt; Colab usually has it):
# !python scripts/export.py --weights models/best_v1.pt --format engine --half --bench

# %% 10. sample images for the dashboard's picker + copy everything to Drive
!mkdir -p samples && for c in crazing inclusion patches pitted_surface rolled-in_scale scratches; do cp /content/YOLO/val/images/${c}_241.jpg samples/; done
!cp -r models reports config samples data /content/drive/MyDrive/SteelSight/

# %% 11. preview the dashboard from Colab (temporary URL; use Hugging Face Spaces for the real one)
!npm install -q localtunnel
!streamlit run app.py --server.port 8501 --server.headless true &> /content/st.log &
!sleep 8 && curl -s ipv4.icanhazip.com && npx localtunnel --port 8501
# open the printed URL; if asked for a password, paste the IP printed above
