# SteelSight AI - camera demo + dashboard on Colab.  Paste each "# %%" block into its own cell.
#
# What this fixes in a bare Gradio webcam demo
#   1. "back camera not working"  -> the demo now asks the phone for the REAR camera, and the upload
#      button opens the phone's native camera app (also rear). See notes at the bottom.
#   2. "can't tell a face from a steel sample" -> a face / blank / (optional) non-steel check runs first.
#   3. wrong image size -> the detector is run at the size it was TRAINED at (200 -> 224).
#
# You need: steelsight.zip (upload it to Colab or to Drive) and your trained best_v1.pt on Drive.

# %% 1. unzip the project + install
from google.colab import drive
drive.mount("/content/drive")
!unzip -q -o /content/steelsight.zip -d /content        # or /content/drive/MyDrive/steelsight.zip
%cd /content/steelsight
!pip install -q -r requirements.txt gradio transformers

# %% 2. settings - the ONLY lines you may need to edit
WEIGHTS = "/content/drive/MyDrive/SteelSight/best_v1.pt"
IMGSZ = 224            # you trained with imgsz=200; Ultralytics rounds that up to 224. Trained at 640? use 640.
USE_CLIP = True        # also reject non-steel scenes (desk, room, screen ...). Downloads ~600 MB once.
CLIP_FLAG = "--clip" if USE_CLIP else ""

# %% 3A. Gradio camera demo (public link, works from a phone)
!python scripts/gradio_demo.py --weights {WEIGHTS} --imgsz {IMGSZ} {CLIP_FLAG} --share
# Open the *.gradio.live link in a NEW browser tab (Chrome / Safari), not inside the Colab output frame
# and not inside WhatsApp / Instagram / Gmail's built-in browser - those block the camera.

# %% 3B. OR the full dashboard (Inspect / Live line / Operator review / Model card)
import os, re, subprocess, time, urllib.request
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared

env = dict(os.environ, STEELSIGHT_WEIGHTS=WEIGHTS, STEELSIGHT_IMGSZ=str(IMGSZ))
app = subprocess.Popen(
    ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true",
     "--browser.gatherUsageStats", "false"],
    env=env, stdout=open("streamlit.log", "w"), stderr=subprocess.STDOUT)
for _ in range(90):                                   # wait until Streamlit answers
    try:
        urllib.request.urlopen("http://localhost:8501/_stcore/health"); break
    except Exception:
        time.sleep(1)

tunnel = subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://localhost:8501", "--no-autoupdate"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in tunnel.stdout:                            # cloudflared prints the public URL in its log
    m = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", line)
    if m:
        print("OPEN THIS ON YOUR PHONE OR LAPTOP:", m.group(0)); break
# The link lives only while this cell and the Colab session are running.
# For the final presentation deploy to a Hugging Face Space instead (see README -> Deploy).

# %% NOTES ------------------------------------------------------------------------------------------
# Phone camera, in order of reliability
#   1. Image box -> upload icon -> "Camera" / "Take photo": opens the phone's own camera app (rear camera).
#      Always works, full resolution, no browser permission needed.
#   2. Image box -> webcam icon: asks the browser for the rear camera. Needs a normal browser tab over https
#      (the gradio.live / trycloudflare.com links are https). If the picture is still the front camera, use the
#      camera drop-down that appears in the webcam view to pick the "back" camera.
# Photograph a CLOSE-UP of the steel, flat-on, under even light. The model was trained on 200 x 200 px
# grayscale close-ups; a whole-desk photo shrinks any defect to a few pixels.
