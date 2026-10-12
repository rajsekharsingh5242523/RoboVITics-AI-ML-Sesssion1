# RoboVITics ML Session 1 — Part 2: Emoji Face Code-Along (40–45 minutes)

## Goal

Build a small live webcam application that estimates facial expressions with DeepFace and overlays a matching emoji on the detected face using OpenCV.

The aim of this code-along is to connect a few practical ideas: reading frames from a webcam, representing images as NumPy arrays, running facial-expression analysis, and blending a transparent image over a video frame.

## What You Will Learn

By the end of this activity, you should be able to:

- Open and read frames from a webcam using OpenCV.
- Understand that an image is represented as a NumPy array.
- Run DeepFace emotion analysis on a frame.
- Choose an emoji based on the predicted expression scores.
- Resize and position an emoji over a face region.
- Blend a transparent PNG image into a webcam frame.
- Keep the display responsive and exit the application safely.

## How the Application Works

**Webcam → Face/expression analysis → Select expression → Select emoji → Overlay emoji → Display live video**

1. OpenCV reads frames from the webcam.
2. The frame is flipped horizontally so it behaves like a mirror.
3. Every third frame is passed to DeepFace for emotion analysis.
4. The program selects the largest detected face and tracks its bounding box across frames.
5. Recent emotion scores are averaged to reduce rapid changes.
6. The emoji corresponding to the highest average score is resized to the face region.
7. Alpha blending combines the emoji with the original frame.
8. The final frame is displayed in a window. Press **q** to quit.

## Requirements

Before starting, make sure you have:

- **Python 3.12** installed. Using the same Python version for installation and execution avoids package-environment confusion.
- A working webcam.
- An internet connection for downloading the emoji PNG images the first time the script runs.
- A desktop environment that supports OpenCV windows (`cv2.imshow`).

### Python packages used

| Package | Purpose |
|---|---|
| `opencv-python` | Webcam access, image processing, resizing, blending, and display |
| `numpy` | Image arrays and numerical calculations |
| `deepface` | Facial-expression analysis |

The script also uses Python standard-library modules: `urllib.request` to download emoji images, `collections.deque` to keep a short history of predictions, and `typing.Any` for a type annotation. These do not need separate installation.

## Step 1 — Check Python

On Windows, open **Command Prompt** or the VS Code terminal and run:

```powershell
py -3.12 --version
```

You should see a Python 3.12 version number. To see the Python installations available on your computer, run:

```powershell
py -0p
```

If Python 3.12 is not listed, install it before continuing. The commands below specifically use Python 3.12.

## Step 2 — Create a Project Folder

```powershell
mkdir emoji-face-project
cd emoji-face-project
```

If the folder already exists, open a terminal in that folder instead of creating it again.

## Step 3 — Install the Dependencies

Upgrade pip and install the required packages:

```powershell
py -3.12 -m pip install --upgrade pip
py -3.12 -m pip install opencv-python numpy deepface
```

Check that the imports work:

```powershell
py -3.12 -c "import cv2, numpy; import deepface; print('All imports successful')"
```

If the command prints `All imports successful`, the basic imports are ready.

### Important OpenCV note

This project uses `cv2.imshow()` to open a desktop window. It needs a GUI-capable OpenCV installation. Packages such as `opencv-python-headless` and `opencv-contrib-python-headless` are intended for headless environments and can conflict with `opencv-python` when installed in the same Python environment.

If you encounter an OpenCV GUI error, remove conflicting headless packages from this same Python 3.12 environment and reinstall the GUI package:

```powershell
py -3.12 -m pip uninstall opencv-python-headless opencv-contrib-python-headless opencv-python
py -3.12 -m pip install opencv-python
```

Then rerun the import check.

## Step 4 — Create the Python File

Create a file named **`emoji_face.py`** inside the project folder. Paste the code below into that file.

```python
import cv2; import numpy as np; import urllib.request
from collections import deque; from typing import Any; from deepface import DeepFace
names=["angry","disgust","fear","happy","sad","surprise","neutral"]
codes="1f620 1f922 1f628 1f60a 1f622 1f62e 1f610".split(); EMOJI={}
for name,code in zip(names,codes):
    url="https://raw.githubusercontent.com/twitter/twemoji/" f"v14.0.2/assets/72x72/{code}.png"
    with urllib.request.urlopen(url,timeout=10) as response: image_bytes=response.read()
    image=cv2.imdecode(np.frombuffer(image_bytes,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
    if image is None: raise RuntimeError(f"Could not decode emoji: {name}")
    EMOJI[name]=image
cap=cv2.VideoCapture(0)
if not cap.isOpened(): raise RuntimeError("Could not open the webcam. Check camera permissions.")
n=0; hist=deque(maxlen=8); box=None; miss=0
while cv2.waitKey(1) != ord("q"):
    ok, f = cap.read()
    if not ok: break 
    f, n = cv2.flip(f, 1), n + 1 
    if n % 3 == 0: 
        try:
            res: Any = DeepFace.analyze(f, actions=["emotion"], detector_backend="yunet")  
            r = max(res, key=lambda r: r["region"]["w"] * r["region"]["h"]) 
            g = r["region"]; new = np.array([g["x"], g["y"], g["w"], g["h"]], float)
            box = new if box is None else 0.6 * box + 0.4 * new; hist.append(r["emotion"]); miss = 0
        except Exception as e:
            miss += 1
            if "detected" not in str(e) and miss == 1: print(e)  
    if miss > 10: box = None; hist.clear()  
    if box is not None and hist: 
        x, y, w, h = map(int, box); avg = {k: np.mean([s[k] for s in hist]) for k in names}
        if min(w, h) > 0:
            em = cv2.resize(EMOJI[max(avg, key=lambda k: avg[k])], (w, h)) 
            x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, f.shape[1]), min(y + h, f.shape[0])
            em = em[y0 - y:y1 - y, x0 - x:x1 - x]; a = em[..., 3:] / 255.0
            f[y0:y1, x0:x1] = (a * em[..., :3] + (1 - a) * f[y0:y1, x0:x1]).astype(np.uint8)
    cv2.imshow("Emoji Face (q to quit)", f) 
cap.release(); cv2.destroyAllWindows()
```

## Step 5 — Run the Application

From the same project folder, run:

```powershell
py -3.12 emoji_face.py
```

Allow camera access if Windows asks for permission. Look toward the camera and try a few different facial expressions. Press **q** while the video window is active to close the application.

The first run may take longer because DeepFace may initialize its models and the script downloads the emoji images from the internet.

## Code Walkthrough

### 1. Importing libraries

- `cv2` is OpenCV. It reads webcam frames, flips and resizes images, blends the emoji, and displays the result.
- `numpy` supports array operations on images and emotion scores.
- `urllib.request` downloads the emoji PNG files.
- `deque` stores a limited history of recent predictions.
- `DeepFace` performs emotion analysis.

### 2. Matching expression names to emoji images

`names` contains the expression labels expected from DeepFace. `codes` contains the matching Twemoji image codes. The loop downloads each PNG, decodes it into an image array, and stores it in the `EMOJI` dictionary using the expression name as the key.

The downloaded emoji images include an alpha channel, which represents transparency.

### 3. Opening the webcam

`cv2.VideoCapture(0)` attempts to open the default camera. The `isOpened()` check stops the script with a useful error if the camera cannot be opened.

### 4. Keeping the video responsive

The main loop reads a frame, checks that the read succeeded, and flips it horizontally. Emotion analysis runs only when `n % 3 == 0`, rather than on every frame. This reduces how often the more expensive analysis step runs.

### 5. Detecting and tracking the face region

DeepFace returns expression scores and a face region. The code chooses the region with the largest width-times-height area, then updates the tracked bounding box using a weighted blend of the previous and newly detected coordinates. This smooths sudden position changes.

The `hist` deque stores up to eight recent emotion-score dictionaries. The program averages each expression's score and chooses the expression with the highest average.

### 6. Resizing and overlaying the emoji

`cv2.resize()` scales the chosen emoji to the face-box dimensions. The code clips the overlay coordinates to the frame boundaries so that an emoji partly outside the frame does not index beyond the image.

The alpha channel controls the blend:

```python
output = alpha * emoji + (1 - alpha) * frame
```

When alpha is 1, the emoji pixel is fully visible. When alpha is 0, the original webcam pixel remains visible. The result is converted to `uint8` before it is written back into the frame.

### 7. Displaying and closing

`cv2.imshow()` displays the live frame. Pressing **q** exits the loop. `cap.release()` releases the camera and `cv2.destroyAllWindows()` closes OpenCV windows.

## Troubleshooting

### `No module named cv2`, `numpy`, or `deepface`

Install packages using the same Python version used to run the script:

```powershell
py -3.12 -m pip install opencv-python numpy deepface
```

### Webcam cannot be opened

- Check whether another application is using the camera.
- Check Windows **Settings → Privacy & security → Camera** and allow desktop apps to access it.
- If your camera is not the default device, change `cv2.VideoCapture(0)` to another camera index such as `1` and test again.

### `cv2.imshow()` says the function is not implemented

This usually indicates that the installed OpenCV package lacks GUI support or that conflicting OpenCV packages are installed. Follow the **Important OpenCV note** in Step 3 and ensure you run the script with the same Python environment where the packages were installed.

### Emoji download fails

The script downloads the images from GitHub when it starts, so check your internet connection and try again. A network proxy or firewall may block the download.

### DeepFace or the `yunet` detector reports an error

Check the full error printed in the terminal. DeepFace versions can differ in which detector backends they support. If your installed version does not support `yunet`, you can test a supported backend by changing `detector_backend="yunet"` to `detector_backend="opencv"` and running the script again.

### The emoji does not appear

- Make sure your face is visible and reasonably well lit.
- Keep your face within the camera frame.
- Wait a moment for the model to initialize and for a few successful analysis frames.
- Check the terminal for model, detector, or network errors.

### A compatibility note about DeepFace results

The script expects `DeepFace.analyze()` to return an iterable of face-result dictionaries. Some DeepFace versions or settings may return a single dictionary instead. If you see an error around `max(res, ...)`, check the return format for your installed version and adapt the result handling to that version.

## Suggested 40–45 Minute Session Plan

| Time | Activity |
|---|---|
| 0–5 min | Introduce the goal and show the expected result |
| 5–10 min | Explain webcam frames and NumPy image arrays |
| 10–18 min | Set up Python and install the dependencies |
| 18–30 min | Walk through the code: analysis, history, emoji selection, and overlay |
| 30–40 min | Run the program and troubleshoot common issues |
| 40–45 min | Recap the pipeline and discuss possible improvements |

## Wrap-Up

This mini-project combines computer vision and a pre-trained facial-expression analysis library in a simple interactive application. It demonstrates a useful image-processing pattern: detect a region, resize an overlay, and combine the overlay with the original frame using transparency.

### Possible extensions

- Add a visible label showing the predicted expression.
- Let the user choose between emoji overlays and bounding boxes.
- Measure the frame rate and compare running analysis on every frame versus every third frame.
- Add a toggle to pause or resume the overlay.

## Privacy and Responsible Use

Run the project only with the knowledge and consent of people in view of the camera. Facial-expression models estimate visual patterns; they do **not** reliably determine a person's true internal emotional state. Treat the result as a coding demonstration, not as a psychological assessment or a basis for important decisions.

## Presenter Checklist

- [ ] Confirm Python 3.12 is available.
- [ ] Install dependencies and run the import test before the session.
- [ ] Test the webcam and confirm that the OpenCV display window works.
- [ ] Check that the computer has internet access for the emoji downloads and model setup.
- [ ] Keep this README open for troubleshooting.
- [ ] Remind participants that they can press **q** to exit.
