# RoboVITics ML Session 1 — Part 2: Emoji Face Code-Along (40–45 minutes)

## Goal

Build a live webcam application that estimates facial expressions using DeepFace and overlays an emoji on the detected face using OpenCV.

**What participants will learn**
- How a Python script accesses a webcam.
- How a pretrained deep-learning model estimates facial expressions.
- How to install Python libraries and run a script.
- How to combine a model's prediction with image processing.
- How to debug common Python, camera, and dependency errors.

**Important:** This program estimates facial expressions; it cannot reliably determine what someone is actually feeling.

## Slide 17 — From an image to an emoji (3 minutes)

Draw this pipeline on the board:

**Webcam → Face detection → Expression prediction → Emoji selection → OpenCV display**

**Say:** We have already learned the theory of machine learning. Now let's use a pretrained deep-learning model in a real application. Instead of training a model from scratch, we will use DeepFace to estimate an expression and OpenCV to display an emoji over the face.

**Ask:** What do you think is responsible for recognizing the face: the webcam, Python, or the AI model?

Wait five seconds, take two answers, then explain:

- The webcam captures the image.
- Python coordinates the steps.
- DeepFace performs facial analysis using a model.
- OpenCV displays the image and overlays the emoji.

**Say:** Our job is to connect these components into one working application.

## Step 1 — Check Python and prepare the folder (5 minutes)

### A. Open a terminal

**Say:** Follow these steps with me. If you already have Python 3.12 installed, do not install it again.

1. Press the **Windows key** on your keyboard.
2. Type `PowerShell`.
3. Click **Windows PowerShell** or **Terminal**.
4. Click inside the terminal window so the cursor is active.
5. Type the following command and press **Enter**:

```powershell
py -3.12 --version
```

**Expected output:** something similar to `Python 3.12.x`.

If the command fails, type:

```powershell
py -0p
```

This lists Python installations that the Windows Python launcher can find. If Python 3.12 is missing, install it before continuing.

### B. Create a project folder

In the same terminal, type each command and press **Enter** after each one:

```powershell
mkdir emoji-face-project
cd emoji-face-project
```

**Say:** We are creating a separate folder so our code and project files stay organized.

### C. Install the libraries

Type:

```powershell
py -3.12 -m pip install --upgrade pip
```

Press **Enter** and wait for the command to finish.

Then type:

```powershell
py -3.12 -m pip install opencv-python numpy deepface
```

Press **Enter** again.

**Explain the libraries:**

- `opencv-python`: accesses the webcam and displays images.
- `numpy`: performs numerical calculations on image arrays.
- `deepface`: provides pretrained facial-analysis models.
- `urllib`: downloads emoji images using Python's standard library, so no installation is required.

**Trainer preparation:** Install these packages on the presentation laptop before the session. DeepFace may download additional model files on its first run, and installation or model downloads can take several minutes. Do not make the audience wait through a lengthy installation.

### D. Test the imports

Type:

```powershell
py -3.12 -c "import cv2, numpy; import deepface; print('All imports successful')"
```

Press **Enter**.

If you see `All imports successful`, continue. If an error appears, save the full error message so you can troubleshoot it rather than guessing.

## Step 2 — Create `emoji_face.py` (7 minutes)

**Say:** Now we will create the Python program. Follow the steps carefully because the filename and indentation matter.

### A. Open a code editor

If you have Visual Studio Code installed:

1. Press the **Windows key**.
2. Type `Visual Studio Code`.
3. Press **Enter**.
4. In VS Code, select **File → Open Folder**.
5. Open the `emoji-face-project` folder you created.
6. Click **New File** in the Explorer panel.
7. Name the file exactly:

`emoji_face.py`

If you do not have VS Code, open Notepad, paste the code below, and use **File → Save As**. Set **Save as type** to **All files**, then enter `emoji_face.py` as the filename. Make sure it does not become `emoji_face.py.txt`.

### B. Paste the complete code

Copy the following code into `emoji_face.py`.

```python
import cv2
import numpy as np
import urllib.request
from collections import deque
from typing import Any
from deepface import DeepFace


# --------------------------------------------------
# 1. Define the expressions and their matching emojis
# --------------------------------------------------

EMOTION_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

EMOJI_CODES = [
    "1f620",  # angry
    "1f922",  # disgust
    "1f628",  # fear
    "1f60a",  # happy
    "1f622",  # sad
    "1f62e",  # surprise
    "1f610",  # neutral
]

EMOJI = {}

# Download and prepare each emoji image.
for name, code in zip(EMOTION_NAMES, EMOJI_CODES):
    url = (
        "https://raw.githubusercontent.com/twitter/"
        f"twemoji/v14.0.2/assets/72x72/{code}.png"
    )

    with urllib.request.urlopen(url, timeout=15) as response:
        image_bytes = response.read()

    image_array = np.frombuffer(
        image_bytes, dtype=np.uint8
    )

    emoji_image = cv2.imdecode(
        image_array, cv2.IMREAD_UNCHANGED
    )

    if emoji_image is None:
        raise RuntimeError(
            f"Could not load emoji for {name}"
        )

    # Ensure the emoji has an alpha channel.
    if emoji_image.ndim == 3 and emoji_image.shape[2] == 3:
        alpha = np.full(
            emoji_image.shape[:2] + (1,),
            255,
            dtype=np.uint8,
        )
        emoji_image = np.concatenate(
            (emoji_image, alpha), axis=2
        )

    EMOJI[name] = emoji_image


# --------------------------------------------------
# 2. Open the webcam
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam. Check camera permissions "
        "or try another camera index."
    )


# --------------------------------------------------
# 3. Prepare variables for smoother predictions
# --------------------------------------------------

frame_number = 0
history = deque(maxlen=8)
face_box = None
miss_count = 0


# --------------------------------------------------
# 4. Process the webcam video
# --------------------------------------------------

try:
    while True:
        ok, frame = cap.read()

        if not ok:
            print("Could not read from webcam.")
            break

        # Mirror the video like a normal selfie camera.
        frame = cv2.flip(frame, 1)
        frame_number += 1

        # Run facial analysis every third frame.
        if frame_number % 3 == 0:
            try:
                result: Any = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    detector_backend="yunet",
                    enforce_detection=True,
                )

                # Normalize the result into a list.
                if isinstance(result, dict):
                    result = [result]

                if result:
                    # If several faces exist, use the largest.
                    largest_face = max(
                        result,
                        key=lambda item:
                            item["region"]["w"]
                            * item["region"]["h"],
                    )

                    region = largest_face["region"]

                    new_box = np.array(
                        [
                            region["x"],
                            region["y"],
                            region["w"],
                            region["h"],
                        ],
                        dtype=float,
                    )

                    # Smooth the face position between frames.
                    if face_box is None:
                        face_box = new_box
                    else:
                        face_box = (
                            0.6 * face_box
                            + 0.4 * new_box
                        )

                    history.append(
                        largest_face["emotion"]
                    )

                    miss_count = 0
                else:
                    miss_count += 1

            except Exception as error:
                miss_count += 1

                # Print the first unexpected error.
                if (
                    "face" not in str(error).lower()
                    and miss_count == 1
                ):
                    print(
                        "Emotion analysis warning:",
                        error,
                    )

        # Clear old predictions if no face is found.
        if miss_count > 10:
            face_box = None
            history.clear()

        # --------------------------------------------------
        # 5. Select an expression and overlay its emoji
        # --------------------------------------------------

        if face_box is not None and history:
            x, y, w, h = map(int, face_box)

            averages = {
                name: float(
                    np.mean(
                        [scores[name] for scores in history]
                    )
                )
                for name in EMOTION_NAMES
            }

            predicted_emotion = max(
                averages, key=averages.get
            )

            if w > 0 and h > 0:
                emoji = cv2.resize(
                    EMOJI[predicted_emotion],
                    (w, h),
                    interpolation=cv2.INTER_AREA,
                )

                frame_height, frame_width = frame.shape[:2]

                # Clip the overlay to the frame boundaries.
                x0 = max(x, 0)
                y0 = max(y, 0)
                x1 = min(x + w, frame_width)
                y1 = min(y + h, frame_height)

                if x0 < x1 and y0 < y1:
                    emoji = emoji[
                        y0 - y:y1 - y,
                        x0 - x:x1 - x,
                    ]

                    # Blend the emoji using its transparency.
                    alpha = (
                        emoji[:, :, 3:4].astype(float)
                        / 255.0
                    )

                    emoji_rgb = emoji[:, :, :3].astype(float)

                    background = frame[
                        y0:y1, x0:x1
                    ].astype(float)

                    frame[y0:y1, x0:x1] = (
                        alpha * emoji_rgb
                        + (1 - alpha) * background
                    ).astype(np.uint8)

        # Display the live video.
        cv2.imshow(
            "Emoji Face - Press q to quit",
            frame,
        )

        # Press q to exit.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Always release the webcam and close OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()
```

### C. Save the file

Press:

**`Ctrl + S`**

**Say:** Before we run the program, check the filename. It must be `emoji_face.py`, not `emoji_face.py.txt`, and it must be inside our project folder.

## Step 3 — Run the program (5 minutes)

### A. Return to the terminal

If you are using VS Code, click **Terminal → New Terminal** in the top menu.

If the terminal is already open, click inside it.

### B. Check that the file exists

Type:

```powershell
dir
```

Press **Enter**.

Look for `emoji_face.py` in the displayed file list.

If it is missing, check whether you saved the file in a different folder or accidentally named it `emoji_face.py.txt`.

### C. Start the application

Type this command exactly:

```powershell
py -3.12 emoji_face.py
```

Press **Enter**.

Wait for the program to start. The first run may take longer because DeepFace may need to download model files. An internet connection is also needed to download the emoji images.

### D. Give camera permission

If Windows displays a camera-permission prompt, allow access. If no prompt appears and the camera does not work, check **Settings → Privacy & security → Camera** and enable camera access for desktop apps.

Look for a window titled:

`Emoji Face - Press q to quit`

Position your face in the camera view and allow the model time to process several frames.

### E. Use the application

- Move your face left and right.
- Try different facial expressions.
- Observe how the emoji changes.
- Move out of view, then return.
- Press the **`q` key while the video window is active** to stop the program.

If the window is active, do not press `Ctrl + C` first. The intended exit key is `q`.

**Expected result:** A webcam window appears, and an emoji is overlaid on the detected face according to the model's expression estimate. Predictions may be delayed or incorrect, and the application may not work with every camera or environment.

## Step 4 — Explain the code (8 minutes)

Do not read the whole script line by line. Show these sections and ask the audience to explain what each does.

### 1. Imports

```python
import cv2
import numpy as np
from collections import deque
from deepface import DeepFace
```

**Say:** These imports give us access to computer vision, numerical operations, a short prediction history, and facial analysis.

**Ask:** Which library is responsible for opening the webcam?

**Answer:** OpenCV, imported as `cv2`.

### 2. The emotion list

```python
EMOTION_NAMES = [
    "angry", "disgust", "fear", "happy",
    "sad", "surprise", "neutral"
]
```

**Say:** DeepFace provides scores for these expression categories. We associate each category with a corresponding emoji.

**Ask:** Why use a dictionary for the emoji images?

**Answer:** We can retrieve an emoji directly by its expression name, such as `EMOJI["happy"]`.

### 3. The webcam

```python
cap = cv2.VideoCapture(0)
```

**Say:** The number `0` usually selects the default webcam. On a system with multiple cameras, another index might be required.

**Ask:** What would happen if the webcam could not be opened?

**Answer:** The program raises an error instead of entering the video loop.

### 4. The prediction history

```python
history = deque(maxlen=8)
```

**Say:** A single prediction can change from frame to frame. We keep a small rolling history and average the scores to make the displayed result less jumpy.

**Ask:** What does `maxlen=8` mean?

**Answer:** The history keeps at most eight entries; older entries are removed as new ones arrive.

### 5. The facial-analysis call

```python
result = DeepFace.analyze(
    frame,
    actions=["emotion"],
    detector_backend="yunet",
    enforce_detection=True,
)
```

**Say:** We send the current video frame to DeepFace and request emotion analysis. YuNet is the selected face detector. This is pretrained inference, not training a new model.

**Ask:** Does this line train the model?

**Answer:** No. It runs inference using a pretrained model.

### 6. Choosing an expression

```python
predicted_emotion = max(
    averages, key=averages.get
)
```

**Say:** We select the expression with the largest average score.

**Ask:** Does the highest score guarantee that the prediction is correct?

**Answer:** No. It only means that this category received the highest score among the available categories.

### 7. Overlaying the emoji

```python
frame[y0:y1, x0:x1] = (
    alpha * emoji_rgb
    + (1 - alpha) * background
).astype(np.uint8)
```

**Say:** This blends the emoji with the original video. The alpha value controls transparency: `1` means the emoji pixel is fully visible, and `0` means the original video pixel remains visible.

**Ask:** Why not simply replace the whole frame with the emoji?

**Answer:** We only want to overlay the emoji in the face region, preserving the rest of the webcam image.

### 8. Exiting safely

```python
if cv2.waitKey(1) & 0xFF == ord("q"):
    break
```

**Say:** OpenCV checks for a key press. If the user presses `q`, the loop stops, and the `finally` block releases the webcam and closes the windows.

## Step 5 — Run a quick interactive challenge (5 minutes)

Divide the room into teams if you have not already done so.

**Say:** We have a working program. Now let's think like ML engineers. The model's output can be noisy, so what changes could improve the user experience?

Give teams one minute to discuss, then take two or three answers.

Possible answers:

1. Make the emoji change less frequently.
2. Keep a longer prediction history.
3. Display the expression name and confidence score.
4. Support multiple faces.
5. Add a button to pause the camera.
6. Let users switch between different emoji styles.

Award one point for a technically sensible answer and ask each team to explain why it might help.

**Optional mini-experiment:** Change:

```python
history = deque(maxlen=8)
```

to:

```python
history = deque(maxlen=15)
```

Save using `Ctrl + S`, stop the running application by pressing `q`, and rerun:

```powershell
py -3.12 emoji_face.py
```

Ask the room whether the predictions appear smoother or respond more slowly. A longer history can smooth fluctuations, but it can also make changes slower to appear.

## Step 6 — Troubleshooting (5 minutes)

### Error A: `No module named 'cv2'`

Run:

```powershell
py -3.12 -m pip install opencv-python
```

Then retry:

```powershell
py -3.12 emoji_face.py
```

### Error B: `No module named 'numpy'`

Run:

```powershell
py -3.12 -m pip install numpy
```

### Error C: `No module named 'deepface'`

Run:

```powershell
py -3.12 -m pip install deepface
```

If installation fails, capture the complete error output. Do not assume that repeating the same command will solve a Python-version or dependency incompatibility.

### Error D: Camera cannot be opened

1. Close other applications that may be using the webcam.
2. Check Windows camera permissions.
3. Confirm that the webcam works in the Windows Camera application.
4. If necessary, try changing:

```python
cap = cv2.VideoCapture(0)
```

to:

```python
cap = cv2.VideoCapture(1)
```

Then save and rerun. Camera index `1` only helps if another camera is available.

### Error E: YuNet detector backend is unavailable

If DeepFace reports that `yunet` is not a supported detector backend, try changing:

```python
detector_backend="yunet"
```

to:

```python
detector_backend="opencv"
```

Save and rerun the script. The OpenCV detector may behave differently.

### Error F: Emoji download fails

Check the internet connection and rerun the program. The script downloads its emoji images when it starts. A timeout or network restriction can prevent this step from completing.

### Error G: The webcam window opens but no emoji appears

1. Face the camera directly.
2. Improve the lighting.
3. Move closer to the camera.
4. Wait several seconds for analysis.
5. Check the terminal for an emotion-analysis warning.

If the error continues, copy the complete terminal traceback. The precise error matters: do not change several parts of the script without identifying the cause.

## Wrap-up (2 minutes)

**Ask:** Which components did we combine to build this application?

Expected answer: Python, OpenCV, NumPy, DeepFace, a webcam, and emoji images.

**Ask:** Did we train the deep-learning model during this workshop?

Expected answer: No. We used a pretrained model to perform inference.

**Say:** This is a practical example of integrating machine learning into an application. The model generates a prediction, and ordinary program logic turns that prediction into something a user can see. In real projects, we must also consider latency, prediction errors, privacy, and responsible use.

**Do:** Ask everyone to name one thing they learned and one improvement they would add. Thank the participants and announce the next RoboVITics ML session.

## Before the workshop — presenter checklist

- [ ] Python 3.12 is installed and `py -3.12 --version` works.
- [ ] All three packages install and import successfully.
- [ ] `emoji_face.py` runs on the actual presentation laptop.
- [ ] Webcam permissions have been checked.
- [ ] The first-run DeepFace model download has completed.
- [ ] The emoji images download successfully.
- [ ] The `q` exit key works.
- [ ] A screenshot or short recording is available as a backup demonstration.
- [ ] The full error output can be captured if something fails.
- [ ] Participants understand that facial-expression estimates are not reliable measurements of a person's internal emotional state.
