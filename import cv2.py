import cv2
import numpy as np
import urllib.request
from collections import deque
from typing import Any
from deepface import DeepFace

names = [
    "angry", "disgust", "fear", "happy",
    "sad", "surprise", "neutral"
]

codes = "1f620 1f922 1f628 1f60a 1f622 1f62e 1f610".split()

EMOJI = {}

for name, code in zip(names, codes):
    url = (
        "https://raw.githubusercontent.com/twitter/twemoji/"
        f"v14.0.2/assets/72x72/{code}.png"
    )
    with urllib.request.urlopen(url, timeout=10) as response:
        image_bytes = response.read()


    image = cv2.imdecode(
        np.frombuffer(image_bytes, dtype=np.uint8),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        raise RuntimeError(f"Could not decode emoji: {name}")

    EMOJI[name] = image

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open the webcam. Check camera permissions."
    )

n = 0
hist = deque(maxlen=8)
box = None
miss = 0

try:
    while True:
        ok, f = cap.read()

        if not ok:
            print("Could not read a camera frame.")
            break

        f = cv2.flip(f, 1)
        n += 1

        if n % 3 == 0:

            try:
                res: Any = DeepFace.analyze(
                    f,
                    actions=["emotion"],
                    detector_backend="yunet"
                )

                if isinstance(res, dict):
                    res = [res]

                faces = [
                    item for item in res
                    if isinstance(item, dict)
                    and "region" in item
                    and "emotion" in item
                ]

                if not faces:
                    raise ValueError("No valid face result was returned.")
                r = max(
                    faces,
                    key=lambda item: (
                        item["region"]["w"] * item["region"]["h"]
                    )
                )
                g = r["region"]
                new = np.array(
                    [g["x"], g["y"], g["w"], g["h"]],
                    dtype=float
                )
                box = (
                    new if box is None
                    else 0.6 * box + 0.4 * new
                )
                hist.append(r["emotion"])
                miss = 0

            except Exception as e:
                miss += 1
                if miss == 1:
                    print("Face analysis failed:", e)

        if miss > 10:
            box = None
            hist.clear()

        if box is not None and hist:

            x, y, w, h = map(int, box)

            if w > 0 and h > 0:

                avg = {
                    k: np.mean([scores[k] for scores in hist])
                    for k in names
                }

                emotion_name = max(avg, key=lambda k: avg[k])

                em = cv2.resize(EMOJI[emotion_name], (w, h))

                x0 = max(x, 0)
                y0 = max(y, 0)
                x1 = min(x + w, f.shape[1])
                y1 = min(y + h, f.shape[0])

                if x1 > x0 and y1 > y0:
                    em = em[
                        y0 - y:y1 - y,
                        x0 - x:x1 - x
                    ]

                    emoji_color = em[..., :3].astype(np.float32)

                    if em.ndim == 3 and em.shape[2] >= 4:
                        alpha = (
                            em[..., 3:4].astype(np.float32) / 255.0
                        )
                    else:
                        alpha = np.ones(
                            (em.shape[0], em.shape[1], 1),
                            dtype=np.float32
                        )

                    
                    roi = f[y0:y1, x0:x1].astype(np.float32)

                    blended = (
                        alpha * emoji_color
                        + (1.0 - alpha) * roi
                    )

                    f[y0:y1, x0:x1] = np.clip(
                        blended, 0, 255
                    ).astype(np.uint8)

        cv2.imshow("Emoji Face (q to quit)", f)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()