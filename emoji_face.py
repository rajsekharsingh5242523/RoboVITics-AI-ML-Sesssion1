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
