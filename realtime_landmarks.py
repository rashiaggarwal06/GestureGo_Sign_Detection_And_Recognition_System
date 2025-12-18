#!/usr/bin/env python3
"""
realtime_landmarks.py
- Loads landmarks_model.keras, landmarks_scaler.pkl, landmarks_classes.npy
- Extracts 21 landmarks per hand, applies same normalization as training
- Uses smoothing + confidence threshold
- Processes UNFLIPPED frames for prediction (so orientation matches dataset),
  but shows a MIRRORED preview (toggleable) so UX is natural.
Key controls:
  q -> quit
  m -> toggle mirrored preview
  p -> toggle debug print of probabilities (useful to tune threshold)
"""

import os, joblib, numpy as np, cv2, mediapipe as mp, tensorflow as tf
from collections import deque
import pyttsx3

output_file = open("output_text.txt", "a")
output_file.write("\n")
final_sentence = ""
last_letter = ""

MODEL_PATH = "landmarks_model.keras"
SCALER_PATH = "landmarks_scaler.pkl"
CLASSES_PATH = "landmarks_classes.npy"

SMOOTH_WINDOW = 5
CONF_THRESHOLD = 0.45   # adjust (lower -> more labels, higher -> fewer false positives)
MIRROR_PREVIEW = True

if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"{MODEL_PATH} not found. Train first with train_landmarks.py")

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
classes = np.load(CLASSES_PATH, allow_pickle=True) if os.path.exists(CLASSES_PATH) else None
if classes is None:
    classes = np.array([str(i) for i in range(model.output_shape[-1])], dtype=object)

# If classes look numeric (dtype kind 'i'), convert to letters if size matches
if classes.dtype.kind in ("i", "u") or all(s.isdigit() for s in classes.astype(str)):
    # try map to alphabet if counts match known sizes
    n = len(classes)
    if n == 29:
        classes = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del", "nothing"], dtype=object)
    elif n == 26:
        classes = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), dtype=object)
    else:
        classes = np.array([str(i) for i in range(n)], dtype=object)

print("Loaded model:", MODEL_PATH)
print("Loaded scaler:", bool(scaler))
print("Loaded classes (sample):", classes[:min(10, len(classes))])

# helpers
def normalize_row_from_landmarks(lm_list):
    arr = np.array([[p.x, p.y, p.z] for p in lm_list], dtype=np.float32)  # (21,3)
    wrist = arr[0].copy()
    arr = arr - wrist
    span_x = arr[:,0].max() - arr[:,0].min()
    span_y = arr[:,1].max() - arr[:,1].min()
    span = max(span_x, span_y)
    if span <= 1e-6:
        span = 1.0
    arr[:,0] /= span
    arr[:,1] /= span
    arr[:,2] /= span
    return arr.flatten()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam")

pred_buffer = deque(maxlen=SMOOTH_WINDOW)
mirror_preview = MIRROR_PREVIEW
debug = False

print("Realtime landmarks: q=quit | m=mirror toggle | p=debug probs toggle")

tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)   # speed of speech
tts_engine.setProperty('volume', 1.0) # volume (0.0 to 1.0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Process the UNFLIPPED frame (to match dataset orientation)
    proc = frame.copy()
    rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    display_label = "No Hand"
    display_prob = 0.0

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        # draw landmarks on proc (so preview also shows them)
        mp_draw.draw_landmarks(proc, lm, mp_hands.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                               mp_draw.DrawingSpec(color=(0,0,255), thickness=2))

        feat = normalize_row_from_landmarks(lm.landmark)
        if scaler is not None:
            feat_s = scaler.transform(feat.reshape(1,-1))
        else:
            feat_s = feat.reshape(1,-1)

        probs = model.predict(feat_s, verbose=0)[0]
        pred_buffer.append(probs)
        avg = np.mean(pred_buffer, axis=0)
        idx = int(np.argmax(avg))
        display_prob = float(avg[idx])
        if display_prob >= CONF_THRESHOLD:
            display_label = str(classes[idx])
        else:
            display_label = "Nothing"

        if display_label != "Nothing" and display_label != last_letter:
            final_sentence += display_label
            output_file.write(display_label)
            output_file.flush()
            last_letter = display_label
            
            # --- TTS --- 
            tts_engine.say(display_label) 
            tts_engine.runAndWait()

        # draw bounding box (landmark-derived)
        h, w = proc.shape[:2]
        xs = [int(p.x * w) for p in lm.landmark]
        ys = [int(p.y * h) for p in lm.landmark]
        x1, x2 = max(min(xs)-20, 0), min(max(xs)+20, w)
        y1, y2 = max(min(ys)-20, 0), min(max(ys)+20, h)
        cv2.rectangle(proc, (x1,y1), (x2,y2), (0,255,0), 2)

        if debug:
            # print top 5 candidates
            top5 = np.argsort(avg)[-5:][::-1]
            print("Top5:", [(classes[i], float(avg[i])) for i in top5])

    # Mirror preview for user (optional)
    preview = cv2.flip(proc, 1) if mirror_preview else proc

    cv2.putText(preview, f"{display_label} ({display_prob*100:.1f}%)",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    
    cv2.putText(preview, f"Sentence: {final_sentence}",
            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("ASL Landmarks Realtime (q to quit)", preview)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        mirror_preview = not mirror_preview
    elif key == ord('p'):
        debug = not debug
    elif key == ord('s'):
        with open("output.txt", "w") as f:
            f.write(final_sentence)
        print("Sentence saved to output.txt")
        
        # Speak full sentence
        tts_engine.say(final_sentence)
        tts_engine.runAndWait()
output_file.close()
cap.release()
cv2.destroyAllWindows()