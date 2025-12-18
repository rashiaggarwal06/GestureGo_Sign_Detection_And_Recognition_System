"""
Generate landmarks (x,y,z for each of 21 hand keypoints = 63 features)
from a directory-based dataset of ASL images.

Folder layout expected:
DATA_DIR/
    A/
       img1.jpg
       img2.jpg
    B/
       ...
"""

import os, glob
import numpy as np
import mediapipe as mp
import cv2
from tqdm import tqdm

DATA_DIR = "/Users/sudo/Documents/GitHub/sign-language/asl_alphabet_train/asl_alphabet_train"  
OUT_FILE = "landmarks_data.npz"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,       
    max_num_hands=1,
    min_detection_confidence=0.5
)

X, y = [], []
class_names = sorted(next(os.walk(DATA_DIR))[1])
print("Detected classes:", class_names)

for label_idx, cls in enumerate(class_names):
    cls_path = os.path.join(DATA_DIR, cls)
    images = glob.glob(os.path.join(cls_path, "*.jpg")) + glob.glob(os.path.join(cls_path, "*.png"))

    for img_path in tqdm(images, desc=f"Processing {cls}"):
        img = cv2.imread(img_path)
        if img is None:  # skip unreadable files
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            # flatten x,y,z for 21 landmarks
            landmark_row = []
            for lm in hand.landmark:
                landmark_row.extend([lm.x, lm.y, lm.z])
            X.append(landmark_row)
            y.append(label_idx)

hands.close()

X = np.array(X)
y = np.array(y)
print("Final landmark dataset shape:", X.shape, y.shape)

np.savez_compressed(OUT_FILE, X=X, y=y)
np.save("landmarks_classes.npy", np.array(class_names))
print(f"Saved to {OUT_FILE}, classes to landmarks_classes.npy")