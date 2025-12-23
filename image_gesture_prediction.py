import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog

IMG_SIZE = 64

model = load_model("hand_gesture_model.h5")

with open("gesture_labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

Tk().withdraw()
image_paths = filedialog.askopenfilenames(
    title="Select multiple hand gesture images",
    filetypes=[("Image files", "*.png *.jpg *.jpeg")]
)

if not image_paths:
    raise ValueError("No images selected")

num_images = len(image_paths)
fig, axes = plt.subplots(num_images, 2, figsize=(10, 4 * num_images))

if num_images == 1:
    axes = [axes]

for i, image_path in enumerate(image_paths):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

    predictions = model.predict(img_normalized, verbose=0)
    predicted_index = np.argmax(predictions)
    predicted_label = labels[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    # IMAGE
    axes[i][0].imshow(img_resized, cmap="gray")
    axes[i][0].set_title(f"Prediction: {predicted_label}")
    axes[i][0].axis("off")

    # BAR
    bars = axes[i][1].barh([predicted_label], [confidence])
    axes[i][1].set_xlim(0, 100)
    axes[i][1].set_xlabel("Confidence (%)")
    axes[i][1].set_title("Confidence")

    for bar in bars:
        axes[i][1].text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{confidence:.2f}%",
            va="center",
            fontsize=11,
            fontweight="bold"
        )

plt.tight_layout()
plt.show()






