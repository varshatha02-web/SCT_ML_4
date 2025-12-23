import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog

# --- 1. SETUP ---
IMG_SIZE = 64
MODEL_PATH = "hand_gesture_model.h5"
DATASET_PATH = r"D:\skillcraft4\leapGestRecog" # Make sure this path is correct

# Load model and labels
model = load_model(MODEL_PATH)
labels = open("gesture_labels.txt").read().splitlines()

# --- 2. SELECT IMAGE ---
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select one image (or Cancel for random batch)")

def predict_and_plot(img_path, ax):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_input = img_resized.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    
    # Prediction
    prediction = model.predict(img_input, verbose=0)
    class_idx = np.argmax(prediction)
    conf = prediction[0][class_idx] * 100

    # Plotting
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title(f"Pred: {labels[class_idx]}")
    ax[0].axis('off')
    
    ax[1].barh([labels[class_idx]], [conf], color='skyblue')
    ax[1].set_xlim(0, 100)
    ax[1].set_title(f"Confidence: {conf:.2f}%")

# --- 3. EXECUTION ---
if file_path:
    # Mode A: Single Image
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    predict_and_plot(file_path, axes)
else:
    # Mode B: Random Batch (The "Combined" look)
    print("No file selected. Picking 4 random images...")
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    
    # Find all images in the dataset
    all_images = []
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.endswith(".png"):
                all_images.append(os.path.join(root, file))
    
    random_samples = random.sample(all_images, 4)
    for i, img_path in enumerate(random_samples):
        predict_and_plot(img_path, axes[i])

plt.tight_layout()
plt.show()
