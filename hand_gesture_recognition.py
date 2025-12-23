import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

# --- 1. CONFIGURATION ---
# Note: Point this to the folder containing the '00', '01', etc. folders
DATASET_PATH = r"D:\skillcraft4\leapGestRecog" 
IMG_SIZE = 64
EPOCHS = 10
BATCH_SIZE = 32

X = []
y = []
labels = []

print("🔍 Starting data ingestion...")

# --- 2. THE TRIPLE LOOP (Deep Directory Traversal) ---
# Loop 1: Subjects (00, 01, 02...)
for subject in os.listdir(DATASET_PATH):
    subject_path = os.path.join(DATASET_PATH, subject)
    if not os.path.isdir(subject_path):
        continue

    # Loop 2: Gestures (01_palm, 02_l, etc.)
    for gesture in os.listdir(subject_path):
        gesture_path = os.path.join(subject_path, gesture)
        if not os.path.isdir(gesture_path):
            continue

        if gesture not in labels:
            labels.append(gesture)
        
        label_index = labels.index(gesture)

        # Loop 3: Actual Image Files (frame_01.png...)
        for img_name in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img_name)
            
            # Read as Grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Preprocessing: Resize and Append
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label_index)

# --- 3. DATA PREPARATION ---
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = np.array(y)

if len(X) == 0:
    print("❌ ERROR: No images found. Check your DATASET_PATH folders.")
    exit()

print(f"✅ Loaded {len(X)} images across {len(labels)} classes.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 4. CNN ARCHITECTURE ---
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    
    # Feature Extraction
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    # Classification
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Regularization to prevent overfitting
    Dense(len(labels), activation='softmax') # Multi-class output
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- 5. TRAINING (Forward & Backward Pass) ---
print("🚀 Starting training...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test)
)

# --- 6. SAVING ARTIFACTS ---
model.save("hand_gesture_model.h5")
with open("gesture_labels.txt", "w") as f:
    for label in labels:
        f.write(label + "\n")

print("\n✨ SUCCESS! Model saved as 'hand_gesture_model.h5'")
print("📝 Labels saved as 'gesture_labels.txt'")

# --- 7. EVALUATION PLOTS ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()
