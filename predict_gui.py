import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import joblib
from PIL import Image, ImageTk

# Load models
model = joblib.load("gesture_model.pkl")
pca = joblib.load("pca_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Image size used during training
IMG_SIZE = 64

# Manually map numeric labels to gesture names
label_map = {
    0: 'c',
    1: 'down',
    2: 'fist',
    3: 'index',
    4: 'l',
    5: 'palm',
    6: 'thumb'
}

# GUI setup
root = tk.Tk()
root.title("Hand Gesture Recognition")

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            # Load and preprocess image
            img = cv2.imread(file_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            flattened = gray.flatten().reshape(1, -1)

            # Apply PCA
            X_pca = pca.transform(flattened)

            # Predict using SVM
            prediction = model.predict(X_pca)
            pred_label = prediction[0]

            # Map predicted numeric label to string name
            label = label_map.get(pred_label, f"Unknown (label {pred_label})")

            # Display prediction
            messagebox.showinfo("Prediction", f"Predicted Gesture: {label}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

# UI Components
btn = tk.Button(root, text="Open Image", command=open_image, font=("Arial", 14))
btn.pack(pady=20)

root.mainloop()
