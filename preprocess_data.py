import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# Update this path to your actual dataset location
DATASET_PATH = "leapGestRecog"  # Make sure it's correct relative to this script

def load_data():
    X = []
    y = []
    label_names = []

    print("Loading data...")

    for person_folder in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person_folder)
        if not os.path.isdir(person_path):
            continue

        print(f"Processing person folder: {person_folder}")

        for gesture_folder in os.listdir(person_path):
            gesture_path = os.path.join(person_path, gesture_folder)
            if not os.path.isdir(gesture_path):
                continue

            label = gesture_folder.split('_')[1]  # Extract 'palm' from '01_palm'

            if label not in label_names:
                label_names.append(label)

            for img_file in os.listdir(gesture_path):
                if img_file.endswith(".png"):
                    img_path = os.path.join(gesture_path, img_file)
                    try:
                        img = Image.open(img_path).convert("L")  # Grayscale
                        img = img.resize((64, 64))
                        img_array = np.array(img)
                        X.append(img_array.flatten())
                        y.append(label)
                    except Exception as e:
                        print(f"Error reading {img_path}: {e}")

    print(f"Total images collected: {len(X)}")
    print(f"Labels found: {set(y)}")

    # Convert to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Save label mapping for later use
    np.save("label_mapping.npy", label_encoder.classes_)

    # Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    return X_train, X_test, Y_train, Y_test
