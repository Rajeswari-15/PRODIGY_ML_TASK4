# 🤖 Hand Gesture Recognition App — Task 4 (Prodigy InfoTech)

This is **Task 4** of my internship with **Prodigy InfoTech**, where I developed a machine learning-powered **Hand Gesture Recognition App** using image data and a trained classification model.

The app allows users to upload an image of a hand gesture and predicts the gesture class using a trained machine learning model. It features a user-friendly GUI to make the experience simple and intuitive.

---

## 🧠 Tech Stack / Tools Used

- **Python**
- **OpenCV** – for image preprocessing
- **Scikit-learn** – for PCA and SVM
- **Tkinter** – for GUI interface
- **NumPy & joblib** – for data processing and model persistence
- **PIL (Pillow)** – for image display in GUI

---

## 📁 Dataset

I used the **[LeapGestRecog Dataset](https://www.kaggle.com/gti-upm/leapgestrecog)** from Kaggle, which contains thousands of grayscale images of various hand gestures such as:

- Palm  
- Fist  
- Thumb  
- Index  
- 'L'  
- 'C'  
- Down  

> ⚠️ The dataset is **not included** in this repository due to its large size.  
> I downloaded it directly from the Kaggle link above and used it for local training and testing.

---

## 📂 Project Structure

To maintain modularity and clarity, I’ve split the work into three main parts:

- **Data Preprocessing** → Refer to [`preprocess_data.py`](./preprocess_data.py)  
   This script loads and processes raw gesture images, converts them to grayscale, resizes, flattens, and prepares the data for training.

- **Model Training** → Refer to [`train_model.py`](./train_model.py)  
   This file trains a Support Vector Machine (SVM) classifier with PCA for dimensionality reduction and saves the trained model.

- **Prediction / GUI Interface** → Refer to [`prediction_gui.py`](./prediction_gui.py)  
   This GUI allows users to upload gesture images, processes them, and predicts the gesture using the saved model.

---

## ✅ Features

- Upload a hand gesture image via a GUI
- Image is resized, converted to grayscale, and flattened
- PCA reduces dimensionality
- SVM predicts the gesture class
- Output displayed through a message box

---

## 🖥️ How to Run the App

1. Clone this repo  
   git clone https://github.com/Rajeswari-15/gesture-recognition-task4.git
   cd gesture-recognition-task4

2. Install required packages

   pip install -r requirements.txt


3. Ensure the following files are present in your directory:

   * `gesture_model.pkl`
   * `pca_model.pkl`
   * `label_encoder.pkl`

4. Launch the GUI
   python prediction_gui.py
  

---

## 🎥 Output Demonstration

The working output of this application is available on my **[LinkedIn post](https://www.linkedin.com/posts/yada-rajeshwari-022b8530b_machinelearning-gesturerecognition-python-activity-7334517317579522049-_4ZG?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE8EndoBu65vprjIb-hNbUtHSPP2hiW1WU8)**.
It shows how the app takes a hand gesture image and accurately predicts the gesture using the model.

---

## 🙌 Acknowledgements

This project was completed as part of my internship with **[@Prodigy InfoTech](https://prodigyinfotech.dev/)**.
Grateful for the opportunity and the learning experience!
