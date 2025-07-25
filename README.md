# 💤 Drowsiness Detection using Machine Learning

This project implements a real-time drowsiness detection system using machine learning. The system monitors eye activity from webcam video, computes Eye Aspect Ratio (EAR), and uses rolling statistics as features to train and deploy a classification model that predicts whether a person is drowsy or alert.


## 🧠 Project Overview

- 🧍 Face and eye landmarks extracted using MediaPipe
- 👁️ EAR (Eye Aspect Ratio) computed from landmarks
- 📊 Rolling mean and standard deviation used for smoothing
- 🤖 Machine Learning model (Random Forest) trained to classify drowsiness
- 🔊 Real-time audio alert system triggered upon detection


## 🔧 Technologies & Libraries

- **Python 3**
- **OpenCV** – Webcam input and visualization
- **MediaPipe** – Facial landmark detection
- **Pandas, NumPy** – Feature engineering and data processing
- **Scikit-learn** – ML model training, prediction, and evaluation
- **Joblib** – Model serialization
- **Pygame** – Playing alarm sound
- **Threading & Queue** – Real-time alert handling


## 📁 Folder Structure

Drowsiness-Detection-ML/

├── Main.py # Real-time detection and prediction script

├── train_model.py # Script to train ML model

├── drowsiness_model.pkl # Trained Random Forest model

├── alarm.wav # Audio alert

├── manual_ear_log.csv # Manually labeled training data

├── model_report.txt # Classification report (optional)

├── requirements.txt # Python dependencies

└── README.md # Project documentation


## 🔍 Feature Engineering

Features used to train the ML model:
- **ear**: Real-time eye aspect ratio
- **rolling_mean**: Mean of last N EAR values
- **rolling_std**: Standard deviation of last N EAR values

These features help the model distinguish between short blinks and sustained drowsiness.

## 🚀 Run Real-Time Detection

Ensure drowsiness_model.pkl and alarm.wav are present. Then:
python Main.py

## 📈 Future Improvements

Add more labeled data for generalization
Integrate yawn or head pose detection
Experiment with LSTM or temporal models
Deploy as a web or desktop app
