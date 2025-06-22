# ALS Detection Using Voice Signals and Deep Learning

This project presents a deep learning-based system to detect Amyotrophic Lateral Sclerosis (ALS) using voice signals. The goal is to provide a non-invasive, efficient, and early diagnosis tool by analyzing acoustic features from speech recordings.

## 🧠 Overview

ALS is a progressive neurodegenerative disease that affects nerve cells and impairs muscle control, including speech. Since changes in speech are early indicators, this project focuses on:

- Extracting voice signal features using **Parselmouth** (Praat interface).
- Augmenting limited data with **CTGAN** from the SDV library.
- Building ML models (XGBoost) to classify ALS vs healthy individuals.
- Hosting a user-friendly **Flask** web interface for real-time predictions.

## 📁 Dataset

- Public dataset containing 1,224 voice samples from 153 individuals:
  - 102 ALS patients (bulbar & spinal onset)
  - 51 healthy controls
- Audio types include:
  - Sustained vowels: `/a/, /e/, /i/, /o/, /u/`
  - Rapid syllables: `/pa/, /ta/, /ka/`

## 🛠️ Tools & Libraries

- Python  
- Flask  
- CTGAN (`sdv` library)  
- Scikit-learn  
- TensorFlow / Keras  
- Parselmouth (Praat)  
- NumPy, Pandas, Matplotlib

## 🔍 Feature Extraction

Using **Parselmouth**, the following acoustic features were extracted:
- **Jitter** (frequency variation)
- **Shimmer** (amplitude variation)
- **Harmonics-to-Noise Ratio (HNR)**
- **Formant frequencies** (F1, F2, F3)
- **Pitch & Intensity**

## 🧪 Model Training

- Classical models: **XGBoost**, **KNN**, **Random Forest**
- CTGAN used to generate synthetic samples (500–1000) to balance dataset
- Performance Metrics: Accuracy, Precision, Recall, F1-score, AUC-ROC

## 📈 Results

- CTGAN helped boost F1-score and accuracy by 5–10%.
- Consistent and improved predictions across multiple runs.

## 🌐 Web Interface (Flask App)

- Allows users (e.g., doctors) to upload voice files.
- Displays real time prediction (ALS Detected / Not Detected) with confidence score.
- Simple, accessible UI 

