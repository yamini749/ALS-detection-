from flask import Flask, render_template, request
import os
import pickle
import numpy as np
import librosa

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
with open('als_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the folder categories we expect
FOLDER_CATEGORIES = [
    'phonatic_a', 'phonatic_o', 'phonatic_i', 'rhythmPA',
    'phonatic_u', 'rhythmTA', 'phonatic_e', 'rhythmKA'
]

def allowed_file(filename):
    return '.' in filename and filename.lower().endswith('.wav')

def extract_features(filepath):
    y, sr = librosa.load(filepath, sr=None)

    # Extract 13 MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # Extract Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    # Extract Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel.T, axis=0)

    # Combine: adjust to only 42 features total (e.g., 13 + 12 + 17)
    features = np.hstack([
        mfccs_mean[:13],
        chroma_mean[:12],
        mel_mean[:17]
    ])

    return features.reshape(1, -1)

@app.route('/')
def home():
    return render_template('index.html', categories=FOLDER_CATEGORIES)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if files were uploaded for each category
    missing_categories = []
    for category in FOLDER_CATEGORIES:
        if f'file_{category}' not in request.files:
            missing_categories.append(category)
    
    if missing_categories:
        return render_template('index.html', 
                              categories=FOLDER_CATEGORIES,
                              prediction='Missing files',
                              details=f'Missing files for: {", ".join(missing_categories)}')

    # Process each file and make separate predictions
    results = []
    predictions_count = {"ALS": 0, "No ALS": 0}
    
    try:
        for category in FOLDER_CATEGORIES:
            file = request.files[f'file_{category}']
            
            if file.filename == '':
                return render_template('index.html', 
                                      categories=FOLDER_CATEGORIES,
                                      prediction='Missing files',
                                      details=f'No file selected for {category}')
            
            if not allowed_file(file.filename):
                return render_template('index.html', 
                                      categories=FOLDER_CATEGORIES,
                                      prediction='Invalid file type',
                                      details=f'Invalid file type for {category}. Please upload a .wav file.')
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{category}_{file.filename}")
            file.save(filepath)
            
            # Extract features and predict for this file
            features = extract_features(filepath)
            prediction = model.predict(features)
            file_result = 'No ALS' if prediction[0] == 1 else 'ALS'
            predictions_count[file_result] += 1
            
            # Store result with confidence if available
            confidence = ""
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features)
                conf_value = probabilities[0][1] if prediction[0] == 1 else probabilities[0][0]
                confidence = f" (Confidence: {conf_value:.2f})"
            
            results.append(f"{category}: {file_result}{confidence}")
        
        # Make final decision based on majority vote
        final_result = "ALS DETECTED" if predictions_count["ALS"] >= predictions_count["No ALS"] else "NO ALS DETECTED"
        majority_count = max(predictions_count["ALS"], predictions_count["No ALS"])
        total_count = majority_count + min(predictions_count["ALS"], predictions_count["No ALS"])
        vote_percentage = (majority_count / total_count) * 100
        
        # Format vote info for details
        vote_info = f"Vote: {majority_count}/{total_count} ({vote_percentage:.1f}%)"
        
        # Combine all results for detailed view
        result_details = f"{vote_info}\n\nIndividual Predictions:\n" + "\n".join(results)
            
    except Exception as e:
        final_result = 'ERROR'
        result_details = f'Error during prediction: {str(e)}'

    return render_template('index.html', 
                          categories=FOLDER_CATEGORIES, 
                          prediction=final_result,
                          details=result_details)

if __name__ == '__main__':
    app.run(debug=True)