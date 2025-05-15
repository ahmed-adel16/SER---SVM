from flask import Flask, request, render_template
import os
import pickle
import numpy as np
from extract import extract_feature
import librosa
import uuid
import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
with open('svc_pipeline.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        logger.debug("Received POST request")
        
        if 'audio' not in request.files:
            logger.error("No audio file in request")
            return render_template('index.html', prediction="Error: No audio file provided")
            
        audio = request.files['audio']
        
        if audio.filename == '':
            logger.error("Empty audio filename")
            return render_template('index.html', prediction="Error: No audio file selected")
            
        logger.debug(f"Processing audio file: {audio.filename}")
        
        # Generate a unique filename
        filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}.wav"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            audio.save(filepath)
            logger.debug(f"Audio saved to {filepath}")
            
            # Extract features
            features = extract_feature(filepath, mfcc=True)
            
            if features is None:
                logger.error("Failed to extract features - None returned")
                return render_template('index.html', prediction="Error: Could not extract features from audio")
                
            if np.isnan(features).any():
                logger.error("NaN values in extracted features")
                return render_template('index.html', prediction="Error: Invalid audio features detected")
            
            # Reshape for prediction
            features = np.array(features).reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(features)[0]
            logger.debug(f"Prediction result: {prediction}")
            
            return render_template('index.html', prediction=prediction)
            
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            return render_template('index.html', prediction=f"Error: {str(e)}")

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
