from flask import Flask, request, render_template
import os
import pickle
import numpy as np
from extract import extract_feature
import librosa

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
with open('svc_pipeline.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        audio = request.files['audio']
        if audio:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], audio.filename)
            audio.save(filepath)

            features = extract_feature(filepath, mfcc=True)
            if features is None or np.isnan(features).any():
                return "Error extracting features. Please upload a valid audio file."

            features = np.array(features).reshape(1, -1)
            prediction = model.predict(features)[0]
            return render_template('index.html', prediction=prediction)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
