import librosa
import numpy as np

def extract_feature(file_name, mfcc=True):
    try:
        X, sample_rate = librosa.load(file_name, sr=22050, mono=True)
        print(f"Loaded {file_name} with {len(X)} samples at {sample_rate} Hz.")

        if len(X) < 22050:
            print("Audio too short (< 1s)")
            return None
        if np.allclose(X, 0):
            print("Audio is silent")
            return None

        if mfcc:
            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            print(f"MFCC shape: {mfccs.shape}, mean shape: {mfccs_mean.shape}")
            return mfccs_mean
        else:
            return None
    except Exception as e:
        print(f"Error in extracting features from {file_name}: {e}")
        return None
