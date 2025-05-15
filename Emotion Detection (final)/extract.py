import librosa
import numpy as np
import logging

logger = logging.getLogger(__name__)

def extract_feature(file_name, mfcc=True):
    try:
        logger.debug(f"Loading audio file: {file_name}")
        X, sample_rate = librosa.load(file_name, sr=22050, mono=True)
        logger.debug(f"Loaded {file_name} with {len(X)} samples at {sample_rate} Hz.")

        if len(X) < 22050:
            logger.warning(f"Audio too short (< 1s): {len(X)} samples")
            return None
            
        if np.allclose(X, 0):
            logger.warning("Audio is silent")
            return None

        if mfcc:
            logger.debug("Extracting MFCC features")
            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            
            if np.isnan(mfccs_mean).any():
                logger.error("NaN values detected in extracted features")
                return None
                
            logger.debug(f"MFCC shape: {mfccs.shape}, mean shape: {mfccs_mean.shape}")
            return mfccs_mean
        else:
            logger.warning("No feature extraction method specified")
            return None
    except Exception as e:
        logger.error(f"Error in extracting features from {file_name}: {e}")
        return None
