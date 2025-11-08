import tensorflow as tf
import xgboost as xgb
import numpy as np
import os
import json
import logging

# Set up logging
logger = logging.getLogger(__name__)

def load_models():
    """Load all three models and return them as a dictionary."""
    try:
        models = {}
        
        # Load primary deep learning model
        if os.path.exists('model/best_dl_model.h5'):
            models['dl_model'] = tf.keras.models.load_model('model/best_dl_model.h5')
            logger.info("Primary DL model loaded successfully")
        else:
            raise FileNotFoundError("Primary DL model not found at 'model/best_dl_model.h5'")
        
        # Load backup deep learning model
        if os.path.exists('model/cnn_bilstm_attention_ecg.h5'):
            models['backup_model'] = tf.keras.models.load_model('model/cnn_bilstm_attention_ecg.h5')
            logger.info("Backup DL model loaded successfully")
        else:
            logger.warning("Backup DL model not found, using primary model as backup")
            models['backup_model'] = models['dl_model']
        
        # Load XGBoost model with multiple format attempts
        models['xgb_model'] = load_xgboost_model()
        
        logger.info("All models loaded successfully")
        return models
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise RuntimeError(f"Error loading models: {str(e)}")

def load_xgboost_model():
    """Load XGBoost model trying different formats and locations."""
    xgb_model_paths = [
        'model/xgboost_ecg.model',
        'model/xgboost_ecg.json',
        'model/xgboost_ecg.ubj',
        'xgboost_ecg.model',
        'xgboost_ecg.json',
        'xgboost_ecg.ubj'
    ]
    
    for model_path in xgb_model_paths:
        if os.path.exists(model_path):
            try:
                xgb_model = xgb.Booster({'nthread': 4})
                xgb_model.load_model(model_path)
                logger.info(f"XGBoost model loaded successfully from: {model_path}")
                return xgb_model
            except Exception as e:
                logger.warning(f"Failed to load XGBoost model from {model_path}: {str(e)}")
                continue
    
    # If all attempts fail, try one more approach with error details
    try:
        xgb_model = xgb.Booster()
        # Try loading from any available file
        for model_path in xgb_model_paths:
            if os.path.exists(model_path):
                xgb_model.load_model(model_path)
                logger.info(f"XGBoost model loaded with default parameters from: {model_path}")
                return xgb_model
    except Exception as e:
        raise RuntimeError(f"Failed to load XGBoost model from any available path. Error: {str(e)}")
    
    raise RuntimeError("XGBoost model not found in any expected location")

def get_ensemble_prediction(processed_signal, models):
    """
    Get ensemble prediction from all models.
    
    Ensemble formula: ensemble_prob = 0.6 × DL_prob + 0.4 × XGBoost_prob
    Where DL_prob comes from best_dl_model.h5 (primary model).
    
    Args:
        processed_signal: Preprocessed ECG signal
        models: Dictionary containing all three models
    
    Returns:
        ensemble_prob: Final ensemble probability (0.0 to 1.0)
        dl_prob: Deep learning model probability
        xgb_prob: XGBoost model probability
        final_prediction: Binary prediction (0 or 1)
    """
    try:
        # Validate input shape
        if processed_signal is None or processed_signal.size == 0:
            raise ValueError("Processed signal is empty or None")
        
        logger.info(f"Input signal shape: {processed_signal.shape}")
        
        # Get prediction from primary deep learning model
        dl_pred = models['dl_model'].predict(processed_signal, verbose=0)
        logger.info(f"Raw DL prediction shape: {dl_pred.shape}")
        
        # Extract probability (assuming binary classification with sigmoid output)
        if len(dl_pred.shape) > 1:
            dl_prob = float(dl_pred[0][0])  # For (1, 1) shape
        else:
            dl_prob = float(dl_pred[0])     # For (1,) shape
        
        logger.info(f"DL model probability: {dl_prob:.6f}")
        
        # Prepare input for XGBoost
        # Flatten the signal while preserving batch dimension
        if len(processed_signal.shape) == 3:
            # Shape: (batch, timesteps, channels) -> (batch, timesteps * channels)
            xgb_input = processed_signal.reshape(processed_signal.shape[0], -1)
        else:
            xgb_input = processed_signal.reshape(1, -1)
        
        logger.info(f"XGBoost input shape: {xgb_input.shape}")
        
        # Get XGBoost prediction
        xgb_dmatrix = xgb.DMatrix(xgb_input)
        xgb_pred = models['xgb_model'].predict(xgb_dmatrix)
        
        # Extract XGBoost probability
        if len(xgb_pred.shape) > 0:
            xgb_prob = float(xgb_pred[0])
        else:
            xgb_prob = float(xgb_pred)
        
        logger.info(f"XGBoost probability: {xgb_prob:.6f}")
        
        # Validate probabilities are within expected range
        if not (0 <= dl_prob <= 1):
            logger.warning(f"DL probability out of range: {dl_prob}, clipping to [0,1]")
            dl_prob = max(0.0, min(1.0, dl_prob))
        
        if not (0 <= xgb_prob <= 1):
            logger.warning(f"XGBoost probability out of range: {xgb_prob}, clipping to [0,1]")
            xgb_prob = max(0.0, min(1.0, xgb_prob))
        
        # Calculate ensemble prediction
        # 60% weight to deep learning model, 40% to XGBoost
        ensemble_prob = 0.6 * dl_prob + 0.4 * xgb_prob
        ensemble_prob = max(0.0, min(1.0, ensemble_prob))  # Ensure within bounds
        
        # Final binary prediction using threshold = 0.5
        final_prediction = 1 if ensemble_prob > 0.5 else 0
        
        logger.info(f"Ensemble probability: {ensemble_prob:.6f}, Final prediction: {final_prediction}")
        
        return ensemble_prob, dl_prob, xgb_prob, final_prediction
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise RuntimeError(f"Prediction error: {str(e)}")

def validate_model_compatibility(models, input_shape):
    """
    Validate that models are compatible with the input shape.
    """
    try:
        # Test DL model with dummy input
        dummy_input = np.random.random(input_shape).astype(np.float32)
        
        # Test primary DL model
        dl_output = models['dl_model'].predict(dummy_input, verbose=0)
        logger.info(f"DL model test - Input: {dummy_input.shape}, Output: {dl_output.shape}")
        
        # Test XGBoost model
        xgb_input = dummy_input.reshape(dummy_input.shape[0], -1)
        xgb_dmatrix = xgb.DMatrix(xgb_input)
        xgb_output = models['xgb_model'].predict(xgb_dmatrix)
        logger.info(f"XGBoost model test - Input: {xgb_input.shape}, Output: {xgb_output.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model compatibility check failed: {str(e)}")
        return False