import tensorflow as tf
import xgboost as xgb
import numpy as np
import os
import json

def load_models():
    """Load all three models and return them as a dictionary."""
    try:
        models = {
            'dl_model': tf.keras.models.load_model('model/best_dl_model.h5'),
            'backup_model': tf.keras.models.load_model('model/cnn_bilstm_attention_ecg.h5'),
            'xgb_model': None
        }
        
        # Try loading XGBoost model with different approaches
        xgb_model = xgb.Booster({'nthread': 4})
        try:
            # First try loading as JSON format
            xgb_model.load_model('model/xgboost_ecg.json')
        except Exception as e1:
            try:
                # Try loading as UBJ format
                xgb_model.load_model('model/xgboost_ecg.ubj')
            except Exception as e2:
                try:
                    # If both fail, try loading old format and save in new format
                    xgb_model.load_model('model/xgboost_ecg.model')
                    # Save in new format (JSON)
                    xgb_model.save_model('model/xgboost_ecg.json')
                    print("XGBoost model converted to JSON format")
                except Exception as e3:
                    raise RuntimeError(f"Failed to load XGBoost model in any format: {str(e3)}")
        
        models['xgb_model'] = xgb_model
        return models
    except Exception as e:
        raise RuntimeError(f"Error loading models: {str(e)}")

def get_ensemble_prediction(processed_signal, models):
    """
    Get ensemble prediction from all models.
    
    Ensemble formula: ensemble_prob = 0.6 × DL_prob + 0.4 × XGBoost_prob
    Where DL_prob comes from best_dl_model.h5 (primary model).
    
    Args:
        processed_signal: Preprocessed ECG signal of shape (1, sequence_length, 12)
        models: Dictionary containing all three models
    
    Returns:
        Final prediction probability (0.0 to 1.0)
    """
    try:
        # Get prediction from primary deep learning model (best_dl_model.h5)
        dl_pred = models['dl_model'].predict(processed_signal, verbose=0)[0][0]
        
        # Ensure prediction is a scalar (not array)
        if isinstance(dl_pred, np.ndarray):
            dl_pred = float(dl_pred)
        else:
            dl_pred = float(dl_pred)
        
        # For XGBoost, we need to reshape the signal to 1D (flatten)
        xgb_input = xgb.DMatrix(processed_signal.reshape(1, -1))
        xgb_pred = models['xgb_model'].predict(xgb_input)[0]
        
        # Ensure prediction is a scalar
        if isinstance(xgb_pred, np.ndarray):
            xgb_pred = float(xgb_pred)
        else:
            xgb_pred = float(xgb_pred)
        
        # Calculate ensemble prediction
        # 60% weight to deep learning model, 40% to XGBoost
        ensemble_prob = 0.6 * dl_pred + 0.4 * xgb_pred
        
        return ensemble_prob
        
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {str(e)}")