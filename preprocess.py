import wfdb
import pandas as pd
import numpy as np
from scipy import signal as sig
import logging
import os

logger = logging.getLogger(__name__)

def process_ecg_file(file_path, return_raw=False, target_length=5000):
    """
    Process ECG file (.dat or .csv) and return normalized signal.
    
    Args:
        file_path: Path to the ECG file
        return_raw: If True, also return the raw signal for visualization
        target_length: Target length for the processed signal
    
    Returns:
        If return_raw=False: processed signal (1, sequence_length, 1)
        If return_raw=True: tuple (processed_signal, raw_signal)
    """
    file_ext = os.path.splitext(file_path)[-1].lower().replace('.', '')
    
    logger.info(f"Processing file: {file_path} with extension: {file_ext}")
    
    if file_ext == 'dat':
        result = process_mit_bih_file(file_path, return_raw=return_raw, target_length=target_length)
    elif file_ext == 'csv':
        result = process_csv_file(file_path, return_raw=return_raw, target_length=target_length)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Please upload .dat or .csv files.")
    
    return result

def process_mit_bih_file(file_path, return_raw=False, target_length=5000):
    """Process MIT-BIH format .dat file."""
    try:
        logger.info("Reading MIT-BIH file...")
        
        # Get the base path without extension
        base_path = os.path.splitext(file_path)[0]
        
        # Check if .hea file exists
        hea_file_path = base_path + '.hea'
        if not os.path.exists(hea_file_path):
            logger.warning(f"Header file not found: {hea_file_path}")
            # Try alternative: check if .hea exists in the same directory with different case
            hea_files = [f for f in os.listdir(os.path.dirname(file_path)) if f.lower().endswith('.hea')]
            if hea_files:
                hea_file_path = os.path.join(os.path.dirname(file_path), hea_files[0])
                base_path = os.path.splitext(hea_file_path)[0]
                logger.info(f"Using alternative header file: {hea_file_path}")
            else:
                # Fallback to direct binary reading
                logger.info("No header file found, attempting direct binary read")
                signal = read_dat_file_directly(file_path)
                raw_signal = signal.copy()
                processed = preprocess_signal(signal, target_length=target_length)
                if return_raw:
                    return processed, raw_signal
                return processed
        
        logger.info(f"Reading MIT-BIH record from: {base_path}")
        record = wfdb.rdrecord(base_path)
        
        # Extract signal from first channel
        if record.p_signal is not None:
            if len(record.p_signal.shape) > 1:
                signal = record.p_signal[:, 0]  # First channel
            else:
                signal = record.p_signal  # Single channel
        else:
            raise ValueError("No signal data found in MIT-BIH file")
        
        logger.info(f"MIT-BIH file loaded: {len(signal)} samples, {record.n_sig} channels")
        logger.info(f"Sampling frequency: {record.fs} Hz")
        
        raw_signal = signal.copy()
        processed = preprocess_signal(signal, target_length=target_length)
        
        if return_raw:
            return processed, raw_signal
        return processed
        
    except Exception as e:
        logger.error(f"Error processing MIT-BIH file: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to process MIT-BIH file: {str(e)}")

def read_dat_file_directly(file_path):
    """
    Read MIT-BIH .dat file directly as binary data (fallback when .hea file is missing).
    Assumes standard MIT-BIH format: 16-bit signed integers.
    """
    try:
        # Get file size to estimate number of samples (2 bytes per sample for 16-bit)
        file_size = os.path.getsize(file_path)
        expected_samples = file_size // 2
        
        logger.info(f"Reading binary .dat file: {file_path}, size: {file_size} bytes, expected samples: {expected_samples}")
        
        # Read as 16-bit signed integers
        signal = np.fromfile(file_path, dtype=np.int16)
        
        if len(signal) == 0:
            raise ValueError("Empty .dat file")
        
        # Convert to float for processing
        signal = signal.astype(np.float64)
        
        logger.info(f"Successfully read {len(signal)} samples from binary .dat file")
        return signal
        
    except Exception as e:
        raise ValueError(f"Could not read .dat file directly: {str(e)}")

def process_csv_file(file_path, return_raw=False, target_length=5000):
    """Process CSV file containing ECG data."""
    try:
        logger.info("Reading CSV file...")
        df = pd.read_csv(file_path)
        
        logger.info(f"CSV file loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Identify ECG signal column
        signal_col = identify_signal_column(df)
        logger.info(f"Selected signal column: '{signal_col}'")
        
        # Extract signal and handle missing values
        signal = df[signal_col].values
        
        # Remove NaN values
        original_length = len(signal)
        signal = signal[~np.isnan(signal)]
        if len(signal) < original_length:
            logger.warning(f"Removed {original_length - len(signal)} NaN values from signal")
        
        if len(signal) == 0:
            raise ValueError("No valid signal data found after removing NaN values")
        
        logger.info(f"Extracted signal length: {len(signal)}")
        
        raw_signal = signal.copy()
        processed = preprocess_signal(signal, target_length=target_length)
        
        if return_raw:
            return processed, raw_signal
        return processed
        
    except Exception as e:
        logger.error(f"Error processing CSV file: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to process CSV file: {str(e)}")

def identify_signal_column(df):
    """Identify the column most likely to contain ECG signal."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in CSV file")
    
    logger.info(f"Available numeric columns: {numeric_cols}")
    
    # Priority list of common ECG column names
    ecg_keywords = [
        'ecg', 'signal', 'lead', 'channel', 'voltage', 'mv', 'v1', 'v2', 'v3', 
        'v4', 'v5', 'v6', 'i', 'ii', 'iii', 'avr', 'avl', 'avf'
    ]
    
    # First, try to find columns with ECG-related names
    for col in df.columns:
        col_lower = col.lower()
        for keyword in ecg_keywords:
            if keyword in col_lower:
                logger.info(f"Found ECG-related column: '{col}'")
                return col
    
    # If no ECG-specific columns found, use the numeric column with highest variance
    variances = {}
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 1:
            variances[col] = np.var(col_data)
    
    if not variances:
        raise ValueError("No valid numeric columns with variance")
    
    selected_col = max(variances, key=variances.get)
    logger.info(f"Selected column '{selected_col}' with highest variance: {variances[selected_col]:.4f}")
    
    return selected_col
def preprocess_signal(signal_data, target_length=1000, sampling_freq=360):
    """
    Preprocess ECG signal for model input.
    Returns signal shaped as (1, 1000, 12) to match model expectations.
    """
    try:
        logger.info(f"Starting preprocessing: input length={len(signal_data)}, target_length={target_length}")
        
        # Ensure signal is 1D
        if len(signal_data.shape) > 1:
            logger.warning(f"Signal has shape {signal_data.shape}, flattening to 1D")
            signal_data = signal_data.flatten()
        
        # Remove any remaining NaN or Inf values
        signal_data = signal_data[np.isfinite(signal_data)]
        
        if len(signal_data) < 10:
            raise ValueError(f"Signal too short after cleaning: {len(signal_data)} samples")
        
        # 1. Remove baseline wander using high-pass filter
        if len(signal_data) > 6:  # Need minimum length for filter
            try:
                nyquist = sampling_freq / 2
                high_cutoff = 0.5  # Hz
                
                # Design high-pass filter
                b, a = sig.butter(3, high_cutoff/nyquist, 'high')
                
                # Apply filter with padding to handle edge effects
                filtered_signal = sig.filtfilt(b, a, signal_data)
                logger.info("Baseline wander removal completed")
            except Exception as filter_error:
                logger.warning(f"Filtering failed: {filter_error}, using original signal")
                filtered_signal = signal_data
        else:
            filtered_signal = signal_data
        
        # 2. Normalize the signal to zero mean and unit variance
        if np.std(filtered_signal) > 0:
            normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
        else:
            normalized_signal = filtered_signal - np.mean(filtered_signal)
            logger.warning("Signal has zero variance, only mean normalization applied")
        
        logger.info(f"Normalization completed: mean={np.mean(normalized_signal):.4f}, std={np.std(normalized_signal):.4f}")
        
        # 3. Resize to target length (1000 samples)
        current_length = len(normalized_signal)
        
        if current_length > target_length:
            # Truncate to target length (take middle section for stability)
            start_idx = (current_length - target_length) // 2
            resized_signal = normalized_signal[start_idx:start_idx + target_length]
            logger.info(f"Signal truncated from {current_length} to {target_length} samples")
            
        elif current_length < target_length:
            # Pad with zeros to target length
            pad_length = target_length - current_length
            resized_signal = np.pad(normalized_signal, (0, pad_length), mode='constant')
            logger.info(f"Signal padded from {current_length} to {target_length} samples")
            
        else:
            resized_signal = normalized_signal
            logger.info(f"Signal already at target length: {target_length}")
        
        # 4. Extract 12 features to create (1000, 12) shape
        feature_matrix = extract_features(resized_signal)
        
        # 5. Reshape for model input: (1, 1000, 12)
        final_signal = feature_matrix.reshape(1, target_length, -1).astype(np.float32)
        
        logger.info(f"Final preprocessed signal shape: {final_signal.shape}")
        logger.info(f"Final signal range: [{np.min(final_signal):.4f}, {np.max(final_signal):.4f}]")
        
        return final_signal
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}", exc_info=True)
        raise ValueError(f"Signal preprocessing failed: {str(e)}")

# Alternative preprocessing for models expecting 12 features
def preprocess_signal_with_features(signal_data, target_length=1000, sampling_freq=360):
    """
    Alternative preprocessing that extracts 12 features per time step.
    Use this if your models expect (1, sequence_length, 12) input.
    """
    try:
        # First preprocess as single channel
        single_channel = preprocess_signal(signal_data, target_length, sampling_freq)
        single_channel = single_channel[0, :, 0]  # Remove batch dimension
        
        # Extract features using sliding window
        features = extract_features(single_channel)
        
        # Reshape for model: (1, sequence_length, num_features)
        final_signal = features.reshape(1, target_length, -1)
        
        logger.info(f"Feature-based preprocessing completed: {final_signal.shape}")
        return final_signal
        
    except Exception as e:
        logger.error(f"Error in feature-based preprocessing: {str(e)}")
        raise

def extract_features(signal, window_size=50):
    """
    Extract multiple features from ECG signal using sliding window approach.
    Returns feature matrix of shape (len(signal), num_features)
    """
    features = []
    
    for i in range(len(signal)):
        # Define window boundaries
        start = max(0, i - window_size // 2)
        end = min(len(signal), i + window_size // 2)
        window = signal[start:end]
        
        if len(window) < 2:
            # If window too small, use global statistics
            window = signal
        
        # Extract 12 features
        feature_vector = [
            np.mean(window),                          # 1. Mean
            np.std(window),                           # 2. Standard deviation
            np.var(window),                           # 3. Variance
            np.min(window),                           # 4. Minimum
            np.max(window),                           # 5. Maximum
            np.median(window),                        # 6. Median
            np.percentile(window, 25),                # 7. 25th percentile
            np.percentile(window, 75),                # 8. 75th percentile
            signal[i],                                # 9. Current value
            np.sum(np.diff(window) > 0) / len(window) if len(window) > 1 else 0,  # 10. Positive slope ratio
            np.abs(np.sum(np.diff(window))) / len(window) if len(window) > 1 else 0,  # 11. Mean absolute difference
            np.sum(np.abs(window)) / len(window)      # 12. Mean absolute value
        ]
        
        features.append(feature_vector)
    
    return np.array(features)