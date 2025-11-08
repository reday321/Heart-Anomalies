import wfdb
import pandas as pd
import numpy as np
from scipy import signal as sig  # Rename to avoid conflict
import logging
import os

logger = logging.getLogger(__name__)

def process_ecg_file(file_path, return_raw=False):
    """
    Process ECG file (.dat or .csv) and return normalized signal.
    
    Args:
        file_path: Path to the ECG file
        return_raw: If True, also return the raw signal for visualization
    
    Returns:
        If return_raw=False: processed signal (1, sequence_length, num_features)
        If return_raw=True: tuple (processed_signal, raw_signal)
    """
    file_ext = file_path.split('.')[-1].lower()
    
    logger.info(f"Processing file with extension: {file_ext}")
    
    if file_ext == 'dat':
        result = process_mit_bih_file(file_path, return_raw=return_raw)
    elif file_ext == 'csv':
        result = process_csv_file(file_path, return_raw=return_raw)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    return result

def process_mit_bih_file(file_path, return_raw=False):
    """Process MIT-BIH format .dat file."""
    try:
        logger.info("Reading MIT-BIH file...")
        # wfdb.rdrecord expects path without extension
        # Get the directory and filename without extension
        dir_path = os.path.dirname(file_path)
        filename_with_ext = os.path.basename(file_path)
        filename_without_ext = os.path.splitext(filename_with_ext)[0]
        
        # Check if .hea file exists
        hea_file_path = os.path.join(dir_path, filename_without_ext + '.hea') if dir_path else (filename_without_ext + '.hea')
        has_header = os.path.exists(hea_file_path)
        
        if has_header:
            # Standard MIT-BIH format with header file
            logger.info(f"Header file found: {hea_file_path}")
            # Construct the base path (without .dat extension)
            if dir_path:
                base_path = os.path.join(dir_path, filename_without_ext)
            else:
                base_path = filename_without_ext
                
            logger.info(f"Reading MIT-BIH record from: {base_path}")
            record = wfdb.rdrecord(base_path)
            
            # Extract first channel (preferably the first channel as per requirements)
            if record.p_signal is not None and len(record.p_signal.shape) > 1 and record.p_signal.shape[1] > 0:
                signal = record.p_signal[:, 0]
            elif record.sig is not None and len(record.sig.shape) > 1 and record.sig.shape[1] > 0:
                # Fallback: use sig attribute if p_signal is not available
                signal = record.sig[:, 0]
            else:
                raise ValueError("Could not extract signal from MIT-BIH file. No valid signal channels found.")
        else:
            # No header file - try to read .dat file directly as binary
            logger.warning(f"Header file (.hea) not found. Attempting to read .dat file directly.")
            logger.info(f"Reading binary .dat file: {file_path}")
            signal = read_dat_file_directly(file_path)
        
        raw_signal = signal.copy()  # Store raw signal for visualization
        logger.info(f"Successfully read MIT-BIH file, signal shape: {signal.shape}")
        
        processed = preprocess_signal(signal)
        if return_raw:
            return processed, raw_signal
        return processed
    except Exception as e:
        logger.error(f"Error reading MIT-BIH file: {str(e)}", exc_info=True)
        error_msg = str(e)
        if "No such file or directory" in error_msg and ".hea" in error_msg:
            raise ValueError("MIT-BIH format requires both .dat and .hea files. Please upload both files, or use CSV format instead.")
        raise ValueError(f"Error reading MIT-BIH file: {str(e)}")

def read_dat_file_directly(file_path):
    """
    Read MIT-BIH .dat file directly as binary data (fallback when .hea file is missing).
    Assumes standard MIT-BIH format: 16-bit signed integers, single channel.
    """
    try:
        # Read binary data (16-bit signed integers)
        with open(file_path, 'rb') as f:
            data = np.fromfile(f, dtype=np.int16)
        
        if len(data) == 0:
            raise ValueError("Empty .dat file")
        
        logger.info(f"Read {len(data)} samples from binary .dat file")
        
        # Convert to float for processing
        signal = data.astype(np.float64)
        
        return signal
    except Exception as e:
        raise ValueError(f"Could not read .dat file directly: {str(e)}. MIT-BIH format typically requires both .dat and .hea files.")

def process_csv_file(file_path, return_raw=False):
    """Process CSV file containing ECG data."""
    try:
        logger.info("Reading CSV file...")
        df = pd.read_csv(file_path)
        logger.info(f"CSV file read successfully. Columns: {df.columns.tolist()}")
        logger.info(f"CSV shape: {df.shape}")
        
        # Try to automatically identify the ECG signal column
        signal_col = identify_signal_column(df)
        logger.info(f"Identified signal column: {signal_col}")
        
        signal = df[signal_col].values
        logger.info(f"Extracted signal shape: {signal.shape}")
        
        raw_signal = signal.copy()  # Store raw signal for visualization
        processed = preprocess_signal(signal)
        if return_raw:
            return processed, raw_signal
        return processed
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}", exc_info=True)
        raise ValueError(f"Error reading CSV file: {str(e)}")

def identify_signal_column(df):
    """Identify the column most likely to contain ECG signal."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in CSV")
    
    logger.info(f"Numeric columns found: {numeric_cols.tolist()}")
    
    # Choose column with highest variance
    variances = df[numeric_cols].var()
    logger.info(f"Column variances: {variances.to_dict()}")
    
    selected_col = variances.idxmax()
    logger.info(f"Selected column with highest variance: {selected_col}")
    return selected_col

def extract_features(signal, window_size=50):
    """
    Extract 12 features from ECG signal using sliding window approach.
    Returns features for each time step.
    """
    features = []
    
    for i in range(len(signal)):
        # Define window boundaries
        start = max(0, i - window_size // 2)
        end = min(len(signal), i + window_size // 2)
        window = signal[start:end]
        
        if len(window) < 2:
            window = signal  # Use entire signal if window too small
        
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
            signal[i],                                 # 9. Current value
            np.sum(np.diff(window) > 0) / len(window) if len(window) > 1 else 0,  # 10. Positive slope ratio
            np.abs(np.sum(np.diff(window))) / len(window) if len(window) > 1 else 0,  # 11. Mean absolute difference
            np.sum(np.abs(window)) / len(window)      # 12. Mean absolute value
        ]
        
        features.append(feature_vector)
    
    return np.array(features)

def preprocess_signal(signal_data, target_length=1000, num_features=12):
    """Preprocess ECG signal for model input."""
    logger.info(f"Preprocessing signal of length {len(signal_data)}")
    
    # Remove baseline wander using high-pass filter
    fs = 360  # Typical sampling frequency for ECG
    b, a = sig.butter(3, 0.5/(fs/2), 'high')  # Using sig.butter instead of signal.butter
    filtered_signal = sig.filtfilt(b, a, signal_data)  # Using sig.filtfilt
    
    # Normalize to [-1, 1] range
    normalized_signal = (filtered_signal - np.mean(filtered_signal)) / (np.max(np.abs(filtered_signal)) + 1e-6)
    logger.info("Signal normalized")
    
    # Resample to target length
    if len(normalized_signal) > target_length:
        indices = np.linspace(0, len(normalized_signal)-1, target_length, dtype=int)
        resampled_signal = normalized_signal[indices]
    elif len(normalized_signal) < target_length:
        resampled_signal = np.interp(
            np.linspace(0, len(normalized_signal)-1, target_length),
            np.arange(len(normalized_signal)),
            normalized_signal
        )
    else:
        resampled_signal = normalized_signal
    logger.info(f"Signal resampled to length {target_length}")
    
    # Extract features to get 12 features per time step
    feature_matrix = extract_features(resampled_signal)
    logger.info(f"Extracted features shape: {feature_matrix.shape}")
    
    # Reshape for model input (batch_size, sequence_length, num_features)
    final_signal = np.reshape(feature_matrix, (1, target_length, num_features))
    logger.info(f"Final signal shape: {final_signal.shape}")
    
    return final_signal