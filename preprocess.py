import wfdb
import pandas as pd
import numpy as np
from scipy import signal as sig  # Rename to avoid conflict
import logging

logger = logging.getLogger(__name__)

def process_ecg_file(file_path):
    """Process ECG file (.dat or .csv) and return normalized signal."""
    file_ext = file_path.split('.')[-1].lower()
    
    logger.info(f"Processing file with extension: {file_ext}")
    
    if file_ext == 'dat':
        return process_mit_bih_file(file_path)
    elif file_ext == 'csv':
        return process_csv_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

def process_mit_bih_file(file_path):
    """Process MIT-BIH format .dat file."""
    try:
        logger.info("Reading MIT-BIH file...")
        record = wfdb.rdrecord(file_path.replace('.dat', ''))
        signal = record.p_signal[:, 0]
        logger.info(f"Successfully read MIT-BIH file, signal shape: {signal.shape}")
        return preprocess_signal(signal)
    except Exception as e:
        logger.error(f"Error reading MIT-BIH file: {str(e)}", exc_info=True)
        raise ValueError(f"Error reading MIT-BIH file: {str(e)}")

def process_csv_file(file_path):
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
        
        return preprocess_signal(signal)
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

def preprocess_signal(signal_data, target_length=1000):
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
    
    # Reshape for model input (batch_size, sequence_length, 1)
    final_signal = np.reshape(resampled_signal, (1, target_length, 1))
    logger.info(f"Final signal shape: {final_signal.shape}")
    
    return final_signal