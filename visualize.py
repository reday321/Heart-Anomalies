import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Flask
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from scipy import signal as sig

logger = logging.getLogger(__name__)

def plot_ecg_signal(signal, output_path, title="ECG Signal Analysis", fs=360):
    """
    Plot ECG signal with enhanced visualization and annotations.
    
    Args:
        signal: ECG signal array (1D numpy array)
        output_path: Path to save the plot image
        title: Title for the plot
        fs: Sampling frequency in Hz (default 360 for MIT-BIH)
    
    Returns:
        Path to the saved image
    """
    try:
        # Validate input
        if signal is None or len(signal) == 0:
            raise ValueError("Empty or None signal provided")
        
        # Ensure signal is 1D
        signal = np.array(signal).flatten()
        
        # Limit signal length for better visualization (show max 10 seconds or entire signal)
        max_samples = min(len(signal), fs * 10)  # 10 seconds maximum
        display_signal = signal[:max_samples]
        
        # Create larger figure with better proportions
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Create time axis
        time_axis = np.arange(len(display_signal)) / fs  # Convert to seconds
        
        # Plot ECG signal with prominent line
        ax.plot(time_axis, display_signal, linewidth=1.5, color='#2E86AB', alpha=0.9)
        
        # Fill area under curve for better visibility
        ax.fill_between(time_axis, display_signal, alpha=0.3, color='#2E86AB')
        
        # Calculate and display key metrics
        signal_mean = np.mean(display_signal)
        signal_std = np.std(display_signal)
        signal_min = np.min(display_signal)
        signal_max = np.max(display_signal)
        signal_range = signal_max - signal_min
        
        # Add horizontal reference line at mean
        ax.axhline(y=signal_mean, color='#A23B72', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {signal_mean:.2f}')
        
        # Add ±1 standard deviation lines
        ax.axhline(y=signal_mean + signal_std, color='#F18F01', linestyle=':', linewidth=1, alpha=0.5, label=f'+1 STD: {signal_mean + signal_std:.2f}')
        ax.axhline(y=signal_mean - signal_std, color='#F18F01', linestyle=':', linewidth=1, alpha=0.5, label=f'-1 STD: {signal_mean - signal_std:.2f}')
        
        # Style the plot
        ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold', color='#333')
        ax.set_ylabel('Amplitude (mV)', fontsize=14, fontweight='bold', color='#333')
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20, color='#2E86AB')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_facecolor('#f8f9fa')
        
        # Customize spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#666')
        
        # Add legend with key metrics
        info_text = f'Samples: {len(display_signal):,} | Duration: {time_axis[-1]:.2f}s | Range: [{signal_min:.2f}, {signal_max:.2f}] | STD: {signal_std:.2f}'
        ax.text(0.5, 0.02, info_text, transform=ax.transAxes, 
                fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#ddd'))
        
        # Add annotation explaining ECG waveform
        annotation_text = "ECG Waveform Analysis - Displaying cardiac electrical activity"
        ax.text(0.5, 0.98, annotation_text, transform=ax.transAxes, 
                fontsize=12, ha='center', style='italic', color='#555',
                bbox=dict(boxstyle='round', facecolor='#e8f4f8', alpha=0.8, edgecolor='#2E86AB'))
        
        # Add subtle background color
        fig.patch.set_facecolor('white')
        
        # Set x-axis limits properly
        ax.set_xlim(0, time_axis[-1])
        
        # Tight layout for better appearance
        plt.tight_layout()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the plot with higher DPI for better quality
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none', 
                   transparent=False, pad_inches=0.1)
        plt.close()
        
        logger.info(f"ECG plot saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating ECG plot: {str(e)}", exc_info=True)
        # Don't raise exception, return None to allow main process to continue
        return None

def plot_ecg_comparison(original_signal, processed_signal, output_path, fs=360):
    """
    Plot original vs processed ECG signal for comparison.
    
    Args:
        original_signal: Raw ECG signal
        processed_signal: Preprocessed ECG signal
        output_path: Path to save the comparison plot
        fs: Sampling frequency
    
    Returns:
        Path to the saved image
    """
    try:
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Limit to first 5 seconds for clarity
        max_samples = min(len(original_signal), fs * 5)
        time_axis = np.arange(max_samples) / fs
        
        # Plot original signal
        ax1.plot(time_axis, original_signal[:max_samples], linewidth=1.5, color='#6c757d', alpha=0.8)
        ax1.set_title('Original ECG Signal', fontsize=16, fontweight='bold', color='#6c757d')
        ax1.set_ylabel('Amplitude (mV)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#f8f9fa')
        
        # Plot processed signal
        ax2.plot(time_axis, processed_signal[:max_samples], linewidth=1.5, color='#28a745', alpha=0.8)
        ax2.set_title('Processed ECG Signal (Filtered & Normalized)', fontsize=16, fontweight='bold', color='#28a745')
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Amplitude (mV)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"ECG comparison plot saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating ECG comparison plot: {str(e)}")
        return None

def plot_prediction_result(signal, output_path, prediction, probability, fs=360):
    """
    Plot ECG signal with prediction result overlay.
    
    Args:
        signal: ECG signal
        output_path: Output path for the plot
        prediction: Prediction result ("Normal" or "Abnormal")
        probability: Prediction probability
        fs: Sampling frequency
    
    Returns:
        Path to the saved image
    """
    try:
        # Create the base plot
        plot_path = plot_ecg_signal(signal, output_path, 
                                   title=f"ECG Analysis - Prediction: {prediction}", 
                                   fs=fs)
        
        if plot_path:
            # Reopen the plot and add prediction annotation
            fig, ax = plt.subplots(figsize=(16, 6))
            
            # Recreate the signal plot
            max_samples = min(len(signal), fs * 10)
            display_signal = signal[:max_samples]
            time_axis = np.arange(len(display_signal)) / fs
            
            # Choose color based on prediction
            if "Abnormal" in prediction:
                signal_color = '#dc3545'  # Red for abnormal
                bg_color = '#f8d7da'
            else:
                signal_color = '#28a745'  # Green for normal
                bg_color = '#d4edda'
            
            ax.plot(time_axis, display_signal, linewidth=1.5, color=signal_color, alpha=0.9)
            ax.fill_between(time_axis, display_signal, alpha=0.3, color=signal_color)
            
            # Add prediction result box
            prediction_text = f"Prediction: {prediction}\nConfidence: {probability:.1%}"
            ax.text(0.02, 0.98, prediction_text, transform=ax.transAxes, 
                    fontsize=14, fontweight='bold', va='top',
                    bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.9, 
                             edgecolor=signal_color, linewidth=2))
            
            # Style the plot
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('Amplitude (mV)', fontsize=12)
            ax.set_title(f'ECG Signal - {prediction}', fontsize=16, fontweight='bold', color=signal_color)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Prediction result plot saved to: {output_path}")
        
        return plot_path
        
    except Exception as e:
        logger.error(f"Error creating prediction result plot: {str(e)}")
        return plot_ecg_signal(signal, output_path, fs=fs)  # Fallback to basic plot

def create_summary_plot(signals_dict, output_path, fs=360):
    """
    Create a summary plot with multiple ECG signals (useful for multi-lead ECG).
    
    Args:
        signals_dict: Dictionary of {lead_name: signal}
        output_path: Output path for the plot
        fs: Sampling frequency
    
    Returns:
        Path to the saved image
    """
    try:
        n_signals = len(signals_dict)
        if n_signals == 0:
            return None
        
        fig, axes = plt.subplots(n_signals, 1, figsize=(16, 3 * n_signals))
        if n_signals == 1:
            axes = [axes]
        
        colors = plt.cm.Set3(np.linspace(0, 1, n_signals))
        
        for idx, (lead_name, signal) in enumerate(signals_dict.items()):
            max_samples = min(len(signal), fs * 5)
            time_axis = np.arange(max_samples) / fs
            
            axes[idx].plot(time_axis, signal[:max_samples], linewidth=1.2, color=colors[idx])
            axes[idx].set_title(f'Lead {lead_name}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Amplitude (mV)', fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_facecolor('#fafafa')
            
            if idx == n_signals - 1:
                axes[idx].set_xlabel('Time (seconds)', fontsize=12)
        
        plt.suptitle('Multi-Lead ECG Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Summary plot saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating summary plot: {str(e)}")
        return None

# Utility function to clean up old plot files
def cleanup_old_plots(plot_directory, max_age_minutes=60):
    """
    Clean up old plot files to prevent disk space issues.
    
    Args:
        plot_directory: Directory containing plot files
        max_age_minutes: Maximum age of files in minutes
    """
    try:
        import time
        import glob
        
        current_time = time.time()
        max_age_seconds = max_age_minutes * 60
        
        plot_files = glob.glob(os.path.join(plot_directory, "*.png"))
        
        for file_path in plot_files:
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                os.remove(file_path)
                logger.info(f"Cleaned up old plot file: {file_path}")
                
    except Exception as e:
        logger.warning(f"Error cleaning up old plots: {str(e)}")