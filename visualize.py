import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Flask
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def plot_ecg_signal(signal, output_path, title="ECG Signal Analysis"):
    """
    Plot ECG signal with enhanced visualization and annotations.
    
    Args:
        signal: ECG signal array (1D numpy array)
        output_path: Path to save the plot image
        title: Title for the plot
    
    Returns:
        Path to the saved image
    """
    try:
        # Limit signal length for better visualization (show max 10 seconds or entire signal)
        max_samples = min(len(signal), 3600)  # 3600 samples = 10 seconds at 360 Hz
        display_signal = signal[:max_samples]
        
        # Create larger figure with better proportions
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Create time axis (assuming 360 Hz sampling rate)
        time_axis = np.arange(len(display_signal)) / 360.0  # Convert to seconds
        
        # Plot ECG signal with prominent line
        ax.plot(time_axis, display_signal, linewidth=2.0, color='#6610f2', alpha=0.9)
        
        # Fill area under curve for better visibility
        ax.fill_between(time_axis, display_signal, alpha=0.3, color='#6610f2')
        
        # Calculate and display key metrics
        signal_mean = np.mean(display_signal)
        signal_std = np.std(display_signal)
        signal_min = np.min(display_signal)
        signal_max = np.max(display_signal)
        
        # Add horizontal reference line at mean
        ax.axhline(y=signal_mean, color='#dc3545', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {signal_mean:.2f}')
        
        # Style the plot
        ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold', color='#333')
        ax.set_ylabel('Amplitude (mV)', fontsize=14, fontweight='bold', color='#333')
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20, color='#6610f2')
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        ax.set_facecolor('#fafafa')
        
        # Add legend with key metrics
        info_text = f'Samples: {len(display_signal)} | Duration: {time_axis[-1]:.2f}s | Range: [{signal_min:.2f}, {signal_max:.2f}]'
        ax.text(0.5, 0.02, info_text, transform=ax.transAxes, 
                fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add annotation explaining ECG waveform
        annotation_text = "ECG Waveform: Each heartbeat shows as a series of peaks (P, QRS, T waves)"
        ax.text(0.5, 0.98, annotation_text, transform=ax.transAxes, 
                fontsize=11, ha='center', style='italic', color='#666',
                bbox=dict(boxstyle='round', facecolor='#fff3cd', alpha=0.7))
        
        # Add subtle background color
        fig.patch.set_facecolor('white')
        
        # Tight layout for better appearance
        plt.tight_layout()
        
        # Save the plot with higher DPI for better quality
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"ECG plot saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating ECG plot: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to create ECG visualization: {str(e)}")

