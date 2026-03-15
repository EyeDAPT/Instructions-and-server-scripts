from flask import Flask, request, jsonify
import dill
import pandas as pd
import numpy as np
import csv
import heartpy as hp
from scipy.signal import savgol_filter, butter, filtfilt
from hmmlearn import hmm
import torch 
import torch.nn as nn
import json
import os

app = Flask(__name__)

# Global buffer for physiological (PPG) data.
# Each entry is a dict: {"timestamp": float, "ppg": float}
shimmer_buffer = []

# Load the model, scaler, and configuration
MODEL_DIR = "model_sequential"  # Update this path as needed

try:
    with open(os.path.join(MODEL_DIR, 'model.dill'), 'rb') as f:
        model = dill.load(f)
    
    with open(os.path.join(MODEL_DIR, 'scaler.dill'), 'rb') as f:
        scaler = dill.load(f)
    
    with open(os.path.join(MODEL_DIR, 'config.json'), 'r') as f:
        config = json.load(f)
    
    print(f"Loaded model and scaler from {MODEL_DIR}")
    print(f"Model config: {config}")
    
    # Ensure model is in evaluation mode
    model.eval()
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    model = None
    scaler = None
    config = {}

@app.route('/upload-gsr', methods=['POST'])
def upload_gsr():
    global shimmer_buffer
    data = request.get_json()
    if isinstance(data, list):
        shimmer_buffer.extend(data)
    else:
        shimmer_buffer.append(data)
    print(f"Received {len(data) if isinstance(data, list) else 1} new GSR samples.")
    return jsonify({"status": "success", "message": "GSR data received"})

#######################################
# PPG Processing Functions
#######################################
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def process_ppg(samples, fs=46.8):
    """
    Processes PPG data using heartpy.
    Applies a lowpass filter, enhances peaks, and computes segmentwise metrics.
    Returns a dictionary with:
      - 'bpm'
      - 'ibi'
      - 'sdnn'
    """
    if not samples:
        return {'bpm': np.nan, 'ibi': np.nan, 'sdnn': np.nan}
    df_ppg = pd.DataFrame(samples)
    sampling_rate = 51
    cutoff_frequency = 3
    filter_order = 3
    df_ppg['ppg'] = butter_lowpass_filter(df_ppg['ppg'], cutoff_frequency, sampling_rate, filter_order)
    df_ppg['ppg'] = hp.enhance_peaks(df_ppg['ppg'], iterations=2)
    sample_rate_ppg = fs
    wd, m = hp.process_segmentwise(df_ppg['ppg'].values, sample_rate=sample_rate_ppg,
                                   segment_width=58, segment_overlap=0)
    aggregated_metrics = {key: np.mean(values) for key, values in m.items()}
    return {
        'bpm': aggregated_metrics.get('bpm', np.nan),
        'ibi': aggregated_metrics.get('ibi', np.nan),
        'sdnn': aggregated_metrics.get('sdnn', np.nan)
    }

#######################################
# Helper Functions for Eye-Tracking Processing
#######################################
def smooth_signal(signal, window_length=11, poly_order=2):
    """Apply Savitzky-Golay filter to smooth signal."""
    # Replace any NaNs or inf values
    if np.isnan(signal).any() or np.isinf(signal).any():
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    # Check that the window length is valid
    if len(signal) < window_length or window_length % 2 == 0:
        return signal
    try:
        return savgol_filter(signal, window_length=window_length, polyorder=poly_order)
    except Exception as e:
        print(f"Smoothing failed: {e}. Returning original signal.")
        return signal

def detect_blinks(pupil_series, eye_freq=90, std_factor=2.0):
    """Detect blinks in pupil diameter series."""
    # Rolling window ~100 ms => ~9 samples at 90Hz
    window_samples = int(0.1 * eye_freq) if int(0.1 * eye_freq) > 0 else 1
    rolling_mean = pupil_series.rolling(window_samples, min_periods=1).mean()
    rolling_std = pupil_series.rolling(window_samples, min_periods=1).std().fillna(0)
    blink_threshold = rolling_mean - std_factor * rolling_std
    mask = (pupil_series < blink_threshold)
    return mask

def mark_blink_segments(blink_mask):
    """Mark continuous blink segments."""
    segments = []
    in_blink = False
    start_idx = None
    for i, val in enumerate(blink_mask):
        if val and not in_blink:
            in_blink = True
            start_idx = i
        elif not val and in_blink:
            end_idx = i - 1
            segments.append((start_idx, end_idx))
            in_blink = False
    if in_blink:
        end_idx = len(blink_mask) - 1
        segments.append((start_idx, end_idx))
    return segments

def apply_blink_margins_and_interpolate(df, blink_segments, eye_freq=90, margin_ms=50):
    """Apply margins around blinks and interpolate."""
    margin_samples = int(round(margin_ms * eye_freq / 1000.0))
    df = df.copy()
    n = len(df)
    for (start, end) in blink_segments:
        s = max(0, start - margin_samples)
        e = min(n - 1, end + margin_samples)
        df.loc[s:e, ['pupil','dir_x','dir_y']] = np.nan
    df[['pupil','dir_x','dir_y']] = df[['pupil','dir_x','dir_y']].interpolate(method='linear').ffill().bfill()
    return df

def compute_velocity(dir_x, dir_y, eye_freq=90):
    """Compute velocity from direction signals."""
    dx = np.diff(dir_x, prepend=dir_x[0])
    dy = np.diff(dir_y, prepend=dir_y[0])
    dist = np.sqrt(dx**2 + dy**2)
    velocity = dist * eye_freq
    return velocity

def compute_acceleration(velocity, eye_freq=90):
    """Compute acceleration from velocity signal."""
    acceleration = np.diff(velocity, prepend=velocity[0]) * eye_freq
    return acceleration

def run_hmm_classification(velocity_array):
    """Use HMM to classify fixations and saccades."""
    vel_reshaped = velocity_array.reshape(-1, 1)
    model = hmm.GaussianHMM(n_components=2, covariance_type='diag', n_iter=100, random_state=42)
    model.fit(vel_reshaped)
    hidden_states = model.predict(vel_reshaped)
    state_means = model.means_.flatten()
    sorted_idx = np.argsort(state_means)
    remap = {sorted_idx[0]: 0, sorted_idx[1]: 1}
    mapped_states = np.array([remap[s] for s in hidden_states])
    return mapped_states, model

def compute_saccade_metrics(velocity, acceleration, states):
    """Compute advanced saccade metrics."""
    # Identify saccade segments
    saccade_segments = []
    in_saccade = False
    start_idx = None
    
    for i, state in enumerate(states):
        if state == 1 and not in_saccade:
            in_saccade = True
            start_idx = i
        elif state == 0 and in_saccade:
            end_idx = i - 1
            saccade_segments.append((start_idx, end_idx))
            in_saccade = False
    
    if in_saccade:
        end_idx = len(states) - 1
        saccade_segments.append((start_idx, end_idx))
    
    # Calculate metrics for each saccade
    peak_velocities = []
    mean_accelerations = []
    mean_decelerations = []
    
    for start, end in saccade_segments:
        if end - start < 2:  # Skip very short saccades
            continue
            
        seg_velocity = velocity[start:end+1]
        seg_acceleration = acceleration[start:end+1]
        
        peak_velocities.append(np.max(seg_velocity))
        
        # Separate acceleration and deceleration phases
        accelerations = seg_acceleration[seg_acceleration > 0]
        decelerations = seg_acceleration[seg_acceleration < 0]
        
        if len(accelerations) > 0:
            mean_accelerations.append(np.mean(accelerations))
        if len(decelerations) > 0:
            mean_decelerations.append(np.abs(np.mean(decelerations)))
    
    # Compute summary metrics
    peak_velocity = np.mean(peak_velocities) if peak_velocities else 0
    mean_acceleration = np.mean(mean_accelerations) if mean_accelerations else 0
    mean_deceleration = np.mean(mean_decelerations) if mean_decelerations else 0
    velocity_std = np.std(peak_velocities) if len(peak_velocities) > 1 else 0
    
    # Accel/decel ratio (higher values indicate more abrupt deceleration)
    accel_decel_ratio = (mean_acceleration / mean_deceleration) if mean_deceleration > 0 else 0
    
    return {
        'saccade_peak_velocity': peak_velocity,
        'saccade_mean_acceleration': mean_acceleration,
        'saccade_mean_deceleration': mean_deceleration,
        'saccade_velocity_std': velocity_std,
        'saccade_accel_decel_ratio': accel_decel_ratio
    }

def compute_pupil_velocities(pupil_values, eye_freq=90):
    """Compute pupil constriction and dilation velocities."""
    # Calculate differences between consecutive samples
    diffs = np.diff(pupil_values, prepend=pupil_values[0])
    
    # Separate constriction (negative) and dilation (positive)
    constriction_velocity = np.mean(diffs[diffs < 0]) if any(diffs < 0) else 0
    dilation_velocity = np.mean(diffs[diffs > 0]) if any(diffs > 0) else 0
    
    # Convert to absolute value for constriction (to make it positive)
    constriction_velocity = abs(constriction_velocity) * eye_freq
    dilation_velocity = dilation_velocity * eye_freq
    
    return constriction_velocity, dilation_velocity

def compute_eye_metrics(eye_chunk, eye_freq=90):
    """Compute comprehensive eye tracking metrics for a data window."""
    try:
        metrics = {}
        
        # Process eye tracking data
        eye_chunk = eye_chunk.copy()
        
        # Create the pupil column (average of left and right)
        if 'pupil_diameter_L' in eye_chunk.columns and 'pupil_diameter_R' in eye_chunk.columns:
            eye_chunk['pupil'] = eye_chunk[['pupil_diameter_L','pupil_diameter_R']].mean(axis=1, skipna=True)
        elif 'avg_pupil_diameter' in eye_chunk.columns:
            eye_chunk['pupil'] = eye_chunk['avg_pupil_diameter']
        else:
            # If no pupil diameter columns, create a dummy one
            eye_chunk['pupil'] = 0
            
        # 2D Gaze Direction: average left/right 3D directions, then project to 2D (ignore z)
        needed_cols = [
            'gaze_direct_L_x', 'gaze_direct_L_y', 'gaze_direct_L_z',
            'gaze_direct_R_x', 'gaze_direct_R_y', 'gaze_direct_R_z'
        ]
        
        if all(col in eye_chunk.columns for col in needed_cols):
            eye_chunk['dir_x_3D'] = (eye_chunk['gaze_direct_L_x'] + eye_chunk['gaze_direct_R_x']) / 2.0
            eye_chunk['dir_y_3D'] = (eye_chunk['gaze_direct_L_y'] + eye_chunk['gaze_direct_R_y']) / 2.0
            eye_chunk['dir_z_3D'] = (eye_chunk['gaze_direct_L_z'] + eye_chunk['gaze_direct_R_z']) / 2.0
            
            mag_3D = np.sqrt(eye_chunk['dir_x_3D']**2 + eye_chunk['dir_y_3D']**2 + eye_chunk['dir_z_3D']**2)
            eye_chunk['dir_x'] = eye_chunk['dir_x_3D'] / mag_3D
            eye_chunk['dir_y'] = eye_chunk['dir_y_3D'] / mag_3D
            
            # Smooth direction signals
            eye_chunk['dir_x'] = smooth_signal(eye_chunk['dir_x'].values)
            eye_chunk['dir_y'] = smooth_signal(eye_chunk['dir_y'].values)
        else:
            # If no gaze direction columns, create dummy ones
            eye_chunk['dir_x'] = 0
            eye_chunk['dir_y'] = 0
        
        # Ensure the eye chunk has all needed columns
        for col in ['pupil', 'dir_x', 'dir_y']:
            if col not in eye_chunk:
                eye_chunk[col] = 0
        
        # Blink detection on raw pupil signal
        raw_blink_mask = detect_blinks(eye_chunk['pupil'], eye_freq)
        eye_chunk['blink'] = raw_blink_mask
        blink_segments = mark_blink_segments(raw_blink_mask)
        eye_chunk = apply_blink_margins_and_interpolate(eye_chunk, blink_segments, eye_freq)
        
        # Compute velocity and acceleration from direction signals
        eye_chunk['velocity'] = compute_velocity(eye_chunk['dir_x'].values, eye_chunk['dir_y'].values, eye_freq)
        eye_chunk['acceleration'] = compute_acceleration(eye_chunk['velocity'].values, eye_freq)
        
        # Classify samples using HMM (0 = fixation, 1 = saccade)
        states, _ = run_hmm_classification(eye_chunk['velocity'].values)
        eye_chunk['state'] = states
        
        # Basic metrics
        # Blink metrics
        blink_count = len(blink_segments)
        blink_rate_per_min = blink_count * (60 / (len(eye_chunk) / eye_freq))
        
        # Fixation metrics: count contiguous segments where state == 0
        fixation_durations = []
        fixation_count = 0
        curr_len = 0
        for s in states:
            if s == 0:
                curr_len += 1
            else:
                if curr_len > 0:
                    fixation_count += 1
                    fixation_durations.append(curr_len)
                curr_len = 0
        if curr_len > 0:
            fixation_count += 1
            fixation_durations.append(curr_len)
        
        mean_fix_dur_samples = np.mean(fixation_durations) if fixation_durations else 0
        mean_fix_dur_ms = (mean_fix_dur_samples / eye_freq) * 1000
        
        # Fixation duration variability (std dev)
        std_fix_dur_ms = (np.std(fixation_durations) / eye_freq * 1000) if len(fixation_durations) > 1 else 0
        
        # Saccade metrics: count and compute amplitude in 2D direction space
        saccade_count = 0
        saccade_amplitudes = []
        i = 0
        n = len(states)
        while i < n:
            if states[i] == 1:
                start_i = i
                while i < n and states[i] == 1:
                    i += 1
                end_i = i - 1
                saccade_count += 1
                if 'dir_x' in eye_chunk.columns and 'dir_y' in eye_chunk.columns:
                    dx = eye_chunk['dir_x'].iloc[end_i] - eye_chunk['dir_x'].iloc[start_i]
                    dy = eye_chunk['dir_y'].iloc[end_i] - eye_chunk['dir_y'].iloc[start_i]
                    amp = np.sqrt(dx**2 + dy**2)
                    saccade_amplitudes.append(amp)
            else:
                i += 1
        mean_sacc_amp = np.mean(saccade_amplitudes) if saccade_amplitudes else 0
        std_sacc_amp = np.std(saccade_amplitudes) if len(saccade_amplitudes) > 1 else 0
        
        # Advanced saccade metrics
        saccade_advanced_metrics = compute_saccade_metrics(
            eye_chunk['velocity'].values, 
            eye_chunk['acceleration'].values, 
            states
        )
        
        # Saccade-fixation ratio
        saccade_fixation_ratio = saccade_count / fixation_count if fixation_count > 0 else 0
        
        # Pupil stats
        pvals = eye_chunk['pupil'].values
        mean_pupil = np.mean(pvals)
        std_pupil = np.std(pvals)
        min_pupil = np.min(pvals)
        max_pupil = np.max(pvals)
        
        # Pupil trend (slope)
        x_idx = np.arange(len(pvals))
        slope = np.polyfit(x_idx, pvals, 1)[0] if len(pvals) > 1 else 0.0
        
        # Advanced pupil metrics: constriction and dilation velocities
        constriction_velocity, dilation_velocity = compute_pupil_velocities(pvals, eye_freq)
        
        # Compile all metrics
        metrics.update({
            'blink_count': blink_count,
            'blink_rate_per_min': blink_rate_per_min,
            'fixation_count': fixation_count,
            'mean_fixation_duration_ms': mean_fix_dur_ms,
            'std_fixation_duration_ms': std_fix_dur_ms,
            'saccade_count': saccade_count,
            'mean_saccade_amplitude': mean_sacc_amp,
            'std_saccade_amplitude': std_sacc_amp,
            'saccade_fixation_ratio': saccade_fixation_ratio,
            'mean_pupil': mean_pupil,
            'std_pupil': std_pupil,
            'min_pupil': min_pupil,
            'max_pupil': max_pupil,
            'pupil_slope': slope,
            'pupil_constriction_velocity': constriction_velocity,
            'pupil_dilation_velocity': dilation_velocity,
        })
        
        # Add advanced saccade metrics
        metrics.update(saccade_advanced_metrics)
        
        return metrics
    except Exception as e:
        print(f"Error in compute_eye_metrics: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return default metrics to avoid crashing
        default_metrics = {
            'blink_count': 0, 'blink_rate_per_min': 0, 'fixation_count': 0,
            'mean_fixation_duration_ms': 0, 'std_fixation_duration_ms': 0,
            'saccade_count': 0, 'mean_saccade_amplitude': 0, 'std_saccade_amplitude': 0,
            'saccade_fixation_ratio': 0, 'mean_pupil': 0, 'std_pupil': 0,
            'min_pupil': 0, 'max_pupil': 0, 'pupil_slope': 0,
            'pupil_constriction_velocity': 0, 'pupil_dilation_velocity': 0,
            'saccade_peak_velocity': 0, 'saccade_mean_acceleration': 0,
            'saccade_mean_deceleration': 0, 'saccade_velocity_std': 0,
            'saccade_accel_decel_ratio': 0
        }
        return default_metrics

def compute_gsr_metrics(gsr_values):
    """Compute GSR metrics for a window."""
    if len(gsr_values) == 0 or np.all(np.isnan(gsr_values)):
        return {
            'gsr_mean': 0,
            'gsr_std': 0,
            'gsr_slope': 0
        }
    
    gsr_series = pd.Series(gsr_values)
    gsr_cleaned = gsr_series.ffill().bfill().values
    
    # Basic statistics
    gsr_mean = np.mean(gsr_cleaned)
    gsr_std = np.std(gsr_cleaned)
    
    # Compute slope (linear trend)
    x_idx = np.arange(len(gsr_cleaned))
    if len(gsr_cleaned) > 1:
        gsr_slope = np.polyfit(x_idx, gsr_cleaned, 1)[0]
    else:
        gsr_slope = 0
    
    return {
        'gsr_mean': gsr_mean,
        'gsr_std': gsr_std,
        'gsr_slope': gsr_slope
    }

def process_hr_with_heartpy(ppg_values, sampling_freq=46.8):
    """Calculate heart rate metrics using HeartPy."""
    if len(ppg_values) == 0 or np.all(np.isnan(ppg_values)):
        return {
            'bpm': 0,
            'sdnn': 0,
            'rmssd': 0,
            'pnn50': 0
        }
    
    # Replace deprecated fillna with ffill/bfill methods
    ppg_series = pd.Series(ppg_values)
    ppg_cleaned = ppg_series.ffill().bfill().values
    
    # Apply bandpass filter to PPG signal
    try:
        # Filter parameters
        lowcut = 0.5  # Hz
        highcut = 3.0  # Hz
        order = 4
        nyquist = 0.5 * sampling_freq
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        ppg_filtered = filtfilt(b, a, ppg_cleaned)
        
        # Process with HeartPy
        try:
            wd, m = hp.process(ppg_filtered, sample_rate=sampling_freq, reject_segmentwise=True, high_precision=True)
            return {
                'bpm': m.get('bpm', 0),
                'sdnn': m.get('sdnn', 0),
                'rmssd': m.get('rmssd', 0),
                'pnn50': m.get('pnn50', 0)
            }
        except Exception as e:
            print(f"HeartPy processing failed: {e}")
            return {
                'bpm': 0,
                'sdnn': 0,
                'rmssd': 0,
                'pnn50': 0
            }
            
    except Exception as e:
        print(f"Error processing PPG: {e}")
        return {
            'bpm': 0,
            'sdnn': 0,
            'rmssd': 0,
            'pnn50': 0
        }
    
def extract_windows(df, window_samples=1325, num_windows=4):
    """
    Extract the last 5300 rows from a dataframe and divide into 4 windows.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing data for a participant session level
    window_samples : int
        Number of samples per window
    num_windows : int
        Number of windows to extract
        
    Returns:
    --------
    list : List of dataframes, one for each window
    """
    try:
        total_samples_needed = window_samples * num_windows
        
        # Extract the last 5300 rows (or all if fewer)
        subset = df.tail(min(len(df), total_samples_needed)).copy()
        subset.reset_index(drop=True, inplace=True)
        
        # Split into windows
        windows = []
        for i in range(num_windows):
            start_idx = i * window_samples
            end_idx = min((i + 1) * window_samples, len(subset))
            
            if start_idx >= len(subset):
                # If we don't have enough data, create an empty window
                window = pd.DataFrame(columns=subset.columns)
            else:
                window = subset.iloc[start_idx:end_idx].copy()
                
            windows.append(window)
        
        return windows
    except Exception as e:
        print(f"Error in extract_windows: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return empty windows in case of error
        empty_windows = []
        for _ in range(num_windows):
            empty_windows.append(pd.DataFrame())
        return empty_windows

#######################################
# Main Feature Processing Functions
#######################################

#######################################
# LSTM Model Definition
#######################################
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, bidirectional=True):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.Linear(self.hidden_size, 1)
        
    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2, bidirectional=True, use_ordinal=False):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_ordinal = use_ordinal
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = AttentionLayer(hidden_size, bidirectional)
        
        if use_ordinal:
            self.fc = nn.Sequential(
                nn.Linear(lstm_output_size, lstm_output_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_output_size // 2, num_classes - 1)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(lstm_output_size, lstm_output_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_output_size // 2, num_classes)
            )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attention_weights = self.attention(lstm_out)
        output = self.fc(context)
        return output
#######################################
# Main Level Data Processing Function
#######################################
def process_level_data(eye_df, shimmer_samples, window_samples=1325, num_windows=4):
    """
    Process all data for a level and return features in the format expected by the model.
    Returns a structured array of shape [1, 4, 28] where:
    - 1 is the batch size
    - 4 is the number of windows (time steps)
    - 28 is the number of features per window
    """
    try:
        print("Starting process_level_data")
        
        # Process eye tracking data
        eye_features = process_eye_windows(eye_df, window_samples, num_windows)
        
        # Process shimmer data (GSR/PPG)
        physio_features = process_shimmer_windows(shimmer_samples, window_samples, num_windows)
        
        # Combine all features - but this time we need to do it carefully to maintain window structure
        all_window_features = []
        
        # Expected features per window = 28
        expected_features_per_window = 28
        
        for window_idx in range(num_windows):
            # Extract features for this window
            window_features = {}
            
            # Add eye tracking features for this window
            for key, value in eye_features.items():
                if key.startswith(f'window_{window_idx}_'):
                    # Remove the window prefix to get the base feature name
                    feature_name = key[len(f'window_{window_idx}_'):]
                    window_features[feature_name] = value
            
            # Add physiological features for this window
            for key, value in physio_features.items():
                if key.startswith(f'window_{window_idx}_'):
                    feature_name = key[len(f'window_{window_idx}_'):]
                    window_features[feature_name] = value
            
            # Convert window features to a flat array
            window_df = pd.DataFrame([window_features])
            window_array = window_df.values.flatten()
            
            # Verify we have the correct number of features
            if len(window_array) != expected_features_per_window:
                print(f"Warning: Window {window_idx} has {len(window_array)} features instead of expected {expected_features_per_window}")
                print(f"Window features: {sorted(window_features.keys())}")
                # If needed, pad or truncate to get the right number of features
                if len(window_array) < expected_features_per_window:
                    padding = np.zeros(expected_features_per_window - len(window_array))
                    window_array = np.concatenate([window_array, padding])
                elif len(window_array) > expected_features_per_window:
                    window_array = window_array[:expected_features_per_window]
            
            all_window_features.append(window_array)
        
        # Stack all windows to create a 2D array of shape [num_windows, features_per_window]
        X_sequence = np.stack(all_window_features)
        
        # Reshape to [1, num_windows, features_per_window] for batch processing
        X_sequence = np.expand_dims(X_sequence, axis=0)
        
        # Scale if a scaler is available
        if scaler is not None:
            try:
                # Apply scaler to each window separately
                X_norm = np.zeros_like(X_sequence)
                for i in range(num_windows):
                    X_norm[0, i, :] = scaler.transform(X_sequence[0, i, :].reshape(1, -1)).flatten()
                print(f"Scaled sequence data to shape: {X_norm.shape}")
            except Exception as e:
                print(f"Error scaling features: {e}")
                print("Using unscaled features")
                X_norm = X_sequence
        else:
            X_norm = X_sequence
            print("No scaler found, using unscaled features")
        
        print(f"Returning sequence data with shape: {X_norm.shape}")
        return X_norm
    
    except Exception as e:
        print(f"Error in process_level_data: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return a dummy array with zeros of the right shape
        dummy_features = np.zeros((1, num_windows, 28))
        return dummy_features
def process_eye_windows(eye_df, window_samples=1325, num_windows=4):
    """Process eye tracking data into sequential windows."""
    try:
        # Parameters
        eye_freq = 90
        
        # Extract windows from the data
        windows = extract_windows(eye_df, window_samples, num_windows)
        
        # Process each window
        all_eye_features = []
        
        for i, window in enumerate(windows):
            if len(window) == 0:
                # Use zeros for empty windows
                eye_metrics = {f'window_{i}_{key}': 0 for key in [
                    'blink_count', 'blink_rate_per_min', 'fixation_count', 
                    'mean_fixation_duration_ms', 'std_fixation_duration_ms',
                    'saccade_count', 'mean_saccade_amplitude', 'std_saccade_amplitude',
                    'saccade_fixation_ratio', 'saccade_peak_velocity',
                    'saccade_mean_acceleration', 'saccade_mean_deceleration',
                    'saccade_velocity_std', 'saccade_accel_decel_ratio',
                    'mean_pupil', 'std_pupil', 'min_pupil', 'max_pupil', 
                    'pupil_slope', 'pupil_constriction_velocity', 'pupil_dilation_velocity'
                ]}
            else:
                # Compute metrics for this window
                metrics = compute_eye_metrics(window, eye_freq)
                
                # Prefix with window index
                eye_metrics = {f'window_{i}_{key}': value for key, value in metrics.items()}
            
            all_eye_features.append(eye_metrics)
        
        # Combine all windows into a single feature dictionary
        combined_features = {}
        for feature_dict in all_eye_features:
            combined_features.update(feature_dict)
        
        return combined_features
    except Exception as e:
        print(f"Error in process_eye_windows: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return default features to avoid crashing
        default_features = {}
        for i in range(num_windows):
            for key in [
                'blink_count', 'blink_rate_per_min', 'fixation_count', 
                'mean_fixation_duration_ms', 'std_fixation_duration_ms',
                'saccade_count', 'mean_saccade_amplitude', 'std_saccade_amplitude',
                'saccade_fixation_ratio', 'saccade_peak_velocity',
                'saccade_mean_acceleration', 'saccade_mean_deceleration',
                'saccade_velocity_std', 'saccade_accel_decel_ratio',
                'mean_pupil', 'std_pupil', 'min_pupil', 'max_pupil', 
                'pupil_slope', 'pupil_constriction_velocity', 'pupil_dilation_velocity'
            ]:
                default_features[f'window_{i}_{key}'] = 0
        
        return default_features

def process_shimmer_windows(shimmer_samples, window_samples=1325, num_windows=4):
    """Process physiological (GSR/PPG) data into sequential windows."""
    try:
        if not shimmer_samples:
            # Generate empty features if no data available
            gsr_features = {}
            for i in range(num_windows):
                gsr_features.update({
                    f'window_{i}_gsr_mean': 0,
                    f'window_{i}_gsr_std': 0,
                    f'window_{i}_gsr_slope': 0,
                    f'window_{i}_bpm': 0,
                    f'window_{i}_sdnn': 0,
                    f'window_{i}_rmssd': 0,
                    f'window_{i}_pnn50': 0
                })
            return gsr_features
        
        # Convert to DataFrame
        shimmer_df = pd.DataFrame(shimmer_samples)
        
        # Sort by timestamp
        if 'timestamp' in shimmer_df.columns:
            shimmer_df.sort_values('timestamp', inplace=True)
        
        # Extract windows
        windows = extract_windows(shimmer_df, window_samples, num_windows)
        
        # Process each window
        all_physio_features = []
        
        for i, window in enumerate(windows):
            if len(window) == 0:
                # Use zeros for empty windows
                physio_metrics = {
                    f'window_{i}_gsr_mean': 0,
                    f'window_{i}_gsr_std': 0,
                    f'window_{i}_gsr_slope': 0,
                    f'window_{i}_bpm': 0,
                    f'window_{i}_sdnn': 0,
                    f'window_{i}_rmssd': 0,
                    f'window_{i}_pnn50': 0
                }
            else:
                # GSR metrics if available
                if 'gsr' in window.columns:
                    gsr_metrics = compute_gsr_metrics(window['gsr'].values)
                    gsr_features = {f'window_{i}_{key}': value for key, value in gsr_metrics.items()}
                else:
                    gsr_features = {
                        f'window_{i}_gsr_mean': 0,
                        f'window_{i}_gsr_std': 0,
                        f'window_{i}_gsr_slope': 0
                    }
                
                # PPG/Heart rate metrics if available
                if 'ppg' in window.columns:
                    hr_metrics = process_hr_with_heartpy(window['ppg'].values)
                    hr_features = {f'window_{i}_{key}': value for key, value in hr_metrics.items()}
                else:
                    hr_features = {
                        f'window_{i}_bpm': 0,
                        f'window_{i}_sdnn': 0,
                        f'window_{i}_rmssd': 0,
                        f'window_{i}_pnn50': 0
                    }
                
                # Combine GSR and HR features for this window
                physio_metrics = {}
                physio_metrics.update(gsr_features)
                physio_metrics.update(hr_features)
            
            all_physio_features.append(physio_metrics)
        
        # Combine all windows into a single feature dictionary
        combined_features = {}
        for feature_dict in all_physio_features:
            combined_features.update(feature_dict)
        
        return combined_features
    
    except Exception as e:
        print(f"Error in process_shimmer_windows: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return default features to avoid crashing
        default_features = {}
        for i in range(num_windows):
            default_features.update({
                f'window_{i}_gsr_mean': 0,
                f'window_{i}_gsr_std': 0,
                f'window_{i}_gsr_slope': 0,
                f'window_{i}_bpm': 0,
                f'window_{i}_sdnn': 0,
                f'window_{i}_rmssd': 0,
                f'window_{i}_pnn50': 0
            })
        
        return default_features
#######################################
# Model Prediction Endpoint
#######################################
@app.route('/upload-eye', methods=['POST'])
def upload_eye():
    global shimmer_buffer, model, scaler
    
    # Check if model and scaler are loaded
    if model is None or scaler is None:
        return jsonify({
            "status": "error", 
            "message": "Model or scaler not loaded. Please check server logs."
        }), 500
    
    try:
        # Parse the incoming data
        data = request.get_json()
        print("Received eye tracking data with keys:", data.keys())
        
        # Extract eye samples
        eye_samples = data.get("items")
        if eye_samples is None:
            return jsonify({"status": "error", "message": "Missing required field: items"}), 400
        
        # Convert to DataFrame
        eye_df = pd.DataFrame(eye_samples)
        
        # Check if there's a 'sample' column that contains nested data
        if 'sample' in eye_df.columns:
            sample_df = eye_df['sample'].apply(pd.Series)
            eye_df = pd.concat([eye_df.drop('sample', axis=1), sample_df], axis=1)
        
        print("Eye tracking DataFrame columns:", eye_df.columns)
        
        # Validate we have the required data
        if 'timestamp' not in eye_df.columns or eye_df.empty:
            return jsonify({"status": "error", "message": "Missing or empty timestamps in eye tracking samples"}), 400
        
        # Determine the time range for this level
        level_start = eye_df['timestamp'].min()
        level_end = eye_df['timestamp'].max()
        print(f"Derived level start: {level_start}, level end: {level_end}")
        
        # Extract relevant shimmer data for this time range
        level_shimmer = [s for s in shimmer_buffer if level_start <= s["timestamp"] <= level_end]
        print(f"Extracted {len(level_shimmer)} PPG samples for level between {level_start} and {level_end}.")
        
        # Clean up the buffer to avoid memory issues - keep only newer data
        shimmer_buffer[:] = [s for s in shimmer_buffer if s["timestamp"] > level_end]
        
        # Process the data - returns a sequence with shape [1, 4, 28]
        X_sequence = process_level_data(eye_df, level_shimmer)
        
        # Convert to PyTorch tensor
        X_tensor = torch.tensor(X_sequence, dtype=torch.float32)
        print(f"Tensor shape before squeeze: {X_tensor.shape}")
        
        # If the tensor has shape [1, 4, 28], we need to squeeze out the batch dimension
        # to get [4, 28], then add the batch dimension back properly
        if X_tensor.shape[0] == 1:
            X_tensor = X_tensor.squeeze(0)  # Remove the first dimension to get [4, 28]
        
        # Check model device and move input tensor to the same device
        device = next(model.parameters()).device
        X_tensor = X_tensor.to(device)
        
        # Now add the batch dimension correctly to get [1, 4, 28]
        X_batched = X_tensor.unsqueeze(0)
        print(f"Input tensor shape: {X_batched.shape}")
        
        # Get prediction from model
        with torch.no_grad():
            outputs = model(X_batched)
            # Move outputs to CPU for numpy conversion
            outputs_cpu = outputs.cpu()
            _, predicted = torch.max(outputs_cpu, 1)
            prediction = int(predicted.item())
            
            # Calculate class probabilities
            probabilities = torch.softmax(outputs_cpu, dim=1).numpy()[0].tolist()
        
        # Return the prediction
        response = {
            "status": "success",
            "prediction": prediction,
            "confidences": probabilities
        }
        
        print("Prediction response:", response)
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in upload_eye endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "status": "error",
            "message": f"An error occurred during processing: {str(e)}"
        }), 500
if __name__ == '__main__':
    # Verify model is loaded
    print("Server starting with model:", "Loaded" if model is not None else "Not loaded")
    print("Listening for connections on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)