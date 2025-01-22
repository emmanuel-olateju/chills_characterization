import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyxdf
import neurokit2 as nk
from neurokit2.signal import signal_filter
from scipy.signal import medfilt, butter, filtfilt
import scipy.interpolate as interpolate
import warnings
from scipy.interpolate import interp1d
import scipy.signal as sciSignal

from functions import ECG_AMPLITUDE, ECG_BASELINEWANDERING, ECG_Z_THRESHOLD

eps = 1E-8

def estimate_sampling_rate(timestamps):
    intervals = np.diff(timestamps)
    
    median_interval = np.median(intervals)
    std_interval = np.std(intervals)
    
    valid_intervals = intervals[np.abs(intervals - median_interval) < 3 * std_interval]
    
    sampling_rate = 1 / np.mean(valid_intervals)
    return int(np.round(sampling_rate))

def normalize_signal(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)
    normalized_signal = (signal - min_val) / ((max_val - min_val)+eps)
    return normalized_signal

def moving_average(signal, window_size):
    if window_size <= 0:
        return signal.copy()
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode='same')

"""
    PROCESSING FUNCTIONS
"""
# --> ECG PREPROCESSING
def myECG_clean(ecg_data, sampling_rate, window_size=2000, overlap=0.5):
    lowcut = 0.5
    highcut = 45.0
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    order = 4
    overlap_samples = int(window_size * overlap)
    filtered_ecg_data = np.zeros_like(ecg_data)
    start = 0
    try:
        while start < len(ecg_data):
            end = min(start + window_size, len(ecg_data))
            window_data = ecg_data[start:end]
            b, a = butter(order, [low, high], btype='band')
            padlen = 3 * max(len(b), len(a))
            if len(window_data) <= padlen:
                padlen = len(window_data) - 1
            filtered_window_data = filtfilt(b, a, window_data, padlen=padlen)
            filtered_ecg_data[start:end] = filtered_window_data
            start += window_size - overlap_samples
    except Exception as e:
        print(e)

    return filtered_ecg_data


def filter_ecg_data(ecg_data, voltage_limit):
    filtered_data = [voltage for voltage in ecg_data if abs(voltage) <= voltage_limit]
    return filtered_data

def handle_ecg_outliers(ecg_data, z_threshold=ECG_Z_THRESHOLD):
    mean = np.mean(ecg_data)
    std_dev = np.std(ecg_data)

    z_scores = (ecg_data - mean)/std_dev
    outliers = (np.abs(z_scores) > z_threshold)

    ecg_data[outliers] = np.nan
    
    filled_data = np.copy(ecg_data) 
    filled_data = np.interp(np.arange(len(ecg_data)), np.where(~np.isnan(filled_data))[0], filled_data[~np.isnan(filled_data)])
    
    return filled_data

def ecg_data_valid(ecg_data, timestamps, z_threshold=ECG_Z_THRESHOLD):
    if len(ecg_data) != len(timestamps):
        print("Mismatch between ECG data and timestamps")
        return False
    
    if np.any(np.isnan(ecg_data)) or np.any(np.isinf(ecg_data)):
        print("Warning: NaN or infinite values detected in ECG data")
        return False
    
    z_scores = np.abs((ecg_data - np.mean(ecg_data)) / np.std(ecg_data))
    outliers = z_scores > z_threshold
    if np.sum(outliers)>int(0.2*len(ecg_data)):
        print(f"Warning: More than 20% ({np.sum(outliers)}) potential outliers detected")
        return False
    
    return True

def compute_relative_wandering(ecg_data):
    baseline = np.mean(ecg_data[:int(0.15*len(ecg_data))])
    relative_wandering = abs(ecg_data - baseline) / baseline
    return np.mean(relative_wandering)

def compute_baseline_wandering(ecg_signal):
    baseline_wandering = np.ptp(ecg_signal)
    return baseline_wandering

# --> EMG & EDA PREPROCESSING
def process_emg_windows(emg_signal, window_size, contamination_threshold):
    from sklearn.neighbors import LocalOutlierFactor
    contaminated_windows = []
    for i in range(0, len(emg_signal), window_size):
        window_data = emg_signal[i:i+window_size]
        X = window_data.reshape(-1, 1)
        warnings.filterwarnings("ignore", message="n_neighbors.*")
        lof = LocalOutlierFactor(n_neighbors=int(window_size/4))
        outlier_scores = lof.fit_predict(X)
        artifactual_points = np.where(outlier_scores == -1)[0]
        proportion_artifacts = len(artifactual_points) / len(window_data)
        if proportion_artifacts >= contamination_threshold:
            end_index = min(i + window_size, len(emg_signal))
            contaminated_windows.append((i, end_index))
    return contaminated_windows


def interpolate_eda_linear(eda_signal, contaminated_windows_emg, window_size):
    interpolated_eda_signal = np.copy(eda_signal)
    for start, end in contaminated_windows_emg:
        left_neighbor_start = max(0, start - window_size)
        right_neighbor_end = min(len(eda_signal), end + window_size)
        x_left = np.arange(left_neighbor_start, start)
        y_left = eda_signal[left_neighbor_start:start]
        x_right = np.arange(end, right_neighbor_end)
        y_right = eda_signal[end:right_neighbor_end]

        if len(x_left) > 0 and len(x_right) > 0:
            x = np.concatenate((x_left, x_right))
            y = np.concatenate((y_left, y_right))
            interp_func = interp1d(x, y, fill_value="extrapolate")
            interpolated_segment = interp_func(np.arange(start, end))

            # Update the interpolated signal with the interpolated segment
            interpolated_eda_signal[start:end] = interpolated_segment

    return interpolated_eda_signal


def compute_quality_index(eda_data):
    eda_data = eda_data/1e6 # converting to microSiemens
    # Rule 1: Check if any EDA value is out of the acceptable range (0.05 - 60 μS)
    out_of_range_count = sum(1 for value in eda_data if value < 0.05 or value > 60)
    # Rule 2: Check for EDA changes faster than ±10 μS/sec within 1s
    rapid_change_indices = set()
    rapid_change_count = 0
    interval_size = 500
    for i in range(interval_size, len(eda_data)):
        change_rate = abs(eda_data[i] - eda_data[i - interval_size])
        if change_rate > 10 or eda_data[i] < 0.05 or eda_data[i] > 60:  # Check if change rate exceeds ±10 μS/sec
            rapid_change_count += 1
            rapid_change_indices.add(i)
            # Mark neighboring 5 seconds as invalid
            rapid_change_indices.update(range(max(0, i - 2500), min(len(eda_data), i + 2501)))

    # Calculate Quality Index
    total_data_points = len(eda_data)
    out_of_range_percentage = (out_of_range_count / total_data_points) * 100
    rapid_change_percentage = (rapid_change_count / (total_data_points - 1)) * 100
    invalid_percentage = (len(rapid_change_indices) / total_data_points) * 100
    quality_index = 100 - max(out_of_range_percentage, rapid_change_percentage, invalid_percentage)

    return max(quality_index, 0)


# --> RESPIRATORY SIGNAL PREPROCESSING
def estimate_respiration_period(signal, sampling_rate):
    """
    Estimate respiration period using Fast Fourier Transform (FFT).

    Parameters:
    - signal: numpy array, input signal
    - sampling_rate: float, sampling rate of the signal

    Returns:
    - float, estimated respiration period
    """

    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1 / sampling_rate)
    peaks, _ = sciSignal.find_peaks(np.abs(fft_signal))
    positive_peaks = peaks[freqs[peaks] >= 0]
    dominant_freq_index = np.argmax(np.abs(fft_signal[positive_peaks]))
    dominant_freq = freqs[peaks[dominant_freq_index]]
    T = 1 / dominant_freq
    return T

def preprocess_respiration_signal(signal, sampling_rate, resampled_rate=100,
                                  window_size_seconds=60, overlap_seconds=15, high_pass_cutoff=0.1, t_multiplier=2):

    b, a = butter(2, high_pass_cutoff / (sampling_rate / 2), btype='high')
    signal_filtered = filtfilt(b, a, signal)
    signal_filtered = sciSignal.resample(signal_filtered, int(len(signal_filtered) * resampled_rate / sampling_rate))

    window_size_samples = int(window_size_seconds * resampled_rate)
    overlap_samples = int(overlap_seconds * resampled_rate)

    # Split signal into windows with overlap
    windows = [signal_filtered[i:i + window_size_samples] for i in range(0, len(signal_filtered), window_size_samples - overlap_samples)]
    filtered_windows = []
    for window in windows:
        t = estimate_respiration_period(window, resampled_rate)
        window_filtered = moving_average(window, int(t * t_multiplier))
        filtered_windows.append(window_filtered)

    # Reassemble windows and smooth using 1-second moving mean filter
    reconstructed_signal = np.concatenate(filtered_windows)
    smoothed_signal = moving_average(reconstructed_signal, resampled_rate)  # Resampled to 100 Hz

    # Resample back to the original sampling rate
    preprocessed_signal = sciSignal.resample(smoothed_signal, len(signal))
    return preprocessed_signal


def clean_resp_data(resp_data, fs):
    cutoff_low = 1.0
    cutoff_high = 0.1
    order=5
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_low / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, resp_data)
    b, a = butter(2, cutoff_high / nyquist, btype='high', analog=False)
    filtered_data = filtfilt(b, a, filtered_data)
    return filtered_data

# --> MERGE ALL PREPROCESSING INTO SINLGE FUNCTION 
def preprocess_data(data):
    output = {}

    try:
        if 'ECG ' in data:
            ecg_data = data['ECG ']
            timestamps_ecg = data['Timestamps']
            timestamps_ecg = np.array(timestamps_ecg)
            effective_fs = estimate_sampling_rate(timestamps_ecg)
            sampling_rate = int(np.round(effective_fs))
            print("Length of ECG data: ", len(ecg_data))
            ecg_data = nk.signal_sanitize(ecg_data)
            ecg_data_notch = signal_filter(ecg_data,
                                           sampling_rate=sampling_rate,
                                           method="powerline",
                                           powerline=60)
            ecg_cleaned = myECG_clean(ecg_data_notch, sampling_rate)
            ecg_cleaned = filter_ecg_data(ecg_cleaned, ECG_AMPLITUDE)
            ecg_valid = ecg_data_valid(ecg_cleaned,timestamps_ecg)
            baseline_wandering = compute_baseline_wandering(ecg_cleaned)
            print(f"ECG Validity:{ecg_valid}")
            print("The baseline wandering of ECG is {}".format(baseline_wandering))
            output['baseline_wandering'] = baseline_wandering
            output['Exclusion_Status_ECG'] = 'Unacceptable' if baseline_wandering > ECG_BASELINEWANDERING else 'Acceptable'
            output['ecg_cleaned'] = ecg_cleaned
        else:
            print(f"No ECG data found in file")

        if 'GSR' in data and 'ExGa 1' in data:
            eda_data = data['GSR']
            emg_data = data['ExGa 1']
            timestamps_eda = data['Timestamps']
            effective_sampling_rate = estimate_sampling_rate(timestamps_eda)
            print("Length of EMG data is {}".format(len(emg_data)))
            print("Length of EDA data is {}".format(len(eda_data)))
            exga1_normalized = normalize_signal(emg_data)
            kernel_size = 8 * effective_sampling_rate  # 8 seconds of data
            if kernel_size % 2 == 0:
                kernel_size += 1
            eda_smooth = medfilt(eda_data, kernel_size=kernel_size)
            nyquist = 0.5 * effective_sampling_rate
            normal_cutoff = 5 / nyquist  # 5 Hz
            b, a = butter(4, normal_cutoff, btype='low', analog=False)
            eda_lowpass = filtfilt(b, a, eda_smooth)
            contaminated_windows_emg = process_emg_windows(exga1_normalized,
                                                           window_size=effective_sampling_rate,
                                                           contamination_threshold=0.2)
            eda_cleaned = interpolate_eda_linear(eda_lowpass,
                                                 contaminated_windows_emg,
                                                 window_size=effective_sampling_rate)
            SQI_EDA = compute_quality_index(eda_cleaned)
            print("The quality index of EDA is: ", SQI_EDA)
            output['EDA_Quality'] = SQI_EDA
            output['Exclusion_Status_EDA'] = 'Unacceptable' if SQI_EDA < 50 else 'Acceptable'

            df = pd.DataFrame({
                'Timestamp': timestamps_eda,
                'EMG': exga1_normalized,
                'GSR_Normalized': eda_data,
                'GSR_Interpolated': eda_cleaned
            })

            output['df'] = df
            output['effective_sampling_rate'] = effective_sampling_rate
        else:
            print(f"No EXG-a or GSR data found in file")

        if 'Resp.' in data:
            resp_data = data['Resp.']
            timestamps_resp = data['Timestamps']
            timestamps_resp = np.array(timestamps_resp)
            effective_fs = estimate_sampling_rate(timestamps_resp)
            sampling_rate = int(np.round(effective_fs))
            resp_filt_data = clean_resp_data(resp_data, sampling_rate)
            output["resp_cleaned"] = resp_filt_data


    except Exception as e:
        print(f"Error processing subject: {e}")
        raise

    return output


"""
    FEATURE EXTRACTION
"""

def mean_HR_HRV(ecg_signal, fs, method_="neurokit"):
    
    clean_ecg = nk.ecg_clean(ecg_signal, sampling_rate=fs, method=method_)
    peaks, _ = nk.ecg_peaks(clean_ecg, sampling_rate=fs, method=method_)
    heart_rate = nk.ecg_rate(peaks, sampling_rate=fs)
    hrv = nk.hrv(peaks, sampling_rate=fs)

    return heart_rate.mean(), hrv