import sys
import math
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal, integrate
from scipy.interpolate import interp1d
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis

from functions import TIME_WINDOW

import numba as nb

@nb.jit(nopython=True)
def simple_peak_detector(signal, min_distance, threshold_factor=0.6):
  peaks = []
  threshold = threshold_factor * np.max(signal)
    
  for i in range(1, len(signal) - 1):
    if signal[i] > threshold and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
      if not peaks or i - peaks[-1] >= min_distance:
        peaks.append(i)
    
  return np.array(peaks)

def interpolate_peaks(peaks, time_duration, fs):

  r_peaks = np.where(peaks==1)[0]
  rr_intervals = np.diff(r_peaks)/fs

  time_rr = np.cumsum(rr_intervals)
  time_rr = np.insert(time_rr, 0, 0)
  time_points = np.arange(time_duration*fs)/fs

  # kind = "cubic" if len(rr_intervals) > 4 else "linear"
  kind = "linear"

  f_interp = interp1d(time_rr, peaks[r_peaks], kind=kind, bounds_error=False,
                    fill_value="extrapolate")
  rr_interp = f_interp(time_points)

  return rr_interp - np.mean(rr_interp)

def HRV_NN(peaks, sampling_rate):
  r_peak_indices = np.where(peaks == 1)[0]
  rr_intervals_samples = np.diff(r_peak_indices)
  rr_intervals = rr_intervals_samples/sampling_rate
  sdnn = np.std(rr_intervals)
  mean_nn = np.mean(rr_intervals)
  median_nn = np.median(rr_intervals)
  return mean_nn, sdnn, median_nn

def HRV_RMSSD(peaks, sampling_rate):

  r_peak_indices = np.where(peaks == 1)[0]
  rr_intervals_samples = np.diff(r_peak_indices)
  rr_intervals = rr_intervals_samples/sampling_rate

  diffs = np.diff(rr_intervals)
  squared_diffs = diffs**2
  mean_squared_diffs = np.mean(squared_diffs)
  rmssd = np.sqrt(mean_squared_diffs)

  return rmssd


def HRV_FREQUENCY(ecg, peaks, fs, max_freq=2.0, freq_ranges={"LF":(0.0, 0.16), "HF":(0.16, 0.4)}):

  assert max_freq <= 2.0

  if peaks.sum()<1:
    results = {}
    for band in freq_ranges:
      results[band] = None
    results["FF"] = None
    results["frequencies"] = None
    results["psd"] = None
  elif peaks.sum()>=1:

    fs_interp = 1.5*max_freq
    rr_intervals = interpolate_peaks(peaks, len(ecg)/fs, fs_interp)

    window = 'hann'
    nperseg = int(len(rr_intervals) / 1)  # Length of each segment
    noverlap = nperseg // 2  # 50% overlap

    frequencies, psd = signal.welch(
        rr_intervals, fs=fs_interp,
        window=window, nperseg=nperseg,
        noverlap=noverlap, detrend='constant',
        scaling='density'
        )

    def power_in_band(band):
          """Calculate power in a given frequency band"""
          mask = (frequencies >= band[0]) & (frequencies <= band[1])
          return np.trapz(psd[mask], frequencies[mask])

    results = {}

    for band in freq_ranges:
      results[band] = power_in_band(freq_ranges[band])
    results["FF"] = power_in_band((0, max_freq))
    results["frequencies"] = frequencies
    results["psd"] = psd
  else:
    results = {}
    for band in freq_ranges:
      results[band] = None
    results["FF"] = None
    results["frequencies"] = None
    results["psd"] = None

  return results

def poincare_indices(peaks, sampling_rate):
  r_peak_indices = np.where(peaks == 1)[0]
  rr_intervals_samples = np.diff(r_peak_indices)
  rr_intervals = rr_intervals_samples/sampling_rate

  rr_n = rr_intervals[:-1]
  rr_n1 = rr_intervals[1:]

  diff_rr = np.diff(rr_intervals) / np.sqrt(2)
  mean_rr = np.mean(rr_intervals)

  sd1 = np.std(diff_rr)
  sd2 = np.sqrt(2 * np.var(rr_intervals) - sd1**2)

  area = np.pi * sd1 * sd2

  ratio = sd1/sd2

  results = {
      "SD1": sd1,
      "SD2": sd2,
      "SD1_SD2_Ratio": ratio,
      "ellipse_area": area,
      "mean_RR": mean_rr
  }

  return results

def RRV_TIME(peaks, troughs, sampling_rate):
  peak_times = peaks/sampling_rate
  trough_times = troughs/sampling_rate

  breath_intervals = np.diff(peak_times)

  sdbb =  np.std(breath_intervals)
  if sdbb!=0 or sdbb!=math.nan:
    rmssd = np.sqrt( np.mean( np.diff(breath_intervals)**2 ) )
    sdsd = np.std( np.diff(breath_intervals) )
    return {
        "SDBB": sdbb,
        "RMSSD": rmssd,
        "SDSD": sdsd
    }
  else:
    return {
        "SDBB": math.nan,
        "RMSSD": math.nan,
        "SDSD": math.nan
    }

def RRV_FREQ(resp, peaks, troughs, sampling_rate, max_freq=2.0, freq_ranges={"LF":(0.0, 0.16), "HF":(0.16, 0.4), "FF":(0.0, 2.0)}):

  if len(peaks==0) or len(troughs)==0:
    return {band: 0 for band in freq_ranges}
  
  samples = np.concatenate([peaks, troughs])
  samples = np.sort(samples)

  peak_times = peaks/sampling_rate
  trough_times = troughs/sampling_rate

  times = np.concatenate([peak_times, trough_times])
  times = np.sort(times)

  breath_intervals = np.diff(times)
  if len(breath_intervals) < 2:
    return {band: 0 for band in freq_ranges}
  
  time_points = np.cumsum(breath_intervals)
  time_points = np.insert(time_points, 0, 0)

  fs_interp = 1.5*max_freq
  t_interp = np.arange(time_points[0], time_points[-1], 1/fs_interp)
  # kind = "cubic" if len(breath_intervals)>4 else "linear"
  kind = "linear"
  f_method = interp1d(time_points, resp[samples], kind=kind, bounds_error=False, fill_value="extrapolate")
  breath_intervals_interp = f_method(t_interp)

  window = signal.windows.hann(len(breath_intervals_interp))
  breath_intervals_windowed = (breath_intervals_interp - np.mean(breath_intervals_interp)) * window

  frequencies, psd = signal.welch(
      breath_intervals_windowed,
      fs=fs_interp,
      nperseg=min(len(breath_intervals_windowed), 256),
      noverlap=min(len(breath_intervals_windowed) // 4, 64),
      scaling="density"
  )

  results = {}
  for band_name, (band_low, band_high) in freq_ranges.items():
    mask = (frequencies >= band_low) & (frequencies <= band_high)
    results[band_name] = np.trapz(psd[mask], frequencies[mask]) if np.any(mask) else 0

  return results

def CARDIORESPIRATORY_COUPLING(resp, peaks, fs):

  if peaks.sum()<1:
    return {
      "coherence": 0,
      "phase_sync": 0
      }

  resp_phase = np.angle(signal.hilbert(resp))
  rr_intervals = interpolate_peaks(peaks, len(resp)/fs, fs)

  f, Cxy = signal.coherence(resp, rr_intervals, fs, nperseg=min(int(0.05*len(resp)), 256),  noverlap=min(int(0.02*len(resp)), 128))
  resp_band = (f>=0.15) & (f<=0.4)
  coherence = np.mean(Cxy[resp_band])

  rr_phase = np.angle(signal.hilbert(rr_intervals))
  phase_diff = resp_phase - rr_phase
  sync_index = np.abs(np.mean(np.exp(1j * phase_diff)))

  return {
      "coherence": coherence,
      "phase_sync": sync_index
  }

@nb.jit(nopython=True)
def calculate_statistics(eda_signal):
  mean_value = np.mean(eda_signal)
  median_value = np.median(eda_signal)
  std_dev = np.std(eda_signal)
  n = len(eda_signal)
  if n == 0 or std_dev == 0:
    skewness_value = 0
  else:
    skewness_value = np.sum(((eda_signal - mean_value) / std_dev) ** 3) / n
  if n == 0 or std_dev == 0:
    kurtosis_value = 0
  else:
    kurtosis_value = np.sum(((eda_signal - mean_value) / std_dev) ** 4) / n - 3

  return {
    'mean': mean_value,
    'median': median_value,
    'std': std_dev,
    'skew': skewness_value,
    'kurtosis': kurtosis_value
  }

def EDA_STATISTICS(eda_signal):
  stats_dict = calculate_statistics(eda_signal)
  return dict(stats_dict)

def SCR_FEATURES(eda_signal, low, high, sampling_rate=128, min_amplitude=0.01):

    eda = np.array(eda_signal).flatten()
    eda_filtered = signal.lfilter(low[0], low[1], eda)
    eda_phasic = signal.filtfilt(high[0], high[1], eda_filtered)
    
    # Step 3: Detect SCR peaks
    # Find all peaks
    peaks, _ = signal.find_peaks(eda_phasic, height=min_amplitude, distance=int(0.5 * sampling_rate))
    
    if len(peaks) < 2:
      return {
          "amplitude_mean": 0,
          "amplitudes_std": 0,
          "risetime_mean": 0,
          "risetime_std": 0,
          "recovery_mean": 0,
          "recovery_std": 0,
          "domnt_freq": 0
      }
    
    # Step 4: Extract SCR features
    amplitudes = []
    rise_times = []
    recovery_times = []
    
    for i, peak in enumerate(peaks[:-1]):
      # Find onset (minimum before peak)
      onset_window = eda_phasic[max(0, peak-int(3 * sampling_rate)):peak]
      if len(onset_window) == 0:
        continue
      onset_idx = max(0, peak-int(3 * sampling_rate)) + np.argmin(onset_window)
        
      # Find recovery (at 63% recovery, or before next peak)
      end_idx = min(len(eda_phasic), peaks[i+1])
      recovery_window = eda_phasic[peak:end_idx]
      if len(recovery_window) == 0:
        continue
          
      # Calculate target amplitude for 63% recovery
      peak_amplitude = eda_phasic[peak]
      onset_amplitude = eda_phasic[onset_idx]
      amplitude = peak_amplitude - onset_amplitude
      
      if amplitude <= min_amplitude:
        continue
          
      target_amp = peak_amplitude - (0.63 * amplitude)
      
      # Find closest point to target amplitude
      recovery_points = np.where(recovery_window <= target_amp)[0]
      if len(recovery_points) == 0:
        recovery_idx = end_idx
      else:
        recovery_idx = peak + recovery_points[0]
      
      # Calculate metrics
      amplitudes.append(amplitude)
      rise_times.append((peak - onset_idx) / sampling_rate)  # in seconds
      recovery_times.append((recovery_idx - peak) / sampling_rate)  # in seconds
    
    # Step 5: Calculate dominant frequency using FFT
    eda_phasic_downsampled = eda_phasic[::2]
    yf = rfft(eda_phasic_downsampled)
    xf = rfftfreq(len(eda_phasic_downsampled), 2/sampling_rate)
    
    # Only consider positive frequencies up to the Nyquist frequency
    positive_freqs = xf[:len(xf)//2]
    power_spectrum = np.abs(yf[:len(xf)//2])
    
    # Find dominant frequency (excluding DC component at 0 Hz)
    mask = positive_freqs > 0.05  # Exclude very low frequencies (< 0.05 Hz)
    dominant_freq = positive_freqs[mask][np.argmax(power_spectrum[mask])]
    
    # Step 6: Compile results
    scr_features = {
      "amplitude_mean": np.mean(amplitudes) if amplitudes else 0,
      "amplitudes_std": np.std(amplitudes) if len(amplitudes) > 1 else 0,
      "risetime_mean": np.mean(rise_times) if rise_times else 0,
      "risetime_std": np.std(rise_times) if len(rise_times) > 1 else 0,
      "recovery_mean": np.mean(recovery_times) if recovery_times else 0,
      "recovery_std": np.std(recovery_times) if len(recovery_times) > 1 else 0,
      "domnt_freq": dominant_freq
    }
    
    return scr_features

def extract_features(arr, fs_):
    
    # --> signals
    ecg_ = arr[:, 3]
    resp_ = arr[:, 2]
    emg_ = arr[:, 1]
    eda_ = arr[:, 0]

    downsample_factor = int(fs_/10)
    if downsample_factor > 1:
      ecg = signal.decimate(ecg_, downsample_factor, ftype='fir', zero_phase=True)
      resp = signal.decimate(resp_, downsample_factor, ftype='fir', zero_phase=True)
      # emg = signal.decimate(emg_, downsample_factor, ftype='fir', zero_phase=True)
      eda = signal.decimate(eda_, downsample_factor, ftype='fir', zero_phase=True)
      fs = fs_ / downsample_factor
    else: 
      fs = fs_
      ecg = ecg_
      resp = resp_
      # emg = emg_
      eda = eda_

    # --> ECG Features
    smooth_window = max(1/fs_, min(len(ecg)//10, 5)/fs_)
      # --> Find ECG Peaks
    ecg_peaks, info = nk.ecg_peaks(ecg_, fs_, correct_artifacts=False, smoothwindow=smooth_window)
    if len(ecg_peaks["ECG_R_Peaks"]) < 3:
      ecg_peaks, info = nk.ecg_peaks(ecg_, fs_, correct_artifacts=True, smoothwindow=smooth_window)
    r_peaks = np.array(ecg_peaks["ECG_R_Peaks"])

      # --> Calculate Heart Rate Values
    heart_rates = nk.ecg_rate(ecg_peaks, fs_)
    mean_hr = np.mean(heart_rates)
    hrv = np.std(heart_rates)

      # --> Calclate NN metrics
    mean_nn, sdnn, median_nn = HRV_NN(r_peaks, fs_)
    rmssd = HRV_RMSSD(r_peaks, fs_)

      # --> heart rate frequency-domain features
    hrv_freq = HRV_FREQUENCY(ecg, r_peaks, fs_)

      # --> heart rate non-linear features
    poincare = poincare_indices(r_peaks, fs_)
    SampEn = nk.entropy_sample(ecg)[0]

    # --> Respiratory Features
    clean_resp = nk.rsp_clean(resp, sampling_rate=fs, method="hampel")
    clean_resp -= np.mean(clean_resp)
      # --> Respiration Rate
    resp_rates = nk.rsp_rate(clean_resp, sampling_rate=fs, window=int(0.25*len(clean_resp)/fs), method="xcorr")
    mean_rr = np.mean(resp_rates)
      # --> Find Respiration Peaks
    signal_std = np.std(clean_resp)
    min_dist = int(0.5*fs)
    resp_peaks, _ = signal.find_peaks(
        clean_resp, height=0.2*signal_std, distance=min_dist,
        prominence=0.5*signal_std)
    troughs, _ = signal.find_peaks(
        -clean_resp, height=-0.2*signal_std, distance=min_dist,
        prominence=0.5*signal_std)
      # --> Respiratory Variability
    rrv_time = RRV_TIME(resp_peaks, troughs, fs)
    rrv_freq = RRV_FREQ(clean_resp, resp_peaks, troughs, fs)
      # --> Cardiorespiratory Coupling
    CRC = CARDIORESPIRATORY_COUPLING(clean_resp, r_peaks, fs)

    # --> EDA features
    eda_clean = nk.eda_clean(eda, sampling_rate=fs)
    eda_stats = EDA_STATISTICS(eda)
    try:
      eda_decomposed = nk.eda_phasic(eda_clean, sampling_rate=fs)
      SCL = np.mean(eda_decomposed["EDA_Tonic"])
    except:
      SCL = np.mean(eda_clean)
    low_coeffs = signal.butter(2, 4/(fs/2), 'lowpass')
    high_coeffs = signal.butter(1, 0.05/(fs/2), 'highpass')
    SCR = SCR_FEATURES(eda, low_coeffs, high_coeffs, fs)

    # --> ARRANGE INSTANCE FOR FEATURE DATASET
    instance = {
      # Heart Rate Features
      "mean_HR": mean_hr, "HRV": hrv,
      "mean_NN": mean_nn, "std_NN": sdnn, "median_NN": median_nn, "HR_RMSSD": rmssd,
      "HRV_LF": hrv_freq['LF'], "HRV_HF": hrv_freq['HF'], "HRV_FF": hrv_freq['FF'],
      "SD1": poincare["SD1"], "SD2": poincare["SD2"], 
      "SD1/SD2": poincare["SD1_SD2_Ratio"], "ellipse_area": poincare["ellipse_area"],
      "ECG_SampEn": SampEn,
      
      # Respiration Features
      "mean_RR": mean_rr,
      "SDBB": rrv_time["SDBB"], "RR_RMSSD": rrv_time["RMSSD"], "SDSD": rrv_time["SDSD"],
      "RRV_LF": rrv_freq['LF'], "RRV_HF": rrv_freq['HF'], "RRV_FF": rrv_freq['FF'],
      "CRC_coherence": CRC["coherence"], "CRC_phase_sync": CRC["phase_sync"],
      
      # EDA Features
      "EDA_SCL": SCL
    }
    
    # --> Complete EDA Features
    for k in eda_stats.keys():
        instance[f"EDA_{k}"] = eda_stats[k]
    for k in SCR.keys():
        instance[f"phasic_{k}"] = SCR[k]
    instance["EDA_SCL"] = SCL

    return instance