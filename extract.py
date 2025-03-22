import sys
import math
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal, integrate
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis

from functions import TIME_WINDOW

def interpolate_peaks(peaks, time_duration, fs):

  r_peaks = np.where(peaks==1)[0]
  rr_intervals = np.diff(r_peaks)/fs

  time_rr = np.cumsum(rr_intervals)
  time_rr = np.insert(time_rr, 0, 0)
  time_points = np.arange(time_duration*fs)/fs

  kind = "cubic" if len(rr_intervals) > 4 else "linear"

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

    fs_interp = 2*max_freq
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

def RRV_FREQ(resp, peaks, troughs, sampling_rate, max_freq=2.0, freq_ranges={"LF":(0.0, 0.16), "HF":(0.16, 0.4)}):

  if len(peaks==0) or len(troughs)==0:
    results = {}
    for band in freq_ranges:
      results[band] = 0
    results["FF"] = 0
    results["frequencies"] = None
    results["psd"] = None
    return results
  
  samples = np.concatenate([peaks, troughs])
  samples = np.sort(samples)

  peak_times = peaks/sampling_rate
  trough_times = troughs/sampling_rate

  times = np.concatenate([peak_times, trough_times])
  times = np.sort(times)

  breath_intervals = np.diff(times)
  time_points = np.cumsum(breath_intervals)
  time_points = np.insert(time_points, 0, 0)

  fs_interp = 2*max_freq
  t_interp = np.arange(time_points[0], time_points[-1], 1/fs_interp)
  kind = "cubic" if len(breath_intervals)>4 else "linear"
  f_method = interp1d(time_points, resp[samples], kind=kind)
  breath_intervals_interp = f_method(t_interp)

  breath_intervals_dereferenced = breath_intervals_interp - np.mean(breath_intervals_interp)

  window = signal.windows.hann(len(breath_intervals_dereferenced))
  breath_intervals_windowed = breath_intervals_dereferenced * window

  frequencies, psd = signal.welch(
      breath_intervals_windowed,
      fs=fs_interp,
      nperseg=len(breath_intervals_windowed),
      scaling="density"
  )

  def power_in_band(band):
    mask = (frequencies >= band[0]) & (frequencies <= band[1])
    return np.trapz(psd[mask], frequencies[mask])

  results = {}
  for band in freq_ranges:
    results[band] = power_in_band(freq_ranges[band])
  results["FF"] = power_in_band((0, max_freq))
  results["frequencies"] = frequencies
  results["psd"] = psd

  return results

def CARDIORESPIRATORY_COUPLING(resp, peaks, fs):

  if peaks.sum()<1:
    return {
      "coherence": None,
      "phase_sync": None
      }
  else:
    resp_analytical = signal.hilbert(resp)
    resp_phase = np.angle(resp_analytical)

    rr_intervals = interpolate_peaks(peaks, len(resp)/fs, fs)

    f, Cxy = signal.coherence(resp, rr_intervals, fs, nperseg=int(0.35*len(resp)))
    resp_band = (f>=0.15) & (f<=0.4)
    coherence = np.mean(Cxy[resp_band])

    phase_diff = resp_phase - np.angle(signal.hilbert(rr_intervals))
    sync_index = np.abs(np.mean(np.exp(1j * phase_diff)))

    return {
        "coherence": coherence,
        "phase_sync": sync_index
    }

def EDA_STATISTICS(eda_signal):
  mean_value = np.mean(eda_signal)
  median_value = np.median(eda_signal)
  std_dev = np.std(eda_signal)
  skewness_value = skew(eda_signal)
  kurtosis_value = kurtosis(eda_signal)

  # Store the results in a dictionary
  stats_dict = {
      'mean': mean_value,
      'median': median_value,
      'std': std_dev,
      'skew': skewness_value,
      'kurtosis': kurtosis_value
  }

  return stats_dict

def extract_features(arr, fs):
    
    # --> signals
    ecg = arr[:, 3]
    resp = arr[:, 2]
    emg = arr[:, 1]
    eda = arr[:, 0]

    # --> heart rate time-domain features
    # try:
    smooth_window = min(len(ecg) // 10, float("inf")) / len(ecg)
    while int(np.rint(smooth_window*fs)) < 1 or int(np.rint(smooth_window*fs))>len(ecg):
      smooth_window = 1/fs
    ecg_peaks, info = nk.ecg_peaks(ecg, fs, correct_artifacts=True, smoothwindow=smooth_window)
    heart_rates = nk.ecg_rate(ecg_peaks, fs)
    mean_hr = np.mean(heart_rates)
    hrv = np.std(heart_rates)
    mean_nn, sdnn, median_nn = HRV_NN(np.array(ecg_peaks["ECG_R_Peaks"]), fs)
    rmssd = HRV_RMSSD(np.array(ecg_peaks["ECG_R_Peaks"]), fs)

    # --> heart rate frequency-domain features
    hrv_freq = HRV_FREQUENCY(ecg, np.array(ecg_peaks["ECG_R_Peaks"]), fs)

    # --> heart rate non-linear features
    poincare = poincare_indices(np.array(ecg_peaks["ECG_R_Peaks"]), fs)
    SampEn = nk.entropy_sample(ecg)[0]

    # --> Respiratory Rate Time Domain Features
    clean_resp = nk.rsp_clean(resp, sampling_rate=fs, method="hampel")
    clean_resp -= np.mean(clean_resp)
    resp_rates = nk.rsp_rate(clean_resp, sampling_rate=fs, window=int(0.25*TIME_WINDOW), method="xcorr")
    mean_rr = np.mean(resp_rates)

    signal_std = np.std(clean_resp)
    resp_peaks, _ = signal.find_peaks(
        clean_resp, height=0.2*signal_std, distance=0.5*fs,
        prominence=0.5*signal_std)
    troughs, _ = signal.find_peaks(
        -clean_resp, height=-0.2*signal_std, distance=0.5*fs,
        prominence=0.5*signal_std)
    rrv_time = RRV_TIME(resp_peaks, troughs, fs)

    # -->  Respiratory Rate Frequency Domain Features
    rrv_freq = RRV_FREQ(clean_resp, resp_peaks, troughs, fs)

    # --> Cardiorespiratory Coupling
    CRC = CARDIORESPIRATORY_COUPLING(
        clean_resp, np.array(ecg_peaks["ECG_R_Peaks"]), fs
        )

    # --> EDA features
    try:
      eda = nk.eda_process(eda, sampling_rate=fs, method_phasic="highpass", method_peaks="neurokit")[0]
          # --> Statistical features: mean, median, std, skew, kurtosis
      eda_stats = EDA_STATISTICS(np.array(eda["EDA_Raw"]))
          # --> Tonic Parameters: Mean Skin Conductance Level
      SCL = np.mean(eda["EDA_Tonic"])
          # --> Phasic Skin Conductance Response Parameters
      SCR = {}
          #--> Amplitude
      amplitude_indices = np.where(eda["SCR_Amplitude"]!=0)[0]
      amplitudes = np.array(eda["SCR_Amplitude"][amplitude_indices])
      SCR["amplitude_mean"] = amplitudes.mean()
      SCR["amplitudes_std"] = amplitudes.std()
          # --> Rise Time
      rise_indices = np.where(eda["SCR_RiseTime"]!=0)[0]
      rise_times = np.array(eda["SCR_RiseTime"][rise_indices])
      SCR["risetime_mean"] = rise_times.mean()
      SCR["risetime_std"] = rise_times.std()
          # --> Recovery Time
      recovery_times = np.where(eda["SCR_Recovery"]==1)[0]/fs
      SCR["recovery_mean"] = recovery_times.mean()
      SCR["recovery_std"] = recovery_times.std()
          # --> Frequency
      phasic_signal = np.array(eda["EDA_Phasic"])
      n = len(phasic_signal)
      frequencies = fftfreq(n, 1/fs)[:n//2]
      fft_values = fft(phasic_signal)[:n//2]
      frequencies = frequencies[:n//2]
      SCR["domnt_freq"] = frequencies[np.argmax(fft_values)]
    except Exception as e:
      eda_cleaned = nk.eda_clean(eda, sampling_rate=fs)
      signals = pd.DataFrame({"EDA_Raw": eda, "EDA_Clean": eda_cleaned})
      eda_stats = EDA_STATISTICS(eda)
      eda_decomposed = nk.eda_phasic(eda_cleaned, sampling_rate=fs)
      SCL = np.mean(eda_decomposed["EDA_Tonic"])

      SCR = {}
      SCR["amplitude_mean"] = math.nan
      SCR["amplitudes_std"] = math.nan
      SCR["risetime_mean"] = math.nan
      SCR["risetime_std"] = math.nan
      SCR["recovery_mean"] = math.nan
      SCR["recovery_std"] = math.nan

    # --> ARRANGE INSTANCE FOR FEATURE DATASET
    instance = {}
    #--> Heart Rate Features
    instance["mean_HR"], instance["HRV"] = (mean_hr, hrv)
    instance["mean_NN"], instance["std_NN"], instance["median_NN"], instance["HR_RMSSD"] = (mean_nn, sdnn, median_nn, rmssd)
    instance["HRV_LF"], instance["HRV_HF"], instance["HRV_FF"] = (hrv_freq['LF'], hrv_freq['HF'], hrv_freq['FF'])
    instance["SD1"], instance["SD2"], instance["SD1/SD2"], instance["ellipse_area"] = (
        poincare["SD1"],
        poincare["SD2"],
        poincare["SD1_SD2_Ratio"],
        poincare["ellipse_area"])
    instance["ECG_SampEn"] = SampEn
    # --> Respiration Rate Features
    instance["mean_RR"] = mean_rr
    instance["SDBB"], instance["RR_RMSSD"], instance["SDSD"] = (rrv_time["SDBB"], rrv_time["RMSSD"], rrv_time["SDSD"] )
    instance["RRV_LF"], instance["RRV_HF"], instance["RRV_FF"] = (rrv_freq['LF'], rrv_freq['HF'], rrv_freq['FF'])
    instance["CRC_coherence"], instance["CRC_phase_sync"] = (CRC["coherence"], CRC["phase_sync"])
    # --> Electrodermal Activity Features
    for k in eda_stats.keys():
        instance[f"EDA_{k}"] = eda_stats[k]
    for k in SCR.keys():
        instance[f"phasic_{k}"] = SCR[k]
    instance["EDA_SCL"] = SCL

    return instance