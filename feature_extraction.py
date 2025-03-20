import os
import yaml

from itertools import chain
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyxdf
import neurokit2 as nk
import joblib

import functions as fnc
import sp as sp

from functions import STIMULUS_FILENAME, XDF_DIR, TARGET_STREAM, STIMULUS_STREAMS, RESTING_STREAMS, MARKER_STREAMS, FRISSON_MARKER_STREAMS, TARGET_EVENTS, LABELS, DURATIONS
from functions import PREPROCESSED_DIR, FEATURES_DIR, DATASET_DIR 
from functions import CHILL_EVENTS_DIR, CHILLS_DATA_DIR, NON_CHILLS_DATA_DIR
from functions import TIME_WINDOW, PLOT_TIME_WINDOW, CHILLS_REPORT_WINDOW

from scipy import signal, integrate
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from extract import *

from IPython.display import clear_output
import time

import logging
import warnings
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")


def main():

    extracted_data = {
        "CHILL": {},
        "NON-CHILL": {}
    }    

    # --> Get Name of All subject xdf files
    subjects_info = fnc.subjects_info(STIMULUS_FILENAME)
    subjects = os.listdir(XDF_DIR)
    subjects = [subject[:-4] for subject in subjects]
    # subjects = ["5006", "5011"]
    # subjects = ["5022"]

    features_df = pd.DataFrame()

    for s, subject in enumerate(subjects):

        '''PREPROCECSSING OF SUBJECTS DATA'''
        print(f"---------------------------------------------------------------SUBJECT: {subject} | {s}/{len(subjects)} ---------------------------------------------------------------------------------------------------------------------------------")
        subject_path = os.path.join(PREPROCESSED_DIR, f"{subject}_processed_data.sav")
        if os.path.exists(os.path.join(subject_path))!=True: # --> Enusre subject xdf file has not been preprocessed before
            print(f"{s+1}/{len(subjects)}: {subject}")
            print(f"-------------------------------------------------------------------")
            
            # --> GET ORDER OF STIMULI SESSIONS
            stimulus_order = fnc.get_stimulus_order(subject, subjects_info)
            if not stimulus_order:  # -->  If no resting and musical stimuli sessions, skip
                continue
            print(f"Stimulus Order for subject {subject}: {stimulus_order}")
            
            # --> LOAD XDF FILE
            xdf_file_path = os.path.join(XDF_DIR, f"{subject}.xdf")
            try:
                streams, _ = pyxdf.load_xdf(xdf_file_path)
                chills_timestamps = fnc.process_markers(streams, MARKER_STREAMS, TARGET_EVENTS)
                frisson_timestamps = fnc.process_markers(streams, FRISSON_MARKER_STREAMS, TARGET_EVENTS)
            except FileNotFoundError:
                print(f"Error: The file {xdf_file_path}.xdf was not found.")
                continue
            except Exception as e:
                print(f"An error occurred while loading {xdf_file_path}.xdf: {e}")
                continue

            # --> GET MUSIC STIMULI/RESTING STATES ONSET
            stimulus_latencies = fnc.process_markers(streams, STIMULUS_STREAMS, TARGET_EVENTS)
            resting_latencies = fnc.process_markers(streams, RESTING_STREAMS, TARGET_EVENTS)
            print(f"Stimulus latencies: {stimulus_latencies}")
            print(f"Resting latencies: {resting_latencies}")
            for stream in STIMULUS_STREAMS:
                stimulus_latencies_ordered = {i: stimulus_latencies[stream][stimulus_order.index(i)] for i in
                                                stimulus_order}
            print(f"Stimulus Latencies: {stimulus_latencies_ordered}")

            # --> EXTRACT DATA
            data, stream = fnc.extract_raw_data(streams, TARGET_STREAM, LABELS)
            if data is None:
                continue

            # --> GET SAMPLING RATE
            timestamps = data["Timestamps"]
            effective_fs = sp.estimate_sampling_rate(timestamps)
            sampling_rate = int(np.round(effective_fs))
            print(f"Sampling Rate: {sampling_rate}")

            if 'desc' in stream['info'] and 'channels' in stream['info']['desc'][0]:
                channel_labels = stream['info']['desc'][0]['channels'][0]['channel']
                channel_names = [ch['label'][0] for ch in channel_labels]
                print("Channel names:", channel_names)

                concatenated_data = fnc.extract_and_concatenate_data(
                    data, timestamps, sampling_rate, stimulus_latencies_ordered, resting_latencies, stimulus_order
                )

                preprocessed_data = dict()
                for epoch in concatenated_data["epochs"]:
                    data = concatenated_data["epochs"][epoch]
                    data["Timestamps"] = concatenated_data["timestamps"][epoch]
                    output = sp.preprocess_data(data)
                    preprocessed_data[epoch] = output
                preprocessed_data["chills"] = chills_timestamps
                preprocessed_data["frisson_chills"] = frisson_timestamps

                if os.path.exists(subject_path) is not True:
                    joblib.dump(preprocessed_data, subject_path)
        else:
            if os.path.exists(subject_path) is False:
                continue
            preprocessed_data = joblib.load(subject_path)

        '''FETCH SUBJECTS EPOCHS/PHASE VALIDITY DATAFRAME'''
        subjects_epochs_validity = pd.read_pickle(os.path.join(DATASET_DIR, "Subject_Epochs_Validity.pkl"))

        '''DEFINE PHASES'''
        phases = ['pre_rest', 'stimulus1', 'stimulus2', 'stimulus3', 'post_rest']
        stimuli_phases = phases[1:-1]
        rest_phases = ['pre_rest', 'post_rest']

        '''FETCH CHILL_EVENTS INFO'''
        chill_events = joblib.load(CHILL_EVENTS_DIR)
        subjects_with_chills = chill_events["subjects"]
        chill_events = chill_events["events"]

        '''GET NON_CHILLS, PRE-CHILLS, POST-CHILLS, AND CHILLS DATA'''
        # --> PREDEFS
        subject_data = preprocessed_data
        chills_data = {phase: [] for phase in stimuli_phases}
        non_chills_data = {phase: [] for phase in rest_phases}
        frisson_chills_data = {phase: [] for phase in stimuli_phases}
        time_window = max(TIME_WINDOW, PLOT_TIME_WINDOW)

        # --> EXTRACT PRE-CHILL AND CHILL DATA FOR SUBJECT
        if subject in subjects_with_chills:

            # --> APPLY WINDOW TO CHILL EVENTS TO GET NON-OVERLAPPING CHILL REPORTS
            ts = subject_data["chills"]["ChillsReport"] #--> timestamps of all subject chill event reports
            consolidated_chills = [ts[0]]
            for t in ts[1:]:
                if t - consolidated_chills[-1] > CHILLS_REPORT_WINDOW:
                    consolidated_chills.append(t)
            chills_ts = np.array(consolidated_chills)
            del consolidated_chills, ts

            # --> APPLY WINDOW TO FRISSON CHILL EVENTS TO GET NON-OVERLAPPING CHILL REPORTS
            fts = subject_data["frisson_chills"]["FrissonStart"]
            frisson_consolidated = [fts[0]]
            for t in fts[1:]:
                if t - frisson_consolidated[-1] > CHILLS_REPORT_WINDOW:
                    frisson_consolidated.append(t)
            frisson_ts = np.array(frisson_consolidated)
            del frisson_consolidated, fts
                

            # --> GET PRE-CHILLS, POST-CHILLS, & CHILLS DATA FOR SUBJECT
            for p, phase in enumerate(stimuli_phases):
                if phase in subject_data:

                    if (subjects_epochs_validity.loc[subject, (phase, "Same_Signal_Length")] is True) and \
                        (subjects_epochs_validity.loc[subject, (phase, "ECG_Fit")] is True) and \
                            (subjects_epochs_validity.loc[subject, (phase, "EDA_Fit")] is True): # --> All signal quality checks must be met
                        # --> FORM MULTIVARIATE SIGNAL OF [GSR/EDA, EMG, RESP, ECG] FOR SUBJECT-PHASE
                        ecg = sp.normalize_signal(np.array(subject_data[phase]["ecg_cleaned"]))
                        resp = sp.normalize_signal(np.array(subject_data[phase]["resp_cleaned"]))
                        emg = sp.normalize_signal(np.array(subject_data[phase]["df"]["EMG"]))
                        gsr = sp.normalize_signal(np.array(subject_data[phase]["df"]["GSR_Interpolated"]))
                        phase_ts = np.array(subject_data[phase]["df"]["Timestamp"])
                        fs = subject_data[phase]["effective_sampling_rate"]
                        signal = np.stack([gsr, emg, resp, ecg], axis=1)

                        # --> EXTRACT SIGNAL OF DURATION TIME_WINDOW PRIOR TO CHILL EVENT, AND TIME_WINDOW//3 AROUND CHILL EVENT REPORT FOR SUBJECT-PHASE
                        for ts_, _data in zip([chills_ts, frisson_ts], [chills_data, frisson_chills_data]):
                            sig = None
                            for ts in ts_:
                                idx_pre = np.where(phase_ts <= ts)[0]
                                idx_post = np.where(phase_ts > ts)[0]
                                n_samples = None

                                # --> GET PRE-CHILL DATA
                                sig_pre, aug_sig, pre_chill = None, None, None
                                if len(idx_pre) != 0:
                                    sig_pre = signal[idx_pre, :]
                                    n_samples = int(fs*time_window) # --> no of samples before chill report
                                    if sig_pre[-n_samples:, :].shape[0] < n_samples: # --> Augment, if pre-chill data length less than n_samples
                                        print("---------------------------------------------------")
                                        print(f"{subject} {phase} chills timestamps {np.where(chills_ts==ts)[0]} out of {len(chills_ts)} with insufficient samples prioir to chill report: {sig_pre[-n_samples:, :].shape[0]}<{n_samples}")
                                        print("---------------------------------------------------")
                                        aug_sig = np.median(sig_pre[-n_samples:, :], axis=0)    # --> median of each channel or signal
                                        aug_sig = np.tile(aug_sig, (n_samples-sig_pre[-n_samples:, :].shape[0], 1)) # --> fill missing samples with median of each signal
                                        pre_chill = np.vstack((aug_sig, sig_pre)) # --> Stack To Form New Signal
                                        print(f"{subject} {phase} chill onset signals extracted")
                                    else:
                                        pre_chill = sig_pre[-n_samples:, :]
                                        print(f"{subject} {phase} chill onset signals sifficient & extracted")

                                # --> GET POST-CHILL DATA
                                sig_post, aug_sig, post_chill = None, None, None
                                if len(idx_post) != 0:
                                    sig_post = signal[idx_post, :]
                                    n_samples = int(fs*time_window) # --> no of samples before and after chill report
                                    if sig_post[-n_samples:, :].shape[0] < n_samples: # --> Augment, if post-chill data length less than n_samples
                                        print("---------------------------------------------------")
                                        print(f"{subject} {phase} chills timestamps {np.where(chills_ts==ts)[0]} out of {len(chills_ts)} with insufficient post_chills samples: {sig_post[-n_samples:, :].shape[0]}<{n_samples}")
                                        print("---------------------------------------------------")
                                        aug_sig = np.median(sig_post[-n_samples:, :], axis=0)    # --> median of each channel or signal
                                        aug_sig = np.tile(aug_sig, (n_samples-sig_post[-n_samples:, :].shape[0], 1)) # --> fill missing samples with median of each signal
                                        post_chill = np.vstack((aug_sig, sig_post)) # --> Stack To Form New Signal
                                        print(f"{subject} {phase} chill offset signal data extracted")
                                    else:
                                        post_chill = sig_post[:n_samples, :]
                                        print(f"{subject} {phase} chill offset signal data sufficient & extracted")
                                    
                                if isinstance(pre_chill, np.ndarray) or isinstance(post_chill, np.ndarray):
                                    _data[phase].append({"pre":pre_chill, "post":post_chill})
                                del sig_pre, sig_post, pre_chill, post_chill, aug_sig, n_samples, idx_pre, idx_post
                            del ecg, resp, emg, gsr, phase_ts, fs, signal

            # --> GET NON-CHILL DATA FOR SUBJECT WITH CHILL REPORT
            for p, phase in enumerate(rest_phases):
                if phase in subject_data:
                    phase_ts = np.array(subject_data[phase]["df"]["Timestamp"])
                    if (np.any(chills_ts[:, None]<=phase_ts) and phase=="pre_rest") or (np.any(chills_ts[:, None]>=phase_ts) and phase=="post_rest"):
                        print(f"{subject} {phase} has false positives, skipped")
                    else:
                        non_chills_data[phase] = []
                        if (subjects_epochs_validity.loc[subject, (phase, "Same_Signal_Length")] is True) and \
                            (subjects_epochs_validity.loc[subject, (phase, "ECG_Fit")] is True) and \
                                (subjects_epochs_validity.loc[subject, (phase, "EDA_Fit")] is True):    # --> All signal quality checks must be met
                            # --> FORM MULTIVARIATE SIGNAL OF [GSR/EDA, EMG, RESP, ECG] FOR SUBJECT-PHASE
                            ecg = sp.normalize_signal(np.array(subject_data[phase]["ecg_cleaned"]))
                            resp = sp.normalize_signal(np.array(subject_data[phase]["resp_cleaned"]))
                            emg = sp.normalize_signal(np.array(subject_data[phase]["df"]["EMG"]))
                            gsr = sp.normalize_signal(np.array(subject_data[phase]["df"]["GSR_Interpolated"]))
                            fs = subject_data[phase]["effective_sampling_rate"]
                            signal = np.stack([gsr, emg, resp, ecg], axis=1)
                            
                            non_chills_data[phase].append(signal)
                            print(f"{subject} {phase} non-chills data extracted")
                            del ecg, resp, emg, gsr, fs, signal
        else:
            # --> GET NON-CHILL DATA FOR SUBJECT NOT HAVING A CHILL REPORT
            for p, phase in enumerate(rest_phases):
                if phase in subject_data:
                    phase_ts = np.array(subject_data[phase]["df"]["Timestamp"])
                    non_chills_data[phase] = []
                    if (subjects_epochs_validity.loc[subject, (phase, "Same_Signal_Length")] is True) and \
                        (subjects_epochs_validity.loc[subject, (phase, "ECG_Fit")] is True) and \
                            (subjects_epochs_validity.loc[subject, (phase, "EDA_Fit")] is True):    # --> All signal quality checks must be met
                        # --> FORM MULTIVARIATE SIGNAL OF [GSR/EDA, EMG, RESP, ECG] FOR SUBJECT-PHASE
                        ecg = sp.normalize_signal(np.array(subject_data[phase]["ecg_cleaned"]))
                        resp = sp.normalize_signal(np.array(subject_data[phase]["resp_cleaned"]))
                        emg = sp.normalize_signal(np.array(subject_data[phase]["df"]["EMG"]))
                        gsr = sp.normalize_signal(np.array(subject_data[phase]["df"]["GSR_Interpolated"]))
                        fs = subject_data[phase]["effective_sampling_rate"]
                        signal = np.stack([gsr, emg, resp, ecg], axis=1)
                        
                        non_chills_data[phase].append(signal)
                        print(f"{subject} {phase} non-chills data extracted")
                        del ecg, resp, emg, gsr, fs, signal
        
        del subject_data, preprocessed_data # --> Delete To Release Memory, CHILL, PRE-CHILL, & NON-CHILL data extracted already

        '''EXTRACT FEATURES'''
        time_window = max(TIME_WINDOW, PLOT_TIME_WINDOW)
        subject_features_df = pd.DataFrame()
        # --> CHILLS FEATURES
        onset_offset = lambda x: "onset" if x=="pre" else "offset"
        for p, phase in enumerate(chills_data.keys()):
            for e, epoch in enumerate(chills_data[phase]):
                for desc in ["pre", "post"]:
                    arr = epoch[desc]
                    if isinstance(arr, np.ndarray):
                        fs = arr.shape[0]/time_window
                        n_samples = int(fs*TIME_WINDOW)
                        arr = arr[-n_samples:, :] if desc=="pre" else arr[:n_samples, :]
                        features = {}
                        features["id"] = f"{subject}_chill_{phases.index(phase)+1}_{e}_{onset_offset(desc)}"
                        features["stimuli"] = phase
                        features["label"] = "CHILL_"+onset_offset(desc).upper()
                        features.update(extract_features(arr, fs))
                        subject_features_df = pd.concat([subject_features_df, pd.DataFrame([features])], ignore_index=True)
        # --> FRISSON CHILLS FEATURES
        onset_offset = lambda x: "onset" if x=="pre" else "offset"
        for p, phase in enumerate(frisson_chills_data.keys()):
            for e, epoch in enumerate(frisson_chills_data[phase]):
                for desc in ["pre", "post"]:
                    if isinstance(arr, np.ndarray):
                        fs = arr.shape[0]/time_window
                        n_samples = int(fs*TIME_WINDOW)
                        arr = arr[-n_samples:, :] if desc=="pre" else arr[:n_samples, :]
                        features = {}   
                        features["id"] = f"{subject}_augmented_chill_{phases.index(phase)+1}_{e}_{onset_offset(desc)}"  
                        features["stimuli"] = phase
                        features["label"] = "AUGMENTED_CHILL_"+onset_offset(desc).upper()
                        features.update(extract_features(arr, fs))
                        subject_features_df = pd.concat([subject_features_df, pd.DataFrame([features])], ignore_index=True)
        # --> NON_CHILLS FEATURES
        for p, phase in enumerate(non_chills_data.keys()):
            for d, data in enumerate(non_chills_data[phase]):
                fs = data.shape[0] / 300

                T, _ = data.shape
                epoch_length = int(TIME_WINDOW*fs)
                num_epochs = T // epoch_length
                overlap = int(0.1*epoch_length)
                step_size = epoch_length - overlap

                for i in range(num_epochs):
                    # --> SELECT TIME_WINDOW(15s) epoch from NON-CHILL data
                    start_idx = i*step_size
                    end_idx = start_idx+epoch_length
                    if end_idx>T:
                        break
                    arr = data[start_idx:end_idx, :]

                    features = {}
                    features["id"] = f"{subject}_{phases.index(phase)+1}_{i}"
                    features["stimuli"] = phase
                    features["label"] = "NON-CHILL"
                    features.update(extract_features(arr, fs))
                    subject_features_df = pd.concat([subject_features_df, pd.DataFrame([features])], ignore_index=True)

        features_df = pd.concat([features_df, subject_features_df], ignore_index=True)
        extracted_data["CHILL"][subject] = chills_data
        extracted_data["NON-CHILL"][subject] = non_chills_data

        clear_output(wait=False)
        # time.sleep(0.025)

    features_df.to_csv(f"{FEATURES_DIR}/all_features.csv")
    joblib.dump(extracted_data["CHILL"], CHILLS_DATA_DIR)
    joblib.dump(extracted_data["NON-CHILL"], NON_CHILLS_DATA_DIR)
    print("Features extracted successfully")

if __name__ == "__main__":
    main()