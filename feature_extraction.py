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

import importlib
import functions as fnc
importlib.reload(fnc)
import sp as sp
importlib.reload(sp)

from functions import STIMULUS_FILENAME, XDF_DIR, TARGET_STREAM, STIMULUS_STREAMS, RESTING_STREAMS, MARKER_STREAMS, TARGET_EVENTS, LABELS, DURATIONS
from functions import PREPROCESSED_DIR, FEATURES_DIR, DATASET_DIR, TIME_WINDOW 
from functions import CHILL_EVENTS_DIR, CHILLS_DATA_DIR, PRE_CHILLS_DATA_DIR, POST_CHILLS_DATA_DIR, NON_CHILLS_DATA_DIR

from scipy import signal, integrate
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from extract import *


def main():

    extracted_data = {
        "PRE-CHILL": {},
        "POST-CHILL": {},
        "CHILL": {},
        "NON-CHILL": {}
    }    

    # --> Get Name of All subject xdf files
    subjects_info = fnc.subjects_info(STIMULUS_FILENAME)
    subjects = os.listdir(XDF_DIR)
    subjects = [subject[:-4] for subject in subjects]
    # subjects = ["5006", "5011"]

    features_df = pd.DataFrame()

    for s, subject in enumerate(subjects):

        '''PREPROCECSSING OF SUBJECTS DATA'''
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

        '''GET NON_CHILLS, PRE-CHILLS, AND CHILLS DATA'''
        # --> PREDEFS
        subject_data = preprocessed_data
        pre_chills_data = {phase: [] for phase in stimuli_phases}
        post_chills_data = {phase: [] for phase in stimuli_phases}
        chills_data = {phase: [] for phase in stimuli_phases}
        non_chills_data = {phase: [] for phase in rest_phases}

        # --> EXTRACT PRE-CHILL AND CHILL DATA FOR SUBJECT
        if subject in subjects_with_chills:

            # --> APPLY WINDOW TO CHILL EVENTS TO GET NON-OVERLAPPING CHILL REPORTS
            ts = subject_data["chills"]["ChillsReport"] #--> timestamps of all subject chill event reports
            pre_time_window = post_time_window = TIME_WINDOW
            # around_time_window = TIME_WINDOW / 2
            consolidated_chills = [ts[0]]
            for t in ts[1:]:
                if t - consolidated_chills[-1] > pre_time_window:
                    consolidated_chills.append(t)
            chills_ts = np.array(consolidated_chills)
            del consolidated_chills, ts

            # --> GET PRE-CHILLS & CHILLS DATA FOR SUBJECT
            for p, phase in enumerate(stimuli_phases):
                if phase in subject_data:
                    pre_chills_data[phase] = []
                    post_chills_data[phase] = []
                    chills_data[phase] = []

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
                        sig = None
                        for ts in chills_ts:
                            idx_pre = np.where(phase_ts <= ts)[0]
                            idx_post = np.where(phase_ts > ts)[0]

                            # --> GET PRE-CHILL DATA
                            if len(idx_pre) != 0:
                                sig_pre = signal[idx_pre, :]
                                n_samples = int(fs*pre_time_window) # --> no of samples before chill report
                                if sig_pre[-n_samples:, :].shape[0] < n_samples: # --> Augment, if pre-chill data length less than n_samples
                                    print("---------------------------------------------------")
                                    print(f"{subject} {phase} chills timestamps {np.where(chills_ts==ts)[0]} out of {len(chills_ts)} with insufficient pre_chills samples: {sig_pre[-n_samples:, :].shape[0]}<{n_samples}")
                                    print("---------------------------------------------------")
                                    aug_sig = np.median(sig_pre[-n_samples:, :], axis=0)    # --> median of each channel or signal
                                    aug_sig = np.tile(aug_sig, (n_samples-sig_pre[-n_samples:, :].shape[0], 1)) # --> fill missing samples with median of each signal
                                    aug_sig = np.vstack((aug_sig, sig_pre)) # --> Stack To Form New Signal
                                    pre_chills_data[phase].append(aug_sig)
                                    del aug_sig
                                else:
                                    pre_chills_data[phase].append(sig_pre[-n_samples:, :])
                                    print(f"{subject} {phase} pre-chill events data extracted")

                                # --> GET POST-CHILL DATA
                                if len(idx_post) != 0:
                                    sig_post = signal[idx_post, :]
                                    n_samples = int(fs*post_time_window) # --> no of samples before and after chill report
                                    temp_sig = np.vstack([
                                            sig_pre[-n_samples//2:, :],
                                            sig_post[:n_samples//2,:]
                                        ])
                                    
                                    if temp_sig.shape[0] < n_samples: # --> Augment, if chill data length less than 2*n_samples
                                        print("---------------------------------------------------")
                                        print(f"{subject} {phase} chills timestamps {np.where(chills_ts==ts)[0]} out of {len(chills_ts)} with insufficient chills samples: {temp_sig.shape[0]}<{2*n_samples}")
                                        print("---------------------------------------------------")
                                        aug_sig = np.median(temp_sig, axis=0)   # --> median of each channel or signal
                                        aug_sig = np.tile(aug_sig, (2*n_samples - temp_sig.shape[0], 1))    # --> fill missing samples with median of each signal
                                        aug_sig = np.vstack((temp_sig, aug_sig))    # --> Stack To Form New Signal
                                        chills_data[phase].append(aug_sig)
                                        del aug_sig
                                    else:
                                        chills_data[phase].append(temp_sig)
                                        print(f"{subject} {phase} chill events data extracted")
                                    
                                    if sig_post[-n_samples:, :].shape[0] < n_samples: # --> Augment, if post-chill data length less than n_samples
                                        print("---------------------------------------------------")
                                        print(f"{subject} {phase} chills timestamps {np.where(chills_ts==ts)[0]} out of {len(chills_ts)} with insufficient post_chills samples: {sig_post[-n_samples:, :].shape[0]}<{n_samples}")
                                        print("---------------------------------------------------")
                                        aug_sig = np.median(sig_post[-n_samples:, :], axis=0)    # --> median of each channel or signal
                                        aug_sig = np.tile(aug_sig, (n_samples-sig_post[-n_samples:, :].shape[0], 1)) # --> fill missing samples with median of each signal
                                        aug_sig = np.vstack((aug_sig, sig_post)) # --> Stack To Form New Signal
                                        post_chills_data[phase].append(aug_sig)
                                        del aug_sig
                                    else:
                                        post_chills_data[phase].append(sig_post[-n_samples:, :])
                                        print(f"{subject} {phase} post-chill events data extracted")

                                    del temp_sig, sig_post
                                del sig_pre, n_samples, idx_pre, idx_post
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
        subject_features_df = pd.DataFrame()
        # --> PRE_CHILLS FEATURES
        for p, phase in enumerate(pre_chills_data.keys()):
            for a, arr in enumerate(pre_chills_data[phase]):
                fs = arr.shape[0]/TIME_WINDOW
                features = {}
                features["id"] = f"{subject}_pre_{phases.index(phase)+1}_{a}"
                features["stimuli"] = phase
                features["label"] = "PRE-CHILL"
                features.update(extract_features(arr, fs))
                subject_features_df = pd.concat([subject_features_df, pd.DataFrame([features])], ignore_index=True)
        # --> POST_CHILLS FEATURES
        for p, phase in enumerate(post_chills_data.keys()):
            for a, arr in enumerate(post_chills_data[phase]):
                fs = arr.shape[0]/TIME_WINDOW
                features = {}
                features["id"] = f"{subject}_post_{phases.index(phase)+1}_{a}"
                features["stimuli"] = phase
                features["label"] = "POST-CHILL"
                features.update(extract_features(arr, fs))
                subject_features_df = pd.concat([subject_features_df, pd.DataFrame([features])], ignore_index=True)
        # --> CHILLS FEATURES
        for p, phase in enumerate(chills_data.keys()):
            for a, arr in enumerate(chills_data[phase]):
                fs = arr.shape[0]/TIME_WINDOW
                features = {}
                features["id"] = f"{subject}_chill_{phases.index(phase)+1}_{a}"
                features["stimuli"] = phase
                features["label"] = "CHILL"
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
        extracted_data["PRE-CHILL"][subject] = pre_chills_data
        extracted_data["POST-CHILL"][subject] = post_chills_data
        extracted_data["CHILL"][subject] = chills_data
        extracted_data["NON-CHILL"][subject] = non_chills_data

    features_df.to_csv(f"{FEATURES_DIR}/all_features.csv")
    joblib.dump(extracted_data["PRE-CHILL"], PRE_CHILLS_DATA_DIR)
    joblib.dump(extracted_data["POST-CHILL"], POST_CHILLS_DATA_DIR)
    joblib.dump(extracted_data["CHILL"], CHILLS_DATA_DIR)
    joblib.dump(extracted_data["NON-CHILL"], NON_CHILLS_DATA_DIR)

if __name__ == "__main__":
    main()