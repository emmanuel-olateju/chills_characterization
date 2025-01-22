import os
import yaml

import numpy as np
import pandas as pd

with open("configs.yaml", "r") as file:
    configs = yaml.safe_load(file)

STIMULUS_FILENAME = configs["STIMULUS_FILENAME"]
DATASET_DIR = configs["DATASET_DIRECTORY"]
XDF_DIR = configs["XDF_DIRECTORY"]
PREPROCESSED_DIR = configs["PREPROCESSED_DIRECTORY"]
FEATURES_DIR = configs["FEATURES_DIRECTORY"]

TARGET_STREAM = configs["TARGET_STREAM"]
STIMULUS_STREAMS = configs["STIMULUS_STREAMS"]
RESTING_STREAMS = configs["RESTING_STREAMS"]
MARKER_STREAMS = configs["MARKER_STREAMS"]
TARGET_EVENTS = configs["TARGET_EVENTS"]
LABELS = configs["LABELS"]
DURATIONS = configs["DURATIONS"]
ECG_AMPLITUDE = configs["ECG_AMPLITUDE"]
ECG_BASELINEWANDERING = configs["ECG_BASELINEWANDERING"]
ECG_Z_THRESHOLD = configs["ECG_Z_THRESHOLD"]

TIME_WINDOW = configs["TIME_WINDOW"]
CHILL_EVENTS_DIR = configs["CHILL_EVENTS_DIR"]
CHILLS_DATA_DIR = configs["CHILLS_DATA_DIR"]
PRE_CHILLS_DATA_DIR = configs["PRE_CHILLS_DATA_DIR"]
POST_CHILLS_DATA_DIR = configs["POST_CHILLS_DATA_DIR"]
NON_CHILLS_DATA_DIR = configs["NON_CHILLS_DATA_DIR"]

def subjects_info(filename):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file {filename} is empty.")
        return None
    except Exception as e:
        print(f"An error occurred while reading {filename}: {e}")
        return None


def get_stimulus_order(subject_id, subject_info):
    subject_row = subject_info[subject_info.iloc[:, 0].astype(str) == subject_id]
    if subject_row.empty:
        print(f"No stimulus order found for subject {subject_id}.")
        return None
    stimulus_order_str = subject_row.iloc[0, 1]
    return list(map(int, stimulus_order_str.split(',')))

def process_markers(streams, target_streams, target_events):
    latencies = {stream: [] for stream in target_streams}
    for stream in streams:
        streamname = stream['info']['name'][0]
        if streamname in target_streams:
            print(f"Markers from stream: {streamname}")
            if streamname not in MARKER_STREAMS:
                latencies[streamname] = [0]*len(target_events[streamname])
                for ts, marker in zip(stream['time_stamps'], [m[0] for m in stream['time_series']]):
                    if marker in target_events[streamname]:
                        idx = target_events[streamname].index(marker)
                        latencies[streamname][idx] = ts
                        print(f"Marker: {marker} at Timestamp: {ts}")
            else:
                latencies[streamname] = []
                for ts, marker in zip(stream['time_stamps'], [m[0] for m in stream['time_series']]):
                    if marker in target_events[streamname]:
                        latencies[streamname].append(ts)
                        print(f"Marker: {marker} at Timestamp: {ts}")

    return latencies

def extract_raw_data(streams, stream_name, labels):
    for stream in streams:
        if (stream['info']['name'][0] == stream_name) and (stream['info']['type'][0] == 'EEG'):
            channel_labels = [ch['label'][0] if isinstance(ch['label'], list) else ch['label']
                              for ch in stream['info']['desc'][0]['channels'][0]['channel']]
            data = {label: [sample[channel_labels.index(label)] for sample in stream['time_series']]
                    for label in labels if label in channel_labels}
            data['Timestamps'] = stream['time_stamps']
            print(f"Length of extracted data: {len(next(iter(data.values())))}")
            print(f"Length of timestamps: {len(data['Timestamps'])}")
            return data, stream
    print(f"Stream '{stream_name}' not found.")
    return None, None

def extract_and_concatenate_data(data, timestamps, sampling_rate, stimulus_latencies, resting_latencies,
                                 stimulus_order):
    def extract_epoch(start_time, duration):
        start_index = np.searchsorted(timestamps, start_time)
        end_index = start_index + int(duration * sampling_rate)
        return {
            label: data[label][start_index:end_index] for label in LABELS if label in data
        }, timestamps[start_index:end_index]

    epochs = {}
    timestamps_epochs = {}

    if len(resting_latencies['RestingStateStart']) != 0:
        if resting_latencies['RestingStateStart'][0] != 0:
            pre_resting_data, pre_resting_timestamps = extract_epoch(resting_latencies['RestingStateStart'][0],
                                                                    DURATIONS['resting_state'])
            epochs["pre_rest"] = pre_resting_data
            timestamps_epochs["pre_rest"] = pre_resting_timestamps

    for stimulus in stimulus_order:
        duration = DURATIONS[f'stimulus{stimulus}']
        print(f"Stimulus: {stimulus}, duration: {duration}")
        if stimulus_latencies[stimulus] != 0:
            stimulus_data, stimulus_timestamps = extract_epoch(stimulus_latencies[stimulus], duration)
            epochs[f"stimulus{stimulus}"] = stimulus_data
            timestamps_epochs[f"stimulus{stimulus}"] = stimulus_timestamps

    if len(resting_latencies['RestingStateStart']) > 1:
        if resting_latencies['RestingStateStart'][1] != 0:
            post_resting_data, post_resting_timestamps = extract_epoch(resting_latencies['RestingStateStart'][1],
                                                                    DURATIONS['resting_state'])
            epochs["post_rest"] = post_resting_data
            timestamps_epochs["post_rest"] = post_resting_timestamps

    concatenated_data = {
        "epochs": epochs,
        "timestamps": timestamps_epochs
    }

    return concatenated_data