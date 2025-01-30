# Feature Extraction Program Documentation

## Overview
The feature extraction program represents a significant evolution from the original peripheral cleaning script (peripheralCleaning_ICAProject.py). This documentation outlines the major architectural changes, new functionalities, and improvements implemented in the new program structure.

## Program Evolution

### 1. Architectural Changes

#### 1.1 Modular Structure
- **Original**: Single script (peripheralCleaning_ICAProject.py) containing all functionality
- **New**: Split into multiple modules:
  - `feature_extraction.py`: Main script for feature extraction pipeline
  - `functions.py`: Utility functions and configuration management
  - `sp.py`: Signal processing functions
  - `extract.py`: Feature extraction functions

#### 1.2 Configuration Management
- **Original**: Constants defined at the beginning of the script
- **New**: 
  - YAML-based configuration system (`configs.yaml`)
  - Configuration loading and management in `functions.py`
  - Centralized configuration for:
    - File paths and directories
    - Stream identifiers
    - Event labels
    - Signal processing parameters
    - Time windows and durations
    - Quality thresholds

### 2. Key Functions in functions.py

#### 2.1 Configuration Loading
```python
with open("configs.yaml", "r") as file:
    configs = yaml.safe_load(file)
```
- Loads all configuration parameters from YAML file
- Provides centralized configuration management
- Easier modification of program parameters

#### 2.2 Subject Information Processing
```python
def subjects_info(filename):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return None
```
- Enhanced error handling for subject data loading
- Specific error messages for different failure cases
- Returns None instead of raising exceptions

#### 2.3 Stimulus Order Management
```python
def get_stimulus_order(subject_id, subject_info):
    subject_row = subject_info[subject_info.iloc[:, 0].astype(str) == subject_id]
    if subject_row.empty:
        print(f"No stimulus order found for subject {subject_id}.")
        return None
    stimulus_order_str = subject_row.iloc[0, 1]
    return list(map(int, stimulus_order_str.split(',')))
```
- Structured return of stimulus order

#### 2.4 Marker Processing
```python
def process_markers(streams, target_streams, target_events):
    latencies = {stream: [] for stream in target_streams}
    # Enhanced marker processing logic
```
- New distinction between marker streams and other streams
- Initialization of latencies with zeros for non-marker streams
- Structured collection of timestamps for different event types

#### 2.5 Data Extraction and Concatenation
```python
def extract_and_concatenate_data(data, timestamps, sampling_rate, stimulus_latencies, resting_latencies, stimulus_order):
    # Enhanced epoch extraction and concatenation
```
- Improved epoch extraction with dictionary-based organization
- Separate handling of pre-rest, stimulus, and post-rest epochs
- Better timestamp management for each epoch

### 3. Signal Processing Module (sp.py)

#### 3.1 Core Signal Processing Functions

##### Signal Analysis
- `estimate_sampling_rate(timestamps)`: Calculates sampling rate from timestamp intervals using median filtering
- `normalize_signal(signal)`: Normalizes signal to range [0,1]
- `moving_average(signal, window_size)`: Implements sliding window average

##### ECG Processing
- `myECG_clean(ecg_data, sampling_rate, window_size=2000, overlap=0.5)`:
  - Bandpass filtering (0.5-45.0 Hz)
  - Windowed processing with overlap
  - Butterworth filter implementation
- `handle_ecg_outliers(ecg_data, z_threshold)`: 
  - Z-score based outlier detection
  - NaN interpolation for outliers
- `ecg_data_valid(ecg_data, timestamps)`:
  - Data quality validation
  - Outlier percentage checking
  - Timestamp alignment verification

##### EMG & EDA Processing
- `process_emg_windows(emg_signal, window_size, contamination_threshold)`:
  - Local Outlier Factor analysis
  - Window-based contamination detection
- `interpolate_eda_linear(eda_signal, contaminated_windows_emg, window_size)`:
  - Linear interpolation for contaminated segments
  - Neighbor-based reconstruction
- `compute_quality_index(eda_data)`:
  - Range validation (0.05 - 60 μS)
  - Change rate analysis
  - Quality scoring system

##### Respiratory Signal Processing
- `estimate_respiration_period(signal, sampling_rate)`:
  - FFT-based period estimation
  - Dominant frequency analysis
- `preprocess_respiration_signal(signal, sampling_rate)`:
  - High-pass filtering
  - Window-based processing
  - Signal reconstruction and smoothing
- `clean_resp_data(resp_data, fs)`:
  - Dual-stage Butterworth filtering
  - Low and high-pass components

#### 3.2 Integrated Preprocessing Pipeline

The `preprocess_data(data)` function provides a comprehensive pipeline for:

1. ECG Processing:
   - Notch filtering for powerline interference
   - Signal cleaning and validation
   - Baseline wandering assessment
   - Quality status determination

2. EDA/EMG Processing:
   - Signal normalization
   - Median filtering
   - Contamination detection
   - Linear interpolation
   - Quality index computation

3. Respiratory Processing:
   - Signal filtering
   - Cleaning and smoothing
   - Sampling rate adjustment

### 4. Feature Extraction Module (extract.py)

#### 4.1 Heart Rate Variability Analysis

##### Time Domain Features
- `HRV_NN(peaks, sampling_rate)`:
  - Calculates mean, standard deviation, and median of RR intervals
  - Returns basic HRV metrics (mean_nn, sdnn, median_nn)

- `HRV_RMSSD(peaks, sampling_rate)`:
  - Computes Root Mean Square of Successive Differences
  - Analyzes beat-to-beat variability

##### Frequency Domain Features
- `HRV_FREQUENCY(ecg, peaks, fs, max_freq=2.0)`:
  - Performs spectral analysis of HRV
  - Calculates power in different frequency bands (LF: 0.0-0.16 Hz, HF: 0.16-0.4 Hz)
  - Uses Welch's method for power spectral density estimation

##### Non-linear Features
- `poincare_indices(peaks, sampling_rate)`:
  - Calculates Poincaré plot metrics
  - Returns SD1, SD2, ratio, and ellipse area
  - Provides geometric analysis of HRV

#### 4.2 Respiratory Rate Variability Analysis

##### Time Domain Features
- `RRV_TIME(peaks, troughs, sampling_rate)`:
  - Computes SDBB (Standard Deviation of Breath-to-Breath intervals)
  - Calculates RMSSD and SDSD for respiratory signals
  - Analyzes breath-to-breath variability

##### Frequency Domain Features
- `RRV_FREQ(resp, peaks, troughs, sampling_rate)`:
  - Performs spectral analysis of respiratory variability
  - Uses interpolation for irregular sampling
  - Calculates power in different frequency bands

#### 4.3 Cardiorespiratory Coupling Analysis
- `CARDIORESPIRATORY_COUPLING(resp, peaks, fs)`:
  - Measures coherence between cardiac and respiratory signals
  - Calculates phase synchronization index
  - Returns coherence and phase synchronization metrics

#### 4.4 Electrodermal Activity Analysis
- `EDA_STATISTICS(eda_signal)`:
  - Computes basic statistical features (mean, median, std, skew, kurtosis)
  - Analyzes tonic and phasic components
  - Extracts SCR (Skin Conductance Response) parameters

#### 4.5 Comprehensive Feature Extraction Pipeline
The `extract_features(arr, fs)` function provides a complete feature extraction pipeline:

1. Signal Processing:
   - ECG peak detection and heart rate calculation
   - Respiratory signal cleaning and rate computation
   - EDA signal processing and component separation

2. Feature Categories:
   - Heart Rate Features:
     - Basic statistics (mean HR, HRV)
     - Time-domain HRV metrics
     - Frequency-domain HRV components
     - Non-linear indices
   
   - Respiratory Features:
     - Mean respiratory rate
     - Variability metrics (SDBB, RMSSD, SDSD)
     - Spectral components
   
   - Cardiorespiratory Features:
     - Coherence measurements
     - Phase synchronization
   
   - EDA Features:
     - Statistical measures
     - Tonic level (SCL)
     - Phasic responses (SCR)
     - Spectral characteristics

### Configuration Management

#### Directory Structure
```yaml
DATASET_DIRECTORY: "drive/MyDrive/chills_dataset/"
XDF_DIRECTORY: "drive/MyDrive/chills_dataset/raw_dataset"
PREPROCESSED_DIRECTORY: "drive/MyDrive/chills_dataset/processed/preprocessed_data"
FEATURES_DIRECTORY: "drive/MyDrive/chills_dataset/processed/features"
CHILL_EVENTS_DIR: "drive/MyDrive/chills_dataset/processed/chill_events.sav"
CHILLS_DATA_DIR: "drive/MyDrive/chills_dataset/processed/chills_signal.sav"
PRE_CHILLS_DATA_DIR: "drive/MyDrive/chills_dataset/processed/pre_chills_signal.sav"
NON_CHILLS_DATA_DIR: "drive/MyDrive/chills_dataset/processed/non_chills_signal.sav"
```

#### Data Stream Configuration
```yaml
LABELS:
  - "ECG "
  - "GSR"
  - "ExGa 1"
  - "Resp."

TARGET_STREAM: "CGX AIM Phys. Mon. AIM-0106"

STIMULUS_STREAMS:
  - "AudioStart"

RESTING_STREAMS:
  - "RestingStateStart"

MARKER_STREAMS: 
  - "ChillsReport"
```

#### Event Definitions
```yaml
TARGET_EVENTS:
  AudioStart:
    - "Stimulus 1 Started"
    - "Stimulus 2 Started"
    - "Stimulus 3 Started"
  RestingStateStart: 
    - "Pre-Intervention Resting State Started"
    - "Post-Intervention Resting State Started"
  ChillsReport:
    - "Chills Reported"
```

#### Timing Parameters
```yaml
DURATIONS:
  stimulus1: 300    # in seconds
  stimulus2: 315
  stimulus3: 273
  resting_state: 300

TIME_WINDOW: 15
```

#### Signal Processing Parameters
```yaml
ECG_AMPLITUDE: 1500     # in mV
ECG_BASELINEWANDERING: 1700  # in mV
ECG_Z_THRESHOLD: 5
```

### Technical Details

#### Configuration Parameters
- All parameters centralized in configs.yaml
- Clear separation of concerns:
  - Directory structure
  - Stream identification
  - Event definitions
  - Processing parameters
  - Timing configurations

#### Data Processing Pipeline
1. Load configuration
2. Process subject information
3. Extract raw data
4. Process markers
5. Signal-specific preprocessing
6. Quality assessment
7. Feature extraction

#### Data Organization
- Raw data stored in XDF_DIRECTORY
- Preprocessed data saved to PREPROCESSED_DIRECTORY
- Features extracted to FEATURES_DIRECTORY
- Separate storage for different event types:
  - Chill events
  - Pre-chill signals
  - Non-chill signals

### Improvements Over Original

#### Error Handling
- Comprehensive error catching in data loading
- Handling of missing files
- Better reporting of processing issues
- Signal-specific validation checks

#### Data Organization
- Dictionary-based data structures
- Clear separation of epochs
- Timestamp management

#### Configuration
- External configuration file
- Centralized parameter management
- Easier maintenance and updates

### Future Considerations

#### Configuration Management
- Version control for YAML configurations
- Environment-specific configurations
- Parameter validation and error checking
- Dynamic parameter adjustment

#### Error Handling
- Comprehensive logging system
- Error recovery strategies
- Data validation checks
- Configuration validation

#### Data Organization
- Automated directory creation
- Data versioning
- Backup strategies
- Quality-based data segregation
