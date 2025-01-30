# 1. Descriptive Statistics

## A. Key Features Across Events (NON-CHILL, PRE-CHILL, POST-CHILL, CHILL)

### Heart Rate & HRV Features:
- `mean_HR`:
  - [Mean & Median](./plots/features_mean_median.png):
    - CHILL: Highest mean (~65-70 bpm) with larger difference between mean/median.
    - Increasing trend through states from NON-CHILL to CHILL.
  - [Violin Distribution](./plots/features_violin_box.png):
    - Distribution with highest deviation: NON-CHILL.
    - Progressive narrowing of distribution through states: from NON-CHILL to CHILL.
    - Most compact distribution: CHILL.
  - [Coefficient of Variation](./plots/coefficient_of_variation.png)
    - Smaller coefficient of variation in PRE_CHILL and POST-CHILL (~0.15) compared to NON-CHILL (~0.18) and CHILL (>2.0).

- `HRV`:
  - [Mean & Median](./plots/features_mean_median.png):
    - Highest mean/median values in NON-CHILL (mean ~5) and CHILL (mean ~4.5).
  - [Violin Distribution](./plots/features_violin_box.png):
    - Fairly similar deviations across states.
  - [Coefficient of Variation](./plots/coefficient_of_variation.png)
    - Higher coefficnet of variation in PRE-CHILL and POST-CHILL (~1.75) comared to NON-CHILL and CHILL (1.3 - 1.5).

- `Summary`
  - `CHILL` `mean_HR` is skewed or has outliers - Larger difference in mean and median.
  - Increase in `mean_HR` from `NON-CHILL` to `CHILL` suggest heightened heart activity as individuals transition to `CHILL` experience.
  - `CHILL` `mean_HR` having the the narrowest distribution indicate more consistent physiological responses in `CHILL` states.
  - Lower CV values in `PRE-CHILL` and `POST-CHILL` suggest relaively stable heart rate in these states.
  - Highest mean and median HRV in  `NON-CHILL` and `CHILL` states suggest association with parasympathetic activity, reflecting more relaxed or engaged state.
  - Consistent HRV distribution widths across states despite differences in mean/median values indicate deviation in HRV are comparable
  - Higher `HRV` CV in `PRE-CHILL` and `POST-CHILL` suggests greater variability in HRV in these states.

- `Observations ==> Inference`
  - High `HRV` and wide `meanHR` ditribution in `NON-CHILL` indicates relaxed and varied state, reflecting baseline conditions or low engagement levels.
  - Increased `meanHR`, compact `meanHR` distribution, and relatively high `HRV` point to increased increased arousal and engagement with relaively stable physiological patterns.
  - Lower `meanHR` and `HRV` values with higher CV suggest transitional physiological states, marked by fluctuations in heart rate and variability.


### EDA (Electrodermal Activity) Features:
- `EDA_mean`, `EDA_median`, and `EDA_SCL `:
  - [Mean & Median](./plots/features_mean_median.png)
    - Higher Mean/Median values in `PRE-CHILL` and `POST-CHILL` (>0.6) compared to `NON-CHILL` and `CHILL` (~0.6).
  - [Violin](./plots/features_violin_box.png) & [Box Distributions](./plots/relevant_features_dist_box.png):
    - `POST_CHILL` has most compact distribution(least deviation) with significant outliers.
    - `POST-CHILL` and `PRE-CHILL` are significantly skewed compared to `CHILL` and `NON-CHILL`.
  - [Coefficient of Variation](./plots/coefficient_of_variation.png)
    - `POST-CHILL` has least CV value (~0.25) followed by `PRE-CHILL` (~0.45) similar to `NON-CHILL` whih is lightly higher, and `CHILL` the highest (>0.5).

- `Summary`
  - EDA levels are generally more elevated in `PRE-CHILL` and `POST-CHILL`phases suggesting arousal during these periods.
  - While majority of `POST-CHILL` observations are clustered around the central values, there are a few extreme cases.
  - Skewness in `POST-CHILL` and `PRE-CHILL` compared to symmetrical `CHILL` and `NON-CHILL` suggests assymetrical response physiological arousal during the `PRE-CHILL` and `POST-CHILL` states.
  - `POST-CHILL` having lowest CV indicates relatively stable `POST-CHILL` responses compared to `PRE-CHILL`, `NON-CHILL` and `CHILL` with highest CV.
  - High variability in `CHILL` could suggest more diverse physiological responses or less consistent emotional statesduring this phase.

- `Observations ==> Inference`
  - `PRE-CHILL` and `POST-CHILL` phases show elevated arousal, with `POST-CHILL` responses being most consistent and clustered around the mean.
  - `CHILL` and `NON_CHILL` phases are associated with lower arousal levels and higher variability, suggesting less consistent/predictable physiological behaviour/engagement.
  - The presence of skewness and outliers in `POST-CHILL` and `PRE-CHILL` may point to individual differences in how participants respond to these conditions, possibly due to externals factors or personal baselines.


### Phasic Features:
- `phasic_recovery_mean`, and `phase_risetime_mean`:
  - [Mean & Median](./plots/features_mean_median.png):
    - `PRE-CHILL` and `POST-CHILL` have highes mean/medians.
    - Mean and medians close, little to no skewness in `phasic_recovery_mean`, and `phase_risetime_mean`.
  - [Violin](./plots/features_violin_box.png) & [Box Distributions](./plots/relevant_features_dist_box.png):
    - `NON-CHILL` and `CHILL` most compact in `phasic_risetime_mean`, followed by
    `POST-CHILL` and then `PRE-CHILL`
    - Similar compactness of `phasic_recovery_mean` in `NON-CHILL`, `CHILL`, and `POST-CHILL`, but 
    `PRE_CHILL` is still the least compact distribution.
  - [Coefficient of Variation](./plots/coefficient_of_variation.png)
    - CV values for `phasic_recovery_mean` generally lower than 
    that of `phasic_risetime_mean`.
    - CV for `phasic_recovery_mean`: `PRE-CHILL`, `CHILL` > `NON-CHILL` > `POST-CHILL`. 
    - CV for `phasic_risetime_mean`: `NON-CHILL` > `CHILL` > `PRE-CHILL` > `POST-CHILL`. 
    - In both `phasic_recovery_mean` and `phasic_risetime_mean`, `POST-CHILL` 
    has the lowest CV.

- `Summary`
  - Higher mean and median values during the PRE-CHILL and POST-CHILL 
  phases indicates a higher baseline or magnitude in these phases 
  compared to CHILL and NON-CHILL.
  - The closeness of mean and median suggests minimal skewness in the 
  data distribution, implying asymmetry.
  - `phase_risetime_mean` variability increases in POST-CHILL, 
  and even more so in PRE-CHILL, indicating broader data spread, 
  or more diverse responses, and more variations in 
  response time in these phases.
  - `phasic_recovery_mean` PRE-CHILL stands out with a less 
  compact (wider) distribution, pointing to higher 
  variability in this phase and less consistency in other phases.
  - `PRE-CHILL`: High mean and median but with less compact 
  and more variable distributions, indicating a phase with 
  diverse behaviors or responses.
  - `CHILL`: Compact distributions for both metrics but with 
  moderate variability (CV), suggesting some consistency 
  but with a degree of fluctuation in responses.
  - `NON-CHILL`: High variability in `phase_risetime_mean` indicates 
  less predictable response times. Lower variability in 
  `phasic_recovery_mean` suggests more stability in recovery 
  responses.
  - `POST-CHILL`: Shows the most consistent behavior 
  overall (lowest CV for both metrics), with relatively 
  high central tendencies, indicating stable and uniform 
  responses.

- `Observations ==> Inference`
  - `POST-CHILL` is characterized by consistent and stable 
  behavior across metrics.
  - `PRE-CHILL` shows the highest variability, especially 
  in `phasic_recovery_mea`n, pointing to diverse responses 
  in this phase.
  - `NON-CHILL` demonstrates variability, particularly in 
  `phase_risetime_mean`, suggesting less predictable 
  response times during this phase.
  - `CHILL` shows moderate variability and compact 
  distributions, reflecting relatively stable but slightly 
  fluctuating responses.

## B. Outliers and Anomalies

### Notable Outliers:
1. Heart Rate Metrics (Box plots in Image 4):
   - `mean_HR`: Multiple outlier dots above 150 bpm
   - `HR_RMSSD`: Numerous outlier points, especially in NON-CHILL

2. EDA Metrics:
   - Box plots (Image 4) show:
     - `EDA_kurtosis`: Extreme outlier dots >200
     - `EDA_skew`: Symmetric outliers in both directions
   - Double bar plots (Image 1) confirm asymmetry between mean and median

3. Phasic Features:
   - Box plots (Image 4) reveal:
     - `phasic_risetime_mean`: Multiple outlier dots >10
     - `phasic_domnt_freq`: Extreme outliers near 250

### Distribution Anomalies:
1. Violin plots (Image 2) show:
   - Right-skewed shapes:
     - `HRV_LF`, `HRV_HF`: Bulbous bottom, thin upper tail
     - `SDBB`, `SDSD`: Asymmetric violin shape
   - Bimodal patterns:
     - `CRC_coherence`: Double bulge in violin shape
     - `EDA_mean`: Two distinct widening points

### Coefficient of Variation:
- Single bar plots (Image 3) demonstrate:
  - Highest bars in NON-CHILL across features
  - Progressive decrease through states
  - Shortest bars in CHILL for most metrics
  - Notable exceptions in EDA features showing inverse pattern

This analysis leverages multiple visualization types (double bars, violin plots, box plots, and single bars) to provide comprehensive insights into the physiological state transitions across conditions.