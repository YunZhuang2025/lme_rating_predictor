# LME Rating Predictor for Pain Assessment

A Linear Mixed-Effects (LME) model-based tool for predicting pain ratings from laser-evoked potential (LEP) features extracted from EEG data.

## Overview

This tool predicts subjective pain ratings using 10 EEG features derived from laser-evoked potentials:
- **Amplitude features**: N1_amp, N2_amp, P2_amp
- **Latency features**: N1_lat, N2_lat, P2_lat  
- **Magnitude features**: ERP_mag, Alpha_mag, Beta_mag, Gamma_mag

The model is trained on a large-scale dataset (772 subjects, 35,851 trials) and provides both z-score predictions and original-scale pain ratings.

## Key Features

✅ **Subject-specific calibration** - Registers subjects using their historical trial data (5-50 trials)  
✅ **Automatic ID assignment** - New subjects are auto-assigned IDs starting from 773  
✅ **Flexible prediction modes** - Single-trial or batch prediction  
✅ **Interactive CLI** - User-friendly command-line interface  
✅ **Confidence intervals** - Provides 95% CI for all predictions  
✅ **No rating history required** - Only needs EEG features

## Requirements

```bash
pip install numpy pandas scipy
```

Python 3.7+ required.

## Quick Start

### 1. Interactive Mode (Recommended for Beginners)

```bash
python lme_rating_predictor_v5_CN_optimized.py interactive
```

This launches an interactive menu with 5 options:
1. Register a new subject
2. Predict single trial
3. Batch register multiple subjects
4. Batch predict multiple trials
5. Exit

### 2. Demo Mode

```bash
python lme_rating_predictor_v5_CN_optimized.py demo
```

Runs a demonstration with example data to show how the tool works.

### 3. Python API

```python
from lme_rating_predictor_v5_CN_optimized import LMERatingPredictor
import pandas as pd

# Load the predictor
predictor = LMERatingPredictor('lme_model_params.json')

# Prepare historical data (5-50 trials recommended)
history_df = pd.DataFrame({
    'N1_amp': [-9.2, -8.8, -9.5, ...],
    'N2_amp': [-17.1, -16.9, -17.8, ...],
    'P2_amp': [13.5, 14.1, 13.2, ...],
    'N1_lat': [152, 151, 153, ...],
    'N2_lat': [200, 198, 202, ...],
    'P2_lat': [390, 392, 388, ...],
    'ERP_mag': [31.2, 30.8, 31.5, ...],
    'Alpha_mag': [79, 80, 78, ...],
    'Beta_mag': [1.7, 1.8, 1.6, ...],
    'Gamma_mag': [2.7, 2.6, 2.8, ...]
})

# Register subject (auto-assigns ID if None)
profile = predictor.register_subject(subject_id=None, historical_features=history_df)

# Predict new trial
new_features = {
    'N1_amp': -9.1,
    'N2_amp': -17.5,
    'P2_amp': 13.8,
    'N1_lat': 152,
    'N2_lat': 200,
    'P2_lat': 392,
    'ERP_mag': 31.2,
    'Alpha_mag': 79,
    'Beta_mag': 1.7,
    'Gamma_mag': 2.7
}

result = predictor.predict(profile.subject_id, new_features)

# Print results
predictor.print_prediction_summary(result)
```

## Input Data Format

### Historical Features (for registration)
CSV file or pandas DataFrame with columns:
```
N1_amp, N2_amp, P2_amp, N1_lat, N2_lat, P2_lat, ERP_mag, Alpha_mag, Beta_mag, Gamma_mag
```

**Recommendations:**
- 5-20 trials: Minimum for basic calibration
- 20-50 trials: Optimal for reliable predictions
- Each trial = one row with all 10 features

### New Trial Features (for prediction)
Dictionary or DataFrame row with the same 10 features.

## Output Format

### Single Prediction
```python
{
    # Z-score scale (relative values)
    'rating_zscore': 0.158,
    'rating_se_zscore': 0.982,
    'ci_lower_zscore': -1.767,
    'ci_upper_zscore': 2.083,
    
    # Original scale (absolute ratings, 0-10 scale)
    'rating_original': 5.62,
    'rating_se_original': 2.31,
    'ci_lower_original': 1.46,
    'ci_upper_original': 9.78,
    
    # Metadata
    'confidence_level': 0.95,
    'subject_id': 773,
    'subject_registered': True
}
```

### Batch Prediction
CSV file with columns:
- `trial_index` - Trial number
- `rating_zscore`, `rating_original` - Predicted ratings
- `ci_lower_original`, `ci_upper_original` - Confidence intervals
- `N1_amp_input`, `N2_amp_input`, ... - Input features (in specified order)
- Other metadata columns

## Understanding the Predictions

### Z-score Scale
- **Relative** comparison to training data
- Mean = 0, SD = 1
- Positive = higher than average pain
- Negative = lower than average pain

### Original Scale  
- **Absolute** pain rating estimate
- Scale: 0-10 (based on training data)
- Mean ≈ 5.25, SD ≈ 2.36
- Accounts for individual differences when subject is registered

### Confidence Intervals
- Default 95% CI provided for all predictions
- Wider intervals = higher uncertainty
- Narrower intervals = more confident predictions

## Workflow Examples

### Example 1: Single Subject Analysis

```python
predictor = LMERatingPredictor('lme_model_params.json')

# Load subject's historical data
history = pd.read_csv('subject_001_history.csv')
predictor.register_subject('001', history)

# Predict multiple new trials
new_trials = pd.read_csv('subject_001_new_trials.csv')
results = predictor.batch_predict('001', new_trials)

# Save results
results.to_csv('subject_001_predictions.csv', index=False)
```

### Example 2: Multi-subject Analysis

```python
predictor = LMERatingPredictor('lme_model_params.json')

# Register multiple subjects
for subj_id in ['001', '002', '003']:
    history = pd.read_csv(f'subject_{subj_id}_history.csv')
    predictor.register_subject(subj_id, history)

# Batch predict for all subjects
for subj_id in ['001', '002', '003']:
    trials = pd.read_csv(f'subject_{subj_id}_trials.csv')
    results = predictor.batch_predict(subj_id, trials)
    results.to_csv(f'predictions_{subj_id}.csv', index=False)
```

## Model Information

- **Training dataset**: 936 subjects, 11 databases, 48910 observations
- **Features**: 10 EEG-derived features from laser-evoked potentials
- **Model type**: Linear Mixed-Effects with random intercepts and slopes
- **Original rating scale**: Mean = 4.45, SD = 2.73 (0-10 scale)

## Feature Order (Important!)

All outputs follow this specific order:
```
N1_amp → N2_amp → P2_amp → N1_lat → N2_lat → P2_lat → 
ERP_mag → Alpha_mag → Beta_mag → Gamma_mag
```

This order groups features logically:
1. **Amplitudes** (early → late LEP components)
2. **Latencies** (corresponding to amplitudes)  
3. **Magnitudes** (overall ERP, then frequency-specific: low → high)

## Citation

If you use this tool in your research, please cite:

```
Zhuang Y., Zhang L.B., Wang X.Q., Geng X.Y., & Hu L. (in preparation). 
From Normative Features to Multidimensional Estimation of Pain: 
A Large-Scale Study of Laser-Evoked Brain Responses.
```

## Troubleshooting

### "Missing feature columns" error
- Ensure your CSV has all 10 required feature columns
- Check column names match exactly (case-sensitive)
- Use the specified feature order

### "Subject not registered" warning
- Register the subject first using `register_subject()`
- Predictions without registration use global normalization (less accurate)

### Low confidence / wide intervals
- Provide more historical trials (>20 recommended)
- Check for outliers in input features
- Ensure feature extraction was performed correctly

## Advanced Usage

### Custom Rating Scale

If your data uses a different rating scale:

```python
# Set custom scale (mean, std)
predictor.set_rating_scale(mean=4.5, std=2.0)
```

### View Registered Subjects

```python
predictor.list_registered_subjects()
```

### Manual Feature Standardization

```python
# For registered subjects, features are auto-standardized
# For unregistered subjects, global standardization is used
```

## License & Support

**Author**: Yun Zhuang  
**Version**: v5.5 (Optimized)  
**Date**: Feb 2026

For questions or issues, please refer to the original publication or contact the authors.

## Notes

- This is a **prediction tool**, not a diagnostic instrument
- Predictions should be interpreted in the context of research
- Individual differences are best captured with subject registration
- The model was trained on controlled experimental data
