# LME Rating Predictor v5.4 - Complete User Guide

## 📖 Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Feature Details](#feature-details)
6. [Python API](#python-api)
7. [File Formats](#file-formats)
8. [FAQ](#faq)
9. [Advanced Usage](#advanced-usage)
10. [Troubleshooting](#troubleshooting)

---

## Overview

### What is LME Rating Predictor?

LME Rating Predictor is a pain subjective rating prediction tool based on Linear Mixed-Effects Model. It predicts subjects' pain ratings by analyzing electroencephalography (EEG) features.

### Core Features

✅ **Within-Subject Standardization** - Uses subject's own historical data for feature standardization  
✅ **Automatic ID Assignment** - New subjects automatically numbered starting from 773  
✅ **Dual-Scale Output** - Provides Z-score (relative value) and original rating (absolute value)  
✅ **Percentile Interpretation** - Intuitive percentile representation of pain level  
✅ **Batch Processing** - Supports batch analysis of multiple subjects and trials  
✅ **Interactive Friendly** - Command-line interactive interface, no programming required  

### Version Information

- **Version:** v5.4
- **Release Date:** 2025-11
- **Python Version:** 3.8+
- **Core Dependencies:** NumPy, Pandas, SciPy

---

## System Requirements

### Required Software

- **Python:** 3.8 or higher
- **Operating System:** Windows / macOS / Linux

### Python Package Dependencies

```bash
pip install numpy pandas scipy
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

### Hardware Requirements

- **Memory:** At least 2GB
- **Disk:** 100MB available space
- **CPU:** No special requirements

---

## Quick Start

### Step 1: Prepare Data

Create a CSV file containing subject's historical feature data:

**subject_773_features.csv**
```csv
erp_N2,erp_P2,erp_N1,latency_N2,latency_P2,latency_N1,TF_gamma,TF_beta,TF_LEP,TF_alpha
-15.2,12.3,-8.5,195,385,148,0.003,1.5,28.5,-1.5
-18.5,14.5,-9.2,202,395,155,0.005,1.8,32.1,-1.7
-16.3,13.1,-8.8,198,390,150,0.004,1.6,30.2,-1.6
-20.1,15.8,-10.1,205,402,158,0.006,2.1,35.8,-1.9
-17.8,13.9,-9.5,200,392,153,0.004,1.7,31.5,-1.6
```

**Requirements:**
- At least 5 rows of data (recommended 10-20 rows)
- Must include all 10 feature columns
- Numeric type, no missing values

### Step 2: Start Program

```bash
python lme_rating_predictor_v5_EN.py
```

### Step 3: Register Subject

```
Selection: 1

Subject ID (leave blank for auto-assignment 773+): [Press Enter]
Feature history CSV file path: subject_773_features.csv

✓ Automatically assigned subject ID: 773
✓ Subject registered: 773
```

### Step 4: Predict

```
Selection: 2

Subject ID: 773

Please enter feature values:
  erp_N2: -17.5
  erp_P2: 13.8
  erp_N1: -9.1
  latency_N2: 200
  latency_P2: 392
  latency_N1: 152
  TF_gamma: 0.004
  TF_beta: 1.7
  TF_LEP: 31.2
  TF_alpha: -1.6

Prediction result displayed...
```

---

## Core Concepts

### 1. Subject Registration

**What is registration?**
- Load subject's historical feature data into memory
- Calculate subject's feature means and standard deviations
- Used for subsequent feature standardization

**Why is registration needed?**
- Implement within-subject standardization to improve prediction accuracy
- Consider individual differences for personalized prediction

**What is needed for registration?**
- Subject's historical feature data (5-50 trials)
- Complete data for 10 EEG features
- Rating data not required

### 2. Feature Standardization

**Within-Subject Standardization:**
```python
feature_z = (feature_raw - feature_mean_subject) / feature_std_subject
```

- Uses subject's own mean and standard deviation
- Eliminates individual baseline differences
- Improves prediction accuracy

**Global Standardization (Alternative):**
```python
feature_z = (feature_raw - feature_mean_global) / feature_std_global
```

- Uses global statistics from training data
- Used when subject is not registered
- Slightly lower accuracy

### 3. Prediction Output

**Z-score Scale (Relative Value):**
- Represents deviation from average level
- Range typically between -3 and +3
- Accompanied by percentile interpretation

**Original Rating Scale (Absolute Value):**
- Inverse standardization to original rating scale
- Range typically between 0-10
- Based on training data statistics (mean=5.2503, std=2.3568)

**Percentile:**
- Represents what percentage of trials this pain level exceeds
- Calculated based on normal distribution
- Intuitive and easy to understand

### 4. Subject ID Assignment

**Automatic Assignment Rules:**
- Training data uses IDs 1-772
- New subjects automatically increment from 773
- Avoids conflicts with training data

**Manual Specification:**
- Can also manually specify any ID
- Recommended to use ≥773 numbering

---

## Feature Details

### Menu 1: Register New Subject (Single)

#### Function Description
Register a single subject's historical feature data to the current session.

#### Operation Steps

1. **Select Menu**
   ```
   Selection: 1
   ```

2. **Enter Subject ID**
   ```
   Subject ID (leave blank for auto-assignment 773+): [Enter or input ID]
   ```
   - Leave blank: Auto-assign (recommended)
   - Enter number: Use specified ID (e.g., 1001)

3. **Provide Feature File**
   ```
   Feature history CSV file path: subject_773_features.csv
   ```

4. **Confirm Registration**
   ```
   ✓ Read 20 historical data records
   ✓ Automatically assigned subject ID: 773
   ✓ Subject registered: 773
     Subject 773
     Historical trials: 20
     Number of features: 10
   ```

#### Notes
- CSV file must include all 10 feature columns
- Recommended 10-20 data rows (minimum 5 rows)
- File path can be relative or absolute

#### Example File Structure
```
project/
├── lme_rating_predictor_v5_EN.py
├── lme_model_params.json
└── data/
    ├── subject_773_features.csv
    ├── subject_774_features.csv
    └── subject_775_features.csv
```

---

### Menu 2: Predict Single Trial

#### Function Description
Predict single trial for registered subject to obtain pain rating estimate.

#### Operation Steps

1. **Select Menu**
   ```
   Selection: 2
   ```

2. **View Registered Subjects**
   ```
   Currently registered subjects: ['773', '774', '775']
   ```

3. **Select Subject**
   ```
   Subject ID: 773
   ```

4. **Enter Feature Values**
   ```
   Please enter feature values:
     erp_N2: -17.5
     erp_P2: 13.8
     erp_N1: -9.1
     latency_N2: 200
     latency_P2: 392
     latency_N1: 152
     TF_gamma: 0.004
     TF_beta: 1.7
     TF_LEP: 31.2
     TF_alpha: -1.6
   ```

5. **View Prediction Result**
   ```
   ============================================================
   Prediction Result - Subject 773
   ============================================================
   
   Z-score Scale (Relative Value):
     Predicted:  0.234 ± 0.812
     95% CI: [-1.357,  1.825]
     Percentile: This pain level exceeds 59.3% of trials
   
   Original Rating Scale (Estimated Value):
     Predicted:   5.80 ± 1.91
     95% CI: [  2.05,   9.55]
   ============================================================
   ```

6. **Save Result (Optional)**
   ```
   Save result to CSV? (y/n): y
   Output filename: prediction_result.csv
   ✓ Saved: prediction_result.csv
   ```

#### Result Interpretation

**Z-score = 0.234**
- Slightly above average level (0 is average)
- Located about 0.2 standard deviations above mean

**Percentile = 59.3%**
- This pain level exceeds 59.3% of trials
- Slightly above median (50%)
- Belongs to medium-high pain level

**Original rating = 5.80**
- Estimated value on 0-10 scale
- Close to moderate pain (5 points)
- 95% confidence interval: 2.05-9.55

#### Notes
- Must register subject first (Menu 1 or 3)
- If subject not registered, can choose to use global standardization
- Input feature values should be actual measurements, not Z-scores

---

### Menu 3: Batch Register New Subjects

#### Function Description
Register multiple subjects at once, suitable for batch data analysis.

#### Method 1: Batch Registration from Folder

**Use Case:** All subjects' feature files stored in the same folder

**Operation Steps:**

1. **Prepare Folder Structure**
   ```
   subjects_data/
   ├── subject_773_features.csv
   ├── subject_774_features.csv
   ├── subject_775_features.csv
   ├── subject_776_features.csv
   └── subject_777_features.csv
   ```

2. **Select Menu and Method**
   ```
   Selection: 3
   Registration method (1=From folder, 2=Manual input file list): 1
   ```

3. **Specify Folder**
   ```
   Feature data folder path: ./subjects_data/
   ```

4. **Confirm Registration**
   ```
   Found 5 CSV files:
     - subject_773_features.csv
     - subject_774_features.csv
     - subject_775_features.csv
     - subject_776_features.csv
     - subject_777_features.csv
   
   Start batch registration? (y/n): y
   
   ✓ Registered: 773 (from subject_773_features.csv)
   ✓ Registered: 774 (from subject_774_features.csv)
   ✓ Registered: 775 (from subject_775_features.csv)
   ✓ Registered: 776 (from subject_776_features.csv)
   ✓ Registered: 777 (from subject_777_features.csv)
   
   Batch registration complete: 5/5 successful
   ```

5. **View Registration Results**
   ```
   Currently registered subjects:
   Number of registered subjects: 5
   
   Subject ID      Trials    Features
   ----------------------------------------
   773                 20        10
   774                 15        10
   775                 30        10
   776                 18        10
   777                 25        10
   ```

**File Naming Convention:**
- Recommended format: `subject_XXX_features.csv`
- Program automatically extracts subject ID from filename
- If extraction fails, automatically assigns new ID

#### Method 2: Manual Input File List

**Use Case:** Files scattered in different locations, or selective registration needed

**Operation Steps:**

1. **Select Method**
   ```
   Selection: 3
   Registration method (1=From folder, 2=Manual input file list): 2
   ```

2. **Input File Paths One by One**
   ```
   Please enter file paths (one per line, empty line to finish):
     File path: ./data/subject_773_features.csv
     File path: ./data/subject_774_features.csv
     File path: ./data/subject_775_features.csv
     File path: [Press Enter to finish]
   ```

3. **Specify ID for Each File**
   ```
   Subject ID for subject_773_features.csv (leave blank for auto-assignment): 773
   ✓ Registered: 773
   
   Subject ID for subject_774_features.csv (leave blank for auto-assignment): 774
   ✓ Registered: 774
   
   Subject ID for subject_775_features.csv (leave blank for auto-assignment): 775
   ✓ Registered: 775
   
   Batch registration complete: 3/3 successful
   ```

#### Notes
- Batch registration skips duplicate subject IDs
- If a file has format errors, displays error but continues processing other files
- After registration completion, automatically displays all registered subjects list

---

### Menu 4: Batch Predict Multiple Trials for Multiple Subjects

#### Function Description
Perform batch prediction for multiple subjects' multiple trials, generating complete prediction result files.

#### Method 1: Batch Prediction for Single Subject

**Use Case:** Predict multiple trials for a single subject

**Operation Steps:**

1. **Select Method**
   ```
   Selection: 4
   Prediction mode (1=Batch predict single subject, 2=Predict multiple subjects separately): 1
   ```

2. **Select Subject**
   ```
   Currently registered subjects: ['773', '774', '775']
   Subject ID: 773
   ```

3. **Provide Feature File**
   ```
   Input CSV file path (containing feature columns): subject_773_trials.csv
   ```

4. **Specify Output File**
   ```
   Output CSV file path (default: predictions_773.csv): [Press Enter or input path]
   ```

5. **View Results**
   ```
   ✓ Read 50 records
   ✓ Batch prediction complete!
     Output file: predictions_773.csv
   ```

**Output File Format:**
```csv
erp_N2,erp_P2,...,TF_alpha,predicted_rating,subject_id
-17.5,13.8,...,-1.6,5.80,773
-16.2,12.9,...,-1.5,5.42,773
...
```

#### Method 2: Predict Multiple Subjects Separately

**Use Case:** Predict multiple subjects, each with their own feature file

**Sub-Method 2.1: Unified File**

All subjects use the same feature file (e.g., all new trials have the same stimuli)

```
Selection: 4
Prediction mode: 2
Subject IDs: 773,774,775
Data file method: 1

Input CSV file path: all_subjects_trials.csv
✓ Read 100 records

Output folder (default: ./predictions/): [Press Enter]

✓ Complete: 773 → ./predictions/predictions_773.csv
✓ Complete: 774 → ./predictions/predictions_774.csv
✓ Complete: 775 → ./predictions/predictions_775.csv

Batch prediction complete: 3/3 successful
```

**Sub-Method 2.2: Specify Separately**

Each subject has their own feature file

```
Selection: 4
Prediction mode: 2
Subject IDs: 773,774,775
Data file method: 2

Input CSV file for 773: ./data/subject_773_trials.csv
Output file (default: predictions_773.csv): [Press Enter]
✓ Complete: 773

Input CSV file for 774: ./data/subject_774_trials.csv
Output file (default: predictions_774.csv): [Press Enter]
✓ Complete: 774

...

Batch prediction complete: 3/3 successful
```

#### Notes
- All subjects must be registered first
- Unregistered subjects will be automatically skipped
- Output files include all input features plus predicted_rating column
- Supports "all" keyword to predict all registered subjects

---

## Python API

### Basic Usage

```python
from lme_rating_predictor_v5_EN import LMERatingPredictor
import pandas as pd

# 1. Initialize predictor
predictor = LMERatingPredictor('lme_model_params.json')

# 2. Prepare historical data
feature_history = pd.read_csv('subject_773_features.csv')

# 3. Register subject
profile = predictor.register_subject(
    subject_id=None,  # Auto-assign
    historical_features=feature_history
)

# 4. Predict single trial
new_features = {
    'erp_N2': -17.5,
    'erp_P2': 13.8,
    'erp_N1': -9.1,
    'latency_N2': 200,
    'latency_P2': 392,
    'latency_N1': 152,
    'TF_gamma': 0.004,
    'TF_beta': 1.7,
    'TF_LEP': 31.2,
    'TF_alpha': -1.6
}

result = predictor.predict(
    subject_id=profile.subject_id,
    features=new_features,
    return_components=True
)

# 5. Print results
predictor.print_prediction_summary(result)
```

### API Reference

#### Class: LMERatingPredictor

**Constructor**
```python
predictor = LMERatingPredictor(
    params_file: str,      # Model parameter JSON file path
    silent: bool = False   # Silent mode (no print)
)
```

**Core Methods**

1. **register_subject** - Register new subject
   ```python
   profile = predictor.register_subject(
       subject_id: Optional[Union[int, str]] = None,  # Subject ID (None for auto)
       historical_features: Union[pd.DataFrame, Dict],  # Historical data
       use_global_stats: bool = False  # Use global standardization
   ) -> SubjectProfile
   ```

2. **predict** - Predict single trial
   ```python
   result = predictor.predict(
       subject_id: Union[int, str],     # Subject ID
       features: Dict[str, float],      # Feature values
       return_components: bool = False  # Return component breakdown
   ) -> Dict
   ```
   
   **Return Dictionary:**
   ```python
   {
       'predicted_rating': float,        # Predicted rating (original scale)
       'predicted_rating_zscore': float, # Predicted rating (Z-score)
       'subject_id': str,                # Subject ID
       'components': {                   # (If return_components=True)
           'fixed_effects': float,
           'database_effect': float,
           'subject_effect': float
       },
       'standardized_features': dict     # (If return_components=True)
   }
   ```

3. **batch_predict** - Batch predict multiple trials
   ```python
   results_df = predictor.batch_predict(
       subject_id: Union[int, str],   # Subject ID
       features_df: pd.DataFrame      # Feature DataFrame
   ) -> pd.DataFrame
   ```

4. **list_registered_subjects** - List all registered subjects
   ```python
   predictor.list_registered_subjects()
   ```

5. **get_subject_info** - Get subject information
   ```python
   info = predictor.get_subject_info(subject_id)
   print(info)
   ```

6. **export_subject_profile** - Export subject profile
   ```python
   predictor.export_subject_profile(
       subject_id: Union[int, str],
       output_file: str
   )
   ```

7. **import_subject_profile** - Import subject profile
   ```python
   profile = predictor.import_subject_profile(
       profile_file: str
   )
   ```

### Advanced Usage Examples

#### 1. Batch Processing Pipeline

```python
import os
import glob

# Initialize
predictor = LMERatingPredictor('lme_model_params.json', silent=True)

# Batch register
csv_files = glob.glob('./subjects_data/*.csv')
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    profile = predictor.register_subject(None, df)
    print(f"✓ Registered: {profile.subject_id}")

# Batch predict
for subject_id in predictor.subject_profiles.keys():
    trials_file = f'./trials/subject_{subject_id}_trials.csv'
    if os.path.exists(trials_file):
        trials_df = pd.read_csv(trials_file)
        results = predictor.batch_predict(subject_id, trials_df)
        results.to_csv(f'./output/predictions_{subject_id}.csv', index=False)
        print(f"✓ Complete: {subject_id}")
```

#### 2. Ensemble Prediction

Use multiple historical data windows for ensemble prediction:

```python
import numpy as np

# Prepare multiple historical data windows
windows = [
    feature_history.iloc[:10],   # First 10 trials
    feature_history.iloc[5:15],  # Middle 10 trials
    feature_history.iloc[10:20]  # Last 10 trials
]

# Register multiple profiles
predictors = []
for i, window in enumerate(windows):
    pred = LMERatingPredictor('lme_model_params.json', silent=True)
    pred.register_subject(f'773_window_{i}', window)
    predictors.append(pred)

# Predict and average
results = []
for pred in predictors:
    result = pred.predict(f'773_window_{i}', new_features)
    results.append(result['predicted_rating'])

ensemble_prediction = np.mean(results)
print(f"Ensemble prediction: {ensemble_prediction:.2f}")
```

### 3. Cross-Validation

Evaluate prediction accuracy:

```python
from sklearn.model_selection import KFold

# Prepare data
full_data = pd.read_csv('subject_773_full.csv')
features = full_data[feature_cols]
true_ratings = full_data['rating']

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
predictions = []
actuals = []

for train_idx, test_idx in kf.split(features):
    # Training set for registration
    train_features = features.iloc[train_idx]
    predictor = LMERatingPredictor('lme_model_params.json', silent=True)
    predictor.register_subject(773, train_features)
    
    # Test set for prediction
    test_features = features.iloc[test_idx]
    for idx in test_features.index:
        feat_dict = test_features.loc[idx].to_dict()
        result = predictor.predict(773, feat_dict)
        predictions.append(result['rating_original'])
        actuals.append(true_ratings[idx])

# Calculate accuracy
from sklearn.metrics import mean_absolute_error, r2_score
mae = mean_absolute_error(actuals, predictions)
r2 = r2_score(actuals, predictions)

print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.3f}")
```

### 4. Batch Processing Script

Create an automated batch processing script:

```python
#!/usr/bin/env python
"""
Batch processing script
Automatically register and predict all subjects
"""

import glob
import os
from lme_rating_predictor_v5_EN import LMERatingPredictor

# Configuration
BASELINE_FOLDER = './baselines/'
TRIALS_FOLDER = './trials/'
OUTPUT_FOLDER = './predictions/'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize
predictor = LMERatingPredictor('lme_model_params.json', silent=True)

# Batch register
baseline_files = glob.glob(os.path.join(BASELINE_FOLDER, 'subject_*_baseline.csv'))
for baseline_file in baseline_files:
    subject_id = os.path.basename(baseline_file).split('_')[1]
    df = pd.read_csv(baseline_file)
    predictor.register_subject(subject_id, df)
    print(f"✓ Registered: {subject_id}")

# Batch predict
trials_files = glob.glob(os.path.join(TRIALS_FOLDER, 'subject_*_trials.csv'))
for trials_file in trials_files:
    subject_id = os.path.basename(trials_file).split('_')[1]
    
    if subject_id not in predictor.subject_profiles:
        print(f"⚠️  Skipping unregistered subject: {subject_id}")
        continue
    
    df = pd.read_csv(trials_file)
    results = predictor.batch_predict(subject_id, df)
    
    output_file = os.path.join(OUTPUT_FOLDER, f'predictions_{subject_id}.csv')
    results.to_csv(output_file, index=False)
    print(f"✓ Complete: {subject_id}")

print("\nAll predictions complete!")
```

### 5. Visualize Prediction Results

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Read prediction results
df = pd.read_csv('predictions_773.csv')

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Z-score distribution
axes[0, 0].hist(df['rating_zscore'], bins=20, edgecolor='black')
axes[0, 0].axvline(0, color='red', linestyle='--', label='Mean')
axes[0, 0].set_title('Rating Z-score Distribution')
axes[0, 0].set_xlabel('Z-score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

# 2. Original rating distribution
axes[0, 1].hist(df['rating_original'], bins=20, edgecolor='black')
axes[0, 1].axvline(5.2503, color='red', linestyle='--', label='Mean')
axes[0, 1].set_title('Original Rating Distribution')
axes[0, 1].set_xlabel('Rating')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()

# 3. Confidence interval width
df['ci_width'] = df['ci_upper_original'] - df['ci_lower_original']
axes[1, 0].scatter(df['rating_original'], df['ci_width'], alpha=0.5)
axes[1, 0].set_title('Prediction Uncertainty')
axes[1, 0].set_xlabel('Predicted Rating')
axes[1, 0].set_ylabel('95% CI Width')

# 4. Time trend
axes[1, 1].plot(df['trial_index'], df['rating_original'], marker='o')
axes[1, 1].fill_between(
    df['trial_index'],
    df['ci_lower_original'],
    df['ci_upper_original'],
    alpha=0.3
)
axes[1, 1].set_title('Prediction Trend')
axes[1, 1].set_xlabel('Trial')
axes[1, 1].set_ylabel('Rating')

plt.tight_layout()
plt.savefig('prediction_analysis.png', dpi=300)
plt.show()
```

---

## Troubleshooting

### Issue 1: Import Error

**Error Message:**
```
ModuleNotFoundError: No module named 'pandas'
```

**Solution:**
```bash
pip install pandas numpy scipy
```

### Issue 2: JSON File Not Found

**Error Message:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'lme_model_params.json'
```

**Solution:**
1. Ensure `lme_model_params.json` is in the current directory
2. Or use absolute path:
   ```python
   predictor = LMERatingPredictor('/path/to/lme_model_params.json')
   ```

### Issue 3: CSV Encoding Problem

**Error Message:**
```
UnicodeDecodeError: 'utf-8' codec can't decode byte
```

**Solution:**
1. Open CSV with Notepad or VSCode
2. Save as UTF-8 encoding
3. Or specify encoding in Python:
   ```python
   df = pd.read_csv('file.csv', encoding='gbk')  # or 'utf-8-sig'
   ```

### Issue 4: Numeric Conversion Failure

**Error Message:**
```
ValueError: could not convert string to float
```

**Solution:**
1. Check if CSV contains non-numeric values (e.g., text, spaces)
2. Use CSV validation tool:
   ```bash
   python csv_validator.py validate subject_773.csv
   python csv_validator.py fix subject_773.csv
   ```

### Issue 5: Out of Memory

**Error Message:**
```
MemoryError
```

**Solution:**
1. Reduce batch processing data volume
2. Process in batches:
   ```python
   # Read in chunks
   chunk_size = 1000
   for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
       results = predictor.batch_predict(subject_id, chunk)
       # Process results...
   ```

### Issue 6: Slow Prediction Speed

**Analysis:**
- Large data volume
- Hardware performance

**Optimization Methods:**
1. Use batch prediction instead of single prediction
2. Turn off unnecessary output:
   ```python
   predictor = LMERatingPredictor('lme_model_params.json', silent=True)
   ```
3. Consider parallel processing:
   ```python
   from concurrent.futures import ThreadPoolExecutor
   
   def predict_subject(subject_id):
       # Prediction logic
       pass
   
   with ThreadPoolExecutor(max_workers=4) as executor:
       results = executor.map(predict_subject, subject_ids)
   ```

### Issue 7: Inconsistent Results

**Symptom:**
Different results from multiple runs

**Cause:**
- Different subject registration order
- Used random seed

**Solution:**
1. Fix subject ID
2. Ensure consistent feature order
3. Check if data was modified

### Getting Help

**When encountering problems:**

1. **Check Logs**
   - Program prints detailed error messages
   - Carefully read error prompts

2. **Verify Input**
   - Validate CSV file format
   - Confirm feature names are correct
   - Check numeric ranges

3. **Simplify Problem**
   - Test with small dataset
   - Gradually troubleshoot

4. **Consult Documentation**
   - This user guide
   - API reference
   - Example code

---

## Appendix

### A. Complete Feature Description

| Feature Name | Chinese Name | Meaning | Unit | Typical Range | Polarity |
|--------------|--------------|---------|------|---------------|----------|
| erp_N2 | N2 Amplitude | Laser-evoked potential N2 component amplitude | μV | -25 ~ -10 | Negative |
| erp_P2 | P2 Amplitude | Laser-evoked potential P2 component amplitude | μV | 8 ~ 20 | Positive |
| erp_N1 | N1 Amplitude | Laser-evoked potential N1 component amplitude | μV | -15 ~ -5 | Negative |
| latency_N2 | N2 Latency | N2 component appearance time | ms | 150 ~ 250 | - |
| latency_P2 | P2 Latency | P2 component appearance time | ms | 300 ~ 450 | - |
| latency_N1 | N1 Latency | N1 component appearance time | ms | 100 ~ 200 | - |
| TF_gamma | Gamma Energy | Gamma band time-frequency energy | a.u. | 0.001 ~ 0.01 | - |
| TF_beta | Beta Energy | Beta band time-frequency energy | a.u. | 0.5 ~ 3.0 | - |
| TF_LEP | LEP Energy | Laser-evoked potential total energy | a.u. | 15 ~ 50 | - |
| TF_alpha | Alpha Energy | Alpha band time-frequency energy | a.u. | -3 ~ 0 | - |

### B. Prediction Flow Diagram

```
┌─────────────────┐
│ Input Raw       │
│ Features        │
│ (10 features)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature         │
│ Standardization │
│ (Within/Global) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LME Model       │
│ Prediction      │
│ (Fixed+Random)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Output Z-score  │
│ (Relative)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Inverse         │
│ Standardization │
│ (mean=5.25,     │
│  std=2.36)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Output Original │
│ (Absolute)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Calculate       │
│ Percentile      │
│ (Normal CDF)    │
└─────────────────┘
```

### C. Command Quick Reference

| Task | Command/Operation |
|------|------------------|
| **Interactive Mode** | `python lme_rating_predictor_v5_EN.py` |
| **Register Single Subject** | Menu 1 → Input CSV file |
| **Batch Register** | Menu 3 → Select method → Input path |
| **Predict Single Trial** | Menu 2 → Input subject ID → Input features |
| **Batch Predict** | Menu 4 → Select method → Input file |
| **Validate CSV** | `python csv_validator.py validate file.csv` |
| **Fix CSV** | `python csv_validator.py fix file.csv` |
| **Create Example** | `python csv_validator.py example` |
| **Run Demo** | `python lme_rating_predictor_v5_EN.py demo` |

### D. Key Constants

| Constant | Value | Description |
|----------|-------|-------------|
| Rating Mean | 5.2503 | Original rating mean from training data |
| Rating Std | 2.3568 | Original rating standard deviation from training data |
| New Subject Starting ID | 773 | Starting number for auto-assignment |
| Number of Training Subjects | 772 | Number of subjects in training data |
| Number of Features | 10 | Required number of features |
| Confidence Level | 0.95 | Default 95% confidence interval |

### E. File List

**Required Files:**
- `lme_rating_predictor_v5_EN.py` - Main program
- `lme_model_params.json` - Model parameters

**Documentation:**
- `Complete_User_Guide.md` - This document (you're reading it)
---

## Version History

### v5.4 (2025-11)
Release version

---

## License and Citation

### License
This software is for academic research use only.

### Citation
If you use this tool for publication, please cite:
```
Zhuang Y., Zhang L.B., Wang X.Q., Geng X.Y., & Hu L., (in preparation) From Normative Features to Multidimensional Estimation of Pain: A Large-Scale Study of Laser-Evoked Brain Responses.
```

---

## Contact

**Technical Support:**
- For issues, please refer to the "Troubleshooting" section of this document
- Check GitHub repository Issues

**Feedback and Suggestions:**
- Welcome to provide usage feedback
- Report bugs or suggest improvements

---

**Last Updated:** 2025-11-27  
**Document Version:** v5.4  
**Applicable Program Version:** lme_rating_predictor v5.4

---

## Quick Reference Card

### Start Program
```bash
python lme_rating_predictor_v5_EN.py
```

### Basic Workflow
```
1. Register Subject → 2. Predict → 3. View Results
```

### CSV Format
```csv
erp_N2,erp_P2,erp_N1,latency_N2,latency_P2,latency_N1,TF_gamma,TF_beta,TF_LEP,TF_alpha
-17.5,13.8,-9.1,200,392,152,0.004,1.7,31.2,-1.6
```

### Interpret Results
- **Z-score = 0**: Average level
- **Z-score > 0**: Above average
- **Z-score < 0**: Below average
- **Percentile 50%**: Median
- **Percentile 84%**: One standard deviation above mean

### Common Commands
```python
# Register
predictor.register_subject(None, df)

# Predict
result = predictor.predict(subject_id, features)

# Batch
results_df = predictor.batch_predict(subject_id, df)
```

**Enjoy using the tool!** 🎉
