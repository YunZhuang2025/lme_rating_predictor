"""
Linear Mixed-Effects (LME) Rating Predictor - Final Version (v5.4)

Core Improvements:
1. Database Effects: Considers average effects across all training databases
2. Subject ID: New subjects automatically use 773+ numbering to avoid conflicts with training data
3. Bash Interaction: Provides complete command-line interactive interface
4. Rating Statistics: Hard-coded original training data statistics (mean=5.2503, std=2.3568)

Use Cases:
- Have subject's historical feature data (5-50 trials)
- No need for rating history data
- Command-line friendly interactive prediction
- Automatic inverse standardization to original rating scale

Prediction Pipeline:
1. Feature Standardization: Raw Features → Z-score Features (using within-subject or global statistics)
2. Model Prediction: Z-score Features → Rating Z-score (LME model)
3. Inverse Standardization: Rating Z-score → Original Rating (using hard-coded original statistics)

Training Data Rating Statistics (from MATLAB):
- mean(all_bigdata_rating) = 5.2503  ← Original rating mean
- std(all_bigdata_rating) = 2.3568   ← Original rating standard deviation
- Note: Rating statistics in JSON are on Z-score scale, not used for inverse standardization

Author: Yun Zhuang
Date: 2025-11
Version: v5.4
If you use this tool in publications, please cite:
Zhuang Y., Zhang L.B., Wang X.Q., Geng X.Y., & Hu L., (in preparation) From Normative Features to Multidimensional Estimation of Pain: A Large-Scale Study of Laser-Evoked Brain Responses.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
import warnings
import sys
import os
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings('ignore')


@dataclass
class SubjectProfile:
    """
    Subject Profile: Stores feature standardization parameters
    """
    subject_id: Union[int, str]
    
    # Historical feature data
    historical_features: Dict[str, np.ndarray]
    
    # Feature standardization parameters
    feature_means: Dict[str, float] = None
    feature_stds: Dict[str, float] = None
    
    def __post_init__(self):
        """Calculate feature standardization parameters"""
        self.feature_means = {}
        self.feature_stds = {}
        
        for feat_name, feat_values in self.historical_features.items():
            self.feature_means[feat_name] = np.mean(feat_values)
            feat_std = np.std(feat_values, ddof=0)
            self.feature_stds[feat_name] = feat_std if feat_std > 0 else 1.0
    
    def standardize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Standardize raw features to Z-scores"""
        standardized = {}
        
        for feat_name, feat_value in features.items():
            if feat_name not in self.feature_means:
                raise ValueError(f"Feature {feat_name} not found in historical data")
            
            mean = self.feature_means[feat_name]
            std = self.feature_stds[feat_name]
            standardized[feat_name] = (feat_value - mean) / std
        
        return standardized
    
    def get_info(self) -> str:
        """Return subject information summary"""
        n_features = len(self.feature_means)
        if n_features > 0:
            feat_name = list(self.historical_features.keys())[0]
            n_trials = len(self.historical_features[feat_name])
        else:
            n_trials = 0
        
        return (f"Subject {self.subject_id}\n"
                f"  Historical trials: {n_trials}\n"
                f"  Number of features: {n_features}")


class LMERatingPredictor:
    """
    Linear Mixed-Effects Rating Predictor - Final Version v5.1
    
    Features:
    - Automatically handles all training databases
    - New subjects automatically assigned 773+ numbering
    - Complete command-line interactive interface
    - Hard-coded rating statistics (mean=5.2503, std=2.3568)
    - Automatic inverse standardization to original rating scale
    
    Important Notes:
    - Rating statistics in JSON file are on Z-score scale (already standardized)
    - Hard-coded statistics are on original rating scale (used for inverse standardization)
    - Prediction flow: Feature Z-score → Model predicts rating Z-score → Inverse standardization to original rating
    """
    
    def __init__(self, params_file: str, silent: bool = False):
        """
        Initialize predictor
        
        Parameters:
            params_file: Path to MATLAB-exported model parameters JSON file
            silent: Whether to run in silent mode (no initialization messages)
        """
        if not silent:
            print("="*80)
            print("LME Rating Predictor - Final Version v5.1")
            print("="*80)
        
        # Load model parameters
        with open(params_file, 'r', encoding='utf-8') as f:
            self.params = json.load(f)
        
        # Extract key parameters
        self.fixed_effects = self.params['fixed_effects']
        self.random_effects = self.params['random_effects']
        self.feature_names = self.params['feature_names'][0]
        
        # Reorder feature names to specified order
        desired_order = ['N1_amp', 'N2_amp', 'P2_amp', 'N1_lat', 'N2_lat', 'P2_lat', 
                        'ERP_mag', 'Alpha_mag', 'Beta_mag', 'Gamma_mag']
        # Verify all features exist
        if set(self.feature_names) == set(desired_order):
            self.feature_names = desired_order
        else:
            if not silent:
                print("⚠️  Warning: Feature names do not match expected order, keeping original order")
        
        self.data_stats = self.params.get('data_stats', {})
        self.model_info = self.params['model_info']
        
        # Extract training data information
        self.training_info = self._extract_training_info()
        
        # Extract global standardization parameters
        self._extract_global_stats(silent)
        
        # Extract fixed effect coefficients
        self._extract_fixed_coefficients(silent)
        
        # Store subject profiles
        self.subject_profiles: Dict[str, SubjectProfile] = {}
        
        # New subject counter (starts from 773)
        self.next_new_subject_id = 773
        
        if not silent:
            print("\n✓ Predictor initialization complete")
            print(f"  Training databases: {self.training_info['num_databases']}")
            print(f"  Training subjects: {self.training_info['num_subjects']}")
            print(f"  New subject ID starts from: {self.next_new_subject_id}")
            print("="*80)
    
    def _extract_training_info(self) -> Dict:
        """Extract training data information"""
        info = {
            'num_databases': self.model_info.get('num_groups_database', 10),
            'num_subjects': self.model_info.get('num_groups_subject', 772),
            'num_observations': self.model_info.get('num_observations', 0)
        }
        
        # Extract detailed info from grouping_info (if available)
        if 'grouping_info' in self.params:
            grouping = self.params['grouping_info']
            info['database_ids'] = grouping.get('unique_databases', [])
            info['subject_ids'] = grouping.get('unique_subjects', [])
        
        return info
    
    def _extract_global_stats(self, silent: bool = False):
        """Extract global standardization parameters"""
        # Global parameters for features
        self.global_feature_means = {}
        self.global_feature_stds = {}
        
        for feat in self.feature_names:
            if feat in self.data_stats:
                self.global_feature_means[feat] = self.data_stats[feat].get('mean', 0.0)
                self.global_feature_stds[feat] = self.data_stats[feat].get('std', 1.0)
            else:
                if not silent:
                    print(f"⚠️  Feature {feat} missing global statistics, using default 0±1")
                self.global_feature_means[feat] = 0.0
                self.global_feature_stds[feat] = 1.0
        
        # Rating inverse standardization parameters (hard-coded original rating statistics)
        # Original training data rating statistics: mean(all_bigdata_rating) = 5.2503, std = 2.3568
        # Note: Rating statistics in JSON file are on Z-score scale (standardized), cannot be used for inverse standardization
        
        # Always use original training data rating statistics (for inverse standardization)
        self.global_rating_mean = 5.2503  # Original rating mean
        self.global_rating_std = 2.3568   # Original rating std
        
        if not silent:
            print(f"\n✓ Rating inverse standardization parameters (hard-coded):")
            print(f"  Original rating mean: {self.global_rating_mean:.4f}")
            print(f"  Original rating std: {self.global_rating_std:.4f}")
            print(f"  Data source: MATLAB all_bigdata_rating")
            
            # If JSON has rating statistics, display but do not use (because they are on Z-score scale)
            if 'rating' in self.data_stats:
                json_mean = self.data_stats['rating'].get('mean', 0.0)
                json_std = self.data_stats['rating'].get('std', 1.0)
                print(f"  Note: Rating statistics in JSON are on Z-score scale（mean={json_mean:.4f}, std={json_std:.4f}）")
                print(f"        Ignored, using original statistics for inverse standardization")
    
    def set_rating_scale(self, mean: float, std: float):
        """
        Set rating scale mean and std
        
        Parameters:
            mean: Rating mean (e.g., 4.5 for 0-10 scale)
            std: Rating std (e.g., 2.0)
        """
        self.global_rating_mean = mean
        self.global_rating_std = std
        
        print(f"\n✓ Rating scale set:")
        print(f"  Mean: {mean:.2f}")
        print(f"  Std: {std:.2f}")
    
    def _extract_fixed_coefficients(self, silent: bool = False):
        """Extract fixed effect coefficients"""
        # Intercept
        intercept_key = [k for k in self.fixed_effects.keys() if 'Intercept' in k]
        if intercept_key:
            self.intercept = self.fixed_effects[intercept_key[0]]['estimate']
            self.intercept_se = self.fixed_effects[intercept_key[0]]['se']
        else:
            self.intercept = 0.0
            self.intercept_se = 0.0
        
        # Coefficients for each feature
        self.coefficients = {}
        self.coefficients_se = {}
        
        for feat in self.feature_names:
            if feat in self.fixed_effects:
                self.coefficients[feat] = self.fixed_effects[feat]['estimate']
                self.coefficients_se[feat] = self.fixed_effects[feat]['se']
            else:
                # Try to match
                found = False
                for key in self.fixed_effects.keys():
                    if feat.lower() in key.lower().replace('_', ''):
                        self.coefficients[feat] = self.fixed_effects[key]['estimate']
                        self.coefficients_se[feat] = self.fixed_effects[key]['se']
                        found = True
                        break
                
                if not found:
                    if not silent:
                        print(f"⚠️  Warning: Coefficient for feature {feat} not found, setting to 0")
                    self.coefficients[feat] = 0.0
                    self.coefficients_se[feat] = 0.0
        
        if not silent:
            # Print coefficient summary
            print(f"\n✓ Model coefficients (top 5 most important features):")
            
            # Sort by absolute coefficient value
            coef_sorted = sorted(
                [(feat, abs(self.coefficients[feat])) for feat in self.feature_names],
                key=lambda x: x[1],
                reverse=True
            )
            
            for i, (feat, abs_coef) in enumerate(coef_sorted[:5], 1):
                coef = self.coefficients[feat]
                pval = self.fixed_effects.get(feat, {}).get('pValue', np.nan)
                sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
                print(f"  {i}. {feat:<15s}: {coef:7.3f} {sig}")
    
    def register_subject(self,
                        subject_id: Optional[Union[int, str]] = None,
                        historical_features: pd.DataFrame = None,
                        feature_cols: Optional[List[str]] = None) -> SubjectProfile:
        """
        Register subject
        
        Parameters:
            subject_id: Subject ID（Optional, auto-assigns 773+ if not provided）
            historical_features: DataFrame containing features
            feature_cols: List of feature column names (default: use all model features)
        
        Returns:
            SubjectProfile object
        """
        # If subject_id not provided, auto-assign new ID
        if subject_id is None:
            subject_id = self.next_new_subject_id
            self.next_new_subject_id += 1
            print(f"\n✓ Auto-assigned subject ID: {subject_id}")
        
        if feature_cols is None:
            feature_cols = self.feature_names
        
        # Check feature columns
        missing_cols = [col for col in feature_cols if col not in historical_features.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns in historical data: {missing_cols}")
        
        # Extract feature data
        hist_features = {
            feat: historical_features[feat].values 
            for feat in feature_cols
        }
        
        # Create subject profile
        profile = SubjectProfile(
            subject_id=subject_id,
            historical_features=hist_features
        )
        
        # Store profile
        self.subject_profiles[str(subject_id)] = profile
        
        print(f"\n✓ Subject registered: {subject_id}")
        print(f"  {profile.get_info()}")
        
        return profile
    
    def predict(self,
               subject_id: Union[int, str],
               features: Dict[str, float],
               confidence_level: float = 0.95) -> Dict:
        """
        Predict rating
        
        Parameters:
            subject_id: Subject ID
            features: Dictionary of raw feature values (10 features)
            confidence_level: Confidence level (default: 0.95)
        
        Returns:
            Dictionary containing prediction results
        """
        subject_key = str(subject_id)
        
        # Check if subject is registered
        if subject_key in self.subject_profiles:
            # Registered: Use subject-specific feature standardization
            profile = self.subject_profiles[subject_key]
            features_zscore = profile.standardize_features(features)
            is_registered = True
        else:
            # Unregistered: Use global feature standardization
            features_zscore = {}
            for feat_name, feat_value in features.items():
                if feat_name not in self.global_feature_means:
                    raise ValueError(f"Feature {feat_name} has no global statistics")
                
                mean = self.global_feature_means[feat_name]
                std = self.global_feature_stds[feat_name]
                features_zscore[feat_name] = (feat_value - mean) / std
            
            is_registered = False
        
        # Predict on Z-score scale
        zscore_result = self._predict_zscore(features_zscore, confidence_level)
        
        # Convert to original rating scale
        rating_original = (zscore_result['rating_mean'] * self.global_rating_std + 
                          self.global_rating_mean)
        rating_se_original = zscore_result['rating_se'] * self.global_rating_std
        ci_lower_original = (zscore_result['ci_lower'] * self.global_rating_std + 
                            self.global_rating_mean)
        ci_upper_original = (zscore_result['ci_upper'] * self.global_rating_std + 
                            self.global_rating_mean)
        
        # Build return result
        result = {
            # Z-score scale (relative values)
            'rating_zscore': zscore_result['rating_mean'],
            'rating_se_zscore': zscore_result['rating_se'],
            'ci_lower_zscore': zscore_result['ci_lower'],
            'ci_upper_zscore': zscore_result['ci_upper'],
            
            # Original scale (estimated absolute values)
            'rating_original': rating_original,
            'rating_se_original': rating_se_original,
            'ci_lower_original': ci_lower_original,
            'ci_upper_original': ci_upper_original,
            
            # Metadata
            'confidence_level': confidence_level,
            'subject_id': subject_id,
            'subject_registered': is_registered
        }
        
        return result
    
    def _predict_zscore(self, 
                       features_zscore: Dict[str, float],
                       confidence_level: float = 0.95) -> Dict:
        """Predict on Z-score scale (internal method)"""
        # Convert to array
        X = self._dict_to_array(features_zscore)
        
        # Calculate fixed effects prediction
        y_pred = self.intercept
        for i, feat in enumerate(self.feature_names):
            y_pred += self.coefficients[feat] * X[i]
        
        # Calculate standard error
        var_pred = self.intercept_se ** 2
        for i, feat in enumerate(self.feature_names):
            var_pred += (self.coefficients_se[feat] * X[i]) ** 2
        
        # Add residual variance
        var_residual = self.random_effects.get('residual_variance', 0)
        var_pred += var_residual
        
        se_pred = np.sqrt(var_pred)
        
        # Calculate confidence interval
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        ci_lower = y_pred - z_score * se_pred
        ci_upper = y_pred + z_score * se_pred
        
        return {
            'rating_mean': float(y_pred),
            'rating_se': float(se_pred),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper)
        }
    
    def _dict_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to array"""
        X = np.zeros(len(self.feature_names))
        
        for i, feat in enumerate(self.feature_names):
            if feat in features:
                X[i] = features[feat]
            else:
                raise ValueError(f"Missing required feature: {feat}")
        
        return X
    
    def batch_predict(self,
                     subject_id: Union[int, str],
                     features_df: pd.DataFrame,
                     confidence_level: float = 0.95) -> pd.DataFrame:
        """Batch prediction"""
        # Check required columns
        missing_cols = [col for col in self.feature_names if col not in features_df.columns]
        if missing_cols:
            raise ValueError(f"Input DataFrame missing columns: {missing_cols}")
        
        results = []
        
        print(f"\nBatch prediction: Subject {subject_id}")
        print(f"Number of samples: {len(features_df)}")
        
        for idx, row in features_df.iterrows():
            features = {feat: row[feat] for feat in self.feature_names}
            
            try:
                result = self.predict(subject_id, features, confidence_level)
                result['trial_index'] = idx
                
                # Add input features
                for feat in self.feature_names:
                    result[f'{feat}_input'] = features[feat]
                
                results.append(result)
            
            except Exception as e:
                print(f"  ⚠️  Trial {idx} prediction failed: {e}")
        
        results_df = pd.DataFrame(results)
        
        print(f"✓ Complete: {len(results)}/{len(features_df)} trials")
        print(f"  Average Z-score: {results_df['rating_zscore'].mean():.3f}")
        print(f"  Average original rating: {results_df['rating_original'].mean():.2f}")
        
        return results_df
    
    def print_prediction_summary(self, result: Dict):
        """Print user-friendly prediction summary"""
        print(f"\n" + "="*60)
        print(f"Prediction Result - Subject {result['subject_id']}")
        print("="*60)
        
        # Z-score results
        print(f"\nZ-score scale (relative values):")
        print(f"  Prediction: {result['rating_zscore']:6.3f} ± {result['rating_se_zscore']:.3f}")
        print(f"  {result['confidence_level']*100:.0f}% CI: "
              f"[{result['ci_lower_zscore']:6.3f}, {result['ci_upper_zscore']:6.3f}]")
        
        # Convert to percentile
        zscore = result['rating_zscore']
        percentile = stats.norm.cdf(zscore) * 100
        print(f"  Percentile: This pain level exceeds {percentile:.1f}% of trials")
        
        # Original scale results
        print(f"\nOriginal rating scale (estimated):")
        print(f"  Prediction: {result['rating_original']:6.2f} ± {result['rating_se_original']:.2f}")
        print(f"  {result['confidence_level']*100:.0f}% CI: "
              f"[{result['ci_lower_original']:6.2f}, {result['ci_upper_original']:6.2f}]")
        
        # Notes
        if not result['subject_registered']:
            print(f"\n⚠️  Subject not registered, using global standardization")
            print(f"  Recommend registering subject for better accuracy")
        
        print("="*60)
    
    def list_registered_subjects(self):
        """List all registered subjects"""
        n_subjects = len(self.subject_profiles)
        print(f"\nRegistered subjects: {n_subjects}")
        
        if n_subjects > 0:
            print(f"\n{'Subject ID':<15s} {'Trials':>10s} {'Number of features':>10s}")
            print("-" * 40)
            
            for subject_id, profile in self.subject_profiles.items():
                feat_name = list(profile.historical_features.keys())[0]
                n_trials = len(profile.historical_features[feat_name])
                n_features = len(profile.feature_means)
                
                print(f"{str(subject_id):<15s} {n_trials:10d} {n_features:10d}")
    
    def save_subject_profile(self, subject_id: Union[int, str], output_file: str):
        """Save subject profile to CSV"""
        subject_key = str(subject_id)
        
        if subject_key not in self.subject_profiles:
            raise ValueError(f"Subject {subject_id} not registered")
        
        profile = self.subject_profiles[subject_key]
        
        # Build DataFrame
        data = {
            'subject_id': [subject_id] * len(self.feature_names),
            'feature': self.feature_names,
            'mean': [profile.feature_means[f] for f in self.feature_names],
            'std': [profile.feature_stds[f] for f in self.feature_names]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        print(f"\n✓ Subject profile saved: {output_file}")
    
    def load_subject_profile(self, subject_id: Union[int, str], input_file: str):
        """Load subject profile from CSV"""
        df = pd.read_csv(input_file)
        
        # Reconstruct historical features (generate random data matching mean and std)
        n_samples = 20  # Generate 20 sample points
        hist_features = {}
        
        for _, row in df.iterrows():
            feat = row['feature']
            mean = row['mean']
            std = row['std']
            
            # Generate samples matching statistics
            hist_features[feat] = np.random.normal(mean, std, n_samples)
        
        # Create profile
        profile = SubjectProfile(
            subject_id=subject_id,
            historical_features=hist_features
        )
        
        # Directly set statistical parameters (more accurate)
        profile.feature_means = {row['feature']: row['mean'] for _, row in df.iterrows()}
        profile.feature_stds = {row['feature']: row['std'] for _, row in df.iterrows()}
        
        # Store profile
        self.subject_profiles[str(subject_id)] = profile
        
        print(f"\n✓ Subject profile loaded from file: {subject_id}")
        print(f"  File: {input_file}")


def interactive_mode():
    """Command-line Interactive mode"""
    print("\n" + "="*80)
    print(" "*20 + "LME Rating Predictor - Interactive Mode")
    print("="*80)
    
    # 1. Load model
    params_file = input("\nModel parameters file path (default: lme_model_params.json): ").strip()
    if not params_file:
        params_file = "lme_model_params.json"
    
    if not os.path.exists(params_file):
        print(f"\n❌ File does not exist: {params_file}")
        return
    
    try:
        predictor = LMERatingPredictor(params_file)
    except Exception as e:
        print(f"\n❌ Load failed: {e}")
        return
    
    # 2. Display rating scale info (hard-coded, no manual setting needed)
    print("\n" + "-"*80)
    print("Rating Scale Information")
    print("-"*80)
    print(f"Using training data rating statistics (hard-coded):")
    print(f"  Mean: {predictor.global_rating_mean:.4f}")
    print(f"  Std: {predictor.global_rating_std:.4f}")
    print(f"  Data source: MATLAB all_bigdata_rating")
    print("\nTo modify, call in Python script:")
    print("  predictor.set_rating_scale(mean=?, std=?)")
    
    # Main loop
    while True:
        print("\n" + "="*80)
        print("Main Menu")
        print("="*80)
        print("1. Register new subject (single)")
        print("2. Predict new trial")
        print("3. Batch register subjects")
        print("4. Batch predict multiple trials for subjects")
        print("5. Exit")
        
        choice = input("\nPlease select (1-5): ").strip()
        
        if choice == '1':
            # Register new subject (single)
            print("\n" + "-"*80)
            print("Register New Subject")
            print("-"*80)
            
            subject_id = input("Subject ID (Leave empty for auto-assignment (773+)): ").strip()
            if not subject_id:
                subject_id = None
            else:
                try:
                    subject_id = int(subject_id)
                except ValueError:
                    pass  # Keep as string
            
            csv_file = input("Historical feature data CSV file path: ").strip()
            
            if not os.path.exists(csv_file):
                print(f"❌ File does not exist: {csv_file}")
                continue
            
            try:
                historical_data = pd.read_csv(csv_file)
                print(f"\n✓ Read {len(historical_data)} historical records")
                print(f"  Columns: {list(historical_data.columns)}")
                
                profile = predictor.register_subject(subject_id, historical_data)
                
            except Exception as e:
                print(f"\n❌ Registration failed: {e}")
        
        elif choice == '2':
            # Predict new trial
            print("\n" + "-"*80)
            print("Predict new trial")
            print("-"*80)
            
            # First display registered subjects
            if len(predictor.subject_profiles) == 0:
                print("⚠️  No registered subjects currently，Please register first (menu 1)")
                continue
            
            print(f"\nCurrently registered subjects: {list(predictor.subject_profiles.keys())}")
            subject_id = input("Subject ID: ").strip()
            
            # Check if subject exists
            if subject_id not in predictor.subject_profiles:
                print(f"⚠️  Subject {subject_id} not registered")
                use_global = input("Use global standardization for prediction? (y/n): ").strip().lower()
                if use_global != 'y':
                    continue
            
            print("\nEnter feature values:")
            features = {}
            
            for feat in predictor.feature_names:
                while True:
                    value = input(f"  {feat}: ").strip()
                    
                    if value.lower() == 'q':
                        break
                    
                    try:
                        features[feat] = float(value)
                        break
                    except ValueError:
                        print(f"    ⚠️  Invalid input, please enter a number")
            
            if len(features) == len(predictor.feature_names):
                try:
                    result = predictor.predict(subject_id, features)
                    predictor.print_prediction_summary(result)
                    
                    # Save options
                    save = input("\nSave results to CSV?(y/n): ").strip().lower()
                    if save == 'y':
                        output_file = input("Output filename (default: prediction_result.csv): ").strip()
                        if not output_file:
                            output_file = "prediction_result.csv"
                        
                        result_df = pd.DataFrame([{**features, **result}])
                        result_df.to_csv(output_file, index=False)
                        print(f"✓ Saved: {output_file}")
                
                except Exception as e:
                    print(f"\n❌ Prediction error: {e}")
        
        elif choice == '3':
            # Batch register subjects
            print("\n" + "-"*80)
            print("Batch register subjects")
            print("-"*80)
            print("\nNote: Please provide folder or file list containing subject IDs and historical feature data")
            
            # Method 1: Batch register from folder
            mode = input("\nRegistration method (1=from folder, 2=manually input file list): ").strip()
            
            if mode == '1':
                folder = input("Feature data folder path: ").strip()
                
                if not os.path.exists(folder):
                    print(f"❌ Folder does not exist: {folder}")
                    continue
                
                # Find all CSV files
                import glob
                csv_files = glob.glob(os.path.join(folder, "*.csv"))
                
                if not csv_files:
                    print(f"❌ No CSV files in folder")
                    continue
                
                print(f"\nFound {len(csv_files)} CSV files:")
                for f in csv_files[:5]:  # Showing first 5
                    print(f"  - {os.path.basename(f)}")
                if len(csv_files) > 5:
                    print(f"  ... And {len(csv_files)-5} more files")
                
                confirm = input("\nStart batch registration? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue
                
                success_count = 0
                for csv_file in csv_files:
                    try:
                        # Extract subject ID from filename (assumed format: subject_XXX_*.csv）
                        filename = os.path.basename(csv_file)
                        # Try multiple naming formats
                        if 'subject_' in filename:
                            subject_id = filename.split('subject_')[1].split('_')[0]
                        else:
                            # Use auto-assignment
                            subject_id = None
                        
                        historical_data = pd.read_csv(csv_file)
                        profile = predictor.register_subject(subject_id, historical_data)
                        success_count += 1
                        print(f"✓ Registered: {profile.subject_id} (from {filename})")
                    
                    except Exception as e:
                        print(f"✗ Failed: {filename} - {e}")
                
                print(f"\nBatch registration complete: {success_count}/{len(csv_files)} successful")
            
            elif mode == '2':
                # Manually input file list
                print("\nEnter file paths (one per line, empty line to end):")
                csv_files = []
                while True:
                    line = input("  File path: ").strip()
                    if not line:
                        break
                    csv_files.append(line)
                
                if not csv_files:
                    print("❌ No files entered")
                    continue
                
                success_count = 0
                for csv_file in csv_files:
                    if not os.path.exists(csv_file):
                        print(f"✗ File does not exist: {csv_file}")
                        continue
                    
                    # Ask for Subject ID
                    subject_id = input(f"  {os.path.basename(csv_file)}  subject ID (leave empty for auto-assignment): ").strip()
                    if not subject_id:
                        subject_id = None
                    
                    try:
                        historical_data = pd.read_csv(csv_file)
                        profile = predictor.register_subject(subject_id, historical_data)
                        success_count += 1
                        print(f"✓ Registered: {profile.subject_id}")
                    
                    except Exception as e:
                        print(f"✗ Failed: {e}")
                
                print(f"\nBatch registration complete: {success_count}/{len(csv_files)} successful")
            
            # Display all registered subjects
            print("\nCurrently registered subjects:")
            predictor.list_registered_subjects()
        
        elif choice == '4':
            # Batch predict multiple trials for subjects
            print("\n" + "-"*80)
            print("Batch predict multiple trials for subjects")
            print("-"*80)
            
            if len(predictor.subject_profiles) == 0:
                print("⚠️  No registered subjects currently，Please register first (menu 1 or 3)")
                continue
            
            print(f"\nCurrently registered subjects: {list(predictor.subject_profiles.keys())}")
            
            # Method 1: Batch predict single subject
            mode = input("\nPrediction mode (1=batch predict single subject, 2=predict separately for multiple subjects): ").strip()
            
            if mode == '1':
                # Batch predict single subject
                subject_id = input("Subject ID: ").strip()
                
                if subject_id not in predictor.subject_profiles:
                    print(f"⚠️  Subject {subject_id} not registered")
                    continue
                
                input_csv = input("Input CSV file path (containing feature columns): ").strip()
                
                if not os.path.exists(input_csv):
                    print(f"❌ File does not exist: {input_csv}")
                    continue
                
                output_csv = input("Output CSV file path (default: predictions_{subject_id}.csv): ").strip()
                if not output_csv:
                    output_csv = f"predictions_{subject_id}.csv"
                
                try:
                    features_df = pd.read_csv(input_csv)
                    print(f"\n✓ Read {len(features_df)} records")
                    
                    results_df = predictor.batch_predict(subject_id, features_df)
                    
                    results_df.to_csv(output_csv, index=False)
                    print(f"\n✓ Batch prediction complete!")
                    print(f"  Output file: {output_csv}")
                
                except Exception as e:
                    print(f"\n❌ Error: {e}")
            
            elif mode == '2':
                # Predict separately for multiple subjects
                print("\nEnter subject IDs to predict (comma-separated, or enter 'all' to predict all):")
                subject_input = input("Subject ID: ").strip()
                
                if subject_input.lower() == 'all':
                    subject_ids = list(predictor.subject_profiles.keys())
                else:
                    subject_ids = [s.strip() for s in subject_input.split(',')]
                
                # Ask for data file method
                data_mode = input("\nData file method (1=unified file, 2=specify separately): ").strip()
                
                if data_mode == '1':
                    # All subjects use same feature file
                    input_csv = input("Input CSV file path (containing feature columns): ").strip()
                    
                    if not os.path.exists(input_csv):
                        print(f"❌ File does not exist: {input_csv}")
                        continue
                    
                    features_df = pd.read_csv(input_csv)
                    print(f"\n✓ Read {len(features_df)} records")
                    
                    output_folder = input("Output folder (default: ./predictions/): ").strip()
                    if not output_folder:
                        output_folder = "./predictions/"
                    
                    os.makedirs(output_folder, exist_ok=True)
                    
                    success_count = 0
                    for subject_id in subject_ids:
                        if subject_id not in predictor.subject_profiles:
                            print(f"⚠️  Skipping unregistered subject: {subject_id}")
                            continue
                        
                        try:
                            output_csv = os.path.join(output_folder, f"predictions_{subject_id}.csv")
                            results_df = predictor.batch_predict(subject_id, features_df)
                            results_df.to_csv(output_csv, index=False)
                            success_count += 1
                            print(f"✓ Complete: {subject_id} → {output_csv}")
                        
                        except Exception as e:
                            print(f"✗ Failed: {subject_id} - {e}")
                    
                    print(f"\nBatch prediction complete: {success_count}/{len(subject_ids)} successful")
                
                elif data_mode == '2':
                    # Specify file separately for each subject
                    success_count = 0
                    for subject_id in subject_ids:
                        if subject_id not in predictor.subject_profiles:
                            print(f"\n⚠️  Skipping unregistered subject: {subject_id}")
                            continue
                        
                        input_csv = input(f"\n{subject_id}  input CSV file: ").strip()
                        
                        if not os.path.exists(input_csv):
                            print(f"✗ File does not exist: {input_csv}")
                            continue
                        
                        output_csv = input(f"Output file (default: predictions_{subject_id}.csv): ").strip()
                        if not output_csv:
                            output_csv = f"predictions_{subject_id}.csv"
                        
                        try:
                            features_df = pd.read_csv(input_csv)
                            results_df = predictor.batch_predict(subject_id, features_df)
                            results_df.to_csv(output_csv, index=False)
                            success_count += 1
                            print(f"✓ Complete: {subject_id}")
                        
                        except Exception as e:
                            print(f"✗ Failed: {subject_id} - {e}")
                    
                    print(f"\nBatch prediction complete: {success_count}/{len(subject_ids)} successful")
        
        elif choice == '5':
            print("\nGoodbye!")
            break
        
        else:
            print("\nInvalid choice, please try again")


def demo():
    """Demonstrate basic usage of predictor"""
    print("\n" + "="*80)
    print(" "*25 + "Usage Demo")
    print("="*80)
    
    # 1. Load predictor (rating statistics hard-coded)
    print("\nStep 1: Load predictor")
    print("-" * 80)
    predictor = LMERatingPredictor('lme_model_params.json')
    
    # 2. Prepare historical feature data
    print("\nStep 2: Prepare subject historical feature data")
    print("-" * 80)
    
    np.random.seed(42)
    feature_history = pd.DataFrame({
        'N1_amp': np.random.normal(-9, 1.5, 20),
        'N2_amp': np.random.normal(-17, 3, 20),
        'P2_amp': np.random.normal(13.5, 2, 20),
        'N1_lat': np.random.normal(152, 10, 20),
        'N2_lat': np.random.normal(200, 15, 20),
        'P2_lat': np.random.normal(390, 20, 20),
        'ERP_mag': np.random.normal(31, 4, 20),
        'Alpha_mag': np.random.normal(79, 8, 20),
        'Beta_mag': np.random.normal(1.7, 0.3, 20),
        'Gamma_mag': np.random.normal(2.6, 0.5, 20)
    })
    
    print(f"Historical feature data: {len(feature_history)} trials")
    
    # 3. Register subject (auto-assign ID)
    print("\nStep 3: Register new subject (auto-assign ID)")
    print("-" * 80)
    profile = predictor.register_subject(subject_id=None, historical_features=feature_history)
    assigned_id = profile.subject_id
    
    # 4. Predict new trial
    print("\nStep 4: Predict new trial")
    print("-" * 80)
    
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
    
    result = predictor.predict(subject_id=assigned_id, features=new_features)
    
    # Print results
    predictor.print_prediction_summary(result)
    
    print("\n" + "="*80)
    print("✓ Demo complete")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'demo':
            demo()
        elif sys.argv[1] == 'interactive' or sys.argv[1] == '-i':
            interactive_mode()
        else:
            print("Usage:")
            print("  python lme_rating_predictor_v5_final.py demo         # Run demo")
            print("  python lme_rating_predictor_v5_final.py interactive # Interactive mode")
            print("  python lme_rating_predictor_v5_final.py -i          # Interactive mode（shorthand）")
    else:
        # Default enters interactive mode
        interactive_mode()
