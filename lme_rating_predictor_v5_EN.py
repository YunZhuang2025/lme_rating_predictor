"""
Linear Mixed-Effects Model (LME) Rating Predictor - Final Version (v5.4)

Core Improvements:
1. Database effect: Consider average effect of all training databases
2. Subject ID: New subjects automatically use 773+ numbering, avoiding conflicts with training data
3. Bash interaction: Provide complete command-line interactive interface
4. Rating statistics: Hard-coded original training data statistics (mean=5.2503, std=2.3568)

Usage Scenarios:
- Has subject's feature history data (5-50 trials)
- No rating history data needed
- Command-line friendly interactive prediction
- Automatic inverse standardization to original rating scale

Prediction Process Description:
1. Feature standardization: Raw features → Z-score features (using within-subject or global statistics)
2. Model prediction: Z-score features → Rating Z-score (LME model)
3. Inverse standardization: Rating Z-score → Raw Rating (using hard-coded original statistics)

Training Data Rating Statistics (MATLAB):
- mean(all_bigdata_rating) = 5.2503  ← Original rating mean
- std(all_bigdata_rating) = 2.3568   ← Original rating standard deviation
- Note: Rating statistics in JSON are in Z-score scale, not used for inverse standardization

Author: Yun Zhuang
Date: 2025-11
Version: v5.4
If you use this tool for publication, please cite:
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
    Subject profile: Store feature standardization parameters
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
                raise ValueError(f"Feature {feat_name} not in historical data")
            
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
    Linear Mixed-Effects Model Rating Predictor - Final Version v5.1
    
    Features:
    - Automatically handles all training databases
    - New subjects automatically assigned 773+ numbering
    - Complete command-line interactive interface
    - Rating statistics hard-coded (mean=5.2503, std=2.3568)
    - Automatic inverse standardization to original rating scale
    
    Important Notes:
    - Rating statistics in JSON file are in Z-score scale (already standardized)
    - Hard-coded statistics are in original rating scale (used for inverse standardization)
    - Prediction flow: Feature Z-score → Model predicts rating Z-score → Inverse standardization to original rating
    """
    
    def __init__(self, params_file: str, silent: bool = False):
        """
        Initialize predictor
        
        Parameters:
            params_file: Path to model parameter JSON file exported from MATLAB
            silent: Whether to use silent mode (no initialization printing)
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
        
        # New subject counter (starting from 773)
        self.next_new_subject_id = 773
        
        if not silent:
            print("\n✓ Predictor initialization complete")
            print(f"  Number of training databases: {self.training_info['num_databases']}")
            print(f"  Number of training subjects: {self.training_info['num_subjects']}")
            print(f"  Starting ID for new subjects: {self.next_new_subject_id}")
            print("="*80)
    
    def _extract_training_info(self) -> Dict:
        """Extract training data information"""
        info = {
            'num_databases': self.model_info.get('num_groups_database', 10),
            'num_subjects': self.model_info.get('num_groups_subject', 772),
            'num_observations': self.model_info.get('num_observations', 0)
        }
        
        # Extract detailed information from grouping_info (if available)
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
                    print(f"⚠️  Feature {feat} missing global statistics, using default values 0±1")
                self.global_feature_means[feat] = 0.0
                self.global_feature_stds[feat] = 1.0
        
        # Rating inverse standardization parameters (hard-coded original rating statistics)
        # Training data original rating statistics: mean(all_bigdata_rating) = 5.2503, std = 2.3568
        # Note: Rating statistics in JSON file are in Z-score scale (already standardized), cannot be used for inverse standardization
        
        # Always use original training data rating statistics (for inverse standardization)
        self.global_rating_mean = 5.2503  # Original rating mean
        self.global_rating_std = 2.3568   # Original rating standard deviation
        
        if not silent:
            print(f"\n✓ Rating inverse standardization parameters (hard-coded):")
            print(f"  Original rating mean: {self.global_rating_mean:.4f}")
            print(f"  Original rating std: {self.global_rating_std:.4f}")
            print(f"  Data source: MATLAB all_bigdata_rating")
            
            # If JSON has rating statistics, display but don't use (because they're in Z-score scale)
            if 'rating' in self.data_stats:
                json_mean = self.data_stats['rating'].get('mean', 0.0)
                json_std = self.data_stats['rating'].get('std', 1.0)
                print(f"  Note: Rating statistics in JSON are in Z-score scale (mean={json_mean:.4f}, std={json_std:.4f})")
                print(f"        Ignored, using original statistics for inverse standardization")
    
    def set_rating_scale(self, mean: float, std: float):
        """
        Set the mean and standard deviation of the rating scale
        
        Parameters:
            mean: Mean of ratings (e.g., 4.5 for 0-10 scale)
            std: Standard deviation of ratings (e.g., 2.0)
        """
        self.global_rating_mean = mean
        self.global_rating_std = std
        
        print(f"\n✓ Rating scale set:")
        print(f"  Mean: {mean:.2f}")
        print(f"  Std: {std:.2f}")
    
    def _extract_fixed_coefficients(self, silent: bool = False):
        """Extract fixed effect coefficients"""
        self.beta_intercept = self.fixed_effects.get('(Intercept)', 0.0)
        
        self.beta_features = {}
        for feat in self.feature_names:
            self.beta_features[feat] = self.fixed_effects.get(feat, 0.0)
        
        if not silent:
            print(f"\n✓ Fixed effect coefficients extracted:")
            print(f"  Intercept: {self.beta_intercept:.4f}")
            print(f"  Features: {len(self.beta_features)} coefficients")
    
    def _compute_database_effect(self) -> float:
        """
        Compute database random effect
        
        Strategy:
        - Training data contains 10 databases with random effects b_database
        - For new subjects, use average effect of all training databases
        - This is a conservative strategy assuming new subjects come from a "typical" database
        
        Returns:
            Database random effect value
        """
        # Extract all database random effects
        database_effects = []
        
        for key, value in self.random_effects.items():
            if key.startswith('database_'):
                database_effects.append(value.get('(Intercept)', 0.0))
        
        if len(database_effects) == 0:
            return 0.0
        
        # Return average database effect
        return np.mean(database_effects)
    
    def _compute_subject_effect(self, subject_id: Union[int, str]) -> float:
        """
        Compute subject random effect
        
        Parameters:
            subject_id: Subject ID (if training subject, use trained b_subject; if new subject, use 0)
        
        Returns:
            Subject random effect value
        """
        # Look up in random_effects
        subject_key = f"subject_{subject_id}"
        
        if subject_key in self.random_effects:
            # Training subject
            return self.random_effects[subject_key].get('(Intercept)', 0.0)
        else:
            # New subject
            return 0.0
    
    def register_subject(
        self,
        subject_id: Optional[Union[int, str]] = None,
        historical_features: Optional[Union[pd.DataFrame, Dict[str, List]]] = None,
        use_global_stats: bool = False
    ) -> SubjectProfile:
        """
        Register new subject
        
        Parameters:
            subject_id: Subject ID (if None, automatically assign 773+)
            historical_features: Historical feature data, can be:
                - pd.DataFrame: Rows are trials, columns are feature names
                - Dict[str, List]: Keys are feature names, values are trial data
            use_global_stats: Whether to use global statistics for standardization (default: within-subject)
        
        Returns:
            SubjectProfile: Registered subject profile
        """
        # Handle subject ID
        if subject_id is None:
            subject_id = self.next_new_subject_id
            self.next_new_subject_id += 1
            print(f"\n✓ Automatically assigned subject ID: {subject_id}")
        else:
            subject_id = str(subject_id)
        
        # Convert historical_features to dict format
        if isinstance(historical_features, pd.DataFrame):
            historical_dict = {col: historical_features[col].values for col in historical_features.columns}
        else:
            historical_dict = {k: np.array(v) for k, v in historical_features.items()}
        
        # Verify all features are present
        missing_features = set(self.feature_names) - set(historical_dict.keys())
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Create subject profile
        profile = SubjectProfile(
            subject_id=subject_id,
            historical_features=historical_dict
        )
        
        # If using global statistics, override profile's standardization parameters
        if use_global_stats:
            profile.feature_means = self.global_feature_means.copy()
            profile.feature_stds = self.global_feature_stds.copy()
            print(f"  Using global standardization parameters")
        else:
            print(f"  Using within-subject standardization parameters")
        
        # Save profile
        self.subject_profiles[str(subject_id)] = profile
        
        # Print registration info
        n_trials = len(list(historical_dict.values())[0])
        print(f"  Historical trials: {n_trials}")
        print(f"  Number of features: {len(historical_dict)}")
        
        return profile
    
    def predict(
        self,
        subject_id: Union[int, str],
        features: Dict[str, float],
        return_components: bool = False
    ) -> Dict:
        """
        Predict rating for single trial
        
        Parameters:
            subject_id: Subject ID
            features: Raw feature values (dict)
            return_components: Whether to return model component breakdown
        
        Returns:
            Prediction result dictionary containing:
            - predicted_rating: Predicted rating (original scale)
            - predicted_rating_zscore: Predicted rating (Z-score scale)
            - subject_id: Subject ID
            And optionally:
            - components: Model component breakdown
            - standardized_features: Standardized feature values
        """
        subject_id = str(subject_id)
        
        # Check if subject is registered
        if subject_id not in self.subject_profiles:
            raise ValueError(f"Subject {subject_id} not registered. Please register first using register_subject()")
        
        profile = self.subject_profiles[subject_id]
        
        # 1. Standardize features (using subject's standardization parameters)
        features_z = profile.standardize_features(features)
        
        # 2. Compute fixed effects contribution
        fixed_contribution = self.beta_intercept
        for feat_name, feat_z in features_z.items():
            fixed_contribution += self.beta_features[feat_name] * feat_z
        
        # 3. Compute random effects
        b_database = self._compute_database_effect()
        b_subject = self._compute_subject_effect(subject_id)
        
        # 4. Compute final predicted rating (Z-score scale)
        rating_zscore = fixed_contribution + b_database + b_subject
        
        # 5. Inverse standardization: Convert Z-score to original rating scale
        # predicted_rating = rating_mean + rating_zscore * rating_std
        predicted_rating = self.global_rating_mean + rating_zscore * self.global_rating_std
        
        # Build result
        result = {
            'predicted_rating': predicted_rating,
            'predicted_rating_zscore': rating_zscore,
            'subject_id': subject_id
        }
        
        if return_components:
            result['components'] = {
                'fixed_effects': fixed_contribution,
                'database_effect': b_database,
                'subject_effect': b_subject,
                'total_zscore': rating_zscore
            }
            result['standardized_features'] = features_z
        
        return result
    
    def batch_predict(
        self,
        subject_id: Union[int, str],
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Batch predict multiple trials for one subject
        
        Parameters:
            subject_id: Subject ID
            features_df: Feature DataFrame, columns are feature names
        
        Returns:
            Results DataFrame with predicted_rating column added
        """
        subject_id = str(subject_id)
        
        # Check if subject is registered
        if subject_id not in self.subject_profiles:
            raise ValueError(f"Subject {subject_id} not registered")
        
        # Predict each trial
        predictions = []
        for idx, row in features_df.iterrows():
            features = row.to_dict()
            result = self.predict(subject_id, features)
            predictions.append(result['predicted_rating'])
        
        # Add prediction column
        result_df = features_df.copy()
        result_df['predicted_rating'] = predictions
        result_df['subject_id'] = subject_id
        
        return result_df
    
    def print_prediction_summary(self, result: Dict):
        """
        Print formatted prediction summary
        
        Parameters:
            result: Prediction result returned by predict()
        """
        print("\n" + "="*80)
        print("Prediction Result")
        print("="*80)
        
        print(f"\nSubject ID: {result['subject_id']}")
        print(f"Predicted Rating: {result['predicted_rating']:.2f}")
        
        if 'components' in result:
            print("\nModel Component Breakdown:")
            print(f"  Fixed effects:     {result['components']['fixed_effects']:>8.4f}")
            print(f"  Database effect:   {result['components']['database_effect']:>8.4f}")
            print(f"  Subject effect:    {result['components']['subject_effect']:>8.4f}")
            print(f"  {'─'*40}")
            print(f"  Total (Z-score):   {result['components']['total_zscore']:>8.4f}")
            print(f"  After inverse standardization → Rating: {result['predicted_rating']:.2f}")
        
        if 'standardized_features' in result:
            print("\nStandardized Features (Z-scores):")
            for feat_name, feat_z in result['standardized_features'].items():
                print(f"  {feat_name:15s}: {feat_z:>8.4f}")
        
        print("="*80)
    
    def get_subject_info(self, subject_id: Union[int, str]) -> str:
        """
        Get subject information
        
        Parameters:
            subject_id: Subject ID
        
        Returns:
            Subject information string
        """
        subject_id = str(subject_id)
        
        if subject_id not in self.subject_profiles:
            return f"Subject {subject_id} not registered"
        
        return self.subject_profiles[subject_id].get_info()
    
    def list_registered_subjects(self):
        """Print all registered subjects"""
        if len(self.subject_profiles) == 0:
            print("No registered subjects")
            return
        
        print("\nRegistered Subjects:")
        print("-" * 80)
        for subject_id, profile in self.subject_profiles.items():
            n_trials = len(list(profile.historical_features.values())[0])
            print(f"  {subject_id}: {n_trials} trials")
    
    def export_subject_profile(self, subject_id: Union[int, str], output_file: str):
        """
        Export subject profile to JSON file
        
        Parameters:
            subject_id: Subject ID
            output_file: Output file path
        """
        subject_id = str(subject_id)
        
        if subject_id not in self.subject_profiles:
            raise ValueError(f"Subject {subject_id} not registered")
        
        profile = self.subject_profiles[subject_id]
        
        # Prepare exportable data
        export_data = {
            'subject_id': subject_id,
            'feature_means': profile.feature_means,
            'feature_stds': profile.feature_stds,
            'historical_features': {
                k: v.tolist() for k, v in profile.historical_features.items()
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Subject profile exported: {output_file}")
    
    def import_subject_profile(self, profile_file: str) -> SubjectProfile:
        """
        Import subject profile from JSON file
        
        Parameters:
            profile_file: Profile file path
        
        Returns:
            Imported subject profile
        """
        with open(profile_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruct subject profile
        historical_features = {
            k: np.array(v) for k, v in data['historical_features'].items()
        }
        
        profile = SubjectProfile(
            subject_id=data['subject_id'],
            historical_features=historical_features
        )
        
        # Manually set standardization parameters
        profile.feature_means = data['feature_means']
        profile.feature_stds = data['feature_stds']
        
        # Save profile
        self.subject_profiles[str(data['subject_id'])] = profile
        
        print(f"\n✓ Subject profile imported: {data['subject_id']}")
        
        return profile


def interactive_mode():
    """Command-line interactive mode"""
    print("\n" + "="*80)
    print(" "*25 + "LME Rating Predictor")
    print(" "*28 + "Interactive Mode")
    print("="*80)
    
    # Load model
    model_file = input("\nModel parameter file path (default: lme_model_params.json): ").strip()
    if not model_file:
        model_file = 'lme_model_params.json'
    
    if not os.path.exists(model_file):
        print(f"❌ File does not exist: {model_file}")
        return
    
    predictor = LMERatingPredictor(model_file)
    
    # Main loop
    while True:
        print("\n" + "="*80)
        print("Main Menu:")
        print("="*80)
        print("1. Register new subject (manual input feature history)")
        print("2. Predict single trial")
        print("3. Batch register subjects (from CSV folder)")
        print("4. Batch predict multiple trials for multiple subjects")
        print("5. Exit")
        print("-"*80)
        
        choice = input("Select operation: ").strip()
        
        if choice == '1':
            # Register new subject
            print("\n" + "-"*80)
            print("Register New Subject")
            print("-"*80)
            
            subject_id = input("\nSubject ID (leave blank for auto-assignment): ").strip()
            if not subject_id:
                subject_id = None
            
            # Input method
            print("\nFeature history input method:")
            print("1. CSV file")
            print("2. Manual input")
            
            input_method = input("Select method: ").strip()
            
            if input_method == '1':
                # Read from CSV
                csv_file = input("CSV file path: ").strip()
                
                if not os.path.exists(csv_file):
                    print(f"❌ File does not exist: {csv_file}")
                    continue
                
                try:
                    historical_data = pd.read_csv(csv_file)
                    print(f"\n✓ Read {len(historical_data)} trials")
                    
                    # Ask about standardization method
                    print("\nStandardization method:")
                    print("1. Within-subject (default, recommended)")
                    print("2. Global statistics")
                    std_method = input("Select: ").strip()
                    use_global = (std_method == '2')
                    
                    profile = predictor.register_subject(
                        subject_id,
                        historical_data,
                        use_global_stats=use_global
                    )
                    
                    print("\n✓ Registration successful!")
                    print(profile.get_info())
                
                except Exception as e:
                    print(f"\n❌ Error: {e}")
            
            elif input_method == '2':
                # Manual input
                print("\nPlease enter feature values for each trial (comma-separated)")
                print(f"Required features: {', '.join(predictor.feature_names)}")
                print("Enter 'done' when finished")
                
                historical_dict = {feat: [] for feat in predictor.feature_names}
                trial_count = 0
                
                while True:
                    trial_count += 1
                    print(f"\nTrial {trial_count}:")
                    
                    complete = True
                    for feat in predictor.feature_names:
                        value_input = input(f"  {feat}: ").strip()
                        
                        if value_input.lower() == 'done':
                            complete = False
                            break
                        
                        try:
                            value = float(value_input)
                            historical_dict[feat].append(value)
                        except ValueError:
                            print(f"  ⚠️  Invalid value, skipping this trial")
                            complete = False
                            break
                    
                    if not complete:
                        # Remove incomplete trial
                        for feat in predictor.feature_names:
                            if len(historical_dict[feat]) > trial_count - 1:
                                historical_dict[feat].pop()
                        break
                
                if trial_count > 1:
                    try:
                        profile = predictor.register_subject(subject_id, historical_dict)
                        print("\n✓ Registration successful!")
                        print(profile.get_info())
                    except Exception as e:
                        print(f"\n❌ Error: {e}")
                else:
                    print("\n⚠️  No valid trials, registration cancelled")
        
        elif choice == '2':
            # Predict single trial
            print("\n" + "-"*80)
            print("Predict Single Trial")
            print("-"*80)
            
            if len(predictor.subject_profiles) == 0:
                print("⚠️  No registered subjects, please register first (Menu 1 or 3)")
                continue
            
            print(f"\nCurrently registered subjects: {list(predictor.subject_profiles.keys())}")
            subject_id = input("Subject ID: ").strip()
            
            if subject_id not in predictor.subject_profiles:
                print(f"⚠️  Subject {subject_id} not registered")
                continue
            
            # Input feature values
            print(f"\nPlease enter feature values for the new trial:")
            features = {}
            
            for feat in predictor.feature_names:
                while True:
                    try:
                        value = float(input(f"  {feat}: ").strip())
                        features[feat] = value
                        break
                    except ValueError:
                        print("  ⚠️  Invalid value, please re-enter")
            
            # Predict
            try:
                result = predictor.predict(
                    subject_id=subject_id,
                    features=features,
                    return_components=True
                )
                
                predictor.print_prediction_summary(result)
            
            except Exception as e:
                print(f"\n❌ Error: {e}")
        
        elif choice == '3':
            # Batch register subjects
            print("\n" + "-"*80)
            print("Batch Register Subjects")
            print("-"*80)
            
            folder = input("\nFolder path containing CSV files: ").strip()
            
            if not os.path.exists(folder):
                print(f"❌ Folder does not exist: {folder}")
                continue
            
            # Find all CSV files
            csv_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
            
            if len(csv_files) == 0:
                print(f"⚠️  No CSV files found in folder: {folder}")
                continue
            
            print(f"\nFound {len(csv_files)} CSV files:")
            for i, f in enumerate(csv_files, 1):
                print(f"  {i}. {os.path.basename(f)}")
            
            # Ask about subject ID assignment
            print("\nSubject ID assignment method:")
            print("1. Auto-assign")
            print("2. Manual input for each file")
            
            id_method = input("Select: ").strip()
            
            # Register each file
            success_count = 0
            
            if id_method == '1':
                for csv_file in csv_files:
                    try:
                        historical_data = pd.read_csv(csv_file)
                        profile = predictor.register_subject(None, historical_data)
                        success_count += 1
                        print(f"✓ Registered: {profile.subject_id} (from {os.path.basename(csv_file)})")
                    
                    except Exception as e:
                        print(f"✗ Failed: {os.path.basename(csv_file)} - {e}")
            
            elif id_method == '2':
                for csv_file in csv_files:
                    subject_id = input(f"  Subject ID for {os.path.basename(csv_file)} (leave blank for auto-assignment): ").strip()
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
            
            # Show all registered subjects
            print("\nCurrently registered subjects:")
            predictor.list_registered_subjects()
        
        elif choice == '4':
            # Batch predict multiple trials for multiple subjects
            print("\n" + "-"*80)
            print("Batch Predict Multiple Trials for Multiple Subjects")
            print("-"*80)
            
            if len(predictor.subject_profiles) == 0:
                print("⚠️  No registered subjects, please register first (Menu 1 or 3)")
                continue
            
            print(f"\nCurrently registered subjects: {list(predictor.subject_profiles.keys())}")
            
            # Method 1: Batch prediction for single subject
            mode = input("\nPrediction mode (1=Batch predict single subject, 2=Predict multiple subjects separately): ").strip()
            
            if mode == '1':
                # Batch prediction for single subject
                subject_id = input("Subject ID: ").strip()
                
                if subject_id not in predictor.subject_profiles:
                    print(f"⚠️  Subject {subject_id} not registered")
                    continue
                
                input_csv = input("Input CSV file path (containing feature columns): ").strip()
                
                if not os.path.exists(input_csv):
                    print(f"❌ File does not exist: {input_csv}")
                    continue
                
                output_csv = input(f"Output CSV file path (default: predictions_{subject_id}.csv): ").strip()
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
                # Predict multiple subjects separately
                print("\nEnter subject IDs to predict (comma-separated, or enter 'all' to predict all):")
                subject_input = input("Subject IDs: ").strip()
                
                if subject_input.lower() == 'all':
                    subject_ids = list(predictor.subject_profiles.keys())
                else:
                    subject_ids = [s.strip() for s in subject_input.split(',')]
                
                # Ask about data file method
                data_mode = input("\nData file method (1=Unified file, 2=Specify separately): ").strip()
                
                if data_mode == '1':
                    # All subjects use the same feature file
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
                        
                        input_csv = input(f"\nInput CSV file for {subject_id}: ").strip()
                        
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
            print("\nInvalid selection, please try again")


def demo():
    """Demonstration of basic predictor usage"""
    print("\n" + "="*80)
    print(" "*25 + "Usage Demonstration")
    print("="*80)
    
    # 1. Load predictor (rating statistics already hard-coded)
    print("\nStep 1: Load predictor")
    print("-" * 80)
    predictor = LMERatingPredictor('lme_model_params.json')
    
    # 2. Prepare feature history data
    print("\nStep 2: Prepare subject's feature history data")
    print("-" * 80)
    
    np.random.seed(42)
    feature_history = pd.DataFrame({
        'erp_N2': np.random.normal(-17, 3, 20),
        'erp_P2': np.random.normal(13.5, 2, 20),
        'erp_N1': np.random.normal(-9, 1.5, 20),
        'latency_N2': np.random.normal(200, 15, 20),
        'latency_P2': np.random.normal(390, 20, 20),
        'latency_N1': np.random.normal(152, 10, 20),
        'TF_gamma': np.random.normal(2.6, 0.5, 20),
        'TF_beta': np.random.normal(1.7, 0.3, 20),
        'TF_LEP': np.random.normal(31, 4, 20),
        'TF_alpha': np.random.normal(79, 8, 20)
    })
    
    print(f"Feature history data: {len(feature_history)} trials")
    
    # 3. Register subject (automatically assign ID)
    print("\nStep 3: Register new subject (automatically assign ID)")
    print("-" * 80)
    profile = predictor.register_subject(subject_id=None, historical_features=feature_history)
    assigned_id = profile.subject_id
    
    # 4. Predict single trial
    print("\nStep 4: Predict new trial")
    print("-" * 80)
    
    new_features = {
        'erp_N2': -17.5,
        'erp_P2': 13.8,
        'erp_N1': -9.1,
        'latency_N2': 200,
        'latency_P2': 392,
        'latency_N1': 152,
        'TF_gamma': 2.7,
        'TF_beta': 1.7,
        'TF_LEP': 31.2,
        'TF_alpha': 79
    }
    
    result = predictor.predict(subject_id=assigned_id, features=new_features)
    
    # Print results
    predictor.print_prediction_summary(result)
    
    print("\n" + "="*80)
    print("✓ Demonstration complete")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'demo':
            demo()
        elif sys.argv[1] == 'interactive' or sys.argv[1] == '-i':
            interactive_mode()
        else:
            print("Usage:")
            print("  python lme_rating_predictor_v5_final.py demo         # Run demonstration")
            print("  python lme_rating_predictor_v5_final.py interactive # Interactive mode")
            print("  python lme_rating_predictor_v5_final.py -i          # Interactive mode (shorthand)")
    else:
        # Default to interactive mode
        interactive_mode()
