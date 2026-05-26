import numpy as np
import pandas as pd
import json
import os
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


class BehavioralCaptchaClassifier:
    """
    Behavioral CAPTCHA classifier using One-Class SVM and Isolation Forest
    
    IMPORTANT: This is ONE-CLASS LEARNING
    - All training data is HUMAN (positive class)
    - No outlier removal during training (all variance is legitimate human diversity)
    - Models learn "what humans look like" including their natural variance
    - Goal: Reject only BOTS (which will have very different patterns)
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.ocsvm_model = None
        self.isolation_forest_model = None
        self.feature_names = []
        
    def load_session_data(self, data_directory='captcha_data'):
        """Load all session data from JSON files"""
        sessions = []
        
        if not os.path.exists(data_directory):
            print(f"Directory '{data_directory}' not found!")
            return sessions
        
        for filename in os.listdir(data_directory):
            if filename.endswith('.json'):
                filepath = os.path.join(data_directory, filename)
                with open(filepath, 'r') as f:
                    session = json.load(f)
                    sessions.append(session)
        
        print(f"Loaded {len(sessions)} sessions from {data_directory}")
        return sessions
    
    def clean_invalid_sessions(self, sessions):
        """
        Remove ONLY clearly invalid/corrupted sessions, NOT behavioral outliers
        
        Removes:
        - Sessions with missing/corrupted data
        - Sessions < 0.3s (accidental double-clicks)
        - Sessions with < 5 movements (no real interaction)
        
        KEEPS:
        - Slow sessions (elderly, careful users)
        - Fast sessions (power users)
        - All natural human variance
        """
        print("\n" + "="*70)
        print("FILTERING INVALID SESSIONS (NOT removing human variance!)")
        print("="*70)
        
        valid_sessions = []
        invalid_count = {'corrupted': 0, 'too_fast': 0, 'no_interaction': 0}
        
        for session in sessions:
            total_time = session.get('total_time', 0)
            num_movements = len(session.get('mouse_movements', []))
            
            # Only remove clearly invalid data
            if total_time < 0.3:  # Accidental click
                invalid_count['too_fast'] += 1
            elif num_movements < 5:  # No real mouse interaction
                invalid_count['no_interaction'] += 1
            elif total_time <= 0 or num_movements == 0:  # Corrupted
                invalid_count['corrupted'] += 1
            else:
                # KEEP everything else - it's all valid human behavior!
                valid_sessions.append(session)
        
        total_invalid = sum(invalid_count.values())
        print(f"\n✅ Valid human sessions: {len(valid_sessions)}")
        print(f"❌ Invalid/corrupted sessions removed: {total_invalid}")
        
        for reason, count in invalid_count.items():
            if count > 0:
                print(f"   - {reason}: {count}")
        
        if len(valid_sessions) != len(sessions):
            print(f"\n💡 Kept {len(valid_sessions)}/{len(sessions)} sessions")
            print("   All variance in valid sessions is LEGITIMATE human diversity!")
        
        return valid_sessions
    
    def analyze_human_diversity(self, features_df):
        """Analyze the natural diversity in human behavior"""
        print("\n" + "="*70)
        print("HUMAN BEHAVIORAL DIVERSITY ANALYSIS")
        print("="*70)
        
        print("\nThis shows the NATURAL variance in human behavior.")
        print("High variance = diverse users (good!), not 'outliers'\n")
        
        key_metrics = ['total_time', 'num_movements', 'mean_velocity', 
                      'mean_acceleration', 'directness_ratio']
        
        print(f"{'Metric':<25} {'Min':<10} {'Mean':<10} {'Max':<10} {'Std Dev'}")
        print("-"*70)
        
        for metric in key_metrics:
            if metric in features_df.columns:
                min_val = features_df[metric].min()
                mean_val = features_df[metric].mean()
                max_val = features_df[metric].max()
                std_val = features_df[metric].std()
                
                print(f"{metric:<25} {min_val:<10.2f} {mean_val:<10.2f} "
                      f"{max_val:<10.2f} {std_val:<10.2f}")
        
        # Coefficient of Variation (CV) - shows relative variability
        print("\n" + "-"*70)
        print("Coefficient of Variation (higher = more diverse):")
        print("-"*70)
        
        for metric in key_metrics:
            if metric in features_df.columns:
                cv = features_df[metric].std() / features_df[metric].mean() if features_df[metric].mean() != 0 else 0
                diversity_level = "Low" if cv < 0.3 else "Medium" if cv < 0.7 else "High"
                print(f"{metric:<25} CV={cv:.2f} ({diversity_level} diversity)")
        
        print("\n💡 High diversity is GOOD - it means you captured various user types!")
    
    def extract_features(self, session_data):
        """Extract behavioral features from session data"""
        features = {}
        
        # Time features
        features['total_time'] = session_data.get('total_time', 0)
        features['time_to_click'] = session_data.get('click_data', {}).get('time_to_click', 0)
        features['num_movements'] = len(session_data.get('mouse_movements', []))
        features['movement_frequency'] = (features['num_movements'] / features['total_time'] 
                                         if features['total_time'] > 0 else 0)
        
        # Distance features
        features['total_distance'] = session_data.get('distance_traveled', 0)
        features['avg_distance_per_movement'] = (features['total_distance'] / features['num_movements']
                                                 if features['num_movements'] > 0 else 0)
        
        # Velocity features
        velocities = session_data.get('velocities', [])
        if velocities:
            features['mean_velocity'] = np.mean(velocities)
            features['std_velocity'] = np.std(velocities)
            features['max_velocity'] = np.max(velocities)
            features['min_velocity'] = np.min(velocities)
            features['velocity_range'] = features['max_velocity'] - features['min_velocity']
        else:
            features['mean_velocity'] = 0
            features['std_velocity'] = 0
            features['max_velocity'] = 0
            features['min_velocity'] = 0
            features['velocity_range'] = 0
        
        # Acceleration features
        accelerations = session_data.get('accelerations', [])
        if accelerations:
            features['mean_acceleration'] = np.mean(accelerations)
            features['std_acceleration'] = np.std(accelerations)
            features['max_acceleration'] = np.max(accelerations)
            features['min_acceleration'] = np.min(accelerations)
            features['acceleration_range'] = features['max_acceleration'] - features['min_acceleration']
        else:
            features['mean_acceleration'] = 0
            features['std_acceleration'] = 0
            features['max_acceleration'] = 0
            features['min_acceleration'] = 0
            features['acceleration_range'] = 0
        
        # Trajectory features
        movements = session_data.get('mouse_movements', [])
        if len(movements) >= 2:
            start = movements[0]
            end = movements[-1]
            straight_distance = np.sqrt((end['x'] - start['x'])**2 + (end['y'] - start['y'])**2)
            features['directness_ratio'] = (straight_distance / features['total_distance']
                                           if features['total_distance'] > 0 else 0)
            features['avg_curvature'] = self.calculate_curvature(movements)
        else:
            features['directness_ratio'] = 0
            features['avg_curvature'] = 0
        
        # Pause detection
        timestamps = session_data.get('timestamps', [])
        if len(timestamps) >= 2:
            time_diffs = np.diff(timestamps)
            features['num_pauses'] = np.sum(time_diffs > 0.5)
            features['avg_pause_duration'] = np.mean(time_diffs[time_diffs > 0.5]) if np.any(time_diffs > 0.5) else 0
        else:
            features['num_pauses'] = 0
            features['avg_pause_duration'] = 0
        
        return features
    
    def calculate_curvature(self, movements):
        """Calculate average curvature of the mouse trajectory"""
        if len(movements) < 3:
            return 0
        
        curvatures = []
        for i in range(1, len(movements) - 1):
            p1 = np.array([movements[i-1]['x'], movements[i-1]['y']])
            p2 = np.array([movements[i]['x'], movements[i]['y']])
            p3 = np.array([movements[i+1]['x'], movements[i+1]['y']])
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                curvatures.append(angle)
        
        return np.mean(curvatures) if curvatures else 0
    
    def prepare_dataset(self, sessions):
        """Convert sessions to feature matrix"""
        features_list = []
        
        for session in sessions:
            features = self.extract_features(session)
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        self.feature_names = df.columns.tolist()
        
        print(f"\nExtracted {len(self.feature_names)} features from {len(sessions)} sessions")
        print("\nFeature Statistics:")
        print(df.describe())
        
        return df
    
    def train_with_cross_validation(self, features_df, sessions):
        """
        Train models with appropriate parameters for one-class learning
        
        Goal: Learn the FULL range of human behavior, reject only clear bots
        """
        print("\n" + "="*70)
        print("TRAINING ONE-CLASS MODELS")
        print("="*70)
        print("\nStrategy: Models will learn ALL human variance as 'normal'")
        print("They should only reject behavior that's clearly non-human\n")
        
        # Use RobustScaler (handles diverse data better)
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(features_df)
        
        # Train One-Class SVM with VERY LOW nu (expect very few anomalies)
        print("-"*70)
        print("One-Class SVM Training")
        print("-"*70)
        print("\nTesting different nu values (nu = expected anomaly rate):")
        
        nu_values = [0.001, 0.005, 0.01, 0.02, 0.03]
        
        for nu in nu_values:
            temp_model = OneClassSVM(nu=nu, kernel='rbf', gamma='auto')
            temp_model.fit(X_scaled)
            
            predictions = temp_model.predict(X_scaled)
            anomalies = (predictions == -1).sum()
            anomaly_rate = anomalies / len(predictions)
            
            print(f"  nu={nu:<6.3f} → Flagged {anomalies}/{len(predictions)} "
                  f"({anomaly_rate:.1%}) as anomalies")
        
        # Use VERY low nu (we expect almost no anomalies in pure human data)
        best_nu = 0.01  # Expect only 1% false positives
        print(f"\n✅ Selected nu={best_nu} (strict threshold, low false positives)")
        
        self.ocsvm_model = OneClassSVM(nu=best_nu, kernel='rbf', gamma='auto')
        self.ocsvm_model.fit(X_scaled)
        
        # Train Isolation Forest with VERY LOW contamination
        print("\n" + "-"*70)
        print("Isolation Forest Training")
        print("-"*70)
        print("\nTesting different contamination values:")
        
        contamination_values = [0.001, 0.005, 0.01, 0.02, 0.03]
        
        for contamination in contamination_values:
            temp_model = IsolationForest(
                contamination=contamination,
                n_estimators=100,
                random_state=42
            )
            temp_model.fit(X_scaled)
            
            predictions = temp_model.predict(X_scaled)
            anomalies = (predictions == -1).sum()
            anomaly_rate = anomalies / len(predictions)
            
            print(f"  contamination={contamination:<6.3f} → Flagged {anomalies}/{len(predictions)} "
                  f"({anomaly_rate:.1%}) as anomalies")
        
        # Use VERY low contamination
        best_contamination = 0.01  # Expect only 1% false positives
        print(f"\n✅ Selected contamination={best_contamination} (strict threshold)")
        
        self.isolation_forest_model = IsolationForest(
            contamination=best_contamination,
            n_estimators=100,
            random_state=42
        )
        self.isolation_forest_model.fit(X_scaled)
        
        # Final evaluation on training data
        print("\n" + "="*70)
        print("FINAL MODEL EVALUATION ON TRAINING DATA")
        print("="*70)
        
        ocsvm_pred = self.ocsvm_model.predict(X_scaled)
        ocsvm_anomalies = (ocsvm_pred == -1).sum()
        
        if_pred = self.isolation_forest_model.predict(X_scaled)
        if_anomalies = (if_pred == -1).sum()
        
        print(f"\nOne-Class SVM: {ocsvm_anomalies}/{len(features_df)} "
              f"({ocsvm_anomalies/len(features_df):.1%}) flagged as anomalies")
        print(f"Isolation Forest: {if_anomalies}/{len(features_df)} "
              f"({if_anomalies/len(features_df):.1%}) flagged as anomalies")
        
        if ocsvm_anomalies > len(features_df) * 0.05:
            print("\n⚠️  WARNING: One-Class SVM is flagging >5% of training data!")
            print("   This means high false positive rate on real humans.")
        else:
            print("\n✅ Good! Low false positive rate on training data.")
        
        if if_anomalies > len(features_df) * 0.05:
            print("⚠️  WARNING: Isolation Forest is flagging >5% of training data!")
            print("   This means high false positive rate on real humans.")
        else:
            print("✅ Good! Low false positive rate on training data.")
        
        # Store sessions for testing
        self.training_sessions = sessions
        
        return X_scaled
    
    def predict(self, session_data, model_type='both'):
        """Predict if a session is human or bot"""
        features = self.extract_features(session_data)
        X = pd.DataFrame([features])[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        results = {}
        
        if model_type in ['ocsvm', 'both'] and self.ocsvm_model:
            ocsvm_pred = self.ocsvm_model.predict(X_scaled)[0]
            ocsvm_score = self.ocsvm_model.score_samples(X_scaled)[0]
            results['ocsvm'] = {
                'prediction': 'Human' if ocsvm_pred == 1 else 'Bot/Anomaly',
                'score': ocsvm_score,
                'confidence': abs(ocsvm_score)
            }
        
        if model_type in ['isolation_forest', 'both'] and self.isolation_forest_model:
            if_pred = self.isolation_forest_model.predict(X_scaled)[0]
            if_score = self.isolation_forest_model.score_samples(X_scaled)[0]
            results['isolation_forest'] = {
                'prediction': 'Human' if if_pred == 1 else 'Bot/Anomaly',
                'score': if_score,
                'confidence': abs(if_score)
            }
        
        return results
    
    def save_models(self, directory='models'):
        """Save trained models"""
        os.makedirs(directory, exist_ok=True)
        
        if self.ocsvm_model:
            joblib.dump(self.ocsvm_model, f'{directory}/ocsvm_model.pkl')
            print(f"\nOne-Class SVM model saved to {directory}/ocsvm_model.pkl")
        
        if self.isolation_forest_model:
            joblib.dump(self.isolation_forest_model, f'{directory}/isolation_forest_model.pkl')
            print(f"Isolation Forest model saved to {directory}/isolation_forest_model.pkl")
        
        joblib.dump(self.scaler, f'{directory}/scaler.pkl')
        joblib.dump(self.feature_names, f'{directory}/feature_names.pkl')
        print(f"Scaler and feature names saved to {directory}/")
    
    def load_models(self, directory='models'):
        """Load trained models"""
        self.ocsvm_model = joblib.load(f'{directory}/ocsvm_model.pkl')
        self.isolation_forest_model = joblib.load(f'{directory}/isolation_forest_model.pkl')
        self.scaler = joblib.load(f'{directory}/scaler.pkl')
        self.feature_names = joblib.load(f'{directory}/feature_names.pkl')
        print("Models loaded successfully!")
    
    def create_diagnostic_plots(self, features_df):
        """Create diagnostic visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Total time distribution
        axes[0, 0].hist(features_df['total_time'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(features_df['total_time'].mean(), color='red', 
                          linestyle='--', label=f'Mean: {features_df["total_time"].mean():.2f}s')
        axes[0, 0].set_xlabel('Total Time (seconds)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Human Session Duration (Natural Variance)')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Velocity distribution
        axes[0, 1].hist(features_df['mean_velocity'], bins=50, edgecolor='black', 
                       alpha=0.7, color='green')
        axes[0, 1].axvline(features_df['mean_velocity'].mean(), color='red',
                          linestyle='--', label=f'Mean: {features_df["mean_velocity"].mean():.0f}')
        axes[0, 1].set_xlabel('Mean Velocity (px/s)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Human Velocity Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Directness ratio
        axes[0, 2].hist(features_df['directness_ratio'], bins=50, 
                       edgecolor='black', alpha=0.7, color='orange')
        axes[0, 2].axvline(features_df['directness_ratio'].mean(), color='red',
                          linestyle='--', label=f'Mean: {features_df["directness_ratio"].mean():.2f}')
        axes[0, 2].set_xlabel('Directness Ratio (0=curved, 1=straight)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Human Trajectory Patterns')
        axes[0, 2].legend()
        axes[0, 2].grid(alpha=0.3)
        
        # 4. Scatter: time vs movements
        axes[1, 0].scatter(features_df['total_time'], features_df['num_movements'], 
                          alpha=0.5, s=30)
        axes[1, 0].set_xlabel('Total Time (seconds)')
        axes[1, 0].set_ylabel('Number of Movements')
        axes[1, 0].set_title('Time vs Movement Count')
        axes[1, 0].grid(alpha=0.3)
        
        # 5. Scatter: velocity vs acceleration
        axes[1, 1].scatter(features_df['mean_velocity'], 
                          features_df['mean_acceleration'], alpha=0.5, s=30, color='purple')
        axes[1, 1].set_xlabel('Mean Velocity (px/s)')
        axes[1, 1].set_ylabel('Mean Acceleration (px/s²)')
        axes[1, 1].set_title('Velocity vs Acceleration (Human Patterns)')
        axes[1, 1].grid(alpha=0.3)
        
        # 6. Box plots
        key_features = ['total_time', 'mean_velocity', 'mean_acceleration', 
                       'directness_ratio', 'movement_frequency']
        
        # Normalize for visualization
        normalized = features_df[key_features].copy()
        for col in normalized.columns:
            normalized[col] = (normalized[col] - normalized[col].mean()) / normalized[col].std()
        
        normalized.boxplot(ax=axes[1, 2])
        axes[1, 2].set_title('Feature Distributions (Standardized)')
        axes[1, 2].set_ylabel('Standard Deviations from Mean')
        axes[1, 2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.setp(axes[1, 2].xaxis.get_majorticklabels(), rotation=45, ha='right')
        axes[1, 2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('human_behavior_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Behavioral analysis plots saved to 'human_behavior_analysis.png'")
        plt.show()
    
    def test_on_training_samples(self, n_samples=10):
        """Test predictions on random training samples"""
        print("\n" + "="*70)
        print(f"TESTING ON {n_samples} RANDOM TRAINING SAMPLES")
        print("="*70)
        print("These are all HUMAN sessions - models should accept them!")
        
        import random
        if not hasattr(self, 'training_sessions'):
            print("No training sessions stored!")
            return
        
        test_samples = random.sample(self.training_sessions, 
                                    min(n_samples, len(self.training_sessions)))
        
        ocsvm_rejects = 0
        if_rejects = 0
        
        for i, session in enumerate(test_samples, 1):
            results = self.predict(session, model_type='both')
            
            print(f"\n{'─'*70}")
            print(f"Sample {i}: Duration={session.get('total_time', 0):.2f}s, "
                  f"Movements={len(session.get('mouse_movements', []))}")
            
            ocsvm_result = results['ocsvm']['prediction']
            if_result = results['isolation_forest']['prediction']
            
            print(f"  One-Class SVM:    {ocsvm_result:<15} (score: {results['ocsvm']['score']:>7.4f})")
            print(f"  Isolation Forest: {if_result:<15} (score: {results['isolation_forest']['score']:>7.4f})")
            
            if ocsvm_result == 'Bot/Anomaly':
                ocsvm_rejects += 1
                print("  ⚠️  One-Class SVM rejected this HUMAN!")
            
            if if_result == 'Bot/Anomaly':
                if_rejects += 1
                print("  ⚠️  Isolation Forest rejected this HUMAN!")
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"One-Class SVM rejected: {ocsvm_rejects}/{n_samples} humans "
              f"({ocsvm_rejects/n_samples:.1%} false positive rate)")
        print(f"Isolation Forest rejected: {if_rejects}/{n_samples} humans "
              f"({if_rejects/n_samples:.1%} false positive rate)")
        
        if ocsvm_rejects > n_samples * 0.05:
            print("\n⚠️  One-Class SVM has HIGH false positive rate!")
        if if_rejects > n_samples * 0.05:
            print("⚠️  Isolation Forest has HIGH false positive rate!")


# Main execution
if __name__ == "__main__":
    print("\n" + "="*70)
    print("BEHAVIORAL CAPTCHA - ONE-CLASS LEARNING")
    print("="*70)
    print("\nTraining Strategy:")
    print("- ALL data is human (legitimate users)")
    print("- NO outlier removal (all variance is natural human diversity)")
    print("- Models learn to accept WIDE range of human behavior")
    print("- Only reject patterns that are clearly non-human (bots)")
    
    # Initialize classifier
    classifier = BehavioralCaptchaClassifier()
    
    # Load data
    sessions = classifier.load_session_data('captcha_data')
    
    if len(sessions) < 50:
        print(f"\n❌ ERROR: Not enough data! You have {len(sessions)} sessions.")
        print("Collect at least 50 sessions before training.")
    else:
        # Clean only invalid/corrupted sessions
        clean_sessions = classifier.clean_invalid_sessions(sessions)
        
        # Extract features
        features_df = classifier.prepare_dataset(clean_sessions)
        
        # Analyze human diversity
        classifier.analyze_human_diversity(features_df)
        
        # Train models with appropriate parameters
        classifier.train_with_cross_validation(features_df, clean_sessions)
        
        # Create diagnostic plots
        classifier.create_diagnostic_plots(features_df)
        
        # Save models
        classifier.save_models('models')
        
        # Test on training samples (should mostly accept them!)
        classifier.test_on_training_samples(n_samples=10)
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print("\nNext steps:")
        print("1. Review 'human_behavior_analysis.png' to see human diversity")
        print("2. Test with simulated bot data to verify detection")
        print("3. If false positive rate is still high, decrease nu/contamination further")