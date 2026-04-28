import numpy as np
import pandas as pd
import json
import os
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.stats import entropy


class FeatureSelectionTraining:
    """
    Train models with different feature sets using train/test split
    
    NEW FEATURES:
    - Velocity Entropy: Measures randomness in velocity (high = human)
    - Direction Changes: Number of trajectory direction changes
    - Jitter: Micro-variations in movement (tremor, natural shakiness)
    
    REMOVED:
    - mean_velocity (replaced with velocity entropy)
    - mean_acceleration (less discriminative)
    """
    
    def __init__(self):
        self.all_sessions = []
        self.train_sessions = []
        self.test_sessions = []
        
        # Define feature groups with NEW features
        self.feature_groups = {
            'temporal': [
                'total_time',
                'time_to_click', 
                'movement_frequency'
            ],
            'kinematic': [
                'std_velocity',
                'velocity_entropy',      # NEW: Randomness in speed
                'std_acceleration',
                'jitter'                 # NEW: Micro-variations
            ],
            'geometric': [
                'total_distance',
                'directness_ratio',
                'avg_curvature',
                'direction_changes',     # NEW: Path complexity
                'avg_distance_per_movement'
            ],
            'behavioral': [
                'num_movements',
                'num_pauses',
                'velocity_range',
                'acceleration_range'
            ]
        }
        
        # Updated core features (8 features - most discriminative)
        self.core_features = [
            'std_velocity',         # Bots: low variance | Humans: high variance
            'velocity_entropy',     # NEW: Bots: low entropy | Humans: high entropy
            'directness_ratio',     # Bots: straight lines | Humans: curves
            'avg_curvature',        # Bots: low curvature | Humans: natural curves
            'total_time',           # Bots: too fast/consistent | Humans: variable
            'direction_changes',    # NEW: Bots: few changes | Humans: many changes
            'jitter',              # NEW: Bots: none/low | Humans: natural tremor
            'num_pauses'           # Bots: no pauses | Humans: hesitations
        ]
        
    def load_data(self, data_directory='captcha_data', test_size=0.2):
        """Load and split data into train/test sets (80/20)"""
        print("="*70)
        print("LOADING AND SPLITTING DATA")
        print("="*70)
        
        sessions = []
        for filename in os.listdir(data_directory):
            if filename.endswith('.json'):
                filepath = os.path.join(data_directory, filename)
                with open(filepath, 'r') as f:
                    sessions.append(json.load(f))
        
        # Clean invalid sessions only
        valid_sessions = []
        for session in sessions:
            if (session.get('total_time', 0) >= 0.3 and 
                len(session.get('mouse_movements', [])) >= 5):
                valid_sessions.append(session)
        
        # Split into train/test (80/20)
        self.train_sessions, self.test_sessions = train_test_split(
            valid_sessions, 
            test_size=test_size, 
            random_state=42
        )
        
        self.all_sessions = valid_sessions
        
        print(f"\nTotal valid sessions: {len(valid_sessions)}")
        print(f"Training set: {len(self.train_sessions)} ({(1-test_size)*100:.0f}%)")
        print(f"Test set: {len(self.test_sessions)} ({test_size*100:.0f}%)")
        
        return self.train_sessions, self.test_sessions
    
    def calculate_velocity_entropy(self, velocities):
        """
        Calculate entropy of velocity distribution
        High entropy = more random/chaotic (human)
        Low entropy = more uniform (bot)
        """
        if len(velocities) < 2:
            return 0
        
        # Create histogram bins
        hist, _ = np.histogram(velocities, bins=10)
        
        # Normalize to probability distribution
        hist = hist / np.sum(hist)
        
        # Remove zeros to avoid log(0)
        hist = hist[hist > 0]
        
        # Calculate Shannon entropy
        return entropy(hist)
    
    def calculate_direction_changes(self, movements):
        """
        Count significant direction changes in trajectory
        Bots: few changes (straight paths)
        Humans: many changes (exploration, correction)
        """
        if len(movements) < 3:
            return 0
        
        direction_changes = 0
        threshold = np.pi / 4  # 45 degrees
        
        for i in range(1, len(movements) - 1):
            p1 = np.array([movements[i-1]['x'], movements[i-1]['y']])
            p2 = np.array([movements[i]['x'], movements[i]['y']])
            p3 = np.array([movements[i+1]['x'], movements[i+1]['y']])
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Calculate angle between vectors
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                
                # Count if significant direction change
                if angle > threshold:
                    direction_changes += 1
        
        return direction_changes
    
    def calculate_jitter(self, movements):
        """
        Calculate micro-variations (jitter) in movement
        Measures natural hand tremor and small corrections
        
        Bots: very low jitter (smooth, calculated paths)
        Humans: measurable jitter (natural shakiness)
        """
        if len(movements) < 3:
            return 0
        
        # Calculate second derivatives (acceleration of position)
        jitters = []
        
        for i in range(1, len(movements) - 1):
            # Position vectors
            p_prev = np.array([movements[i-1]['x'], movements[i-1]['y']])
            p_curr = np.array([movements[i]['x'], movements[i]['y']])
            p_next = np.array([movements[i+1]['x'], movements[i+1]['y']])
            
            # First derivatives (velocity)
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            
            # Second derivative (acceleration/jitter)
            accel = v2 - v1
            jitter_magnitude = np.linalg.norm(accel)
            
            jitters.append(jitter_magnitude)
        
        # Return average jitter magnitude
        return np.mean(jitters) if jitters else 0
    
    def extract_features(self, session_data):
        """Extract all features including NEW ones"""
        features = {}
        
        # Temporal features
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
            features['std_velocity'] = np.std(velocities)
            features['max_velocity'] = np.max(velocities)
            features['min_velocity'] = np.min(velocities)
            features['velocity_range'] = features['max_velocity'] - features['min_velocity']
            features['velocity_entropy'] = self.calculate_velocity_entropy(velocities)  # NEW
        else:
            features['std_velocity'] = 0
            features['max_velocity'] = 0
            features['min_velocity'] = 0
            features['velocity_range'] = 0
            features['velocity_entropy'] = 0  # NEW
        
        # Acceleration features
        accelerations = session_data.get('accelerations', [])
        if accelerations:
            features['std_acceleration'] = np.std(accelerations)
            features['max_acceleration'] = np.max(accelerations)
            features['min_acceleration'] = np.min(accelerations)
            features['acceleration_range'] = features['max_acceleration'] - features['min_acceleration']
        else:
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
            features['direction_changes'] = self.calculate_direction_changes(movements)  # NEW
            features['jitter'] = self.calculate_jitter(movements)  # NEW
        else:
            features['directness_ratio'] = 0
            features['avg_curvature'] = 0
            features['direction_changes'] = 0  # NEW
            features['jitter'] = 0  # NEW
        
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
        """Calculate trajectory curvature"""
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
    
    def prepare_feature_sets(self, sessions):
        """Create different feature configurations from sessions"""
        print("\n" + "="*70)
        print("PREPARING FEATURE SETS")
        print("="*70)
        
        # Extract all features
        all_features_list = []
        for session in sessions:
            features = self.extract_features(session)
            all_features_list.append(features)
        
        all_features_df = pd.DataFrame(all_features_list)
        
        # Configuration 1: Core 8 features (UPDATED - most important)
        print("\n1. CORE FEATURES (8 features) - Recommended for production")
        print("   Most discriminative between humans and bots:")
        for i, feat in enumerate(self.core_features, 1):
            print(f"   {i}. {feat}")
        
        # Configuration 2: Kinematic + Geometric (12 features)
        combo_features = (self.feature_groups['kinematic'] + 
                         self.feature_groups['geometric'])
        print(f"\n2. KINEMATIC + GEOMETRIC ({len(combo_features)} features)")
        print("   Physical movement patterns:")
        for feat in combo_features:
            print(f"   - {feat}")
        
        # Configuration 3: All features (22 features - increased from 20)
        print(f"\n3. ALL FEATURES ({len(all_features_df.columns)} features)")
        print("   Complete feature set")
        
        return {
            'core_8': all_features_df[self.core_features],
            'kinematic_geometric_12': all_features_df[combo_features],
            'all_features': all_features_df
        }
    
    def train_and_evaluate(self, train_sessions, test_sessions):
        """Train models with different feature configurations and evaluate on test set"""
        print("\n" + "="*70)
        print("TRAINING AND EVALUATION WITH TRAIN/TEST SPLIT")
        print("="*70)
        
        # Prepare feature sets for train and test
        train_feature_sets = self.prepare_feature_sets(train_sessions)
        test_feature_sets = self.prepare_feature_sets(test_sessions)
        
        results = {}
        
        for config_name in train_feature_sets.keys():
            print(f"\n{'='*70}")
            print(f"CONFIGURATION: {config_name.upper()}")
            print(f"Features: {len(train_feature_sets[config_name].columns)}")
            print(f"{'='*70}")
            
            # Get train and test data
            X_train = train_feature_sets[config_name]
            X_test = test_feature_sets[config_name]
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train One-Class SVM (UPDATED PARAMETERS)
            print("\n--- One-Class SVM (nu=0.05, gamma='scale') ---")
            ocsvm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
            ocsvm.fit(X_train_scaled)
            
            # Evaluate on training set
            ocsvm_train_pred = ocsvm.predict(X_train_scaled)
            ocsvm_train_fp = (ocsvm_train_pred == -1).sum()
            ocsvm_train_fp_rate = ocsvm_train_fp / len(X_train)
            
            # Evaluate on test set
            ocsvm_test_pred = ocsvm.predict(X_test_scaled)
            ocsvm_test_fp = (ocsvm_test_pred == -1).sum()
            ocsvm_test_fp_rate = ocsvm_test_fp / len(X_test)
            
            print(f"Training FP: {ocsvm_train_fp}/{len(X_train)} ({ocsvm_train_fp_rate:.1%})")
            print(f"Test FP:     {ocsvm_test_fp}/{len(X_test)} ({ocsvm_test_fp_rate:.1%})")
            
            # Train Isolation Forest (UPDATED PARAMETERS)
            print("\n--- Isolation Forest (contamination=0.05) ---")
            iso_forest = IsolationForest(contamination=0.1, n_estimators=100, random_state=42)
            iso_forest.fit(X_train_scaled)
            
            # Evaluate on training set
            if_train_pred = iso_forest.predict(X_train_scaled)
            if_train_fp = (if_train_pred == -1).sum()
            if_train_fp_rate = if_train_fp / len(X_train)
            
            # Evaluate on test set
            if_test_pred = iso_forest.predict(X_test_scaled)
            if_test_fp = (if_test_pred == -1).sum()
            if_test_fp_rate = if_test_fp / len(X_test)
            
            print(f"Training FP: {if_train_fp}/{len(X_train)} ({if_train_fp_rate:.1%})")
            print(f"Test FP:     {if_test_fp}/{len(X_test)} ({if_test_fp_rate:.1%})")
            
            # Store results
            results[config_name] = {
                'features': list(X_train.columns),
                'n_features': len(X_train.columns),
                'scaler': scaler,
                'ocsvm_model': ocsvm,
                'ocsvm_train_fp_rate': ocsvm_train_fp_rate,
                'ocsvm_test_fp_rate': ocsvm_test_fp_rate,
                'if_model': iso_forest,
                'if_train_fp_rate': if_train_fp_rate,
                'if_test_fp_rate': if_test_fp_rate,
                'feature_df': X_train
            }
        
        return results
    
    def visualize_comparison(self, results):
        """Create comparison visualizations"""
        print("\n" + "="*70)
        print("CREATING COMPARISON VISUALIZATIONS")
        print("="*70)
        
        configs = list(results.keys())
        n_features = [results[c]['n_features'] for c in configs]
        
        # Get test FP rates (most important)
        ocsvm_test_fp = [results[c]['ocsvm_test_fp_rate'] * 100 for c in configs]
        if_test_fp = [results[c]['if_test_fp_rate'] * 100 for c in configs]
        
        # Get train FP rates
        ocsvm_train_fp = [results[c]['ocsvm_train_fp_rate'] * 100 for c in configs]
        if_train_fp = [results[c]['if_train_fp_rate'] * 100 for c in configs]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Test Set False Positive Rates (MOST IMPORTANT)
        x = np.arange(len(configs))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, ocsvm_test_fp, width, label='One-Class SVM', color='#ff6b6b')
        axes[0, 0].bar(x + width/2, if_test_fp, width, label='Isolation Forest', color='#4ecdc4')
        
        axes[0, 0].set_xlabel('Feature Configuration', fontweight='bold', fontsize=11)
        axes[0, 0].set_ylabel('False Positive Rate (%)', fontweight='bold', fontsize=11)
        axes[0, 0].set_title('TEST SET Performance (Most Important)', fontweight='bold', fontsize=13)
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([c.replace('_', '\n') for c in configs])
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3, axis='y')
        axes[0, 0].axhline(y=5, color='orange', linestyle='--', alpha=0.5, linewidth=2)
        
        # Plot 2: Training Set False Positive Rates
        axes[0, 1].bar(x - width/2, ocsvm_train_fp, width, label='One-Class SVM', color='#ff6b6b', alpha=0.7)
        axes[0, 1].bar(x + width/2, if_train_fp, width, label='Isolation Forest', color='#4ecdc4', alpha=0.7)
        
        axes[0, 1].set_xlabel('Feature Configuration', fontweight='bold', fontsize=11)
        axes[0, 1].set_ylabel('False Positive Rate (%)', fontweight='bold', fontsize=11)
        axes[0, 1].set_title('TRAINING SET Performance', fontweight='bold', fontsize=13)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([c.replace('_', '\n') for c in configs])
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3, axis='y')
        axes[0, 1].axhline(y=5, color='orange', linestyle='--', alpha=0.5, linewidth=2)
        
        # Plot 3: Generalization (Train vs Test)
        for i, config in enumerate(configs):
            axes[1, 0].plot([0, 1], [ocsvm_train_fp[i], ocsvm_test_fp[i]], 
                          'o-', linewidth=2, markersize=10, label=config, alpha=0.7)
        
        axes[1, 0].set_xticks([0, 1])
        axes[1, 0].set_xticklabels(['Training', 'Test'], fontsize=11)
        axes[1, 0].set_ylabel('False Positive Rate (%)', fontweight='bold', fontsize=11)
        axes[1, 0].set_title('One-Class SVM: Train vs Test', fontweight='bold', fontsize=13)
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Plot 4: Dimensionality effect on test performance
        axes[1, 1].plot(n_features, ocsvm_test_fp, 'o-', linewidth=3, markersize=12, 
                       label='One-Class SVM', color='#ff6b6b')
        axes[1, 1].plot(n_features, if_test_fp, 's-', linewidth=3, markersize=12,
                       label='Isolation Forest', color='#4ecdc4')
        
        axes[1, 1].set_xlabel('Number of Features', fontweight='bold', fontsize=11)
        axes[1, 1].set_ylabel('Test FP Rate (%)', fontweight='bold', fontsize=11)
        axes[1, 1].set_title('Dimensionality Effect on Test Performance', fontweight='bold', fontsize=13)
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].axhline(y=5, color='orange', linestyle='--', alpha=0.5, linewidth=2)
        
        plt.tight_layout()
        plt.savefig('feature_comparison.png', dpi=300, bbox_inches='tight')
        print("📊 Comparison plot saved to 'feature_comparison.png'")
        plt.show()
    
    def save_best_models(self, results, output_dir='models'):
        """Save all model configurations"""
        print("\n" + "="*70)
        print("SAVING MODELS")
        print("="*70)
        
        # Find best configuration based on TEST performance
        best_config = min(results.keys(), 
                         key=lambda k: results[k]['ocsvm_test_fp_rate'] + results[k]['if_test_fp_rate'])
        
        print(f"\nBest configuration (lowest test FP): {best_config}")
        print(f"  One-Class SVM Test FP: {results[best_config]['ocsvm_test_fp_rate']:.1%}")
        print(f"  Isolation Forest Test FP: {results[best_config]['if_test_fp_rate']:.1%}")
        
        # Save all configurations
        for config_name, config_data in results.items():
            config_dir = os.path.join(output_dir, config_name)
            os.makedirs(config_dir, exist_ok=True)
            
            # Save models
            joblib.dump(config_data['ocsvm_model'], 
                       f'{config_dir}/ocsvm_model.pkl')
            joblib.dump(config_data['if_model'], 
                       f'{config_dir}/isolation_forest_model.pkl')
            joblib.dump(config_data['scaler'], 
                       f'{config_dir}/scaler.pkl')
            joblib.dump(config_data['features'], 
                       f'{config_dir}/feature_names.pkl')
            
            print(f"\n✅ Saved {config_name} models to '{config_dir}/'")
        
        # Create detailed summary report
        with open(f'{output_dir}/training_summary.txt', 'w') as f:
            f.write("="*70 + "\n")
            f.write("FEATURE SELECTION TRAINING SUMMARY\n")
            f.write("Updated Parameters: nu=0.05, gamma='scale', contamination=0.05\n")
            f.write("Train/Test Split: 80/20\n")
            f.write("="*70 + "\n\n")
            
            for config_name, config_data in results.items():
                f.write(f"\n{config_name.upper()}\n")
                f.write("-"*70 + "\n")
                f.write(f"Number of features: {config_data['n_features']}\n")
                f.write(f"Features: {', '.join(config_data['features'])}\n\n")
                
                f.write("One-Class SVM:\n")
                f.write(f"  Training FP Rate: {config_data['ocsvm_train_fp_rate']:.2%}\n")
                f.write(f"  Test FP Rate:     {config_data['ocsvm_test_fp_rate']:.2%}\n\n")
                
                f.write("Isolation Forest:\n")
                f.write(f"  Training FP Rate: {config_data['if_train_fp_rate']:.2%}\n")
                f.write(f"  Test FP Rate:     {config_data['if_test_fp_rate']:.2%}\n")
                f.write("\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write(f"RECOMMENDED (Best Test Performance): {best_config}\n")
            f.write("="*70 + "\n")
        
        print(f"\n📄 Summary saved to '{output_dir}/training_summary.txt'")
        
        return best_config
    
    def explain_new_features(self):
        """Explain the NEW features and why they're important"""
        print("\n" + "="*70)
        print("NEW FEATURES EXPLANATION")
        print("="*70)
        
        explanations = {
            'velocity_entropy': {
                'human': 'High entropy - chaotic, unpredictable speed changes',
                'bot': 'Low entropy - uniform, predictable velocities',
                'importance': '⭐⭐⭐⭐⭐',
                'note': 'Replaces mean_velocity with better discrimination'
            },
            'direction_changes': {
                'human': 'Many changes - exploration, correction, natural path',
                'bot': 'Few changes - direct, calculated trajectory',
                'importance': '⭐⭐⭐⭐⭐',
                'note': 'Captures path complexity'
            },
            'jitter': {
                'human': 'Measurable jitter - natural hand tremor, micro-corrections',
                'bot': 'Very low jitter - perfectly smooth, calculated movement',
                'importance': '⭐⭐⭐⭐',
                'note': 'Detects unnatural smoothness in bot paths'
            }
        }
        
        print("\nNEW FEATURES (replacing mean_velocity and mean_acceleration):\n")
        for feat, exp in explanations.items():
            print(f"{feat.upper()} {exp['importance']}")
            print(f"   👤 Human: {exp['human']}")
            print(f"   🤖 Bot:   {exp['bot']}")
            print(f"   💡 Note:  {exp['note']}")
            print()


# Main execution
if __name__ == "__main__":
    print("="*70)
    print("FEATURE SELECTION & MODEL TRAINING")
    print("Updated: nu=0.05, gamma='scale', contamination=0.05")
    print("New Features: velocity_entropy, direction_changes, jitter")
    print("Train/Test Split: 80/20")
    print("="*70)
    
    trainer = FeatureSelectionTraining()
    
    # Load data with train/test split
    train_sessions, test_sessions = trainer.load_data('captcha_data', test_size=0.2)
    
    if len(train_sessions) < 50:
        print(f"\n❌ Need at least 50 training sessions, you have {len(train_sessions)}")
    else:
        # Explain new features
        trainer.explain_new_features()
        
        # Train and evaluate all configurations
        results = trainer.train_and_evaluate(train_sessions, test_sessions)
        
        # Visualize comparison
        trainer.visualize_comparison(results)
        
        # Save best models
        best_config = trainer.save_best_models(results)
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"\nBest configuration: {best_config}")
        print("\nKey improvements:")
        print("  ✓ Train/test split (80/20) for proper evaluation")
        print("  ✓ New features: velocity_entropy, direction_changes, jitter")
        print("  ✓ Updated parameters: nu=0.05, gamma='scale', contamination=0.05")
        print("\nTest performance is what matters for real-world deployment!")