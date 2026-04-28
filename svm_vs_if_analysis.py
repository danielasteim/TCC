import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
import joblib
import json
import os
import math


class ModelComparisonAnalysis:
    """
    Analyze why One-Class SVM and Isolation Forest behave differently
    Updated with new features: velocity_entropy, direction_changes, jitter
    Uses train/test split for proper evaluation
    """
    
    def __init__(self):
        self.train_sessions = []
        self.test_sessions = []
        
    def load_and_prepare(self, data_directory='captcha_data', test_size=0.2):
        """Load data with train/test split and trained models"""
        print("="*70)
        print("LOADING DATA AND MODELS")
        print("="*70)
        
        # Load sessions
        sessions = []
        for filename in os.listdir(data_directory):
            if filename.endswith('.json'):
                filepath = os.path.join(data_directory, filename)
                with open(filepath, 'r') as f:
                    sessions.append(json.load(f))
        
        # Clean invalid sessions
        valid_sessions = []
        for session in sessions:
            if (session.get('total_time', 0) >= 0.3 and 
                len(session.get('mouse_movements', [])) >= 5):
                valid_sessions.append(session)
        
        # Split into train/test (same split as training)
        self.train_sessions, self.test_sessions = train_test_split(
            valid_sessions, 
            test_size=test_size, 
            random_state=42
        )
        
        print(f"\nTotal valid sessions: {len(valid_sessions)}")
        print(f"Training set: {len(self.train_sessions)}")
        print(f"Test set: {len(self.test_sessions)}")
        
        # Extract features
        train_features_df = self.prepare_dataset(self.train_sessions)
        test_features_df = self.prepare_dataset(self.test_sessions)
        
        # Load trained models
        try:
            self.ocsvm = joblib.load('models/all_features/ocsvm_model.pkl')
            self.iso_forest = joblib.load('models/all_features/isolation_forest_model.pkl')
            self.scaler = joblib.load('models/all_features/scaler.pkl')
            self.feature_names = joblib.load('models/all_features/feature_names.pkl')
            print("\n✅ Models loaded successfully")
            print(f"   Features: {', '.join(self.feature_names)}")
        except Exception as e:
            print(f"\n❌ Could not load models: {e}")
            print("   Train models first with feature_selection_training.py!")
            return None, None
        
        return train_features_df, test_features_df
    
    def calculate_velocity_entropy(self, velocities):
        """Calculate entropy of velocity distribution"""
        if len(velocities) < 2:
            return 0
        hist, _ = np.histogram(velocities, bins=10)
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        return entropy(hist)
    
    def calculate_direction_changes(self, movements):
        """Count significant direction changes"""
        if len(movements) < 3:
            return 0
        
        direction_changes = 0
        threshold = np.pi / 4
        
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
                
                if angle > threshold:
                    direction_changes += 1
        
        return direction_changes
    
    def calculate_jitter(self, movements):
        """Calculate micro-variations (jitter)"""
        if len(movements) < 3:
            return 0
        
        jitters = []
        for i in range(1, len(movements) - 1):
            p_prev = np.array([movements[i-1]['x'], movements[i-1]['y']])
            p_curr = np.array([movements[i]['x'], movements[i]['y']])
            p_next = np.array([movements[i+1]['x'], movements[i+1]['y']])
            
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            accel = v2 - v1
            jitter_magnitude = np.linalg.norm(accel)
            jitters.append(jitter_magnitude)
        
        return np.mean(jitters) if jitters else 0
    
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
    
    def extract_features(self, session_data):
        """Extract features including NEW ones"""
        features = {}
        
        features['total_time'] = session_data.get('total_time', 0)
        features['num_movements'] = len(session_data.get('mouse_movements', []))
        features['movement_frequency'] = (features['num_movements'] / features['total_time'] 
                                         if features['total_time'] > 0 else 0)
        
        features['total_distance'] = session_data.get('distance_traveled', 0)
        
        velocities = session_data.get('velocities', [])
        features['std_velocity'] = np.std(velocities) if velocities else 0
        features['velocity_entropy'] = self.calculate_velocity_entropy(velocities)
        
        accelerations = session_data.get('accelerations', [])
        features['std_acceleration'] = np.std(accelerations) if accelerations else 0
        
        movements = session_data.get('mouse_movements', [])
        if len(movements) >= 2:
            start = movements[0]
            end = movements[-1]
            straight_distance = np.sqrt((end['x'] - start['x'])**2 + (end['y'] - start['y'])**2)
            features['directness_ratio'] = (straight_distance / features['total_distance']
                                           if features['total_distance'] > 0 else 0)
            features['avg_curvature'] = self.calculate_curvature(movements)
            features['direction_changes'] = self.calculate_direction_changes(movements)
            features['jitter'] = self.calculate_jitter(movements)
        else:
            features['directness_ratio'] = 0
            features['avg_curvature'] = 0
            features['direction_changes'] = 0
            features['jitter'] = 0
        
        timestamps = session_data.get('timestamps', [])
        if len(timestamps) >= 2:
            time_diffs = np.diff(timestamps)
            features['num_pauses'] = np.sum(time_diffs > 0.5)
        else:
            features['num_pauses'] = 0
        
        return features
    
    def prepare_dataset(self, sessions):
        """Convert sessions to feature matrix"""
        features_list = []
        for session in sessions:
            features = self.extract_features(session)
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def analyze_decision_boundaries(self, train_features_df, test_features_df):
        """Analyze how each model draws decision boundaries on TRAIN and TEST sets"""
        print("\n" + "="*70)
        print("DECISION BOUNDARY ANALYSIS (Train vs Test)")
        print("="*70)
        
        # Get only the features used by models
        train_X = train_features_df[self.feature_names]
        test_X = test_features_df[self.feature_names]
        
        # Scale
        train_X_scaled = self.scaler.transform(train_X)
        test_X_scaled = self.scaler.transform(test_X)
        
        # Get predictions and scores
        ocsvm_train_pred = self.ocsvm.predict(train_X_scaled)
        ocsvm_train_scores = self.ocsvm.score_samples(train_X_scaled)
        ocsvm_test_pred = self.ocsvm.predict(test_X_scaled)
        ocsvm_test_scores = self.ocsvm.score_samples(test_X_scaled)
        
        if_train_pred = self.iso_forest.predict(train_X_scaled)
        if_train_scores = self.iso_forest.score_samples(train_X_scaled)
        if_test_pred = self.iso_forest.predict(test_X_scaled)
        if_test_scores = self.iso_forest.score_samples(test_X_scaled)
        
        # Analyze score distributions
        print("\n1. ANOMALY SCORE DISTRIBUTIONS")
        print("-"*70)
        
        print("\nOne-Class SVM:")
        print(f"  Training - Mean: {ocsvm_train_scores.mean():.4f}, Std: {ocsvm_train_scores.std():.4f}")
        print(f"  Test     - Mean: {ocsvm_test_scores.mean():.4f}, Std: {ocsvm_test_scores.std():.4f}")
        
        print("\nIsolation Forest:")
        print(f"  Training - Mean: {if_train_scores.mean():.4f}, Std: {if_train_scores.std():.4f}")
        print(f"  Test     - Mean: {if_test_scores.mean():.4f}, Std: {if_test_scores.std():.4f}")
        
        # Analyze rejections
        print("\n2. REJECTION RATES (False Positives on Human Data)")
        print("-"*70)
        
        ocsvm_train_reject = (ocsvm_train_pred == -1).sum()
        ocsvm_test_reject = (ocsvm_test_pred == -1).sum()
        if_train_reject = (if_train_pred == -1).sum()
        if_test_reject = (if_test_pred == -1).sum()
        
        print(f"\nOne-Class SVM:")
        print(f"  Training: {ocsvm_train_reject}/{len(train_X)} ({ocsvm_train_reject/len(train_X)*100:.1f}%)")
        print(f"  Test:     {ocsvm_test_reject}/{len(test_X)} ({ocsvm_test_reject/len(test_X)*100:.1f}%)")
        
        print(f"\nIsolation Forest:")
        print(f"  Training: {if_train_reject}/{len(train_X)} ({if_train_reject/len(train_X)*100:.1f}%)")
        print(f"  Test:     {if_test_reject}/{len(test_X)} ({if_test_reject/len(test_X)*100:.1f}%)")
        
        # Generalization analysis
        print("\n3. GENERALIZATION ANALYSIS")
        print("-"*70)
        
        ocsvm_gap = abs((ocsvm_test_reject/len(test_X)) - (ocsvm_train_reject/len(train_X)))
        if_gap = abs((if_test_reject/len(test_X)) - (if_train_reject/len(train_X)))
        
        print(f"\nTrain-Test Gap (lower is better):")
        print(f"  One-Class SVM: {ocsvm_gap*100:.1f}%")
        print(f"  Isolation Forest: {if_gap*100:.1f}%")
        
        if ocsvm_gap < 0.05 and if_gap < 0.05:
            print("\n✅ Both models generalize well (gap < 5%)")
        elif ocsvm_gap > if_gap:
            print(f"\n⚠️  SVM has worse generalization (gap {ocsvm_gap*100:.1f}% vs {if_gap*100:.1f}%)")
        else:
            print(f"\n⚠️  IF has worse generalization (gap {if_gap*100:.1f}% vs {ocsvm_gap*100:.1f}%)")
        
        # Analyze disagreements on TEST set (most important)
        print("\n4. MODEL DISAGREEMENT ON TEST SET")
        print("-"*70)
        
        both_accept = (ocsvm_test_pred == 1) & (if_test_pred == 1)
        both_reject = (ocsvm_test_pred == -1) & (if_test_pred == -1)
        ocsvm_only_reject = (ocsvm_test_pred == -1) & (if_test_pred == 1)
        if_only_reject = (ocsvm_test_pred == 1) & (if_test_pred == -1)
        
        print(f"\nBoth models ACCEPT:  {both_accept.sum():>4} ({both_accept.sum()/len(test_X)*100:>5.1f}%)")
        print(f"Both models REJECT:  {both_reject.sum():>4} ({both_reject.sum()/len(test_X)*100:>5.1f}%)")
        print(f"Only SVM rejects:    {ocsvm_only_reject.sum():>4} ({ocsvm_only_reject.sum()/len(test_X)*100:>5.1f}%) ⚠️")
        print(f"Only IF rejects:     {if_only_reject.sum():>4} ({if_only_reject.sum()/len(test_X)*100:>5.1f}%)")
        
        # Analyze NEW features in rejected samples
        if ocsvm_only_reject.sum() > 0:
            print("\n5. NEW FEATURES IN SVM-REJECTED TEST SAMPLES")
            print("-"*70)
            
            rejected_features = test_X[ocsvm_only_reject]
            accepted_features = test_X[both_accept]
            
            new_features = ['velocity_entropy', 'direction_changes', 'jitter']
            
            print(f"\n{'Feature':<20} {'Accepted Mean':<15} {'Rejected Mean':<15} {'Diff %'}")
            print("-"*70)
            
            for feat in new_features:
                if feat in rejected_features.columns:
                    accepted_mean = accepted_features[feat].mean()
                    rejected_mean = rejected_features[feat].mean()
                    diff_pct = ((rejected_mean - accepted_mean) / accepted_mean * 100) if accepted_mean != 0 else 0
                    
                    print(f"{feat:<20} {accepted_mean:>10.3f}     {rejected_mean:>10.3f}     {diff_pct:>+6.1f}%")
        
        return {
            'train': (ocsvm_train_pred, if_train_pred),
            'test': (ocsvm_test_pred, if_test_pred),
            'train_scores': (ocsvm_train_scores, if_train_scores),
            'test_scores': (ocsvm_test_scores, if_test_scores)
        }
    
    def visualize_in_2d(self, train_features_df, test_features_df, predictions):
        """Visualize decision boundaries using PCA"""
        print("\n" + "="*70)
        print("CREATING 2D VISUALIZATIONS")
        print("="*70)
        
        # Get only model features
        train_X = train_features_df[self.feature_names]
        test_X = test_features_df[self.feature_names]
        
        # Scale
        train_X_scaled = self.scaler.transform(train_X)
        test_X_scaled = self.scaler.transform(test_X)
        
        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        pca.fit(train_X_scaled)
        
        train_2d = pca.transform(train_X_scaled)
        test_2d = pca.transform(test_X_scaled)
        
        explained_var = pca.explained_variance_ratio_
        print(f"\nPCA Variance Explained: {explained_var[0]:.1%} + {explained_var[1]:.1%} = {explained_var.sum():.1%}")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(20, 13))
        
        ocsvm_train_pred, if_train_pred = predictions['train']
        ocsvm_test_pred, if_test_pred = predictions['test']
        
        # Row 1: Training Set
        # Plot 1: SVM Training
        colors_ocsvm_train = ['red' if p == -1 else 'blue' for p in ocsvm_train_pred]
        axes[0, 0].scatter(train_2d[:, 0], train_2d[:, 1], c=colors_ocsvm_train, alpha=0.6, s=30)
        axes[0, 0].set_title(f'One-Class SVM - TRAINING\n({(ocsvm_train_pred == -1).sum()} rejected)', 
                            fontsize=11, fontweight='bold')
        axes[0, 0].set_xlabel(f'PC1 ({explained_var[0]:.1%})')
        axes[0, 0].set_ylabel(f'PC2 ({explained_var[1]:.1%})')
        axes[0, 0].grid(alpha=0.3)
        
        # Plot 2: IF Training
        colors_if_train = ['red' if p == -1 else 'green' for p in if_train_pred]
        axes[0, 1].scatter(train_2d[:, 0], train_2d[:, 1], c=colors_if_train, alpha=0.6, s=30)
        axes[0, 1].set_title(f'Isolation Forest - TRAINING\n({(if_train_pred == -1).sum()} rejected)', 
                            fontsize=11, fontweight='bold')
        axes[0, 1].set_xlabel(f'PC1 ({explained_var[0]:.1%})')
        axes[0, 1].set_ylabel(f'PC2 ({explained_var[1]:.1%})')
        axes[0, 1].grid(alpha=0.3)
        
        # Plot 3: Training Disagreement
        both_accept_train = (ocsvm_train_pred == 1) & (if_train_pred == 1)
        both_reject_train = (ocsvm_train_pred == -1) & (if_train_pred == -1)
        svm_only_train = (ocsvm_train_pred == -1) & (if_train_pred == 1)
        if_only_train = (ocsvm_train_pred == 1) & (if_train_pred == -1)
        
        colors_combined_train = []
        for i in range(len(ocsvm_train_pred)):
            if both_accept_train[i]:
                colors_combined_train.append('blue')
            elif both_reject_train[i]:
                colors_combined_train.append('red')
            elif svm_only_train[i]:
                colors_combined_train.append('orange')
            else:
                colors_combined_train.append('purple')
        
        axes[0, 2].scatter(train_2d[:, 0], train_2d[:, 1], c=colors_combined_train, alpha=0.6, s=30)
        axes[0, 2].set_title('Model Agreement - TRAINING', fontsize=11, fontweight='bold')
        axes[0, 2].set_xlabel(f'PC1 ({explained_var[0]:.1%})')
        axes[0, 2].set_ylabel(f'PC2 ({explained_var[1]:.1%})')
        axes[0, 2].grid(alpha=0.3)
        
        # Row 2: Test Set (MOST IMPORTANT)
        # Plot 4: SVM Test
        colors_ocsvm_test = ['red' if p == -1 else 'blue' for p in ocsvm_test_pred]
        axes[1, 0].scatter(test_2d[:, 0], test_2d[:, 1], c=colors_ocsvm_test, alpha=0.6, s=30)
        axes[1, 0].set_title(f'One-Class SVM - TEST ⭐\n({(ocsvm_test_pred == -1).sum()} rejected)', 
                            fontsize=11, fontweight='bold', color='darkred')
        axes[1, 0].set_xlabel(f'PC1 ({explained_var[0]:.1%})')
        axes[1, 0].set_ylabel(f'PC2 ({explained_var[1]:.1%})')
        axes[1, 0].grid(alpha=0.3)
        
        # Plot 5: IF Test
        colors_if_test = ['red' if p == -1 else 'green' for p in if_test_pred]
        axes[1, 1].scatter(test_2d[:, 0], test_2d[:, 1], c=colors_if_test, alpha=0.6, s=30)
        axes[1, 1].set_title(f'Isolation Forest - TEST ⭐\n({(if_test_pred == -1).sum()} rejected)', 
                            fontsize=11, fontweight='bold', color='darkgreen')
        axes[1, 1].set_xlabel(f'PC1 ({explained_var[0]:.1%})')
        axes[1, 1].set_ylabel(f'PC2 ({explained_var[1]:.1%})')
        axes[1, 1].grid(alpha=0.3)
        
        # Plot 6: Test Disagreement
        both_accept_test = (ocsvm_test_pred == 1) & (if_test_pred == 1)
        both_reject_test = (ocsvm_test_pred == -1) & (if_test_pred == -1)
        svm_only_test = (ocsvm_test_pred == -1) & (if_test_pred == 1)
        if_only_test = (ocsvm_test_pred == 1) & (if_test_pred == -1)
        
        colors_combined_test = []
        for i in range(len(ocsvm_test_pred)):
            if both_accept_test[i]:
                colors_combined_test.append('blue')
            elif both_reject_test[i]:
                colors_combined_test.append('red')
            elif svm_only_test[i]:
                colors_combined_test.append('orange')
            else:
                colors_combined_test.append('purple')
        
        axes[1, 2].scatter(test_2d[:, 0], test_2d[:, 1], c=colors_combined_test, alpha=0.6, s=30)
        axes[1, 2].set_title('Model Agreement - TEST ⭐', fontsize=11, fontweight='bold')
        axes[1, 2].set_xlabel(f'PC1 ({explained_var[0]:.1%})')
        axes[1, 2].set_ylabel(f'PC2 ({explained_var[1]:.1%})')
        axes[1, 2].grid(alpha=0.3)
        
        # Add legend to last plot
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Both Accept'),
            Patch(facecolor='red', label='Both Reject'),
            Patch(facecolor='orange', label='Only SVM Rejects'),
            Patch(facecolor='purple', label='Only IF Rejects')
        ]
        axes[1, 2].legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        plt.savefig('model_comparison_2d.png', dpi=300, bbox_inches='tight')
        print("\n📊 2D visualization saved to 'model_comparison_2d.png'")
        plt.show()
    
    def generate_research_summary(self, train_features_df, test_features_df, predictions):
        """Generate comprehensive research summary"""
        print("\n" + "="*70)
        print("RESEARCH PAPER SUMMARY")
        print("="*70)
        
        train_X = train_features_df[self.feature_names]
        test_X = test_features_df[self.feature_names]
        
        ocsvm_train_pred, if_train_pred = predictions['train']
        ocsvm_test_pred, if_test_pred = predictions['test']
        
        train_total = len(train_X)
        test_total = len(test_X)
        
        svm_train_reject = (ocsvm_train_pred == -1).sum()
        svm_test_reject = (ocsvm_test_pred == -1).sum()
        if_train_reject = (if_train_pred == -1).sum()
        if_test_reject = (if_test_pred == -1).sum()
        
        print("\n" + "─"*70)
        print("QUANTITATIVE RESULTS")
        print("─"*70)
        
        print(f"\nDataset: {train_total + test_total} human behavioral samples")
        print(f"  Training: {train_total} sessions")
        print(f"  Test: {test_total} sessions")
        print(f"\nFeatures: {len(self.feature_names)} behavioral metrics")
        print(f"  NEW: velocity_entropy, direction_changes, jitter")
        print(f"\nParameters: nu=0.05, gamma='scale', contamination=0.05")
        
        print(f"\n{'Model':<20} {'Training FP':<15} {'Test FP':<15} {'Gap'}")
        print("─"*70)
        print(f"{'One-Class SVM':<20} {svm_train_reject/train_total:>10.1%}     "
              f"{svm_test_reject/test_total:>10.1%}     "
              f"{abs(svm_test_reject/test_total - svm_train_reject/train_total):>6.1%}")
        print(f"{'Isolation Forest':<20} {if_train_reject/train_total:>10.1%}     "
              f"{if_test_reject/test_total:>10.1%}     "
              f"{abs(if_test_reject/test_total - if_train_reject/train_total):>6.1%}")
        
        # Save detailed report
        with open('research_findings_updated.txt', 'w') as f:
            f.write("="*70 + "\n")
            f.write("ONE-CLASS SVM VS ISOLATION FOREST - UPDATED ANALYSIS\n")
            f.write("="*70 + "\n\n")
            
            f.write("UPDATES:\n")
            f.write("- Train/Test Split: 80/20 for proper evaluation\n")
            f.write("- New Features: velocity_entropy, direction_changes, jitter\n")
            f.write("- Parameters: nu=0.05, gamma='scale', contamination=0.05\n\n")
            
            f.write(f"DATASET:\n")
            f.write(f"  Total sessions: {train_total + test_total}\n")
            f.write(f"  Training: {train_total}\n")
            f.write(f"  Test: {test_total}\n\n")
            
            f.write(f"FEATURES ({len(self.feature_names)}):\n")
            for feat in self.feature_names:
                f.write(f"  - {feat}\n")
            f.write("\n")
            
            f.write("RESULTS:\n")
            f.write("-"*70 + "\n")
            f.write(f"One-Class SVM:\n")
            f.write(f"  Training FP: {svm_train_reject/train_total:.2%}\n")
            f.write(f"  Test FP:     {svm_test_reject/test_total:.2%}\n")
            f.write(f"  Generalization Gap: {abs(svm_test_reject/test_total - svm_train_reject/train_total):.2%}\n\n")
            
            f.write(f"Isolation Forest:\n")
            f.write(f"  Training FP: {if_train_reject/train_total:.2%}\n")
            f.write(f"  Test FP:     {if_test_reject/test_total:.2%}\n")
            f.write(f"  Generalization Gap: {abs(if_test_reject/test_total - if_train_reject/train_total):.2%}\n\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("KEY FINDINGS FOR RESEARCH PAPER:\n")
            f.write("="*70 + "\n\n")
            
            f.write("1. TEST SET PERFORMANCE (Most Important):\n")
            if svm_test_reject < if_test_reject:
                f.write("   One-Class SVM achieved lower false positive rate on unseen data\n")
            else:
                f.write("   Isolation Forest achieved lower false positive rate on unseen data\n")
            
            f.write("\n2. GENERALIZATION:\n")
            svm_gap = abs(svm_test_reject/test_total - svm_train_reject/train_total)
            if_gap = abs(if_test_reject/test_total - if_train_reject/train_total)
            
            if svm_gap < if_gap:
                f.write("   One-Class SVM generalizes better (smaller train-test gap)\n")
            else:
                f.write("   Isolation Forest generalizes better (smaller train-test gap)\n")
            
            f.write("\n3. NEW FEATURES IMPACT:\n")
            f.write("   - velocity_entropy: Captures randomness in human speed patterns\n")
            f.write("   - direction_changes: Quantifies path exploration behavior\n")
            f.write("   - jitter: Detects natural hand tremor vs bot smoothness\n")
            
            f.write("\n4. PARAMETER TUNING:\n")
            f.write("   - Increased nu from 0.01 to 0.05 (expect 5% anomalies)\n")
            f.write("   - Changed gamma from 'auto' to 'scale' for better generalization\n")
            f.write("   - Increased contamination to 0.05 for consistency\n")
        
        print("\n📄 Detailed research findings saved to 'research_findings_updated.txt'")


# Main execution
if __name__ == "__main__":
    print("\n" + "="*70)
    print("ONE-CLASS SVM VS ISOLATION FOREST - UPDATED ANALYSIS")
    print("="*70)
    print("\nUpdates:")
    print("  ✓ Train/Test split (80/20)")
    print("  ✓ New features: velocity_entropy, direction_changes, jitter")
    print("  ✓ Parameters: nu=0.05, gamma='scale', contamination=0.05\n")
    
    analyzer = ModelComparisonAnalysis()
    
    # Load data and models
    train_features_df, test_features_df = analyzer.load_and_prepare('captcha_data', test_size=0.2)
    
    if train_features_df is not None:
        # Analyze decision boundaries
        predictions = analyzer.analyze_decision_boundaries(train_features_df, test_features_df)
        
        # Visualize in 2D
        analyzer.visualize_in_2d(train_features_df, test_features_df, predictions)
        
        # Generate research summary
        analyzer.generate_research_summary(train_features_df, test_features_df, predictions)
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70)
        print("\nGenerated files:")
        print("  - model_comparison_2d.png (6 visualizations: train + test)")
        print("  - research_findings_updated.txt (for your paper)")