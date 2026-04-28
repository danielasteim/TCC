import tkinter as tk
import time
import json
import math
import random
import numpy as np
from datetime import datetime
import joblib


class BotSimulator:
    """
    Simulate different types of bots to test CAPTCHA detection
    
    Bot Types:
    1. Linear Bot - Moves in straight lines, instant clicks
    2. Fast Bot - Superhuman speed, no hesitation
    3. Perfect Bot - Too precise, mechanical movements
    4. Slow Bot - Suspiciously slow and deliberate
    5. Sophisticated Bot - Tries to mimic human (hardest to detect)
    """
    
    def __init__(self):
        self.bot_types = {
            'linear': self.linear_bot,
            'fast': self.fast_bot,
            'perfect': self.perfect_bot,
            'slow': self.slow_bot,
            'sophisticated': self.sophisticated_bot
        }
    
    def linear_bot(self, start_pos, target_pos):
        """
        Linear Bot: Moves in perfectly straight line
        - No curves
        - Constant velocity
        - Instant targeting
        """
        print("\n🤖 Simulating: LINEAR BOT")
        print("   Characteristics: Straight lines, no curves, mechanical")
        
        session_data = {
            'session_id': f'bot_linear_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'mouse_movements': [],
            'timestamps': [],
            'velocities': [],
            'accelerations': [],
            'click_data': {},
            'total_time': 0,
            'distance_traveled': 0
        }
        
        # Generate linear path
        num_points = 15  # Very few points (bots are efficient)
        start_time = time.time()
        
        for i in range(num_points):
            t = i / (num_points - 1)
            
            # Linear interpolation (straight line)
            x = start_pos[0] + t * (target_pos[0] - start_pos[0])
            y = start_pos[1] + t * (target_pos[1] - start_pos[1])
            
            current_time = start_time + i * 0.01  # Constant 10ms intervals
            
            session_data['mouse_movements'].append({
                'x': int(x),
                'y': int(y),
                'time_offset': current_time - start_time
            })
            session_data['timestamps'].append(current_time)
        
        # Calculate metrics
        session_data = self._calculate_metrics(session_data)
        
        # Instant click
        session_data['click_data'] = {
            'time_to_click': 0.15,  # Suspiciously fast
            'click_position': target_pos,
            'click_timestamp': start_time + 0.15
        }
        session_data['total_time'] = 0.15
        
        return session_data
    
    def fast_bot(self, start_pos, target_pos):
        """
        Fast Bot: Superhuman speed
        - Very high velocity
        - Minimal time
        - Few movements
        """
        print("\n🤖 Simulating: FAST BOT")
        print("   Characteristics: Superhuman speed, too fast")
        
        session_data = {
            'session_id': f'bot_fast_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'mouse_movements': [],
            'timestamps': [],
            'velocities': [],
            'accelerations': [],
            'click_data': {},
            'total_time': 0,
            'distance_traveled': 0
        }
        
        # Generate very fast path
        num_points = 8  # Very few points
        start_time = time.time()
        
        for i in range(num_points):
            t = i / (num_points - 1)
            
            # Add tiny random jitter (trying to look human)
            jitter = 2
            x = start_pos[0] + t * (target_pos[0] - start_pos[0]) + random.randint(-jitter, jitter)
            y = start_pos[1] + t * (target_pos[1] - start_pos[1]) + random.randint(-jitter, jitter)
            
            current_time = start_time + i * 0.005  # 5ms intervals (too fast!)
            
            session_data['mouse_movements'].append({
                'x': int(x),
                'y': int(y),
                'time_offset': current_time - start_time
            })
            session_data['timestamps'].append(current_time)
        
        session_data = self._calculate_metrics(session_data)
        
        session_data['click_data'] = {
            'time_to_click': 0.04,  # Impossibly fast
            'click_position': target_pos,
            'click_timestamp': start_time + 0.04
        }
        session_data['total_time'] = 0.04
        
        return session_data
    
    def perfect_bot(self, start_pos, target_pos):
        """
        Perfect Bot: Too precise
        - Perfect timing intervals
        - Exact same velocity
        - No natural variance
        """
        print("\n🤖 Simulating: PERFECT BOT")
        print("   Characteristics: Too precise, mechanical consistency")
        
        session_data = {
            'session_id': f'bot_perfect_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'mouse_movements': [],
            'timestamps': [],
            'velocities': [],
            'accelerations': [],
            'click_data': {},
            'total_time': 0,
            'distance_traveled': 0
        }
        
        # Generate perfectly timed path
        num_points = 25
        start_time = time.time()
        
        for i in range(num_points):
            t = i / (num_points - 1)
            
            # Bezier curve (smooth but TOO perfect)
            control = ((start_pos[0] + target_pos[0]) / 2, 
                      (start_pos[1] + target_pos[1]) / 2 - 30)
            
            x = (1-t)**2 * start_pos[0] + 2*(1-t)*t * control[0] + t**2 * target_pos[0]
            y = (1-t)**2 * start_pos[1] + 2*(1-t)*t * control[1] + t**2 * target_pos[1]
            
            # EXACTLY 20ms intervals (too perfect!)
            current_time = start_time + i * 0.020
            
            session_data['mouse_movements'].append({
                'x': int(x),
                'y': int(y),
                'time_offset': current_time - start_time
            })
            session_data['timestamps'].append(current_time)
        
        session_data = self._calculate_metrics(session_data)
        
        session_data['click_data'] = {
            'time_to_click': 0.50,  # Exactly 500ms
            'click_position': target_pos,
            'click_timestamp': start_time + 0.50
        }
        session_data['total_time'] = 0.50
        
        return session_data
    
    def slow_bot(self, start_pos, target_pos):
        """
        Slow Bot: Suspiciously slow
        - Too deliberate
        - Unnatural pauses
        - Too many points
        """
        print("\n🤖 Simulating: SLOW BOT")
        print("   Characteristics: Unnaturally slow, too deliberate")
        
        session_data = {
            'session_id': f'bot_slow_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'mouse_movements': [],
            'timestamps': [],
            'velocities': [],
            'accelerations': [],
            'click_data': {},
            'total_time': 0,
            'distance_traveled': 0
        }
        
        # Generate very slow path
        num_points = 100  # Too many points
        start_time = time.time()
        
        for i in range(num_points):
            t = i / (num_points - 1)
            
            x = start_pos[0] + t * (target_pos[0] - start_pos[0])
            y = start_pos[1] + t * (target_pos[1] - start_pos[1])
            
            # Very slow, regular intervals
            current_time = start_time + i * 0.05  # 50ms intervals
            
            session_data['mouse_movements'].append({
                'x': int(x),
                'y': int(y),
                'time_offset': current_time - start_time
            })
            session_data['timestamps'].append(current_time)
        
        session_data = self._calculate_metrics(session_data)
        
        session_data['click_data'] = {
            'time_to_click': 5.0,  # Suspiciously slow
            'click_position': target_pos,
            'click_timestamp': start_time + 5.0
        }
        session_data['total_time'] = 5.0
        
        return session_data
    
    def sophisticated_bot(self, start_pos, target_pos):
        """
        Sophisticated Bot: Attempts to mimic human
        - Adds random curves
        - Variable timing
        - Still has tells (low variance, etc.)
        """
        print("\n🤖 Simulating: SOPHISTICATED BOT")
        print("   Characteristics: Tries to mimic human, hardest to detect")
        
        session_data = {
            'session_id': f'bot_sophisticated_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'mouse_movements': [],
            'timestamps': [],
            'velocities': [],
            'accelerations': [],
            'click_data': {},
            'total_time': 0,
            'distance_traveled': 0
        }
        
        # Generate semi-natural path
        num_points = random.randint(40, 60)  # Variable count
        start_time = time.time()
        
        # Add control points for curves
        mid_x = (start_pos[0] + target_pos[0]) / 2 + random.randint(-40, 40)
        mid_y = (start_pos[1] + target_pos[1]) / 2 + random.randint(-40, 40)
        
        for i in range(num_points):
            t = i / (num_points - 1)
            
            # Quadratic bezier with noise
            x = (1-t)**2 * start_pos[0] + 2*(1-t)*t * mid_x + t**2 * target_pos[0]
            y = (1-t)**2 * start_pos[1] + 2*(1-t)*t * mid_y + t**2 * target_pos[1]
            
            # Add noise
            noise = 5
            x += random.randint(-noise, noise)
            y += random.randint(-noise, noise)
            
            # Variable timing (but still less variance than humans)
            interval = random.uniform(0.015, 0.025)  # 15-25ms (less variance than humans)
            current_time = start_time + sum([random.uniform(0.015, 0.025) for _ in range(i)])
            
            session_data['mouse_movements'].append({
                'x': int(x),
                'y': int(y),
                'time_offset': current_time - start_time
            })
            session_data['timestamps'].append(current_time)
        
        session_data = self._calculate_metrics(session_data)
        
        # Semi-realistic click time
        click_time = random.uniform(0.9, 1.5)
        session_data['click_data'] = {
            'time_to_click': click_time,
            'click_position': target_pos,
            'click_timestamp': start_time + click_time
        }
        session_data['total_time'] = click_time
        
        return session_data
    
    def _calculate_metrics(self, session_data):
        """Calculate velocities, accelerations, distance"""
        movements = session_data['mouse_movements']
        
        if len(movements) < 2:
            return session_data
        
        # Calculate distance and velocity
        total_distance = 0
        velocities = []
        
        for i in range(1, len(movements)):
            dx = movements[i]['x'] - movements[i-1]['x']
            dy = movements[i]['y'] - movements[i-1]['y']
            distance = math.sqrt(dx**2 + dy**2)
            total_distance += distance
            
            time_delta = session_data['timestamps'][i] - session_data['timestamps'][i-1]
            if time_delta > 0:
                velocity = distance / time_delta
                velocities.append(velocity)
        
        session_data['distance_traveled'] = total_distance
        session_data['velocities'] = velocities
        
        # Calculate accelerations
        accelerations = []
        for i in range(1, len(velocities)):
            time_delta = session_data['timestamps'][i+1] - session_data['timestamps'][i]
            if time_delta > 0:
                accel = (velocities[i] - velocities[i-1]) / time_delta
                accelerations.append(accel)
        
        session_data['accelerations'] = accelerations
        
        return session_data
    
    def test_bot_against_models(self, bot_data, model_dir='models/core_8'):
        """Test bot against both models"""
        print("\n" + "="*60)
        print("TESTING BOT AGAINST MODELS")
        print("="*60)
        
        # Load models
        try:
            ocsvm = joblib.load(f'{model_dir}/ocsvm_model.pkl')
            iso_forest = joblib.load(f'{model_dir}/isolation_forest_model.pkl')
            scaler = joblib.load(f'{model_dir}/scaler.pkl')
            feature_names = joblib.load(f'{model_dir}/feature_names.pkl')
        except:
            print("❌ Could not load models. Train them first!")
            return
        
        # Extract features
        features = self._extract_features(bot_data)
        
        # Create feature vector
        feature_vector = []
        for feat_name in feature_names:
            feature_vector.append(features.get(feat_name, 0))
        
        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        # Test with One-Class SVM
        svm_pred = ocsvm.predict(X_scaled)[0]
        svm_score = ocsvm.score_samples(X_scaled)[0]
        
        # Test with Isolation Forest
        if_pred = iso_forest.predict(X_scaled)[0]
        if_score = iso_forest.score_samples(X_scaled)[0]
        
        print("\nBot Characteristics:")
        print(f"  Total Time: {bot_data['total_time']:.3f}s")
        print(f"  Movements: {len(bot_data['mouse_movements'])}")
        print(f"  Distance: {bot_data['distance_traveled']:.1f}px")
        print(f"  Mean Velocity: {features.get('mean_velocity', 0):.1f}px/s")
        print(f"  Velocity Std: {features.get('std_velocity', 0):.1f}")
        print(f"  Directness: {features.get('directness_ratio', 0):.3f}")
        
        print("\n--- One-Class SVM ---")
        svm_detected = (svm_pred == -1)
        print(f"  Prediction: {'🚫 BOT DETECTED' if svm_detected else '✓ Passed as human'}")
        print(f"  Score: {svm_score:.4f}")
        
        print("\n--- Isolation Forest ---")
        if_detected = (if_pred == -1)
        print(f"  Prediction: {'🚫 BOT DETECTED' if if_detected else '✓ Passed as human'}")
        print(f"  Score: {if_score:.4f}")
        
        print("\n" + "="*60)
        
        if svm_detected and if_detected:
            print("✅ BOTH MODELS DETECTED THE BOT!")
        elif svm_detected or if_detected:
            print("⚠️  ONE MODEL DETECTED THE BOT")
            if svm_detected:
                print("   (SVM detected, IF missed)")
            else:
                print("   (IF detected, SVM missed)")
        else:
            print("❌ BOT PASSED BOTH MODELS (False Negative)")
        
        return {
            'svm_detected': svm_detected,
            'if_detected': if_detected,
            'svm_score': svm_score,
            'if_score': if_score
        }
    
    def _extract_features(self, session_data):
        """Extract features from bot session"""
        features = {}
        
        features['total_time'] = session_data['total_time']
        features['num_movements'] = len(session_data['mouse_movements'])
        features['movement_frequency'] = (features['num_movements'] / features['total_time'] 
                                         if features['total_time'] > 0 else 0)
        
        features['total_distance'] = session_data['distance_traveled']
        
        velocities = session_data['velocities']
        features['mean_velocity'] = np.mean(velocities) if velocities else 0
        features['std_velocity'] = np.std(velocities) if velocities else 0
        
        accelerations = session_data['accelerations']
        features['mean_acceleration'] = np.mean(accelerations) if accelerations else 0
        
        movements = session_data['mouse_movements']
        if len(movements) >= 2:
            start = movements[0]
            end = movements[-1]
            straight_distance = math.sqrt((end['x'] - start['x'])**2 + (end['y'] - start['y'])**2)
            features['directness_ratio'] = (straight_distance / features['total_distance']
                                           if features['total_distance'] > 0 else 0)
            features['avg_curvature'] = self._calculate_curvature(movements)
        else:
            features['directness_ratio'] = 0
            features['avg_curvature'] = 0
        
        timestamps = session_data['timestamps']
        if len(timestamps) >= 2:
            time_diffs = np.diff(timestamps)
            features['num_pauses'] = np.sum(time_diffs > 0.5)
        else:
            features['num_pauses'] = 0
        
        return features
    
    def _calculate_curvature(self, movements):
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
    
    def run_all_tests(self, model_dir='models/core_8'):
        """Test all bot types"""
        print("\n" + "="*70)
        print("TESTING ALL BOT TYPES")
        print("="*70)
        
        # Test positions
        start_pos = (100, 300)
        target_pos = (400, 200)
        
        results = {}
        
        for bot_name, bot_func in self.bot_types.items():
            print("\n" + "─"*70)
            
            # Generate bot data
            bot_data = bot_func(start_pos, target_pos)
            
            # Test against models
            result = self.test_bot_against_models(bot_data, model_dir)
            results[bot_name] = result
        
        # Summary
        print("\n" + "="*70)
        print("DETECTION SUMMARY")
        print("="*70)
        
        print(f"\n{'Bot Type':<20} {'SVM Detected':<15} {'IF Detected':<15} {'Status'}")
        print("─"*70)
        
        for bot_name, result in results.items():
            svm_status = "✓" if result['svm_detected'] else "✗"
            if_status = "✓" if result['if_detected'] else "✗"
            
            if result['svm_detected'] and result['if_detected']:
                status = "✅ Both detected"
            elif result['svm_detected'] or result['if_detected']:
                status = "⚠️  One detected"
            else:
                status = "❌ Both missed"
            
            print(f"{bot_name:<20} {svm_status:<15} {if_status:<15} {status}")
        
        # Calculate detection rates
        svm_detected = sum(1 for r in results.values() if r['svm_detected'])
        if_detected = sum(1 for r in results.values() if r['if_detected'])
        total = len(results)
        
        print("\n" + "="*70)
        print(f"SVM Detection Rate: {svm_detected}/{total} ({svm_detected/total*100:.0f}%)")
        print(f"IF Detection Rate:  {if_detected}/{total} ({if_detected/total*100:.0f}%)")
        print("="*70)


# Main execution
if __name__ == "__main__":
    print("="*70)
    print("BOT SIMULATOR - CAPTCHA TESTING")
    print("="*70)
    print("\nThis tool simulates 5 different types of bots:")
    print("1. Linear Bot - Straight lines, no curves")
    print("2. Fast Bot - Superhuman speed")
    print("3. Perfect Bot - Too precise timing")
    print("4. Slow Bot - Unnaturally slow")
    print("5. Sophisticated Bot - Tries to mimic human")
    
    simulator = BotSimulator()
    
    print("\nMake sure you have trained models in 'models/core_8/'")
    input("Press Enter to start testing...")
    
    # Run all tests
    simulator.run_all_tests(model_dir='models/all_features')