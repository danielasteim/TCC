import tkinter as tk
from tkinter import messagebox
import time
import json
import math
import os
from datetime import datetime
from collections import deque
import numpy as np
import joblib


class ProductionCaptchaIF:
    """
    Production CAPTCHA using Isolation Forest
    BLUE THEME - for easy visual identification
    """
    
    def __init__(self, root, model_dir='models/kinematic_geometric_11'):
        self.root = root
        self.root.title("CAPTCHA Verification - Isolation Forest")
        self.root.geometry("500x450")
        self.root.configure(bg='#e6f2ff')  # Light blue background
        
        # Load trained model
        self.load_model(model_dir)
        
        # Session data storage
        self.session_data = {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S_%f'),
            'mouse_movements': [],
            'timestamps': [],
            'velocities': [],
            'accelerations': [],
            'click_data': {},
            'total_time': 0,
            'distance_traveled': 0
        }
        
        # Tracking variables
        self.start_time = time.time()
        self.last_position = None
        self.last_time = time.time()
        self.is_tracking = True
        self.checkbox_checked = False
        self.recent_positions = deque(maxlen=3)
        
        self.setup_ui()
        
    def load_model(self, model_dir):
        """Load trained Isolation Forest model"""
        try:
            self.model = joblib.load(f'{model_dir}/isolation_forest_model.pkl')
            self.scaler = joblib.load(f'{model_dir}/scaler.pkl')
            self.feature_names = joblib.load(f'{model_dir}/feature_names.pkl')
            print(f"✅ Isolation Forest Model loaded from {model_dir}")
            print(f"   Features: {', '.join(self.feature_names)}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("   Using dummy mode (will always pass)")
            self.model = None
            
    def setup_ui(self):
        # Main frame - BLUE THEME
        main_frame = tk.Frame(self.root, bg='#cce5ff', relief=tk.RAISED, borderwidth=3)
        main_frame.place(relx=0.5, rely=0.5, anchor='center', width=420, height=320)
        
        # Model indicator
        model_label = tk.Label(
            main_frame,
            text="🔵 ISOLATION FOREST",
            font=('Arial', 10, 'bold'),
            bg='#0066cc',
            fg='white',
            padx=10,
            pady=5
        )
        model_label.pack(pady=(10, 5))
        
        # Title
        title_label = tk.Label(
            main_frame, 
            text="Security Verification",
            font=('Arial', 14, 'bold'),
            bg='#cce5ff',
            fg='#003366'
        )
        title_label.pack(pady=10)
        
        # Checkbox frame
        checkbox_frame = tk.Frame(main_frame, bg='white', relief=tk.GROOVE, borderwidth=2)
        checkbox_frame.pack(pady=20, padx=40, fill='x')
        
        # Checkbox
        self.checkbox_var = tk.BooleanVar()
        self.checkbox = tk.Checkbutton(
            checkbox_frame,
            text="  I'm not a robot",
            variable=self.checkbox_var,
            font=('Arial', 12),
            bg='white',
            command=self.on_checkbox_click,
            cursor='hand2'
        )
        self.checkbox.pack(side='left', padx=10, pady=15)
        
        # Icon
        icon_label = tk.Label(
            checkbox_frame,
            text="🤖",
            font=('Arial', 24),
            bg='white'
        )
        icon_label.pack(side='right', padx=10)
        
        # Status label
        self.status_label = tk.Label(
            main_frame,
            text="Move your mouse naturally and check the box",
            font=('Arial', 9),
            bg='#cce5ff',
            fg='#666666'
        )
        self.status_label.pack(pady=10)
        
        # Movement counter
        self.counter_label = tk.Label(
            main_frame,
            text="Movements: 0",
            font=('Arial', 8),
            bg='#cce5ff',
            fg='#999999'
        )
        self.counter_label.pack(pady=2)
        
        # Result frame (hidden initially)
        self.result_frame = tk.Frame(main_frame, bg='#cce5ff')
        self.result_label = tk.Label(
            self.result_frame,
            text="",
            font=('Arial', 11, 'bold'),
            bg='#cce5ff'
        )
        self.result_label.pack()
        
        # Bind mouse events
        self.root.bind('<Motion>', self.track_mouse_movement)
        
    def track_mouse_movement(self, event):
        """Track mouse movements"""
        if not self.is_tracking:
            return
            
        current_time = time.time()
        current_pos = (event.x, event.y)
        
        if self.last_position is None:
            self.last_position = current_pos
            self.last_time = current_time
            self.recent_positions.append((current_pos, current_time))
            return
        
        # Record movement
        self.session_data['mouse_movements'].append({
            'x': event.x,
            'y': event.y,
            'time_offset': current_time - self.start_time
        })
        self.session_data['timestamps'].append(current_time)
        
        self.counter_label.config(text=f"Movements: {len(self.session_data['mouse_movements'])}")
        
        # Calculate metrics
        dx = current_pos[0] - self.last_position[0]
        dy = current_pos[1] - self.last_position[1]
        distance = math.sqrt(dx**2 + dy**2)
        self.session_data['distance_traveled'] += distance
        
        time_delta = current_time - self.last_time
        if time_delta > 0:
            velocity = distance / time_delta
            self.session_data['velocities'].append(velocity)
            
            self.recent_positions.append((current_pos, current_time))
            if len(self.recent_positions) >= 3:
                acceleration = self.calculate_acceleration()
                self.session_data['accelerations'].append(acceleration)
        
        self.last_position = current_pos
        self.last_time = current_time
        
    def calculate_acceleration(self):
        """Calculate acceleration"""
        if len(self.recent_positions) < 3:
            return 0
        
        positions = list(self.recent_positions)
        try:
            time_diff_1 = positions[1][1] - positions[0][1]
            time_diff_2 = positions[2][1] - positions[1][1]
            
            if time_diff_1 == 0 or time_diff_2 == 0:
                return 0
            
            v1_x = (positions[1][0][0] - positions[0][0][0]) / time_diff_1
            v1_y = (positions[1][0][1] - positions[0][0][1]) / time_diff_1
            v1 = math.sqrt(v1_x**2 + v1_y**2)
            
            v2_x = (positions[2][0][0] - positions[1][0][0]) / time_diff_2
            v2_y = (positions[2][0][1] - positions[1][0][1]) / time_diff_2
            v2 = math.sqrt(v2_x**2 + v2_y**2)
            
            time_delta = positions[2][1] - positions[1][1]
            if time_delta > 0:
                return (v2 - v1) / time_delta
        except:
            pass
        
        return 0
    
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
    
    def extract_features(self):
        """Extract features for prediction"""
        features = {}
        
        # Core features
        features['total_time'] = self.session_data['total_time']
        features['num_movements'] = len(self.session_data['mouse_movements'])
        features['movement_frequency'] = (features['num_movements'] / features['total_time'] 
                                         if features['total_time'] > 0 else 0)
        
        features['total_distance'] = self.session_data['distance_traveled']
        
        velocities = self.session_data['velocities']
        features['mean_velocity'] = np.mean(velocities) if velocities else 0
        features['std_velocity'] = np.std(velocities) if velocities else 0
        
        accelerations = self.session_data['accelerations']
        features['mean_acceleration'] = np.mean(accelerations) if accelerations else 0
        
        movements = self.session_data['mouse_movements']
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
        
        timestamps = self.session_data['timestamps']
        if len(timestamps) >= 2:
            time_diffs = np.diff(timestamps)
            features['num_pauses'] = np.sum(time_diffs > 0.5)
        else:
            features['num_pauses'] = 0
        
        return features
    
    def on_checkbox_click(self):
        """Handle checkbox click"""
        if not self.checkbox_checked and self.checkbox_var.get():
            self.checkbox_checked = True
            click_time = time.time()
            
            self.session_data['click_data'] = {
                'time_to_click': click_time - self.start_time,
                'click_position': self.last_position if self.last_position else (0, 0),
                'click_timestamp': click_time
            }
            
            self.session_data['total_time'] = click_time - self.start_time
            self.is_tracking = False
            
            # Verify with model
            self.verify_human()
    
    def verify_human(self):
        """Verify if user is human using Isolation Forest"""
        if len(self.session_data['mouse_movements']) < 5:
            self.show_result(False, "Insufficient movement data")
            return
        
        # Extract features
        features_dict = self.extract_features()
        
        # Create feature vector in correct order
        feature_vector = []
        for feat_name in self.feature_names:
            feature_vector.append(features_dict.get(feat_name, 0))
        
        # Scale and predict
        if self.model is not None:
            X = np.array(feature_vector).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            prediction = self.model.predict(X_scaled)[0]
            score = self.model.score_samples(X_scaled)[0]
            
            is_human = (prediction == 1)
            
            print(f"\n{'='*50}")
            print(f"Isolation Forest Verification Result:")
            print(f"  Prediction: {'HUMAN' if is_human else 'BOT/ANOMALY'}")
            print(f"  Score: {score:.4f}")
            print(f"  Time: {self.session_data['total_time']:.2f}s")
            print(f"  Movements: {len(self.session_data['mouse_movements'])}")
            print(f"{'='*50}")
            
            self.show_result(is_human, score)
        else:
            # Dummy mode
            self.show_result(True, 0.0)
    
    def show_result(self, is_human, score):
        """Display verification result"""
        self.result_frame.pack(pady=10)
        
        if is_human:
            self.result_label.config(
                text=f"✓ Verified Human\nConfidence: {abs(score):.4f}",
                fg='#00aa00'
            )
            self.status_label.config(text="Verification successful!", fg='#00aa00')
        else:
            self.result_label.config(
                text=f"✗ Verification Failed\nScore: {score:.4f}",
                fg='#cc0000'
            )
            self.status_label.config(text="Suspicious behavior detected", fg='#cc0000')


if __name__ == "__main__":
    print("="*50)
    print("PRODUCTION CAPTCHA - ISOLATION FOREST")
    print("="*50)
    
    root = tk.Tk()
    
    # Try to load model, fall back to dummy mode if not found
    app = ProductionCaptchaIF(root, model_dir='models/all_features')
    
    root.mainloop()