import tkinter as tk
from tkinter import ttk, messagebox
import time
import json
import math
import os
from datetime import datetime
from collections import deque
import numpy as np


class Captcha:
    def __init__(self, root):
        self.root = root
        self.root.title("Behavioral CAPTCHA")
        self.root.geometry("500x400")
        self.root.configure(bg='#f0f0f0')
        
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
        
        # Recent positions for acceleration calculation
        self.recent_positions = deque(maxlen=6)
        
        self.setup_ui()
        
        print("CAPTCHA initialized | CAPTCHA iniciado")
        
    def setup_ui(self):
        """ Sets up the UI enviroment including title, frames, checkbox, icon and status of verification"""
        """ Realiza a configuração da interface incluindo titulo, frames, checkbox, icone e status da verificação"""
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#ffffff', relief=tk.RAISED, borderwidth=2)
        main_frame.place(relx=0.5, rely=0.5, anchor='center', width=400, height=250)
        
        # Title
        title_label = tk.Label(
            main_frame, 
            text="Verifique se é humano",
            font=('Arial', 14, 'bold'),
            bg='#ffffff',
            fg='#333333'
        )
        title_label.pack(pady=20)
        
        # Checkbox frame
        checkbox_frame = tk.Frame(main_frame, bg='#ffffff', relief=tk.GROOVE, borderwidth=2)
        checkbox_frame.pack(pady=30, padx=40, fill='x')
        
        # Custom checkbox
        self.checkbox_var = tk.BooleanVar()
        self.checkbox = tk.Checkbutton(
            checkbox_frame,
            text="  Autenticação",
            variable=self.checkbox_var,
            font=('Arial', 12),
            bg='#ffffff',
            command=self.on_checkbox_click,
            cursor='hand2'
        )
        self.checkbox.pack(side='left', padx=10, pady=15)
        
        # Icon placeholder
        icon_label = tk.Label(
            checkbox_frame,
            text="👻",
            font=('Arial', 24),
            bg='#ffffff'
        )
        icon_label.pack(side='right', padx=10)
        
        # Status label
        self.status_label = tk.Label(
            main_frame,
            text="Mova seu mouse e clique na checkbox para verificar sua autenticidade",
            font=('Arial', 9),
            bg='#ffffff',
            fg='#666666'
        )
        self.status_label.pack(pady=10)
        
        #! Movement counter (remember to remove after models are implemented)
        self.counter_label = tk.Label(
            main_frame,
            text="Movimentos: 0",
            font=('Arial', 8),
            bg='#ffffff',
            fg='#999999'
        )
        self.counter_label.pack(pady=2)
        # Bind mouse events to the entire window
        self.root.bind('<Motion>', self.track_mouse_movement)
        
    def track_mouse_movement(self, event):
        """Track mouse movements and calculate behavioral metrics"""
        """ Calculo os movimentos do mouse e as métricas comportacionais """
        if not self.is_tracking:
            return
            
        current_time = time.time()
        current_pos = (event.x, event.y)
        
        # Initialize first position
        if self.last_position is None:
            self.last_position = current_pos
            self.last_time = current_time
            self.recent_positions.append((current_pos, current_time))
            return
        
        # Record position and timestamp
        self.session_data['mouse_movements'].append({
            'x': event.x,
            'y': event.y,
            'time_offset': current_time - self.start_time
        })
        self.session_data['timestamps'].append(current_time)
        
        # Update counter
        self.counter_label.config(text=f"Movements: {len(self.session_data['mouse_movements'])}")
        
        # Distance dx dy
        dx = current_pos[0] - self.last_position[0]
        dy = current_pos[1] - self.last_position[1]
        distance = math.sqrt(dx**2 + dy**2)
        self.session_data['distance_traveled'] += distance
        
        # Velocity and Aceleration
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
        """Calculate acceleration from recent positions"""
        """Calcula aceleração a partir de posições recentes"""
        if len(self.recent_positions) < 3:
            return 0
        
        positions = list(self.recent_positions)
        
        try:
            # Calculate velocities between consecutive points
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
            
            # Calculate acceleration
            time_delta = positions[2][1] - positions[1][1]
            if time_delta > 0:
                acceleration = (v2 - v1) / time_delta
                return acceleration
        except:
            pass
        
        return 0
    
    def on_checkbox_click(self):
        """Handle checkbox click event"""
        """Evento de clique do checkbox"""
        print("Checkbox clicked!")  # Debug

        
        if not self.checkbox_checked and self.checkbox_var.get():
            self.checkbox_checked = True
            click_time = time.time()
            
            # Record click time and position inside checkox frame
            self.session_data['click_data'] = {
                'time_to_click': click_time - self.start_time,
                'click_position': self.last_position if self.last_position else (0, 0),
                'click_timestamp': click_time
            }
            
            self.session_data['total_time'] = click_time - self.start_time
            
            print(f"Session data captured | Dados da sessão capturados")
            
            # Submit data after verification
            self.status_label.config(text="✓ Verificação completa, salvando dados...", fg='#4CAF50')
            self.submit_data()
            

        elif not self.checkbox_var.get():
            # If unchecked
            self.checkbox_checked = False
            self.status_label.config(text="Por favor, verifique o checkbox novamente", fg='#ff6666')
            self.is_tracking = True  
            
    def submit_data(self):
        """Save session data and show results"""
        """Salvar os dados da sessão"""
        # Verify is there is data
        if len(self.session_data['mouse_movements']) < 5:
            messagebox.showwarning(
                "Insufficient Data | Dados insuficientes",
                "Not enough mouse movement data captured, click on the window and try again | Dados de movimento insuficentes, por favor clique na janela e tente novamente"
            )
            return
        
        # Create data directory if it doesn't exist
        os.makedirs('captcha_data', exist_ok=True)
        
        # Calculate additional metrics
        self.session_data['metrics'] = self.calculate_metrics()
        
        # Save to JSON file
        filename = f"captcha_data/session_{self.session_data['session_id']}.json"
        with open(filename, 'w') as f:
            json.dump(self.session_data, f, indent=2)
        
        print(f"\nData saved to: {filename}")
        
        # Show summary
        metrics = self.session_data['metrics']
        summary = f"""Session {self.session_data['session_id']} Data Saved Successfully!"""
        
        messagebox.showinfo("Success!", summary)
        
        # Record another session poup
        response = messagebox.askyesno(
            "New session? | Nova sessão?",
            "Would you like to record new session? | Gostaria de registrar uma nvoa seção?"
        )
        
        if response:
            self.reset_session()
        else:
            self.root.quit()
        
    def calculate_metrics(self):
        """Calculate summary metrics from session data"""
        """Calcular as métricas da sessão"""
        velocities = self.session_data['velocities']
        accelerations = self.session_data['accelerations']
        
        return {
            'total_time': self.session_data['total_time'],
            'num_movements': len(self.session_data['mouse_movements']),
            'total_distance': self.session_data['distance_traveled'],
            'avg_velocity': np.mean(velocities) if velocities else 0,
            'max_velocity': np.max(velocities) if velocities else 0,
            'avg_acceleration': np.mean(accelerations) if accelerations else 0,
            'max_acceleration': np.max(accelerations) if accelerations else 0,
            'time_to_click': self.session_data['click_data'].get('time_to_click', 0)
        }
        
    def reset_session(self):
        """Reset for new session"""
        """Resetar para nova seção"""
        print("\n" + "="*50)
        print("Starting new session...")
        print("="*50)
        
        self.checkbox_var.set(False)
        self.checkbox_checked = False
        self.is_tracking = True
        self.start_time = time.time()
        self.last_position = None
        self.last_time = time.time()
        self.recent_positions.clear()
        
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
        



if __name__ == "__main__":
    print("="*50)
    print("CAPTCHA - Data Collection | Registro de Dados")
    print("="*50)
    print("\nStarting application... | Iniciando aplicação...\n")
    
    root = tk.Tk()
    app = Captcha(root)
    root.mainloop()
    
    print("\nApplication closed. Check the 'captcha_data' folder for saved sessions. | Aplicação fechada, registros salvos em 'captcha_data'")