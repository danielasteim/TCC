"""
data_collection.py

Janela de coleta de dados comportamentais (movimentação do mouse) em
um sistema CAPTCHA para futuro treinamento de dois modelos.

Inicialmente a ideia era coletar de vários usuários, por isso as descrições em inglês
No entanto por ética de dados, não será utilizado coleta de usuários externos

Versão 2 do coletor (COLLECTOR_VERSION):
  - Janela em tela cheia; o painel com o checkbox é sorteado numa posição
    aleatória da tela a cada sessão (antes era a janela 500x400 que mudava de lugar).
  - Todas as coordenadas (movimentos e clique) estão no referencial da janela,
    calculadas a partir de x_root/y_root. Na v1, event.x/event.y ficavam
    relativos ao widget sob o cursor (frame, checkbox), misturando referenciais.
  - O clique é a posição real do <ButtonPress-1> no checkbox, não a última
    posição de movimento.
  - O primeiro evento de movimento também é salvo.
  - Tamanho da janela e geometria do checkbox-alvo são salvos por sessão.
  - As sessões v2 são salvas em captcha_data_v2/, separadas das sessões v1.

Uso:
    python data_collection.py
    (Esc fecha a aplicação)

Dependências:
    pip install numpy tkinter
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import time
import json
import math
import os
import random
from datetime import datetime
from collections import deque
import numpy as np


COLLECTOR_VERSION = 2
OUTPUT_DIR = 'captcha_data_v2'

# Tamanho do painel que contém o checkbox
PANEL_W, PANEL_H = 400, 250
# Distância mínima entre o painel e a borda da tela
PANEL_MARGIN = 20


class Captcha:
    def __init__(self, root):
        self.root = root
        self.root.title("Behavioral CAPTCHA")
        self.root.configure(bg='#f0f0f0')

        # Dados da sessão
        self.session_data = self._new_session_data(session_user=[])

        # Recorda sessão do usuário
        name = simpledialog.askstring(
            title="Identificação",
            prompt="Digite seu nome e último sobrenome:",
            parent=self.root
        )
        if name is None:
            self.session_data["session_user"].append("UNKNOWN")
        else:
            self.session_data["session_user"].append(name.strip().upper())

        print(self.session_data["session_user"])

        self.root.attributes('-fullscreen', True)
        self.root.bind('<Escape>', lambda event: self.root.quit())
        self.root.update()

        # Variaveis
        self.start_time = time.time()
        self.last_position = None
        self.last_time = time.time()
        self.is_tracking = True
        self.checkbox_checked = False
        self.press_position = None
        self.sessions_count = 1

        self.recent_positions = deque(maxlen=6)

        self.setup_ui()

        print("CAPTCHA initialized | CAPTCHA iniciado")

    def _new_session_data(self, session_user):
        return {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S_%f'),
            'collector_version': COLLECTOR_VERSION,
            'session_user': session_user,
            'mouse_movements': [],
            'timestamps': [],
            'velocities': [],
            'accelerations': [],
            'click_data': {},
            'total_time': 0,
            'distance_traveled': 0
        }

    def _window_position(self, event):
        """Converts an event position to window coordinates, regardless of the widget under the cursor.

        Converte a posição do evento para o referencial da janela,
        independente do widget que está sob o cursor.
        """
        return (event.x_root - self.root.winfo_rootx(),
                event.y_root - self.root.winfo_rooty())

    def _random_panel_position(self):
        """Pick a random center for the panel, keeping it fully on screen.

        Sorteia o centro do painel do checkbox, mantendo-o inteiro na tela.
        """
        win_w = self.root.winfo_width()
        win_h = self.root.winfo_height()

        min_x = PANEL_W // 2 + PANEL_MARGIN
        min_y = PANEL_H // 2 + PANEL_MARGIN
        max_x = max(min_x, win_w - PANEL_W // 2 - PANEL_MARGIN)
        max_y = max(min_y, win_h - PANEL_H // 2 - PANEL_MARGIN)

        return random.randint(min_x, max_x), random.randint(min_y, max_y)

    def _record_geometry(self):
        """Save window size and checkbox geometry, in window coordinates.

        Salva o tamanho da janela e a geometria do checkbox-alvo, no
        referencial da janela.
        """
        self.root.update_idletasks()

        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        cb_x = self.checkbox.winfo_rootx() - root_x
        cb_y = self.checkbox.winfo_rooty() - root_y
        cb_w = self.checkbox.winfo_width()
        cb_h = self.checkbox.winfo_height()

        self.session_data['window_size'] = {
            'width': self.root.winfo_width(),
            'height': self.root.winfo_height()
        }
        self.session_data['screen_size'] = {
            'width': self.root.winfo_screenwidth(),
            'height': self.root.winfo_screenheight()
        }
        self.session_data['window_origin'] = {'x': root_x, 'y': root_y}
        self.session_data['target'] = {
            'x': cb_x,
            'y': cb_y,
            'width': cb_w,
            'height': cb_h,
            'center_x': cb_x + cb_w / 2,
            'center_y': cb_y + cb_h / 2
        }

    def setup_ui(self):
        """Sets up the UI environment including title, frames, checkbox, icon and status of verification.

        Realiza a configuração da interface incluindo titulo, frames, checkbox, icone e status da verificação.
        """


        for widget in self.root.winfo_children():
            widget.destroy()

        # Painel em posição aleatória da tela
        panel_x, panel_y = self._random_panel_position()
        main_frame = tk.Frame(self.root, bg='#ffffff', relief=tk.RAISED, borderwidth=2)
        main_frame.place(x=panel_x, y=panel_y, anchor='center', width=PANEL_W, height=PANEL_H)

        # Titulo
        title_label = tk.Label(
            main_frame,
            text="Verifique se é humano",
            font=('Arial', 14, 'bold'),
            bg='#ffffff',
            fg='#333333'
        )
        title_label.pack(pady=20)

        # Checkbox fram
        checkbox_frame = tk.Frame(main_frame, bg='#ffffff', relief=tk.GROOVE, borderwidth=2)
        checkbox_frame.pack(pady=30, padx=40, fill='x')

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
        self.checkbox.bind('<ButtonPress-1>', self.on_checkbox_press)

        # Icone
        icon_label = tk.Label(
            checkbox_frame,
            text="👻",
            font=('Arial', 24),
            bg='#ffffff'
        )
        icon_label.pack(side='right', padx=10)

        # Status da interação
        self.status_label = tk.Label(
            main_frame,
            text="Mova seu mouse e clique na checkbox para verificar sua autenticidade",
            font=('Arial', 9),
            bg='#ffffff',
            fg='#666666'
        )
        self.status_label.pack(pady=10)

        # Counter de movimentos
        self.counter_label = tk.Label(
            main_frame,
            text="Movimentos: 0",
            font=('Arial', 8),
            bg='#ffffff',
            fg='#999999'
        )
        self.counter_label.pack(pady=2)

        self._record_geometry()

        self.root.bind('<Motion>', self.track_mouse_movement)


    def track_mouse_movement(self, event):
        """Track mouse movements and calculate behavioral metrics.

        Calculo os movimentos do mouse e as métricas comportacionais.
        """
        if not self.is_tracking:
            return

        current_time = time.time()
        current_pos = self._window_position(event)

        # Posição e timestamp (inclui o primeiro evento)
        self.session_data['mouse_movements'].append({
            'x': current_pos[0],
            'y': current_pos[1],
            'time_offset': current_time - self.start_time
        })
        self.session_data['timestamps'].append(current_time)

        # Atualiza contador
        self.counter_label.config(
            text=f"Movimentos: {len(self.session_data['mouse_movements'])}"
        )

        if self.last_position is None:
            self.last_position = current_pos
            self.last_time = current_time
            self.recent_positions.append((current_pos, current_time))
            return

        # Distancia dx dy
        dx = current_pos[0] - self.last_position[0]
        dy = current_pos[1] - self.last_position[1]
        distance = math.sqrt(dx**2 + dy**2)
        self.session_data['distance_traveled'] += distance

        # Velocidade e aceleração
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
        """Calculate acceleration from recent positions.

        Calcula aceleração a partir de posições recentes.
        """
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
        except Exception:
            pass

        return 0


    def on_checkbox_press(self, event):
        """Store the real click position | Guarda a posição real do clique."""
        self.press_position = self._window_position(event)
        self.press_time = time.time()

    def on_checkbox_click(self):
        """Handle checkbox click event | Evento de clique do checkbox."""
        print("Checkbox clicked!")

        if not self.checkbox_checked and self.checkbox_var.get():
            self.checkbox_checked = True
            self.is_tracking = False
            click_time = self.press_time if self.press_position else time.time()
            click_position = self.press_position or self.last_position or (0, 0)

            self.session_data['click_data'] = {
                'time_to_click': click_time - self.start_time,
                'click_position': click_position,
                'click_timestamp': click_time
            }
            self.session_data['total_time'] = click_time - self.start_time

            print("Session data captured | Dados da sessão capturados")

            self.status_label.config(
                text="✓ Verificação completa, salvando dados...", fg='#4CAF50'
            )
            self.submit_data()

        elif not self.checkbox_var.get():
            self.checkbox_checked = False
            self.status_label.config(
                text="Por favor, verifique o checkbox novamente", fg='#ff6666'
            )
            self.is_tracking = True


    def submit_data(self):
        """Save session data and show results | Salvar os dados da sessão."""
        if len(self.session_data['mouse_movements']) < 5:
            messagebox.showwarning(
                "Insufficient Data | Dados insuficientes",
                "Not enough mouse movement data captured, click on the window and try again | "
                "Dados de movimento insuficentes, por favor clique na janela e tente novamente",
                parent=self.root
            )
            # Desmarca o checkbox e continua registrando na mesma sessão
            self.checkbox_var.set(False)
            self.checkbox_checked = False
            self.press_position = None
            self.is_tracking = True
            return

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        self.session_data['metrics'] = self.calculate_metrics()

        name = self.session_data['session_user'][0].replace(' ', '_')
        filename = (
            f"{OUTPUT_DIR}/session_{self.session_data['session_id']}_{name}.json"
        )
        with open(filename, 'w') as f:
            json.dump(self.session_data, f, indent=2)

        print(f"\nData saved to: {filename}")

        messagebox.showinfo(
            "Success!",
            f"Session {self.session_data['session_id']} Data Saved Successfully!",
            parent=self.root
        )

        response = messagebox.askyesno(
            "New session? | Nova sessão?",
            f"Would you like to record a new session? | Gostaria de registrar uma nova sessão?\n\n"
            f"Sessions so far: {self.sessions_count}",
            parent=self.root
        )

        if response:
            self.sessions_count += 1
            self.reset_session()
        else:
            self.root.quit()

    def calculate_metrics(self):
        """Calculate summary metrics from session data | Calcular as métricas da sessão."""
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
        """Reset for new session | Resetar para nova sessão."""
        print("\n" + "=" * 50)
        print("Starting new session...")
        print("=" * 50)

        self.checkbox_checked = False
        self.is_tracking = True
        self.press_position = None
        self.start_time = time.time()
        self.last_position = None
        self.last_time = time.time()
        self.recent_positions.clear()

        self.session_data = self._new_session_data(
            session_user=self.session_data["session_user"]
        )

        # Gera o painel em um local aleatório da tela
        self.setup_ui()




if __name__ == "__main__":
    print("=" * 50)
    print("CAPTCHA - Data Collection | Registro de Dados")
    print("=" * 50)
    print("\nStarting application... | Iniciando aplicação...\n")

    root = tk.Tk()
    app = Captcha(root)
    root.mainloop()

    print(
        f"\nApplication closed. Check the '{OUTPUT_DIR}' folder for saved sessions. | "
        f"Aplicação fechada, registros salvos em '{OUTPUT_DIR}'"
    )
