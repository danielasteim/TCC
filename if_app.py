"""
floresta_isolacao_app.py

Interface CAPTCHA comportamental que autentica o usuário em tempo real
usando o modelo de Floresta de Isolação já treinado.

Fluxo: captura os dados de mouse (mesma lógica de data_collection.py)
-> extrai as features (Features, feature_processor.py) -> aplica o
mesmo pipeline de pré-processamento do treino (Winsorizador +
StandardScaler) -> pontua com o modelo -> mostra o resultado da
autenticação na tela.

Uso:
    python floresta_isolacao_app.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import time
import math
import random
from datetime import datetime
from collections import deque
from pathlib import Path

import joblib
import pandas as pd

from features import Features

DIR_MODELOS = Path(__file__).parent / "modelos"


class CaptchaFlorestaIsolacao:
    # --- Tema azul ---
    COR_FUNDO_JANELA = '#E3F2FD'
    COR_FUNDO_CARTAO = '#FFFFFF'
    COR_PRIMARIA = '#1565C0'
    COR_TEXTO_TITULO = '#0D47A1'
    COR_TEXTO_SECUNDARIO = '#5C7A99'
    COR_APROVADO = '#1B5E20'
    COR_REJEITADO = '#B71C1C'

    NOME_MODELO = "Floresta de Isolação"
    ARQUIVO_MODELO = "floresta_isolacao.joblib"

    def __init__(self, root):
        self.root = root
        self.root.title(f"CAPTCHA Comportamental — {self.NOME_MODELO}")
        self.root.configure(bg=self.COR_FUNDO_JANELA)

        # --- Carrega o pipeline treinado ---
        self.extrator = Features()
        self.winsorizador = joblib.load(DIR_MODELOS / "winsorizador.joblib")
        self.escalonador = joblib.load(DIR_MODELOS / "escalonador.joblib")
        self.modelo = joblib.load(DIR_MODELOS / self.ARQUIVO_MODELO)
        self.colunas_features = joblib.load(DIR_MODELOS / "colunas_features.joblib")

        # --- Dados da sessão (mesma estrutura de data_collection.py) ---
        self.session_data = {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S_%f'),
            'session_user': ['TESTE_AUTENTICACAO'],
            'mouse_movements': [],
            'timestamps': [],
            'velocities': [],
            'accelerations': [],
            'click_data': {},
            'total_time': 0,
            'distance_traveled': 0,
        }

        self.start_time = time.time()
        self.last_position = None
        self.last_time = time.time()
        self.is_tracking = True
        self.checkbox_checked = False
        self.recent_positions = deque(maxlen=6)

        self._randomizar_posicao_janela()
        self.montar_interface()

    def _randomizar_posicao_janela(self):
        """Posiciona a janela em um local aleatório da tela (igual ao coletor)."""
        LARG_JANELA, ALT_JANELA = 500, 400
        self.root.update_idletasks()
        largura_tela = self.root.winfo_screenwidth()
        altura_tela = self.root.winfo_screenheight()
        max_x = max(0, largura_tela - LARG_JANELA)
        max_y = max(0, altura_tela - ALT_JANELA)
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        self.root.geometry(f"{LARG_JANELA}x{ALT_JANELA}+{x}+{y}")
        self.session_data['window_origin'] = {'x': x, 'y': y}

    def montar_interface(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        cartao = tk.Frame(self.root, bg=self.COR_FUNDO_CARTAO, relief=tk.RAISED, borderwidth=2)
        cartao.place(relx=0.5, rely=0.5, anchor='center', width=400, height=250)

        titulo = tk.Label(
            cartao,
            text="Verifique se é humano",
            font=('Arial', 14, 'bold'),
            bg=self.COR_FUNDO_CARTAO,
            fg=self.COR_TEXTO_TITULO,
        )
        titulo.pack(pady=20)

        quadro_checkbox = tk.Frame(cartao, bg=self.COR_FUNDO_CARTAO, relief=tk.GROOVE, borderwidth=2,
                                    highlightbackground=self.COR_PRIMARIA, highlightthickness=1)
        quadro_checkbox.pack(pady=30, padx=40, fill='x')

        self.checkbox_var = tk.BooleanVar()
        self.checkbox = tk.Checkbutton(
            quadro_checkbox,
            text="  Autenticação",
            variable=self.checkbox_var,
            font=('Arial', 12),
            bg=self.COR_FUNDO_CARTAO,
            activebackground=self.COR_FUNDO_CARTAO,
            command=self.ao_clicar_checkbox,
            cursor='hand2',
        )
        self.checkbox.pack(side='left', padx=10, pady=15)

        icone = tk.Label(quadro_checkbox, text="🌲", font=('Arial', 24), bg=self.COR_FUNDO_CARTAO)
        icone.pack(side='right', padx=10)

        self.status_label = tk.Label(
            cartao,
            text="Mova seu mouse e clique na checkbox para verificar sua autenticidade",
            font=('Arial', 9),
            bg=self.COR_FUNDO_CARTAO,
            fg=self.COR_TEXTO_SECUNDARIO,
            wraplength=340,
            justify='center',
        )
        self.status_label.pack(pady=10)

        self.contador_label = tk.Label(
            cartao,
            text="Movimentos: 0",
            font=('Arial', 8),
            bg=self.COR_FUNDO_CARTAO,
            fg='#999999',
        )
        self.contador_label.pack(pady=2)

        self.root.bind('<Motion>', self.rastrear_movimento_mouse)

    def rastrear_movimento_mouse(self, event):
        """Mesma lógica de captura de data_collection.py."""
        if not self.is_tracking:
            return

        agora = time.time()
        posicao_atual = (event.x, event.y)

        if self.last_position is None:
            self.last_position = posicao_atual
            self.last_time = agora
            self.recent_positions.append((posicao_atual, agora))
            return

        self.session_data['mouse_movements'].append({
            'x': event.x,
            'y': event.y,
            'time_offset': agora - self.start_time,
        })
        self.session_data['timestamps'].append(agora)

        self.contador_label.config(text=f"Movimentos: {len(self.session_data['mouse_movements'])}")

        dx = posicao_atual[0] - self.last_position[0]
        dy = posicao_atual[1] - self.last_position[1]
        distancia = math.sqrt(dx ** 2 + dy ** 2)
        self.session_data['distance_traveled'] += distancia

        delta_t = agora - self.last_time
        if delta_t > 0:
            velocidade = distancia / delta_t
            self.session_data['velocities'].append(velocidade)
            self.recent_positions.append((posicao_atual, agora))
            if len(self.recent_positions) >= 3:
                aceleracao = self.calcular_aceleracao()
                self.session_data['accelerations'].append(aceleracao)

        self.last_position = posicao_atual
        self.last_time = agora

    def calcular_aceleracao(self):
        """Mesma lógica de data_collection.py (mantida só por completude/compatibilidade)."""
        if len(self.recent_positions) < 3:
            return 0
        posicoes = list(self.recent_positions)
        try:
            dt1 = posicoes[1][1] - posicoes[0][1]
            dt2 = posicoes[2][1] - posicoes[1][1]
            if dt1 == 0 or dt2 == 0:
                return 0
            v1x = (posicoes[1][0][0] - posicoes[0][0][0]) / dt1
            v1y = (posicoes[1][0][1] - posicoes[0][0][1]) / dt1
            v1 = math.sqrt(v1x ** 2 + v1y ** 2)
            v2x = (posicoes[2][0][0] - posicoes[1][0][0]) / dt2
            v2y = (posicoes[2][0][1] - posicoes[1][0][1]) / dt2
            v2 = math.sqrt(v2x ** 2 + v2y ** 2)
            dt = posicoes[2][1] - posicoes[1][1]
            if dt > 0:
                return (v2 - v1) / dt
        except Exception:
            pass
        return 0

    def ao_clicar_checkbox(self):
        if not self.checkbox_checked and self.checkbox_var.get():
            self.checkbox_checked = True
            momento_clique = time.time()

            self.session_data['click_data'] = {
                'time_to_click': momento_clique - self.start_time,
                'click_position': self.last_position if self.last_position else (0, 0),
                'click_timestamp': momento_clique,
            }
            self.session_data['total_time'] = momento_clique - self.start_time

            self.is_tracking = False
            self.autenticar()

        elif not self.checkbox_var.get():
            self.checkbox_checked = False
            self.status_label.config(text="Por favor, verifique o checkbox novamente", fg=self.COR_REJEITADO)
            self.is_tracking = True

    def autenticar(self):
        """Extrai as features e pontua a sessão com o modelo treinado."""
        if len(self.session_data['mouse_movements']) < 5:
            messagebox.showwarning(
                "Dados insuficientes",
                "Movimento de mouse insuficiente. Clique na janela e tente novamente.",
            )
            self.reiniciar_sessao()
            return

        vetor = self.extrator.transform(self.session_data)
        valores = vetor[self.colunas_features].values.reshape(1, -1)
        X = pd.DataFrame(valores, columns=self.colunas_features)

        X_winsorizado = self.winsorizador.transformar(X)
        X_escalonado = self.escalonador.transform(X_winsorizado)

        predicao = self.modelo.predict(X_escalonado)[0]   # 1 = humano, -1 = anômalo
        score = self.modelo.decision_function(X_escalonado)[0]

        if predicao == 1:
            mensagem = f"✓ Autenticado como humano\n({self.NOME_MODELO} — score: {score:.3f})"
            cor = self.COR_APROVADO
        else:
            mensagem = f"✗ Rejeitado — comportamento anômalo\n({self.NOME_MODELO} — score: {score:.3f})"
            cor = self.COR_REJEITADO

        self.status_label.config(text=mensagem, fg=cor)

        continuar = messagebox.askyesno("Resultado", f"{mensagem}\n\nTestar novamente?")
        if continuar:
            self.reiniciar_sessao()
        else:
            self.root.quit()

    def reiniciar_sessao(self):
        self.checkbox_checked = False
        self.is_tracking = True
        self.start_time = time.time()
        self.last_position = None
        self.last_time = time.time()
        self.recent_positions.clear()

        self.session_data = {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S_%f'),
            'session_user': ['TESTE_AUTENTICACAO'],
            'mouse_movements': [],
            'timestamps': [],
            'velocities': [],
            'accelerations': [],
            'click_data': {},
            'total_time': 0,
            'distance_traveled': 0,
        }
        self._randomizar_posicao_janela()
        self.montar_interface()


if __name__ == "__main__":
    root = tk.Tk()
    app = CaptchaFlorestaIsolacao(root)
    root.mainloop()
