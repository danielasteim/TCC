"""
if.py
======
CAPTCHA comportamental usando Isolation Forest.
Tema azul.

Usa captcha_feature_pipeline.extract_features() para garantir
que as features sejam idênticas às usadas no treinamento.
"""

import tkinter as tk
import time
import math
import numpy as np
import joblib
import pandas as pd
from datetime import datetime
from captcha_feature_pipeline import extract_features

MODELS_DIR = "models"


class ProductionCaptchaIF:
    """CAPTCHA com Isolation Forest — tema azul."""

    # ── Tema visual ───────────────────────────────────────────────────────────
    THEME = {
        "bg_root":      "#e6f2ff",
        "bg_frame":     "#cce5ff",
        "bg_badge":     "#0066cc",
        "fg_title":     "#003366",
        "badge_text":   "🔵 ISOLATION FOREST",
        "model_file":   "isolation_forest_model.pkl",
        "window_title": "CAPTCHA Verification — Isolation Forest",
    }

    def __init__(self, root, model_dir=MODELS_DIR):
        self.root = root
        self.root.title(self.THEME["window_title"])
        self.root.geometry("500x450")
        self.root.configure(bg=self.THEME["bg_root"])

        self._load_model(model_dir)
        self._reset_session()
        self._setup_ui()

    # ── Carregamento do modelo ────────────────────────────────────────────────

    def _load_model(self, model_dir):
        try:
            self.model         = joblib.load(f"{model_dir}/isolation_forest_model.pkl")
            self.scaler        = joblib.load(f"{model_dir}/scaler.pkl")
            self.feature_names = joblib.load(f"{model_dir}/feature_names.pkl")
            print(f"✅ Isolation Forest carregado de '{model_dir}/'")
            print(f"   Features esperadas: {len(self.feature_names)}")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            print("   Modo dummy ativado — sempre aprovará.")
            self.model = None
            self.feature_names = []

    # ── Estado da sessão ──────────────────────────────────────────────────────

    def _reset_session(self):
        self.session_data = {
            "session_id":      datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            "session_user":    ["captcha_user"],
            "mouse_movements": [],
            "timestamps":      [],
            "velocities":      [],
            "accelerations":   [],
            "click_data":      {},
            "total_time":      0,
            "distance_traveled": 0,
            "window_origin":   {"x": 500, "y": 450},
        }
        self.start_time       = time.time()
        self.last_position    = None
        self.last_time        = self.start_time
        self.is_tracking      = True
        self.checkbox_checked = False

    def _finalize_session(self, click_time):
        """
        Calcula velocidades e acelerações a partir dos eventos gravados
        e registra o clique — mesmo formato dos JSONs coletados.
        """
        movements  = self.session_data["mouse_movements"]
        timestamps = self.session_data["timestamps"]
        n = len(movements)

        velocities    = []
        accelerations = []

        for i in range(1, n):
            dx = movements[i]["x"] - movements[i-1]["x"]
            dy = movements[i]["y"] - movements[i-1]["y"]
            dt = max(timestamps[i] - timestamps[i-1], 1e-9)
            velocities.append(math.sqrt(dx**2 + dy**2) / dt)

        for i in range(1, len(velocities)):
            dt = max(timestamps[i+1] - timestamps[i], 1e-9)
            accelerations.append((velocities[i] - velocities[i-1]) / dt)

        self.session_data["velocities"]    = velocities
        self.session_data["accelerations"] = accelerations
        self.session_data["total_time"]    = click_time - self.start_time
        self.session_data["click_data"]    = {
            "time_to_click":   click_time - self.start_time,
            "click_position":  list(self.last_position) if self.last_position else [0, 0],
            "click_timestamp": click_time,
        }

    # ── Interface gráfica ─────────────────────────────────────────────────────

    def _setup_ui(self):
        bg = self.THEME["bg_frame"]

        main_frame = tk.Frame(
            self.root, bg=bg, relief=tk.RAISED, borderwidth=3
        )
        main_frame.place(relx=0.5, rely=0.5, anchor="center", width=420, height=320)

        tk.Label(
            main_frame, text=self.THEME["badge_text"],
            font=("Arial", 10, "bold"),
            bg=self.THEME["bg_badge"], fg="white", padx=10, pady=5
        ).pack(pady=(10, 5))

        tk.Label(
            main_frame, text="Security Verification",
            font=("Arial", 14, "bold"),
            bg=bg, fg=self.THEME["fg_title"]
        ).pack(pady=10)

        cb_frame = tk.Frame(main_frame, bg="white", relief=tk.GROOVE, borderwidth=2)
        cb_frame.pack(pady=20, padx=40, fill="x")

        self.checkbox_var = tk.BooleanVar()
        tk.Checkbutton(
            cb_frame, text="  I'm not a robot",
            variable=self.checkbox_var,
            font=("Arial", 12), bg="white",
            command=self._on_checkbox_click, cursor="hand2"
        ).pack(side="left", padx=10, pady=15)

        tk.Label(cb_frame, text="🤖", font=("Arial", 24), bg="white").pack(
            side="right", padx=10
        )

        self.status_label = tk.Label(
            main_frame,
            text="Mova o mouse naturalmente e marque a caixa",
            font=("Arial", 9), bg=bg, fg="#666666"
        )
        self.status_label.pack(pady=10)

        self.counter_label = tk.Label(
            main_frame, text="Movimentos: 0",
            font=("Arial", 8), bg=bg, fg="#999999"
        )
        self.counter_label.pack(pady=2)

        self.result_frame = tk.Frame(main_frame, bg=bg)
        self.result_label = tk.Label(
            self.result_frame, text="",
            font=("Arial", 11, "bold"), bg=bg
        )
        self.result_label.pack()

        self.root.bind("<Motion>", self._track_mouse)

    # ── Rastreamento do mouse ─────────────────────────────────────────────────

    def _track_mouse(self, event):
        if not self.is_tracking:
            return

        current_time = time.time()

        if self.last_position is None:
            self.last_position = (event.x, event.y)
            self.last_time     = current_time
            return

        self.session_data["mouse_movements"].append({
            "x": event.x, "y": event.y,
            "time_offset": current_time - self.start_time,
        })
        self.session_data["timestamps"].append(current_time)

        dx = event.x - self.last_position[0]
        dy = event.y - self.last_position[1]
        self.session_data["distance_traveled"] += math.sqrt(dx**2 + dy**2)

        self.last_position = (event.x, event.y)
        self.last_time     = current_time

        n = len(self.session_data["mouse_movements"])
        self.counter_label.config(text=f"Movimentos: {n}")

    # ── Clique no checkbox ────────────────────────────────────────────────────

    def _on_checkbox_click(self):
        if self.checkbox_checked or not self.checkbox_var.get():
            return

        self.checkbox_checked = True
        self.is_tracking      = False
        click_time            = time.time()

        self._finalize_session(click_time)
        self._verify()

    # ── Verificação ───────────────────────────────────────────────────────────

    def _verify(self):
        if len(self.session_data["mouse_movements"]) < 5:
            self._show_result(False, 0.0, motivo="Dados insuficientes")
            return

        feats = extract_features(self.session_data)

        if self.model is None:
            self._show_result(True, 0.0)
            return

        row = pd.Series({col: feats.get(col, 0.0) for col in self.feature_names})
        row.replace([np.inf, -np.inf], np.nan, inplace=True)
        row.fillna(0.0, inplace=True)

        X        = self.scaler.transform(row.values.reshape(1, -1))
        pred     = self.model.predict(X)[0]
        score    = self.model.score_samples(X)[0]
        is_human = (pred == 1)

        print(f"\n{'='*50}")
        print(f"Isolation Forest — Resultado:")
        print(f"  Predição  : {'HUMANO' if is_human else 'BOT/ANOMALIA'}")
        print(f"  Score     : {score:.4f}")
        print(f"  Tempo     : {self.session_data['total_time']:.2f}s")
        print(f"  Movimentos: {len(self.session_data['mouse_movements'])}")
        key = ["time_to_click","ratio_dist_desl","cv_temporal",
               "taxa_retrocesso","jerk_std","entropia_vel_norm"]
        for k in key:
            if k in feats:
                print(f"  {k:<28}: {feats[k]:.5f}")
        print("=" * 50)

        self._show_result(is_human, score)

    # ── Exibição do resultado ─────────────────────────────────────────────────

    def _show_result(self, is_human, score, motivo=None):
        self.result_frame.pack(pady=10)
        if is_human:
            self.result_label.config(
                text=f"✓ Verificado: Humano\nConfiança: {abs(score):.4f}",
                fg="#00aa00"
            )
            self.status_label.config(text="Verificação bem-sucedida!", fg="#00aa00")
        else:
            msg = motivo or f"Score: {score:.4f}"
            self.result_label.config(
                text=f"✗ Verificação falhou\n{msg}", fg="#cc0000"
            )
            self.status_label.config(text="Comportamento suspeito detectado", fg="#cc0000")


if __name__ == "__main__":
    print("=" * 50)
    print("CAPTCHA — ISOLATION FOREST")
    print("=" * 50)
    root = tk.Tk()
    app  = ProductionCaptchaIF(root, model_dir=MODELS_DIR)
    root.mainloop()
