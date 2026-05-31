"""
bot.py
=======
Simulador de bots para testar a detecção do CAPTCHA comportamental.

Gera sessões sintéticas de 5 tipos de bot, extrai as mesmas features
do captcha_feature_pipeline.py e testa contra os modelos treinados.

Uso:
    python bot.py

Dependências:
    pip install numpy pandas scikit-learn joblib
"""

import math
import random
import time
import json
import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Importa o pipeline de features — deve estar na mesma pasta
from captcha_feature_pipeline import extract_features


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ──────────────────────────────────────────────────────────────────────────────

MODELS_DIR  = "models"
WINDOW      = {"x": 800, "y": 600}   # janela simulada para os bots


# ──────────────────────────────────────────────────────────────────────────────
# GERAÇÃO DAS SESSÕES DE BOT
# ──────────────────────────────────────────────────────────────────────────────

def _build_session(session_id, movements, timestamps, click_pos, ttc):
    """
    Monta o dict de sessão no mesmo formato dos JSONs coletados,
    calculando velocities e accelerations internamente.
    """
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

    t0 = timestamps[0]
    return {
        "session_id":      session_id,
        "session_user":    ["bot"],
        "mouse_movements": movements,
        "timestamps":      timestamps,
        "velocities":      velocities,
        "accelerations":   accelerations,
        "click_data": {
            "time_to_click":    ttc,
            "click_position":   list(click_pos),
            "click_timestamp":  t0 + ttc,
        },
        "total_time":       ttc,
        "distance_traveled": sum(
            math.sqrt((movements[i]["x"]-movements[i-1]["x"])**2 +
                      (movements[i]["y"]-movements[i-1]["y"])**2)
            for i in range(1, n)
        ),
        "window_origin": WINDOW,
    }


class BotSimulator:
    """
    Simula 5 tipos de bots para testar a detecção comportamental do CAPTCHA.

    Tipos:
      linear       — linha reta, velocidade constante
      fast         — velocidade sobre-humana, poucos pontos
      perfect      — timing exato, curva Bézier perfeita
      slow         — lento demais, intervalos regulares
      sophisticated — tenta imitar humano com ruído e curvas
    """

    def __init__(self):
        self.bot_types = {
            "linear":       self.linear_bot,
            "fast":         self.fast_bot,
            "perfect":      self.perfect_bot,
            "slow":         self.slow_bot,
            "sophisticated": self.sophisticated_bot,
        }

    # ── Bot 1: Linear ─────────────────────────────────────────────────────────

    def linear_bot(self, start, target):
        """
        Linha reta, velocidade constante, intervalos idênticos de 10ms.
        Sinais: ratio_dist_desl ≈ 1.0, cv_temporal ≈ 0, jerk ≈ 0.
        """
        print("\n🤖 LINEAR BOT — linha reta, sem curvas, timing mecânico")
        n   = 15
        t0  = time.time()
        mov, ts = [], []

        for i in range(n):
            frac = i / (n - 1)
            mov.append({"x": int(start[0] + frac*(target[0]-start[0])),
                        "y": int(start[1] + frac*(target[1]-start[1])),
                        "time_offset": i * 0.010})
            ts.append(t0 + i * 0.010)

        return _build_session(
            f"bot_linear_{datetime.now().strftime('%H%M%S')}",
            mov, ts, target, ttc=0.15
        )

    # ── Bot 2: Fast ───────────────────────────────────────────────────────────

    def fast_bot(self, start, target):
        """
        Velocidade sobre-humana, 5ms entre pontos, jitter mínimo.
        Sinais: time_to_click impossível (< 0.05s), vel muito alta, cv_passos baixo.
        """
        print("\n🤖 FAST BOT — velocidade sobre-humana")
        n   = 8
        t0  = time.time()
        mov, ts = [], []

        for i in range(n):
            frac = i / (n - 1)
            mov.append({"x": int(start[0] + frac*(target[0]-start[0])) + random.randint(-2, 2),
                        "y": int(start[1] + frac*(target[1]-start[1])) + random.randint(-2, 2),
                        "time_offset": i * 0.005})
            ts.append(t0 + i * 0.005)

        return _build_session(
            f"bot_fast_{datetime.now().strftime('%H%M%S')}",
            mov, ts, target, ttc=0.04
        )

    # ── Bot 3: Perfect ────────────────────────────────────────────────────────

    def perfect_bot(self, start, target):
        """
        Curva Bézier perfeita, intervalos exatamente iguais de 20ms.
        Sinais: cv_temporal ≈ 0, jerk ≈ 0, d_centro_clique ≈ 0.
        """
        print("\n🤖 PERFECT BOT — Bézier perfeita, timing exato")
        n   = 25
        t0  = time.time()
        ctrl = ((start[0]+target[0])/2, (start[1]+target[1])/2 - 30)
        mov, ts = [], []

        for i in range(n):
            frac = i / (n - 1)
            x = (1-frac)**2*start[0] + 2*(1-frac)*frac*ctrl[0] + frac**2*target[0]
            y = (1-frac)**2*start[1] + 2*(1-frac)*frac*ctrl[1] + frac**2*target[1]
            mov.append({"x": int(x), "y": int(y), "time_offset": i * 0.020})
            ts.append(t0 + i * 0.020)

        return _build_session(
            f"bot_perfect_{datetime.now().strftime('%H%M%S')}",
            mov, ts, target, ttc=0.50
        )

    # ── Bot 4: Slow ───────────────────────────────────────────────────────────

    def slow_bot(self, start, target):
        """
        Muitos pontos, intervalos de 50ms uniformes, ttc = 5s.
        Sinais: cv_temporal ≈ 0, n_movimentos alto e regular, ttc outlier.
        """
        print("\n🤖 SLOW BOT — lento demais, intervalos regulares")
        n   = 100
        t0  = time.time()
        mov, ts = [], []

        for i in range(n):
            frac = i / (n - 1)
            mov.append({"x": int(start[0] + frac*(target[0]-start[0])),
                        "y": int(start[1] + frac*(target[1]-start[1])),
                        "time_offset": i * 0.050})
            ts.append(t0 + i * 0.050)

        return _build_session(
            f"bot_slow_{datetime.now().strftime('%H%M%S')}",
            mov, ts, target, ttc=5.0
        )

    # ── Bot 5: Sophisticated ──────────────────────────────────────────────────

    def sophisticated_bot(self, start, target):
        """
        Tenta imitar humano: Bézier com ruído, timing variável.
        Mas: variância temporal muito baixa, sem retrocessos, entropia baixa.
        """
        print("\n🤖 SOPHISTICATED BOT — imita humano, mais difícil de detectar")
        n   = random.randint(40, 60)
        t0  = time.time()
        ctrl = (
            (start[0]+target[0])/2 + random.randint(-40, 40),
            (start[1]+target[1])/2 + random.randint(-40, 40),
        )
        mov, ts = [], []
        t_acc = 0.0

        for i in range(n):
            frac = i / (n - 1)
            x = (1-frac)**2*start[0] + 2*(1-frac)*frac*ctrl[0] + frac**2*target[0]
            y = (1-frac)**2*start[1] + 2*(1-frac)*frac*ctrl[1] + frac**2*target[1]
            x += random.randint(-5, 5)
            y += random.randint(-5, 5)

            # Intervalo ligeiramente variável mas MUITO menos que humano (cv ≈ 0.05)
            interval = random.uniform(0.015, 0.025)
            t_acc   += interval

            mov.append({"x": int(x), "y": int(y), "time_offset": t_acc})
            ts.append(t0 + t_acc)

        ttc = random.uniform(0.9, 1.5)
        return _build_session(
            f"bot_sophisticated_{datetime.now().strftime('%H%M%S')}",
            mov, ts, target, ttc=ttc
        )


# ──────────────────────────────────────────────────────────────────────────────
# TESTE CONTRA OS MODELOS
# ──────────────────────────────────────────────────────────────────────────────

def load_models(directory=MODELS_DIR):
    """Carrega modelos treinados pelo captcha_ml_models.py."""
    try:
        ocsvm    = joblib.load(f"{directory}/ocsvm_model.pkl")
        iso      = joblib.load(f"{directory}/isolation_forest_model.pkl")
        scaler   = joblib.load(f"{directory}/scaler.pkl")
        feat_names = joblib.load(f"{directory}/feature_names.pkl")
        return ocsvm, iso, scaler, feat_names
    except FileNotFoundError:
        print(f"\n❌ Modelos não encontrados em '{directory}/'.")
        print("   Execute captcha_ml_models.py primeiro para treinar os modelos.")
        return None, None, None, None


def test_session(session_data, ocsvm, iso_forest, scaler, feature_names):
    """
    Extrai as features da sessão do bot usando o mesmo pipeline
    do captcha_feature_pipeline.py e testa contra os modelos.
    """
    # Extrai features com o mesmo pipeline usado no treino
    feats = extract_features(session_data)

    # Monta vetor na mesma ordem das features do treino
    row = pd.Series({col: feats.get(col, 0.0) for col in feature_names})
    row.replace([np.inf, -np.inf], np.nan, inplace=True)
    row.fillna(0.0, inplace=True)

    X = scaler.transform(row.values.reshape(1, -1))

    svm_pred  = ocsvm.predict(X)[0]
    svm_score = ocsvm.score_samples(X)[0]
    if_pred   = iso_forest.predict(X)[0]
    if_score  = iso_forest.score_samples(X)[0]

    svm_detected = (svm_pred == -1)
    if_detected  = (if_pred  == -1)

    # Imprime diagnóstico das features mais discriminativas
    print(f"\n  Features extraídas (principais):")
    key_feats = [
        "time_to_click", "ratio_dist_desl", "cv_temporal",
        "taxa_retrocesso", "jerk_std", "entropia_vel_norm",
        "ratio_desaceleracao", "desvio_lateral_max",
    ]
    for f in key_feats:
        val = feats.get(f, 0.0)
        print(f"    {f:<28} {val:.5f}")

    print(f"\n  One-Class SVM    : {'🚫 BOT DETECTADO' if svm_detected else '✅ Passou como humano'}"
          f"  (score: {svm_score:.4f})")
    print(f"  Isolation Forest : {'🚫 BOT DETECTADO' if if_detected else '✅ Passou como humano'}"
          f"  (score: {if_score:.4f})")

    return {
        "svm_detected": svm_detected,
        "if_detected":  if_detected,
        "svm_score":    svm_score,
        "if_score":     if_score,
    }


# ──────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("BOT SIMULATOR — CAPTCHA BEHAVIORAL DETECTION TEST")
    print("=" * 70)

    ocsvm, iso, scaler, feat_names = load_models(MODELS_DIR)
    if ocsvm is None:
        exit(1)

    print(f"\nModelos carregados. Features esperadas: {len(feat_names)}")

    simulator = BotSimulator()
    start_pos  = (100, 300)
    target_pos = (400, 200)

    results = {}

    for bot_name, bot_func in simulator.bot_types.items():
        print("\n" + "─" * 70)
        session = bot_func(start_pos, target_pos)
        result  = test_session(session, ocsvm, iso, scaler, feat_names)
        results[bot_name] = result

    # ── Resumo final ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESUMO DE DETECÇÃO")
    print("=" * 70)
    print(f"\n  {'Tipo de Bot':<22} {'SVM':^12} {'IF':^12} {'Status'}")
    print("  " + "─" * 62)

    for name, r in results.items():
        svm_s  = "🚫 detectado" if r["svm_detected"] else "✅ passou"
        if_s   = "🚫 detectado" if r["if_detected"]  else "✅ passou"

        if r["svm_detected"] and r["if_detected"]:
            status = "✅ Ambos detectaram"
        elif r["svm_detected"] or r["if_detected"]:
            status = "⚠️  Um detectou"
        else:
            status = "❌ Nenhum detectou"

        print(f"  {name:<22} {svm_s:^12}  {if_s:^12}  {status}")

    total    = len(results)
    svm_det  = sum(1 for r in results.values() if r["svm_detected"])
    if_det   = sum(1 for r in results.values() if r["if_detected"])
    both_det = sum(1 for r in results.values() if r["svm_detected"] and r["if_detected"])

    print("\n" + "─" * 70)
    print(f"  Taxa SVM             : {svm_det}/{total}  ({svm_det/total:.0%})")
    print(f"  Taxa Isolation Forest: {if_det}/{total}  ({if_det/total:.0%})")
    print(f"  Taxa consenso (ambos): {both_det}/{total}  ({both_det/total:.0%})")
    print("=" * 70)
