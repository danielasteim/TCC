"""
mouse_behavior_preprocessor.py

Classe de pré-processamento para o CAPTCHA comportamental baseado em
dinâmica de mouse (Isolation Forest + One-Class SVM).

Responsabilidades desta classe:
  1. Normalizar TODAS as grandezas com dimensão espacial de uma sessão
     (posição, velocidade, aceleração, distância percorrida e posição
     de clique) pela diagonal da resolução da janela daquela sessão.
     Isso torna os valores adimensionais e comparáveis entre sessões
     capturadas em dispositivos com resoluções diferentes.
  2. Extrair, a partir da sessão normalizada, um vetor de features
     agregadas de tamanho fixo — pronto para alimentar `.fit()` /
     `.predict()` / `.decision_function()` do scikit-learn.

A mesma classe é usada em dois momentos:
  - Offline, para montar o dataset de treino a partir das ~10.000
    sessões já capturadas (transform_batch / transform_batch_from_dir).
  - Online, em produção, para pré-processar UMA sessão recém-capturada
    antes de pontuá-la com os modelos já treinados (transform).
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


class MouseBehaviorPreprocessor:
    """Normaliza sessões de captura de mouse e extrai features para os modelos."""

    # Ordem fixa das colunas do vetor de features. Mantenha essa ordem
    # estável entre o treino e a inferência em produção.
    FEATURE_NAMES = [
        "num_movements",
        "total_time",
        "time_to_click",
        "total_distance_norm",
        "straightness_ratio",
        "avg_velocity_norm",
        "std_velocity_norm",
        "max_velocity_norm",
        "avg_acceleration_norm",
        "std_acceleration_norm",
        "max_acceleration_norm",
        "click_offset_norm",
        "avg_time_between_movements",
        "std_time_between_movements",
    ]

    def __init__(self, eps: float = 1e-9):
        # eps evita divisão por zero em sessões degeneradas (ex.: diagonal
        # nula, sessão sem nenhum movimento antes do clique etc.)
        self.eps = eps

    # ------------------------------------------------------------------
    # 1) Normalização espacial pela diagonal da janela
    # ------------------------------------------------------------------
    def _compute_diagonal(self, session: dict) -> float:
        width = session["window_origin"]["x"]
        height = session["window_origin"]["y"]
        diagonal = math.hypot(width, height)
        return diagonal if diagonal > self.eps else self.eps

    def normalize_session(self, session: dict) -> dict:
        """
        Retorna uma CÓPIA da sessão com todas as grandezas espaciais
        divididas pela diagonal da janela daquela sessão:
          - mouse_movements[i].x / .y
          - velocities[i]
          - accelerations[i]
          - distance_traveled
          - click_data.click_position
          - metrics.total_distance / avg_velocity / max_velocity /
            avg_acceleration / max_acceleration

        Grandezas temporais (timestamps, total_time, time_to_click,
        num_movements) NÃO são alteradas aqui — elas não têm dimensão
        espacial.
        """
        diagonal = self._compute_diagonal(session)
        s = json.loads(json.dumps(session))  # deep copy simples (dict só com tipos básicos)

        for m in s.get("mouse_movements", []):
            m["x"] = m["x"] / diagonal
            m["y"] = m["y"] / diagonal

        s["velocities"] = [v / diagonal for v in s.get("velocities", [])]
        s["accelerations"] = [a / diagonal for a in s.get("accelerations", [])]

        if "distance_traveled" in s:
            s["distance_traveled"] = s["distance_traveled"] / diagonal

        click_data = s.get("click_data", {})
        if "click_position" in click_data:
            cx, cy = click_data["click_position"]
            click_data["click_position"] = [cx / diagonal, cy / diagonal]

        metrics = s.get("metrics", {})
        for key in ("total_distance", "avg_velocity", "max_velocity",
                    "avg_acceleration", "max_acceleration"):
            if key in metrics:
                metrics[key] = metrics[key] / diagonal

        s["_diagonal"] = diagonal  # guardado apenas para auditoria/debug
        return s

    # ------------------------------------------------------------------
    # 2) Extração de features agregadas (vetor de tamanho fixo)
    # ------------------------------------------------------------------
    def extract_features(self, normalized_session: dict) -> dict:
        """
        Recebe uma sessão JÁ NORMALIZADA (saída de normalize_session) e
        retorna um dict de features agregadas, na ordem de FEATURE_NAMES.
        """
        movements = normalized_session.get("mouse_movements", [])
        velocities = np.asarray(normalized_session.get("velocities", []), dtype=float)
        accelerations = np.asarray(normalized_session.get("accelerations", []), dtype=float)
        timestamps = np.asarray(normalized_session.get("timestamps", []), dtype=float)

        num_movements = len(movements)
        total_time = float(normalized_session.get("total_time", 0.0))
        time_to_click = float(normalized_session.get("click_data", {}).get("time_to_click", 0.0))
        total_distance_norm = float(normalized_session.get("distance_traveled", 0.0))

        # Razão de "linearidade" do trajeto: distância em linha reta entre o
        # primeiro e o último ponto, dividida pela distância total percorrida.
        # É a razão entre duas grandezas já normalizadas, então o resultado
        # já é adimensional por construção (não precisa normalizar de novo).
        # 1.0 = caminho perfeitamente reto; valores baixos = caminho tortuoso.
        if num_movements >= 2 and total_distance_norm > self.eps:
            x0, y0 = movements[0]["x"], movements[0]["y"]
            x1, y1 = movements[-1]["x"], movements[-1]["y"]
            straight_dist = math.hypot(x1 - x0, y1 - y0)
            straightness_ratio = straight_dist / (total_distance_norm + self.eps)
        else:
            straightness_ratio = 0.0

        avg_velocity_norm = float(np.mean(velocities)) if velocities.size else 0.0
        std_velocity_norm = float(np.std(velocities)) if velocities.size else 0.0
        max_velocity_norm = float(np.max(velocities)) if velocities.size else 0.0

        avg_acceleration_norm = float(np.mean(accelerations)) if accelerations.size else 0.0
        std_acceleration_norm = float(np.std(accelerations)) if accelerations.size else 0.0
        max_acceleration_norm = float(np.max(np.abs(accelerations))) if accelerations.size else 0.0

        # Distância (já normalizada) entre o último ponto do mouse antes do
        # clique e a posição real do clique. Humanos clicam praticamente
        # onde o cursor já está; cliques "teleportados" são suspeitos.
        click_pos = normalized_session.get("click_data", {}).get("click_position")
        if click_pos is not None and num_movements >= 1:
            last_x, last_y = movements[-1]["x"], movements[-1]["y"]
            click_offset_norm = math.hypot(click_pos[0] - last_x, click_pos[1] - last_y)
        else:
            click_offset_norm = 0.0

        # Intervalos de tempo entre eventos sucessivos de movimento.
        if timestamps.size >= 2:
            diffs = np.diff(timestamps)
            avg_time_between_movements = float(np.mean(diffs))
            std_time_between_movements = float(np.std(diffs))
        else:
            avg_time_between_movements = 0.0
            std_time_between_movements = 0.0

        return {
            "num_movements": num_movements,
            "total_time": total_time,
            "time_to_click": time_to_click,
            "total_distance_norm": total_distance_norm,
            "straightness_ratio": straightness_ratio,
            "avg_velocity_norm": avg_velocity_norm,
            "std_velocity_norm": std_velocity_norm,
            "max_velocity_norm": max_velocity_norm,
            "avg_acceleration_norm": avg_acceleration_norm,
            "std_acceleration_norm": std_acceleration_norm,
            "max_acceleration_norm": max_acceleration_norm,
            "click_offset_norm": click_offset_norm,
            "avg_time_between_movements": avg_time_between_movements,
            "std_time_between_movements": std_time_between_movements,
        }

    # ------------------------------------------------------------------
    # 3) API de alto nível (use estes métodos no seu código)
    # ------------------------------------------------------------------
    def transform(self, session: dict) -> pd.Series:
        """
        Pipeline completo para UMA sessão: normaliza + extrai features.
        Retorna um pandas.Series de uma linha, na ordem de FEATURE_NAMES.

        Use este método tanto para montar o dataset de treino quanto,
        em produção, para pontuar uma sessão real no momento do CAPTCHA.
        """
        normalized = self.normalize_session(session)
        features = self.extract_features(normalized)
        return pd.Series(features, index=self.FEATURE_NAMES)

    def transform_batch(self, sessions: list) -> pd.DataFrame:
        """Aplica transform() numa lista de sessões (dicts) e retorna um DataFrame."""
        rows = [self.transform(s) for s in sessions]
        return pd.DataFrame(rows, columns=self.FEATURE_NAMES)

    def transform_batch_from_dir(self, directory: str, pattern: str = "*.json") -> pd.DataFrame:
        """
        Lê todos os arquivos .json de sessão de um diretório, aplica o
        pipeline completo e retorna um DataFrame (uma linha por sessão,
        com a coluna 'session_id' na frente). Pensado para montar o
        dataset de treino a partir das 10.000 sessões já capturadas.
        """
        sessions, session_ids = [], []
        for path in sorted(Path(directory).glob(pattern)):
            with open(path, "r", encoding="utf-8") as f:
                session = json.load(f)
            sessions.append(session)
            sid = session.get("session_id", path.stem)
            session_ids.append(sid)

        df = self.transform_batch(sessions)
        df.insert(0, "session_id", session_ids)
        return df


if __name__ == "__main__":
    # Exemplo rápido de uso com um único arquivo de sessão.
    import sys

    pre = MouseBehaviorPreprocessor()
    example_path = sys.argv[1] if len(sys.argv) > 1 else None
    if example_path:
        with open(example_path, "r", encoding="utf-8") as f:
            session = json.load(f)
        print(pre.transform(session))
