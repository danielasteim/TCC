"""
feature_processor.py

Classe de pré-processamento para o CAPTCHA comportamental baseado em
dinâmica de mouse (Isolation Forest + One-Class SVM).

Responsabilidades:
  1. Normalizar todas as grandezas com dimensão espacial de uma sessão
     (posição, velocidade, aceleração, distância percorrida e posição
     de clique) pela diagonal da resolução da janela daquela sessão —
     tornando-as adimensionais e comparáveis entre sessões capturadas
     em dispositivos com resoluções diferentes.
  2. Extrair, a partir da sessão normalizada, um vetor de features —
     pronto para alimentar `.fit()` / `.predict()` do scikit-learn.

Organização das features:
  Cada feature é calculada por um método próprio, com prefixo
  `_feature_`, que recebe a sessão JÁ NORMALIZADA e devolve um valor
  escalar. O método `extract_features` funciona como o "main" desse
  processo: chama cada um dos métodos de feature já implementados e
  monta o vetor final.

  Para adicionar uma nova feature:
    1. Implemente um método `_feature_<nome>(self, session)`.
    2. Adicione a linha correspondente dentro de `extract_features`.

A mesma classe é usada em dois momentos:
  - Offline, para montar o dataset de treino a partir das sessões já
    capturadas (transform_batch / transform_batch_from_dir).
  - Online, em produção, para pré-processar UMA sessão recém-capturada
    antes de pontuá-la com os modelos já treinados (transform).
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


class Features:
    """Normaliza sessões de captura de mouse e extrai features para os modelos."""

    def __init__(self, eps: float = 1e-9):
        # eps evita divisão por zero em sessões degeneradas (ex.: diagonal
        # nula, sessão sem nenhum movimento antes do clique etc.)
        self.eps = eps

    # Normalização  pela diagonal da janela
    def _compute_diagonal(self, session: dict) -> float:
        width = session["window_origin"]["x"]
        height = session["window_origin"]["y"]
        diagonal = math.hypot(width, height)
        return diagonal if diagonal > self.eps else self.eps

    def normalize_session(self, session: dict) -> dict:
        """
        Retorna uma CÓPIA em forma de dicionário da sessão com todas as grandezas espaciais
        normalizadas pela diagonal da janela daquela sessão:
          - mouse_movements[i].x / .y
          - velocities[i]
          - accelerations[i]
          - distance_traveled
          - click_data.click_position
          - metrics.total_distance / avg_velocity / max_velocity /
            avg_acceleration / max_acceleration

        Grandezas temporais (timestamps, total_time, time_to_click, num_movements) NÃO são alteradas
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

        s["_diagonal"] = diagonal 
        return s


    # Features 
    def _feature_num_movements(self, session: dict) -> float:
        return float(len(session.get("mouse_movements", [])))

    def _feature_total_time(self, session: dict) -> float:
        return float(session.get("total_time", 0.0))

    def _feature_time_to_click(self, session: dict) -> float:
        return float(session.get("click_data", {}).get("time_to_click", 0.0))

    def _feature_total_distance_norm(self, session: dict) -> float:
        return float(session.get("distance_traveled", 0.0))

    def _feature_straightness_ratio(self, session: dict) -> float:
        """
        Distância em linha reta entre o primeiro e o último ponto do
        trajeto, dividida pela distância total percorrida. É a razão
        entre duas grandezas já normalizadas, então o resultado já é
        adimensional por construção. 1.0 = caminho perfeitamente reto;
        valores baixos = caminho tortuoso.
        """
        movements = session.get("mouse_movements", [])
        total_distance_norm = self._feature_total_distance_norm(session)
        if len(movements) < 2 or total_distance_norm <= self.eps:
            return 0.0
        x0, y0 = movements[0]["x"], movements[0]["y"]
        x1, y1 = movements[-1]["x"], movements[-1]["y"]
        straight_dist = math.hypot(x1 - x0, y1 - y0)
        return straight_dist / (total_distance_norm + self.eps)

    def _feature_avg_velocity_norm(self, session: dict) -> float:
        velocities = np.asarray(session.get("velocities", []), dtype=float)
        return float(np.mean(velocities)) if velocities.size else 0.0

    def _feature_std_velocity_norm(self, session: dict) -> float:
        velocities = np.asarray(session.get("velocities", []), dtype=float)
        return float(np.std(velocities)) if velocities.size else 0.0

    def _feature_max_velocity_norm(self, session: dict) -> float:
        velocities = np.asarray(session.get("velocities", []), dtype=float)
        return float(np.max(velocities)) if velocities.size else 0.0

    def _feature_avg_acceleration_norm(self, session: dict) -> float:
        accelerations = np.asarray(session.get("accelerations", []), dtype=float)
        return float(np.mean(accelerations)) if accelerations.size else 0.0

    def _feature_std_acceleration_norm(self, session: dict) -> float:
        accelerations = np.asarray(session.get("accelerations", []), dtype=float)
        return float(np.std(accelerations)) if accelerations.size else 0.0

    def _feature_max_acceleration_norm(self, session: dict) -> float:
        accelerations = np.asarray(session.get("accelerations", []), dtype=float)
        return float(np.max(np.abs(accelerations))) if accelerations.size else 0.0

    def _feature_click_offset_norm(self, session: dict) -> float:
        """
        Distância (já normalizada) entre o último ponto do mouse antes
        do clique e a posição real do clique. Humanos clicam
        praticamente onde o cursor já está; cliques "teleportados" são
        suspeitos.
        """
        movements = session.get("mouse_movements", [])
        click_pos = session.get("click_data", {}).get("click_position")
        if click_pos is None or not movements:
            return 0.0
        last_x, last_y = movements[-1]["x"], movements[-1]["y"]
        return math.hypot(click_pos[0] - last_x, click_pos[1] - last_y)

    def _feature_avg_time_between_movements(self, session: dict) -> float:
        timestamps = np.asarray(session.get("timestamps", []), dtype=float)
        if timestamps.size < 2:
            return 0.0
        return float(np.mean(np.diff(timestamps)))

    def _feature_std_time_between_movements(self, session: dict) -> float:
        timestamps = np.asarray(session.get("timestamps", []), dtype=float)
        if timestamps.size < 2:
            return 0.0
        return float(np.std(np.diff(timestamps)))

    # Orquestrador
    def extract_features(self, normalized_session: dict) -> dict:
        """
        Chama cada um dos métodos _feature_* já implementados e monta o
        vetor de features final
        """
        return {
            "num_movements": self._feature_num_movements(normalized_session),
            "total_time": self._feature_total_time(normalized_session),
            "time_to_click": self._feature_time_to_click(normalized_session),
            "total_distance_norm": self._feature_total_distance_norm(normalized_session),
            "straightness_ratio": self._feature_straightness_ratio(normalized_session),
            "avg_velocity_norm": self._feature_avg_velocity_norm(normalized_session),
            "std_velocity_norm": self._feature_std_velocity_norm(normalized_session),
            "max_velocity_norm": self._feature_max_velocity_norm(normalized_session),
            "avg_acceleration_norm": self._feature_avg_acceleration_norm(normalized_session),
            "std_acceleration_norm": self._feature_std_acceleration_norm(normalized_session),
            "max_acceleration_norm": self._feature_max_acceleration_norm(normalized_session),
            "click_offset_norm": self._feature_click_offset_norm(normalized_session),
            "avg_time_between_movements": self._feature_avg_time_between_movements(normalized_session),
            "std_time_between_movements": self._feature_std_time_between_movements(normalized_session),
        }


    # Pipeline completa
    def transform(self, session: dict) -> pd.Series:
        """
        Pipeline completo para UMA sessão: normaliza + extrai features.
        Retorna um pandas.Series de uma linha.

        Use este método tanto para montar o dataset de treino quanto,
        em produção, para pontuar uma sessão real no momento do CAPTCHA.
        """
        normalized = self.normalize_session(session)
        features = self.extract_features(normalized)
        return pd.Series(features)

    def transform_batch(self, sessions: list) -> pd.DataFrame:
        """Aplica transform() numa lista de sessões (dicts) e retorna um DataFrame."""
        rows = [self.transform(s) for s in sessions]
        return pd.DataFrame(rows)

    def transform_batch_from_dir(self, directory: str, pattern: str = "*.json") -> pd.DataFrame:
        """
        Lê todos os arquivos .json de sessão de um diretório, aplica o
        pipeline completo e retorna um DataFrame (uma linha por sessão,
        com a coluna 'session_id' na frente). Pensado para montar o
        dataset de treino a partir das sessões já capturadas.
        """
        sessions, session_ids = [], []
        for path in sorted(Path(directory).glob(pattern)):
            with open(path, "r", encoding="utf-8") as f:
                session = json.load(f)
            sessions.append(session)
            session_ids.append(session.get("session_id", path.stem))

        df = self.transform_batch(sessions)
        df.insert(0, "session_id", session_ids)
        return df
