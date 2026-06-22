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
    WINDOW_WIDTH = 500
    WINDOW_HEIGHT = 400

    def _compute_diagonal(self, session: dict) -> float:
        diagonal = math.hypot(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
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
        s = json.loads(json.dumps(session))  

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
        1.0 = caminho perfeitamente reto;
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
    
    def _jerk(self, session: dict) -> np.ndarray:

        accelerations = np.asarray(session.get("accelerations", []), dtype=float)
        timestamps = np.asarray(session.get("timestamps", []), dtype=float)

        if accelerations.size < 2 or timestamps.size < accelerations.size + 1:
            return np.array([])

        accel_times = timestamps[1: 1 + accelerations.size]
        delta_a = np.diff(accelerations)
        delta_t = np.diff(accel_times)

        delta_t = np.where(np.abs(delta_t) < self.eps, self.eps, delta_t)

        jerk = delta_a / delta_t
        return np.abs(jerk)


    def _feature_jerk_mean(self, session: dict) -> float:
        jerk = self._jerk(session)
        return float(np.mean(jerk)) if jerk.size else 0.0


    def _feature_jerk_std(self, session: dict) -> float:
        jerk = self._jerk(session)
        return float(np.std(jerk)) if jerk.size else 0.0

    # Orquestrador
    def extract_features(self, normalized_session: dict) -> dict:
        """
        Chama cada um dos métodos _feature_* já implementados e monta o
        vetor de features final
        """
        return {
            "jerk_mean": self._feature_jerk_mean(normalized_session),
            "jerk_std": self._feature_jerk_std(normalized_session),
        }



    # Pipeline completa
    def transform(self, session: dict) -> pd.Series:
        """
        Pipeline completo para UMA sessão: normaliza + extrai features.
        Retorna um pandas.Series de uma linha.
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
        com a coluna 'session_id' na frente).
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


if __name__ == "__main__":
    """
    Uso:
        python feature_processor.py caminho/da/sessao.json
        python feature_processor.py caminho/da/pasta_com_sessoes/

    Se receber um arquivo único: mostra o vetor de features dessa sessão.
    Se receber uma pasta: roda em todas as sessões, mostra quantas
    processaram sem erro e quais falharam (se alguma falhar).
    """
    import sys

    if len(sys.argv) < 2:
        print("Uso: python feature_processor.py <arquivo.json | pasta/>")
        sys.exit(1)

    target = Path(sys.argv[1])
    fp = Features()

    if target.is_dir():
        ok, errors = 0, []
        rows = []
        for path in sorted(target.glob("*.json")):
            with open(path, "r", encoding="utf-8") as f:
                session = json.load(f)
            try:
                rows.append(fp.transform(session))
                ok += 1
            except Exception as e:
                errors.append((path.name, str(e)))

        print(f"Processadas com sucesso: {ok}")
        print(f"Erros: {len(errors)}")
        for name, err in errors[:10]:
            print(f"  {name} -> {err}")

        if rows:
            df = pd.DataFrame(rows)
            print()
            print(df.describe())
    else:
        with open(target, "r", encoding="utf-8") as f:
            session = json.load(f)
        print(fp.transform(session))