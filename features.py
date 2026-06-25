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

    # Piso mínimo de Δt (segundos) entre duas amostras de mouse confiáveis.
    # Intervalos menores que isso são artefatos de coalescência de eventos
    # do sistema (mais de uma amostra "real" chega com timestamps quase
    # idênticos), não movimento humano — ver investigação: distâncias de
    # até ~180px registradas em <1ms, fisicamente impossível. Velocidade,
    # aceleração e jerk são recalculados a partir da sequência já filtrada
    # por esse piso, em vez de usar os arrays 'velocities'/'accelerations'
    # originais do JSON.
    MIN_DT = 0.001  # 1 ms

    # ------------------------------------------------------------------
    # 1) Normalização espacial pela diagonal da janela
    # ------------------------------------------------------------------
    # Tamanho fixo da janela usada no data_collection.py (WIN_W, WIN_H).
    # `window_origin` NÃO é largura/altura — é a posição da janela na tela
    # (canto superior esquerdo), por isso não é usado aqui. Nessa coleta a
    # janela tem sempre o mesmo tamanho, então a diagonal é constante.
    # Se uma futura versão do coletor (ex.: captcha web com viewport
    # variável) passar a salvar o tamanho real por sessão, troque este
    # método para ler esse campo em vez da constante abaixo.
    WINDOW_WIDTH = 500
    WINDOW_HEIGHT = 400

    # Coordenadas do centro do checkbox-alvo e sua largura, extraídas
    # rodando data_collection.py headless (Xvfb) e lendo winfo_x/y/width
    # dos widgets reais. Mesmo espaço de coordenadas dos eventos de
    # mouse (relativo ao canto superior esquerdo da janela 500x400).
    CHECKBOX_CENTER_X = 172.0
    CHECKBOX_CENTER_Y = 204.0
    CHECKBOX_WIDTH = 136.0

    def _compute_diagonal(self, session: dict) -> float:
        diagonal = math.hypot(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
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
    # 1.5) Cinemática robusta (com piso de Δt) — recalculada a partir de
    # mouse_movements + timestamps, em vez dos arrays 'velocities' e
    # 'accelerations' originais do JSON. Usada por todas as features de
    # velocidade, aceleração e jerk abaixo.
    # ------------------------------------------------------------------
    def _clean_kinematic_sequence(self, session: dict):
        """
        Retorna (xs, ys, ts) já filtrados: descarta qualquer amostra cujo
        intervalo de tempo até a ÚLTIMA amostra mantida seja menor que
        MIN_DT. Equivale a colapsar amostras coalescidas (mesmo instante
        físico, timestamps quase iguais) num único ponto.
        """
        movements = session.get("mouse_movements", [])
        timestamps = session.get("timestamps", [])
        if not movements or len(movements) != len(timestamps):
            return np.array([]), np.array([]), np.array([])

        xs, ys, ts = [], [], []
        last_t = None
        for m, t in zip(movements, timestamps):
            if last_t is None or (t - last_t) >= self.MIN_DT:
                xs.append(m["x"])
                ys.append(m["y"])
                ts.append(t)
                last_t = t
        return np.array(xs), np.array(ys), np.array(ts)

    def _velocity_series(self, session: dict) -> np.ndarray:
        xs, ys, ts = self._clean_kinematic_sequence(session)
        if xs.size < 2:
            return np.array([])
        dist = np.hypot(np.diff(xs), np.diff(ys))
        dt = np.diff(ts)  # já >= MIN_DT por construção
        return dist / dt

    def _velocity_times(self, session: dict) -> np.ndarray:
        _, _, ts = self._clean_kinematic_sequence(session)
        return ts[1:] if ts.size >= 2 else np.array([])

    def _acceleration_series(self, session: dict) -> np.ndarray:
        velocity = self._velocity_series(session)
        vel_times = self._velocity_times(session)
        if velocity.size < 2:
            return np.array([])
        return np.diff(velocity) / np.diff(vel_times)

    def _acceleration_times(self, session: dict) -> np.ndarray:
        vel_times = self._velocity_times(session)
        return vel_times[1:] if vel_times.size >= 2 else np.array([])

    # ------------------------------------------------------------------
    # 2) Features — uma por método. Todas recebem a sessão NORMALIZADA.
    # ------------------------------------------------------------------
    def _feature_num_movements(self, session: dict) -> float:
        return float(len(session.get("mouse_movements", [])))

    def _feature_total_time(self, session: dict) -> float:
        return float(session.get("total_time", 0.0))

    def _feature_time_to_click(self, session: dict) -> float:
        return float(session.get("click_data", {}).get("time_to_click", 0.0))

    def _feature_total_distance_norm(self, session: dict) -> float:
        return float(session.get("distance_traveled", 0.0))

    def _feature_ratio(self, session: dict) -> float:
        """
        Razão distância/deslocamento (d_caminho / d_euclidiana), conforme
        o TCC. ≥ 1.0; quanto maior, mais "torto" foi o caminho em relação
        à linha reta entre o primeiro e o último ponto. 1.0 = caminho
        perfeitamente reto.
        """
        movements = session.get("mouse_movements", [])
        total_distance_norm = self._feature_total_distance_norm(session)
        if len(movements) < 2:
            return 1.0
        x0, y0 = movements[0]["x"], movements[0]["y"]
        x1, y1 = movements[-1]["x"], movements[-1]["y"]
        straight_dist = math.hypot(x1 - x0, y1 - y0)
        if straight_dist <= self.eps:
            # início e fim praticamente no mesmo ponto: a razão tende ao
            # infinito se houve qualquer caminho percorrido; usamos eps
            # no denominador só para evitar divisão por zero literal.
            straight_dist = self.eps
        return total_distance_norm / straight_dist

    def _feature_click_center_distance(self, session: dict) -> float:
        """
        Distância (normalizada pela diagonal) entre o ponto de clique e
        o centro geométrico do checkbox-alvo. Coordenadas do centro
        extraídas empiricamente do layout real do data_collection.py
        (ver CHECKBOX_CENTER_X/Y).
        """
        click_pos = session.get("click_data", {}).get("click_position")
        if click_pos is None:
            return 0.0
        diagonal = self._compute_diagonal(session)
        center_x_norm = self.CHECKBOX_CENTER_X / diagonal
        center_y_norm = self.CHECKBOX_CENTER_Y / diagonal
        return math.hypot(click_pos[0] - center_x_norm, click_pos[1] - center_y_norm)

    def _feature_reaction_time_fitts_residual(self, session: dict) -> float:
        """
        Resíduo do tempo de reação em relação à Lei de Fitts:
            ID = log2(D/W + 1)
            residual = time_to_click - ID

        D = distância do primeiro evento de mouse até o centro do
        checkbox-alvo; W = largura do checkbox (CHECKBOX_WIDTH).
        D e W são mantidos na mesma escala (ambos normalizados pela
        diagonal, ou ambos em pixels — a razão D/W é invariante à
        normalização, então o resultado de ID não muda).
        """
        movements = session.get("mouse_movements", [])
        time_to_click = self._feature_time_to_click(session)
        if not movements:
            return time_to_click

        diagonal = self._compute_diagonal(session)
        center_x_norm = self.CHECKBOX_CENTER_X / diagonal
        center_y_norm = self.CHECKBOX_CENTER_Y / diagonal
        width_norm = self.CHECKBOX_WIDTH / diagonal

        x0, y0 = movements[0]["x"], movements[0]["y"]
        d = math.hypot(x0 - center_x_norm, y0 - center_y_norm)

        index_of_difficulty = math.log2(d / width_norm + 1.0)
        return time_to_click - index_of_difficulty

    def _feature_avg_velocity_norm(self, session: dict) -> float:
        velocities = self._velocity_series(session)
        return float(np.mean(velocities)) if velocities.size else 0.0

    def _feature_std_velocity_norm(self, session: dict) -> float:
        velocities = self._velocity_series(session)
        return float(np.std(velocities)) if velocities.size else 0.0

    def _feature_max_velocity_norm(self, session: dict) -> float:
        velocities = self._velocity_series(session)
        return float(np.max(velocities)) if velocities.size else 0.0

    def _feature_avg_acceleration_norm(self, session: dict) -> float:
        accelerations = self._acceleration_series(session)
        return float(np.mean(accelerations)) if accelerations.size else 0.0

    def _feature_std_acceleration_norm(self, session: dict) -> float:
        accelerations = self._acceleration_series(session)
        return float(np.std(accelerations)) if accelerations.size else 0.0

    def _feature_max_acceleration_norm(self, session: dict) -> float:
        accelerations = self._acceleration_series(session)
        return float(np.max(np.abs(accelerations))) if accelerations.size else 0.0

    def _jerk(self, session: dict) -> np.ndarray:
        """
        Helper privado, não é uma feature em si — usado por
        _feature_jerk_mean e _feature_jerk_std.

        Jerk é a derivada temporal da aceleração:
            jerk[i] = (a[i+1] - a[i]) / (t[i+1] - t[i])

        Usa as séries já recalculadas com piso de Δt (_acceleration_series
        / _acceleration_times), então não precisa mais de clamp de
        divisão por zero — por construção, todo Δt aqui já é >= MIN_DT.
        """
        accelerations = self._acceleration_series(session)
        acc_times = self._acceleration_times(session)
        if accelerations.size < 2:
            return np.array([])
        da = np.diff(accelerations)
        dt = np.diff(acc_times)
        return np.abs(da / dt)

    def _feature_jerk_mean(self, session: dict) -> float:
        """Jerk médio absoluto da sessão: mean(|jerk(t)|)."""
        jerk = self._jerk(session)
        return float(np.mean(jerk)) if jerk.size else 0.0

    def _feature_jerk_std(self, session: dict) -> float:
        """Variância (desvio padrão) do jerk absoluto da sessão: std(|jerk(t)|)."""
        jerk = self._jerk(session)
        return float(np.std(jerk)) if jerk.size else 0.0

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
        """Usa a sequência já filtrada pelo piso de Δt (mesma de velocidade/aceleração)."""
        _, _, ts = self._clean_kinematic_sequence(session)
        if ts.size < 2:
            return 0.0
        return float(np.mean(np.diff(ts)))

    def _feature_std_time_between_movements(self, session: dict) -> float:
        """Usa a sequência já filtrada pelo piso de Δt (mesma de velocidade/aceleração)."""
        _, _, ts = self._clean_kinematic_sequence(session)
        if ts.size < 2:
            return 0.0
        return float(np.std(np.diff(ts)))

    # ------------------------------------------------------------------
    # 2.5) As 12 features oficiais do capítulo de Metodologia do TCC
    # ------------------------------------------------------------------
    def _shannon_entropy(self, values: np.ndarray, bins: int = 20) -> float:
        """
        Helper compartilhado por entropia_velocidade e entropia_temporal.
        Entropia de Shannon de uma distribuição discretizada em `bins`
        intervalos igualmente espaçados, com a convenção 0*log2(0) = 0
        (bins vazios são ignorados na soma).
        """
        values = np.asarray(values, dtype=float)
        if values.size < 2:
            return 0.0
        counts, _ = np.histogram(values, bins=bins)
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts / total
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    # --- Cinemática ---
    def _feature_jerk_medio(self, session: dict) -> float:
        """
        jerk_medio = mean(|jerk(t)|)

        Interpretação: jerk médio muito baixo indica movimento "suave demais"
        (aceleração quase constante), comum em trajetórias geradas por
        interpolação algorítmica. Humanos produzem variações de força
        muscular constantes, resultando em jerk médio mais alto e variável.
        """
        jerk = self._jerk(session)
        return float(np.mean(jerk)) if jerk.size else 0.0

    def _feature_jerk_dp(self, session: dict) -> float:
        """
        jerk_dp = std(|jerk(t)|)

        Interpretação: desvio padrão baixo do jerk sugere um padrão de
        "brusquidão" constante ao longo de toda a sessão (suspeito de bot,
        que tende a manter o mesmo perfil de movimento do início ao fim).
        Humanos variam a intensidade dos movimentos bruscos durante a
        trajetória (mais brusco ao iniciar, mais fino ao se aproximar do
        alvo, por exemplo).
        """
        jerk = self._jerk(session)
        return float(np.std(jerk)) if jerk.size else 0.0

    def _feature_razao_trajetoria(self, session: dict) -> float:
        """
        razao_trajetoria = d_caminho / d_euclidiana

        Interpretação: valores próximos de 1.0 indicam uma trajetória quase
        perfeitamente reta entre o início e o clique — pouco comum em
        humanos, que tendem a fazer pequenos ajustes e curvas mesmo em
        movimentos curtos. Valores bem maiores que 1 refletem o desvio
        natural humano em relação ao caminho mais curto possível.
        """
        movements = session.get("mouse_movements", [])
        if len(movements) < 2:
            return 1.0

        total_distance_norm = self._feature_total_distance_norm(session)
        x0, y0 = movements[0]["x"], movements[0]["y"]
        x1, y1 = movements[-1]["x"], movements[-1]["y"]
        straight_dist = math.hypot(x1 - x0, y1 - y0)
        if straight_dist <= self.eps:
            straight_dist = self.eps
        return total_distance_norm / straight_dist

    def _feature_entropia_velocidade(self, session: dict) -> float:
        """
        entropia_velocidade = -sum_k p[k]*log2(p[k]), distribuição de
        velocidade discretizada em B=20 bins (sem normalizar por H_max).

        Interpretação: entropia baixa indica que a velocidade do cursor se
        concentra em poucos valores repetidos — característico de movimento
        mecânico/uniforme gerado por script. Entropia alta reflete a ampla
        variabilidade natural da velocidade humana ao longo da sessão.
        """
        velocities = self._velocity_series(session)
        return self._shannon_entropy(velocities, bins=20)

    # --- Trajetória ---
    def _curvature_series(self, session: dict) -> np.ndarray:
        """Helper: ângulos de curvatura θ[i] para cada tripla de pontos consecutivos."""
        movements = session.get("mouse_movements", [])
        if len(movements) < 3:
            return np.array([])

        pts = np.array([[m["x"], m["y"]] for m in movements])
        v1 = pts[1:-1] - pts[:-2]
        v2 = pts[2:] - pts[1:-1]
        norms1 = np.linalg.norm(v1, axis=1)
        norms2 = np.linalg.norm(v2, axis=1)
        valid = (norms1 > self.eps) & (norms2 > self.eps)
        if not np.any(valid):
            return np.array([])

        dot = np.sum(v1[valid] * v2[valid], axis=1)
        cos_theta = np.clip(dot / (norms1[valid] * norms2[valid]), -1.0, 1.0)
        return np.arccos(cos_theta)

    def _feature_curvatura_media(self, session: dict) -> float:
        """
        curvatura_media = mean(θ)

        Interpretação: curvatura média próxima de 0 (ângulos quase retos
        entre segmentos consecutivos) indica trajetória excessivamente
        linear — mais provável de ser um bot seguindo uma rota direta.
        Humanos apresentam mudanças de direção constantes mesmo em
        movimentos simples, elevando a curvatura média.
        """
        theta = self._curvature_series(session)
        return float(np.mean(theta)) if theta.size else 0.0

    def _feature_curvatura_dp(self, session: dict) -> float:
        """
        curvatura_dp = std(θ)

        Interpretação: desvio padrão baixo sugere um padrão de mudança de
        direção repetitivo/uniforme (suspeito). Humanos variam a
        intensidade das curvas conforme a fase do movimento (mais erráticos
        no início, mais retos perto do alvo, por exemplo).
        """
        theta = self._curvature_series(session)
        return float(np.std(theta)) if theta.size else 0.0

    def _feature_taxa_retrocesso(self, session: dict) -> float:
        """
        taxa_retrocesso = n_retrocessos / N

        Interpretação: taxa muito baixa ou nula indica ausência de
        correções de curso — mais característico de bots, que seguem uma
        rota pré-calculada sem hesitação. Humanos comumente revertem a
        direção do cursor (mesmo que levemente) ao longo do trajeto.
        """
        movements = session.get("mouse_movements", [])
        n = len(movements)
        if n < 3:
            return 0.0

        xs = np.array([m["x"] for m in movements])
        ys = np.array([m["y"] for m in movements])
        dx = np.diff(xs)
        dy = np.diff(ys)

        inversion = (dx[1:] * dx[:-1] < 0) | (dy[1:] * dy[:-1] < 0)
        n_retrocessos = int(np.sum(inversion))
        return n_retrocessos / n

    # --- Clique ---
    def _feature_tempo_reacao(self, session: dict) -> float:
        """
        tempo_reacao = tc - t0  (tempo entre o primeiro evento registrado e o clique)

        Interpretação: tempos de reação muito curtos e muito consistentes
        entre sessões sugerem automação (reação "perfeita", sem o tempo de
        processamento cognitivo humano). Humanos apresentam tempos de
        reação mais longos e variáveis, influenciados por atenção e
        tomada de decisão.

        OBS: diferente do campo `time_to_click` já existente na classe —
        aquele mede a partir da abertura da janela (start_time); este mede
        a partir do primeiro evento de mouse efetivamente registrado.
        """
        timestamps = session.get("timestamps", [])
        click_timestamp = session.get("click_data", {}).get("click_timestamp")
        if not timestamps or click_timestamp is None:
            return 0.0
        return float(click_timestamp - timestamps[0])

    def _feature_dist_clique(self, session: dict) -> float:
        """
        dist_clique = distância euclidiana entre o ponto de clique e o
        centro geométrico do checkbox-alvo (normalizada pela diagonal).

        Interpretação: distância muito próxima de zero em todas as sessões
        (clique sempre no centro exato) é suspeita — característica de
        scripts com coordenadas fixas. Humanos apresentam dispersão natural
        ao redor do alvo, raramente acertando o centro exato.
        """
        click_pos = session.get("click_data", {}).get("click_position")
        if click_pos is None:
            return 0.0
        diagonal = self._compute_diagonal(session)
        center_x_norm = self.CHECKBOX_CENTER_X / diagonal
        center_y_norm = self.CHECKBOX_CENTER_Y / diagonal
        return math.hypot(click_pos[0] - center_x_norm, click_pos[1] - center_y_norm)

    def _feature_vel_aproximacao(self, session: dict) -> float:
        """
        vel_aproximacao = vel_aproximacao_abs / vel_media  (razão de desaceleração, eq. 27)

        Interpretação: valores abaixo de 1 indicam desaceleração ao se
        aproximar do alvo (consistente com a Lei de Fitts e o ajuste
        motor fino humano). Valores próximos ou acima de 1 sugerem
        manutenção/aumento da velocidade até o clique — mais
        característico de bots, que não precisam desacelerar para
        garantir precisão.
        """
        velocities = self._velocity_series(session)
        if velocities.size == 0:
            return 0.0
        k = min(5, velocities.size)
        vel_aproximacao_abs = float(np.mean(velocities[-k:]))
        vel_media = float(np.mean(velocities))
        if abs(vel_media) <= self.eps:
            return 0.0
        return vel_aproximacao_abs / vel_media

    # --- Temporal ---
    def _feature_dp_temporal(self, session: dict) -> float:
        """
        dp_temporal = sqrt(mean((Δt - mean(Δt))^2))

        Interpretação: desvio padrão muito baixo (ritmo extremamente regular
        entre eventos) é característico de scripts com intervalos fixos
        entre movimentos. Humanos têm ritmo naturalmente irregular, variando
        a cadência dos movimentos ao longo da sessão.
        """
        _, _, ts = self._clean_kinematic_sequence(session)
        if ts.size < 2:
            return 0.0
        return float(np.std(np.diff(ts)))

    def _feature_entropia_temporal(self, session: dict) -> float:
        """
        entropia_temporal = -sum_k p[k]*log2(p[k]), distribuição de Δt
        discretizada em B=20 bins.

        Interpretação: entropia baixa indica padrão repetitivo nos
        intervalos entre eventos (suspeito de geração programática com
        delays fixos ou poucos valores possíveis). Entropia alta reflete a
        irregularidade natural do ritmo de movimento humano.
        """
        _, _, ts = self._clean_kinematic_sequence(session)
        if ts.size < 2:
            return 0.0
        delta_t = np.diff(ts)
        return self._shannon_entropy(delta_t, bins=20)

    # ------------------------------------------------------------------
    # 3) Orquestrador ("main" da extração de features)
    # ------------------------------------------------------------------
    def extract_features(self, normalized_session: dict) -> dict:
        """
        Chama cada um dos métodos _feature_* já implementados e monta o
        vetor de features final, na forma de um dict. Este é o ponto
        único de entrada para extração — conforme novas features forem
        adicionadas como métodos novos, basta incluir a chamada
        correspondente aqui.
        """
        return {
            "num_movements": self._feature_num_movements(normalized_session),
            "total_time": self._feature_total_time(normalized_session),
            "time_to_click": self._feature_time_to_click(normalized_session),
            "reaction_time_fitts_residual": self._feature_reaction_time_fitts_residual(normalized_session),
            "total_distance_norm": self._feature_total_distance_norm(normalized_session),
            "ratio": self._feature_ratio(normalized_session),
            "avg_velocity_norm": self._feature_avg_velocity_norm(normalized_session),
            "std_velocity_norm": self._feature_std_velocity_norm(normalized_session),
            "max_velocity_norm": self._feature_max_velocity_norm(normalized_session),
            "avg_acceleration_norm": self._feature_avg_acceleration_norm(normalized_session),
            "std_acceleration_norm": self._feature_std_acceleration_norm(normalized_session),
            "max_acceleration_norm": self._feature_max_acceleration_norm(normalized_session),
            "jerk_mean": self._feature_jerk_mean(normalized_session),
            "jerk_std": self._feature_jerk_std(normalized_session),
            "click_offset_norm": self._feature_click_offset_norm(normalized_session),
            "click_center_distance": self._feature_click_center_distance(normalized_session),
            "avg_time_between_movements": self._feature_avg_time_between_movements(normalized_session),
            "std_time_between_movements": self._feature_std_time_between_movements(normalized_session),
            # --- As 12 features oficiais do capítulo de Metodologia do TCC ---
            "jerk_medio": self._feature_jerk_medio(normalized_session),
            "jerk_dp": self._feature_jerk_dp(normalized_session),
            "razao_trajetoria": self._feature_razao_trajetoria(normalized_session),
            "entropia_velocidade": self._feature_entropia_velocidade(normalized_session),
            "curvatura_media": self._feature_curvatura_media(normalized_session),
            "curvatura_dp": self._feature_curvatura_dp(normalized_session),
            "taxa_retrocesso": self._feature_taxa_retrocesso(normalized_session),
            "tempo_reacao": self._feature_tempo_reacao(normalized_session),
            "dist_clique": self._feature_dist_clique(normalized_session),
            "vel_aproximacao": self._feature_vel_aproximacao(normalized_session),
            "dp_temporal": self._feature_dp_temporal(normalized_session),
            "entropia_temporal": self._feature_entropia_temporal(normalized_session),
        }

    # ------------------------------------------------------------------
    # 4) API de alto nível (use estes métodos no seu código)
    # ------------------------------------------------------------------
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


class Winsorizador:
    """
    Limita (winsoriza) cada feature aos percentis aprendidos no
    conjunto de treino, antes do escalonamento (StandardScaler).
    Evita que outliers extremos (ex.: razao_trajetoria ou jerk_medio em
    sessões degeneradas) distorçam a média/desvio padrão do scaler.

    Os limites são aprendidos uma vez (ajustar, sobre os dados de
    treino) e reaplicados sempre da mesma forma — tanto ao montar o
    dataset de treino quanto, em produção, ao pontuar uma sessão real.
    """

    def __init__(self, percentil_inferior: float = 0.005, percentil_superior: float = 0.995):
        self.percentil_inferior = percentil_inferior
        self.percentil_superior = percentil_superior
        self.limites_inferiores_ = None
        self.limites_superiores_ = None
        self.colunas_ = None

    def ajustar(self, X: pd.DataFrame) -> "Winsorizador":
        self.colunas_ = list(X.columns)
        self.limites_inferiores_ = X.quantile(self.percentil_inferior)
        self.limites_superiores_ = X.quantile(self.percentil_superior)
        return self

    def transformar(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.limites_inferiores_ is None or self.limites_superiores_ is None:
            raise RuntimeError("Winsorizador não foi ajustado (chame ajustar() antes).")
        X = X[self.colunas_].copy()
        return X.clip(lower=self.limites_inferiores_, upper=self.limites_superiores_, axis=1)

    def ajustar_transformar(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.ajustar(X).transformar(X)

    def salvar(self, caminho: str) -> None:
        import joblib
        joblib.dump(self, caminho)

    @staticmethod
    def carregar(caminho: str) -> "Winsorizador":
        import joblib
        return joblib.load(caminho)