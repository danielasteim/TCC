"""
caracteristicas_f.py

Extração das 12 características usadas pelos modelos de uma classe
(SVM de Uma Classe e Floresta de Isolação) a partir de uma sessão no
formato JSON do coletor v3 — o mesmo formato que as sessões sintéticas
(bots) devem seguir.

Grupos e características (fundamentação no Capítulo 2):

  Geométricas
    razao_trajetoria     comprimento percorrido / distância P0 -> clique
    desvio_maximo        maior distância perpendicular à reta P0 -> clique, / L
    curvatura_media      ângulo médio de mudança de direção (trajetória
                         reamostrada por comprimento de arco)
  Cinemáticas
    jerk_adimensional    log dimensionless jerk (LDLJ)
    n_submovimentos      número de picos da velocidade suavizada
    razao_pico_media     velocidade de pico / velocidade média
  Temporais
    residuo_fitts        MT - (a + b·ID); a e b ajustados no treino
    instante_pico        instante do pico de velocidade / duração
    razao_metades        velocidade média da 2ª metade / da 1ª metade
  Aproximação e clique
    desaceleracao_final  velocidade média dos k últimos eventos / média geral
    dist_clique          distância clique -> centro do alvo, / diagonal da janela
    tempo_parado         intervalo entre o último movimento e o clique

O resíduo de Fitts depende de coeficientes estimados nos dados de treino.
Por isso a extração (`Caracteristicas`) devolve o tempo de movimento e o
índice de dificuldade, e o transformador `ResiduoFitts` — ajustado dentro
de cada partição — os converte em `residuo_fitts`. Use-o num Pipeline do
scikit-learn, antes do `WinsorizadorF` e do `StandardScaler`:

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM

    X = Caracteristicas().extrair_pasta("captcha_data_f")
    modelo = Pipeline([
        ("fitts", ResiduoFitts()),
        ("winsor", WinsorizadorF(0.005, 0.995)),
        ("padronizacao", StandardScaler()),
        ("svm", OneClassSVM(kernel="rbf", nu=0.05, gamma="auto")),
    ])
    modelo.fit(X[COLUNAS_EXTRAIDAS])

Campos da sessão usados: mouse_movements[{x, y}] (px, referencial da
janela), timestamps (s), click_data{click_position, click_timestamp},
target{width, height, center_x, center_y} e window_size{width, height}.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

GRUPOS = {
    "geometricas": ["razao_trajetoria", "desvio_maximo", "curvatura_media"],
    "cinematicas": ["jerk_adimensional", "n_submovimentos", "razao_pico_media"],
    "temporais": ["residuo_fitts", "instante_pico", "razao_metades"],
    "aproximacao_clique": ["desaceleracao_final", "dist_clique", "tempo_parado"],
}
CARACTERISTICAS = [c for grupo in GRUPOS.values() for c in grupo]  # as 12 finais

# Colunas brutas que o ResiduoFitts converte em residuo_fitts
COLUNAS_FITTS = ["tempo_movimento", "indice_dificuldade"]
# O que Caracteristicas.extrair devolve (entrada do Pipeline)
COLUNAS_EXTRAIDAS = [c for c in CARACTERISTICAS if c != "residuo_fitts"] + COLUNAS_FITTS


class Caracteristicas:
    """Extrai as características de uma sessão (sem estado: nada é ajustado aqui)."""

    def __init__(self,
                 min_dt: float = 0.001,
                 dt_reamostragem: float = 0.010,
                 janela_suavizacao: int = 5,
                 passo_curvatura_px: float = 5.0,
                 limiar_pico: float = 0.10,
                 limiar_vale: float = 0.50,
                 k_final: int = 5):
        """
        min_dt              amostras a menos de min_dt (s) da anterior são fundidas
                            (eventos acumulados na fila da interface)
        dt_reamostragem     passo (s) da reamostragem temporal usada nas
                            características derivadas da velocidade
        janela_suavizacao   pontos da média móvel aplicada às posições reamostradas
        passo_curvatura_px  passo (px) da reamostragem por comprimento de arco
                            usada na curvatura (evita a grade de pixels)
        limiar_pico         um pico de velocidade só conta se for >= limiar_pico × pico global
        limiar_vale         dois picos são submovimentos distintos se o vale entre
                            eles for < limiar_vale × o menor dos dois
        k_final             eventos finais usados na desaceleracao_final
        """
        self.min_dt = min_dt
        self.dt = dt_reamostragem
        self.janela = janela_suavizacao
        self.passo_curv = passo_curvatura_px
        self.limiar_pico = limiar_pico
        self.limiar_vale = limiar_vale
        self.k_final = k_final

    @classmethod
    def de_config(cls, config: dict) -> "Caracteristicas":
        """Cria o extrator a partir do bloco `caracteristicas` do config_f.yaml."""
        return cls(**config.get("caracteristicas", {}))

    # ------------------------------------------------------------------
    # Preparação
    # ------------------------------------------------------------------
    def _preparar(self, s: dict) -> dict:
        P = np.array([[m["x"], m["y"]] for m in s["mouse_movements"]], dtype=float)
        t = np.asarray(s["timestamps"], dtype=float)
        if len(P) != len(t):
            raise ValueError("mouse_movements e timestamps com tamanhos diferentes")
        cd = s["click_data"]
        c = np.asarray(cd["click_position"], dtype=float)
        tc = float(cd["click_timestamp"])

        # eventos posteriores ao clique não fazem parte da trajetória
        antes = t <= tc
        P, t = P[antes], t[antes]

        # funde amostras acumuladas (Δt < min_dt) mantendo a primeira
        manter = [0]
        for i in range(1, len(t)):
            if t[i] - t[manter[-1]] >= self.min_dt:
                manter.append(i)
        P, t = P[manter], t[manter]
        if len(P) < 3:
            raise ValueError("menos de 3 eventos de movimento")

        alvo = s["target"]
        janela = s["window_size"]
        return {
            "P": P, "t": t, "c": c, "tc": tc,
            "centro": np.array([alvo["center_x"], alvo["center_y"]], dtype=float),
            "W": float(min(alvo["width"], alvo["height"])),
            "diagonal": float(np.hypot(janela["width"], janela["height"])),
        }

    def _suavizar(self, z: np.ndarray) -> np.ndarray:
        k = self.janela
        if k <= 1 or len(z) < k:
            return z
        zp = np.pad(z, k // 2, mode="edge")  # borda replicada: sem picos falsos nas pontas
        return np.convolve(zp, np.ones(k) / k, mode="valid")[: len(z)]

    def _cinematica(self, d: dict) -> dict:
        """Velocidade e jerk sobre a trajetória reamostrada a cada `dt` e suavizada."""
        t, P = d["t"], d["P"]
        tt = np.arange(t[0], t[-1] + 1e-9, self.dt)
        if len(tt) < 4:
            return None
        x = self._suavizar(np.interp(tt, t, P[:, 0]))
        y = self._suavizar(np.interp(tt, t, P[:, 1]))
        vx, vy = np.gradient(x, self.dt), np.gradient(y, self.dt)
        v = np.hypot(vx, vy)
        jx = np.gradient(np.gradient(vx, self.dt), self.dt)
        jy = np.gradient(np.gradient(vy, self.dt), self.dt)
        return {"v": v, "j2": jx ** 2 + jy ** 2, "T": tt[-1] - tt[0]}

    # ------------------------------------------------------------------
    # Geométricas
    # ------------------------------------------------------------------
    @staticmethod
    def _polilinha(d: dict) -> np.ndarray:
        """Eventos de movimento seguidos do ponto do clique."""
        return np.vstack([d["P"], d["c"]])

    def razao_trajetoria(self, d: dict) -> float:
        L = np.linalg.norm(d["c"] - d["P"][0])
        comp = np.linalg.norm(np.diff(self._polilinha(d), axis=0), axis=1).sum()
        return comp / L if L > 0 else np.nan

    def desvio_maximo(self, d: dict) -> float:
        p0, c = d["P"][0], d["c"]
        L = np.linalg.norm(c - p0)
        if L == 0:
            return np.nan
        u = (c - p0) / L
        rel = d["P"] - p0
        return float(np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0]).max() / L)

    def curvatura_media(self, d: dict) -> float:
        Q = self._polilinha(d)
        passos = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        Q = Q[np.r_[True, passos > 0]]                 # remove pontos repetidos
        acum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
        s = np.arange(0, acum[-1], self.passo_curv)
        if len(s) < 3:
            return 0.0
        R = np.c_[np.interp(s, acum, Q[:, 0]), np.interp(s, acum, Q[:, 1])]
        ang = np.arctan2(*np.diff(R, axis=0).T[::-1])
        giro = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)
        return float(giro.mean())

    # ------------------------------------------------------------------
    # Cinemáticas
    # ------------------------------------------------------------------
    @staticmethod
    def jerk_adimensional(k: dict, dt: float) -> float:
        """LDLJ = -ln[(T³ / v_pico²) · ∫|jerk|² dt]  (Balasubramanian et al., 2015)."""
        v_pico = k["v"].max()
        integral = k["j2"].sum() * dt
        if v_pico <= 0 or integral <= 0:
            return np.nan
        return float(-np.log(k["T"] ** 3 / v_pico ** 2 * integral))

    def n_submovimentos(self, k: dict) -> int:
        v = k["v"]
        v_pico = v.max()
        picos = [i for i in range(1, len(v) - 1)
                 if v[i] >= v[i - 1] and v[i] > v[i + 1] and v[i] >= self.limiar_pico * v_pico]
        n = 1
        for a, b in zip(picos[:-1], picos[1:]):
            if v[a:b + 1].min() < self.limiar_vale * min(v[a], v[b]):
                n += 1
        return n

    @staticmethod
    def razao_pico_media(k: dict, d: dict) -> float:
        comp = np.linalg.norm(np.diff(d["P"], axis=0), axis=1).sum()
        v_media = comp / k["T"] if k["T"] > 0 else 0
        return float(k["v"].max() / v_media) if v_media > 0 else np.nan

    # ------------------------------------------------------------------
    # Temporais
    # ------------------------------------------------------------------
    @staticmethod
    def tempo_movimento(d: dict) -> float:
        return float(d["tc"] - d["t"][0])

    @staticmethod
    def indice_dificuldade(d: dict) -> float:
        D = np.linalg.norm(d["c"] - d["P"][0])
        return float(np.log2(D / d["W"] + 1))

    @staticmethod
    def instante_pico(k: dict, dt: float) -> float:
        return float(np.argmax(k["v"]) * dt / k["T"]) if k["T"] > 0 else np.nan

    @staticmethod
    def razao_metades(k: dict) -> float:
        v = k["v"]
        meio = len(v) // 2
        primeira = v[:meio].mean()
        return float(v[meio:].mean() / primeira) if primeira > 0 else np.nan

    # ------------------------------------------------------------------
    # Aproximação e clique
    # ------------------------------------------------------------------
    def desaceleracao_final(self, d: dict) -> float:
        v = np.linalg.norm(np.diff(d["P"], axis=0), axis=1) / np.diff(d["t"])
        media = v.mean()
        return float(v[-self.k_final:].mean() / media) if media > 0 else np.nan

    @staticmethod
    def dist_clique(d: dict) -> float:
        return float(np.linalg.norm(d["c"] - d["centro"]) / d["diagonal"])

    @staticmethod
    def tempo_parado(d: dict) -> float:
        return float(d["tc"] - d["t"][-1])

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def extrair(self, sessao: dict) -> dict:
        """
        Devolve um dict com COLUNAS_EXTRAIDAS. Características que não podem
        ser calculadas (sessão curta demais) ficam como NaN.
        """
        d = self._preparar(sessao)
        k = self._cinematica(d)
        nan = np.nan
        return {
            "razao_trajetoria": self.razao_trajetoria(d),
            "desvio_maximo": self.desvio_maximo(d),
            "curvatura_media": self.curvatura_media(d),
            "jerk_adimensional": self.jerk_adimensional(k, self.dt) if k else nan,
            "n_submovimentos": self.n_submovimentos(k) if k else nan,
            "razao_pico_media": self.razao_pico_media(k, d) if k else nan,
            "instante_pico": self.instante_pico(k, self.dt) if k else nan,
            "razao_metades": self.razao_metades(k) if k else nan,
            "desaceleracao_final": self.desaceleracao_final(d),
            "dist_clique": self.dist_clique(d),
            "tempo_parado": self.tempo_parado(d),
            "tempo_movimento": self.tempo_movimento(d),
            "indice_dificuldade": self.indice_dificuldade(d),
        }

    def extrair_lote(self, sessoes, nomes=None, remover_invalidas: bool = True) -> pd.DataFrame:
        """
        Extrai uma lista de sessões. Sessões que não podem ser processadas
        (ou com alguma característica NaN, se remover_invalidas) são
        removidas e listadas em `df.attrs['invalidas']`.
        """
        linhas, indices, invalidas = [], [], []
        for i, s in enumerate(sessoes):
            nome = nomes[i] if nomes is not None else i
            try:
                linha = self.extrair(s)
            except (ValueError, KeyError, ZeroDivisionError) as erro:
                invalidas.append((nome, str(erro)))
                continue
            if remover_invalidas and any(pd.isna(v) for v in linha.values()):
                invalidas.append((nome, "característica indefinida"))
                continue
            linhas.append(linha)
            indices.append(nome)
        df = pd.DataFrame(linhas, index=pd.Index(indices, name="arquivo"), columns=COLUNAS_EXTRAIDAS)
        df.attrs["invalidas"] = invalidas
        return df

    def extrair_pasta(self, pasta, padrao: str = "*.json", **kwargs) -> pd.DataFrame:
        arquivos = sorted(Path(pasta).glob(padrao))
        sessoes = (json.load(open(a, encoding="utf-8")) for a in arquivos)
        return self.extrair_lote(list(sessoes), nomes=[a.name for a in arquivos], **kwargs)


class ResiduoFitts(BaseEstimator, TransformerMixin):
    """
    Converte tempo_movimento e indice_dificuldade em residuo_fitts =
    MT - (a + b·ID), com a e b estimados por mínimos quadrados em fit()
    (só nos dados de treino). Devolve as 12 CARACTERISTICAS, na ordem.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.b_, self.a_ = np.polyfit(X["indice_dificuldade"], X["tempo_movimento"], 1)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["residuo_fitts"] = X["tempo_movimento"] - (self.a_ + self.b_ * X["indice_dificuldade"])
        return X[CARACTERISTICAS]


class WinsorizadorF(BaseEstimator, TransformerMixin):
    """Limita cada coluna aos percentis aprendidos em fit() (só no treino)."""

    def __init__(self, inferior: float = 0.005, superior: float = 0.995):
        self.inferior = inferior
        self.superior = superior

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.limites_ = (X.quantile(self.inferior), X.quantile(self.superior))
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.clip(lower=self.limites_[0], upper=self.limites_[1], axis=1)
