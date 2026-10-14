"""
gerador_f.py

Gerador de trajetórias sintéticas baseado em conhecimento, seguindo o
Método 1 de Acien et al. (2022), seção 3.1: uma trajetória é a combinação
de uma FORMA (linear, quadrática, exponencial) com um PERFIL DE VELOCIDADE
(constante, crescente/logarítmico, gaussiano).

Adaptação em relação ao artigo: as formas são definidas no sistema local
P0 -> Pc (s = u/L, z = w/L), e não como y = f(x) em coordenadas de tela,
para valer em qualquer direção de movimento (inclusive vertical).

    P(s) = P0 + s·L·û + z(s)·L·v̂,   s ∈ [0, 1]

Formas (todas com z(0) = z(1) = 0):
    linear:      z(s) = 0
    quadrática:  z(s) = a·s·(s − 1)
    exponencial: z(s) = a·[(e^(b·s) − 1)/(e^b − 1) − s]

Perfis (incrementos d_i entre pontos consecutivos, i = 1..M−1, normalizados
para somar 1):
    constante:   d_i = 1
    crescente:   d_i ∝ log(1 + i)
    gaussiano:   d_i ∝ exp(−(i − M/2)² / (2σ²)),  σ = (M − 1)/4
"""

import numpy as np

# ----------------------------------------------------------------------
# Formas
# ----------------------------------------------------------------------
def _base_exponencial(s: np.ndarray, b: float) -> np.ndarray:
    """(e^(b·s) − 1)/(e^b − 1) − s; tende a 0 quando b → 0."""
    if abs(b) < 1e-6:
        return np.zeros_like(s, dtype=float)
    return np.expm1(b * s) / np.expm1(b) - s


def desvio_forma(s: np.ndarray, forma: int, a: float = 0.0, b: float = 0.0) -> np.ndarray:
    """Desvio perpendicular adimensional z(s) para a `forma` (0, 1 ou 2)."""
    s = np.asarray(s, dtype=float)
    if forma == 0:
        return np.zeros_like(s)
    if forma == 1:
        return a * s * (s - 1.0)
    if forma == 2:
        return a * _base_exponencial(s, b)
    raise ValueError(f"forma desconhecida: {forma}")


# ----------------------------------------------------------------------
# Perfis de velocidade
# ----------------------------------------------------------------------
def posicoes_perfil(M: int, perfil: int) -> np.ndarray:
    """
    Posições s_1..s_M em [0, 1] (s_1 = 0, s_M = 1) espaçadas segundo o
    `perfil` (0 constante, 1 crescente, 2 gaussiano).
    """
    if M < 2:
        raise ValueError("M precisa ser >= 2")
    i = np.arange(1, M, dtype=float)
    if perfil == 0:
        d = np.ones_like(i)
    elif perfil == 1:
        d = np.log1p(i)
    elif perfil == 2:
        sigma = (M - 1) / 4.0
        d = np.exp(-((i - M / 2.0) ** 2) / (2.0 * sigma ** 2))
    else:
        raise ValueError(f"perfil desconhecido: {perfil}")
    d = d / d.sum()
    return np.concatenate([[0.0], np.cumsum(d)])


# ----------------------------------------------------------------------
# Trajetória
# ----------------------------------------------------------------------
def trajetoria(p0, pc, forma: int, perfil: int, M: int, a: float = 0.0, b: float = 0.0) -> np.ndarray:
    """Trajetória (M x 2) em pixels, de P0 até Pc."""
    p0 = np.asarray(p0, dtype=float)
    pc = np.asarray(pc, dtype=float)
    direcao = pc - p0
    L = float(np.linalg.norm(direcao))
    if L == 0:
        return np.repeat(p0[None, :], M, axis=0)
    u_hat = direcao / L
    v_hat = np.array([-u_hat[1], u_hat[0]])
    s = posicoes_perfil(M, perfil)
    z = desvio_forma(s, forma, a, b)
    return p0 + np.outer(s * L, u_hat) + np.outer(z * L, v_hat)


# ----------------------------------------------------------------------
# Ajuste das formas a trajetórias humanas (usado em 00_estimate_params_f.py)
# ----------------------------------------------------------------------
def ajustar_quadratica(s: np.ndarray, z: np.ndarray):
    """Mínimos quadrados de z ≈ a·s(s−1). Retorna (a, sse)."""
    g = s * (s - 1.0)
    den = float(g @ g)
    if den <= 0:
        return 0.0, float(z @ z)
    a = float(g @ z) / den
    r = z - a * g
    return a, float(r @ r)


def ajustar_exponencial(s: np.ndarray, z: np.ndarray, grade_b: np.ndarray):
    """
    Mínimos quadrados de z ≈ a·[(e^(b·s) − 1)/(e^b − 1) − s]: para cada b da
    grade, `a` tem solução fechada; escolhe o b de menor erro. Retorna (a, b, sse).
    """
    melhor = (0.0, 0.0, float(z @ z))
    for b in grade_b:
        g = _base_exponencial(s, b)
        den = float(g @ g)
        if den <= 0:
            continue
        a = float(g @ z) / den
        r = z - a * g
        sse = float(r @ r)
        if sse < melhor[2]:
            melhor = (a, float(b), sse)
    return melhor


# ----------------------------------------------------------------------
# Amostragem dos parâmetros a partir de results/params_humanos.json
# ----------------------------------------------------------------------
class AmostradorParametros:
    """Sorteia P0, alvo, deslocamento do clique, M, a e b das distribuições humanas."""

    def __init__(self, params: dict, M_minimo: int = 10):
        self.params = params
        self.M_minimo = M_minimo
        self.largura, self.altura = params["janela"]["largura"], params["janela"]["altura"]

    @staticmethod
    def _quantil(rng, tabela: dict) -> float:
        """Amostra da distribuição empírica restrita aos percentis guardados (p5..p95)."""
        q = np.asarray(tabela["quantis"])
        return float(np.interp(rng.uniform(0, 1), np.linspace(0, 1, len(q)), q))

    def p0(self, rng) -> np.ndarray:
        pts = self.params["p0_relativo"]
        x, y = pts[rng.integers(len(pts))]
        return np.array([x * self.largura, y * self.altura])

    def alvo(self, rng) -> np.ndarray:
        pts = self.params["alvo_relativo"]
        x, y = pts[rng.integers(len(pts))]
        return np.array([x * self.largura, y * self.altura])

    def deslocamento_clique(self, rng) -> np.ndarray:
        pts = self.params["deslocamento_clique"]
        return np.array(pts[rng.integers(len(pts))], dtype=float)

    def M(self, rng, L: float, setor: int) -> int:
        tabela = self.params["M"]
        faixa = int(np.searchsorted(tabela["limites_L"], L))
        celula = tabela["celulas"][str(setor)][faixa]
        M = int(round(rng.normal(celula["media"], celula["desvio"])))
        return max(M, self.M_minimo)

    def coeficientes(self, rng, forma: int):
        if forma == 0:
            return 0.0, 0.0
        if forma == 1:
            return self._quantil(rng, self.params["quadratica"]["a"]), 0.0
        # a e b da exponencial são sorteados juntos (par de uma mesma trajetória
        # humana), porque o ajuste os torna dependentes entre si
        pares = self.params["exponencial"]["pares_ab"]
        a, b = pares[rng.integers(len(pares))]
        return float(a), float(b)
