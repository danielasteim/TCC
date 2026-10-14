"""
common_f.py

Funções compartilhadas pelos scripts numerados: leitura do config_f.yaml,
caminhos, carga das sessões e conversão de uma trajetória para o sistema
de coordenadas local (u, w) usado pelo gerador (seção 3.5.2).
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

# trabalho_f/scripts/common_f.py -> RAIZ é a pasta do TCC, PASTA_F é trabalho_f/
PASTA_F = Path(__file__).resolve().parents[1]
RAIZ = PASTA_F.parent
# append (e não insert) para que o math.py da raiz não sombreie nada
sys.path.append(str(RAIZ))

from features import Features  # noqa: E402

FAMILIAS_FORMA = {0: "linear", 1: "quadratica", 2: "exponencial"}
FAMILIAS_PERFIL = {0: "constante", 1: "crescente", 2: "gaussiano"}


def carregar_config(caminho: Path = PASTA_F / "config_f.yaml") -> dict:
    with open(caminho, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def caminho(config: dict, chave: str) -> Path:
    """Caminho absoluto de uma entrada de config['caminhos'] (relativa à pasta do TCC)."""
    return RAIZ / config["caminhos"][chave]


def carregar_sessoes(diretorio: Path, limite: int = None, semente: int = 0):
    """
    Lê os JSONs de sessão de `diretorio`. Com `limite`, lê uma amostra
    aleatória (reprodutível) desse tamanho — útil para testes rápidos.
    """
    arquivos = sorted(Path(diretorio).glob("*.json"))
    if limite is not None and limite < len(arquivos):
        rng = np.random.default_rng(semente)
        arquivos = [arquivos[i] for i in sorted(rng.choice(len(arquivos), limite, replace=False))]
    for arquivo in arquivos:
        with open(arquivo, "r", encoding="utf-8") as f:
            yield arquivo.name, json.load(f)


def tamanho_janela(sessao: dict):
    tamanho = sessao.get("window_size", {})
    return (tamanho.get("width", Features.WINDOW_WIDTH),
            tamanho.get("height", Features.WINDOW_HEIGHT))


def centro_alvo(sessao: dict):
    cx, cy, _ = Features()._geometria_alvo(sessao)
    return np.array([cx, cy], dtype=float)


def pontos_e_tempos(sessao: dict):
    movimentos = sessao.get("mouse_movements", [])
    pontos = np.array([[m["x"], m["y"]] for m in movimentos], dtype=float)
    tempos = np.array(sessao.get("timestamps", []), dtype=float)
    return pontos, tempos


def para_sistema_local(pontos: np.ndarray, p0: np.ndarray, pc: np.ndarray):
    """
    Projeta `pontos` no sistema local definido por P0 -> Pc e devolve
    (s, z) adimensionais: s = u / L (avanço ao longo do segmento) e
    z = w / L (desvio perpendicular), com L = ||Pc - P0||.
    """
    direcao = pc - p0
    L = float(np.linalg.norm(direcao))
    u_hat = direcao / L
    v_hat = np.array([-u_hat[1], u_hat[0]])
    rel = pontos - p0
    return rel @ u_hat / L, rel @ v_hat / L, L


def setor_direcao(p0: np.ndarray, pc: np.ndarray, n_setores: int = 8) -> int:
    """Setor angular (0..n-1) da direção P0 -> Pc; setor 0 centrado em 0° (para a direita)."""
    angulo = np.arctan2(pc[1] - p0[1], pc[0] - p0[0]) % (2 * np.pi)
    largura = 2 * np.pi / n_setores
    return int(((angulo + largura / 2) // largura) % n_setores)
