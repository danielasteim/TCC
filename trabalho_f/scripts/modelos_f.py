"""
modelos_f.py

Peças comuns ao treino (02_treinar_f.py) e à avaliação (05_avaliar_f.py):
montagem do Pipeline de cada modelo, calibração do limiar e extração das
características com cache em CSV.

SelecionarColunas fica aqui (e não no script) para que os modelos salvos com
joblib possam ser carregados por qualquer script que importe este módulo.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from caracteristicas_f import Caracteristicas, ResiduoFitts, WinsorizadorF

MODELOS = {"svm": "SVM de Uma Classe", "floresta": "Floresta de Isolação"}


class SelecionarColunas(BaseEstimator, TransformerMixin):
    """Mantém só as colunas pedidas (usado na ablação)."""

    def __init__(self, colunas=None):
        self.colunas = colunas

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.colunas] if self.colunas is not None else X


def montar_modelo(nome: str, config: dict, colunas=None) -> Pipeline:
    if nome == "svm":
        c = config["svm"]
        modelo = OneClassSVM(kernel=c["kernel"], nu=c["nu"], gamma=c["gamma"])
    else:
        c = config["floresta"]
        modelo = IsolationForest(n_estimators=c["n_estimators"], max_samples=c["max_samples"],
                                 contamination=c["contamination"], random_state=config["semente"])
    inf, sup = config["winsorizacao"]
    return Pipeline([
        ("fitts", ResiduoFitts()),
        ("selecao", SelecionarColunas(colunas)),
        ("winsor", WinsorizadorF(inf, sup)),
        ("padronizacao", StandardScaler()),
        ("modelo", modelo),
    ])


def calibrar_limiar(modelo: Pipeline, X_treino: pd.DataFrame, fpr_alvo: float) -> float:
    """
    Limiar do escore de decisão que rejeita a fração fpr_alvo dos humanos de
    treino: o quantil fpr_alvo de f(x). Iguala o ponto de operação dos dois
    modelos, que nos limiares padrão (f(x) < 0) rejeitam frações diferentes.
    """
    return float(np.quantile(modelo.decision_function(X_treino), fpr_alvo))


# ----------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------
def selecionar_humanos(H: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Base humana usada no treino e na avaliação: as config["n_humanos"]
    primeiras sessões em ordem cronológica (o nome do arquivo começa pela data
    e hora da sessão). Sem a chave, ou com null, usa todas.
    """
    n = config.get("n_humanos")
    H = H.sort_index()
    if n is None or n >= len(H):
        return H
    return H.iloc[:n]


def familia_do_bot(nome_arquivo: str, sessao: dict) -> str:
    if "bot_familia" in sessao:
        return str(sessao["bot_familia"])
    m = re.search(r"shape=(\w+?)_vp=(\w+?)(?:[_.]|$)", nome_arquivo)
    return f"shape={m.group(1)}_vp={m.group(2)}" if m else "bot"


def extrair_com_cache(pasta: Path, destino: Path, extrator: Caracteristicas,
                      recalcular: bool, com_familia: bool = False) -> pd.DataFrame:
    if destino.exists() and not recalcular:
        return pd.read_csv(destino, index_col="arquivo")
    arquivos = sorted(pasta.glob("*.json"))
    sessoes = [json.load(open(a, encoding="utf-8")) for a in arquivos]
    X = extrator.extrair_lote(sessoes, nomes=[a.name for a in arquivos])
    invalidas = X.attrs["invalidas"]
    if invalidas:
        print(f"  {len(invalidas)} sessões inválidas em {pasta.name} (ex.: {invalidas[:3]})")
    if com_familia:
        familias = {a.name: familia_do_bot(a.name, s) for a, s in zip(arquivos, sessoes)}
        X["familia"] = [familias[i] for i in X.index]
    destino.parent.mkdir(parents=True, exist_ok=True)
    X.to_csv(destino)
    return X
