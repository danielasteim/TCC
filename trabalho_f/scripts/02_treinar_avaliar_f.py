"""
02_treinar_avaliar_f.py

Treina e avalia o SVM de Uma Classe e a Floresta de Isolação com as 12
características (caracteristicas_f.py), em validação cruzada de k partições
sobre as sessões humanas (seção 3.6 da Metodologia).

Em cada partição, os dois modelos (Pipeline: ResiduoFitts -> WinsorizadorF
-> StandardScaler -> modelo) são ajustados só com os humanos de treino e
aplicados:
  - aos humanos de teste  -> taxa de falsos positivos (FPR, humanos rejeitados);
    o limiar de decisão é calibrado em cada partição para rejeitar
    config["fpr_alvo"] dos humanos de TREINO (mesmo ponto de operação nos dois);
  - às sessões de bots, se existirem -> taxa de detecção por família e AUC-ROC
    (humanos de teste x bots, pelo escore de decisão).

Análises:
  3.6.1  desempenho por modelo (média e desvio-padrão entre partições);
  3.6.2  ablação: retira uma característica (e um grupo) por vez e refaz o k-fold;
  3.6.3  comparação direta: sessões em que só um dos modelos rejeita, com
         teste exato de McNemar sobre os pares discordantes.

As sessões de bots são lidas de config['caminhos']['bots_sessoes'] (mesmo
formato JSON do coletor). A família de cada bot vem do campo "bot_familia" da
sessão ou, se ausente, do padrão "shape=A_vp=B" no nome do arquivo.

Saídas em trabalho_f/resultados/tabelas/:
  caracteristicas_humanos.csv   (cache da extração; use --recalcular para refazer)
  caracteristicas_bots.csv
  avaliacao_particoes.csv       uma linha por modelo x partição
  resumo_modelos.csv            média ± desvio por modelo
  deteccao_familias.csv         taxa de detecção por modelo x família (se houver bots)
  ablacao.csv                   variação das métricas ao retirar cada característica/grupo
  divergencias.csv              contagem de discordâncias e p-valor de McNemar
e os modelos ajustados com todos os humanos em trabalho_f/modelos/.

Uso:
    python trabalho_f/scripts/02_treinar_avaliar_f.py
    python trabalho_f/scripts/02_treinar_avaliar_f.py --sem-ablacao
    python trabalho_f/scripts/02_treinar_avaliar_f.py --bots caminho/das/sessoes_de_bots
"""

import argparse
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from caracteristicas_f import (CARACTERISTICAS, COLUNAS_EXTRAIDAS, GRUPOS,
                               Caracteristicas, ResiduoFitts, WinsorizadorF)
from common_f import caminho, carregar_config


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


MODELOS = {"svm": "SVM de Uma Classe", "floresta": "Floresta de Isolação"}


# ----------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Avaliação em k partições
# ----------------------------------------------------------------------
def avaliar(H: pd.DataFrame, B: pd.DataFrame, config: dict, colunas=None, guardar_predicoes=False):
    """
    Devolve (linhas por modelo x partição, predições) para um subconjunto de
    colunas. As predições (−1 = rejeitado) servem à comparação direta.
    """
    kf = KFold(config["k_folds"], shuffle=True, random_state=config["semente"])
    linhas, predicoes = [], []
    XB = B[COLUNAS_EXTRAIDAS] if B is not None and len(B) else None
    for particao, (tr, te) in enumerate(kf.split(H)):
        preds_particao = {}
        for nome in MODELOS:
            Xtr, Xte = H.iloc[tr][COLUNAS_EXTRAIDAS], H.iloc[te][COLUNAS_EXTRAIDAS]
            modelo = montar_modelo(nome, config, colunas).fit(Xtr)
            limiar = calibrar_limiar(modelo, Xtr, config["fpr_alvo"])
            fh = modelo.decision_function(Xte)
            ph = np.where(fh < limiar, -1, 1)
            linha = {"modelo": nome, "particao": particao, "limiar": limiar,
                     "fpr": float(np.mean(ph == -1)),
                     "fpr_limiar_padrao": float(np.mean(fh < 0))}
            preds_particao[nome] = {"humanos": ph}
            if XB is not None:
                fb = modelo.decision_function(XB)
                pb = np.where(fb < limiar, -1, 1)
                linha["deteccao"] = float(np.mean(pb == -1))
                linha["deteccao_limiar_padrao"] = float(np.mean(fb < 0))
                for familia, idx in B.groupby("familia").indices.items():
                    linha[f"det:{familia}"] = float(np.mean(pb[idx] == -1))
                # escore de anormalidade: quanto maior, mais "bot" (não depende do limiar)
                rotulos = np.r_[np.zeros(len(fh)), np.ones(len(fb))]
                linha["auc"] = float(roc_auc_score(rotulos, -np.r_[fh, fb]))
                preds_particao[nome]["bots"] = pb
            linhas.append(linha)
        if guardar_predicoes:
            predicoes.append(preds_particao)
    return pd.DataFrame(linhas), predicoes


def calibrar_limiar(modelo: Pipeline, X_treino: pd.DataFrame, fpr_alvo: float) -> float:
    """
    Limiar do escore de decisão que rejeita a fração fpr_alvo dos humanos de
    treino: o quantil fpr_alvo de f(x). Iguala o ponto de operação dos dois
    modelos, que nos limiares padrão (f(x) < 0) rejeitam frações diferentes.
    """
    return float(np.quantile(modelo.decision_function(X_treino), fpr_alvo))


def resumir(df: pd.DataFrame) -> pd.DataFrame:
    metricas = [c for c in df.columns if c not in ("modelo", "particao")]
    agg = df.groupby("modelo")[metricas].agg(["mean", "std"])
    agg.columns = [f"{m}_{e}" for m, e in agg.columns]
    return agg


def divergencias(predicoes) -> pd.DataFrame:
    """Pares discordantes entre os modelos, somados nas partições, e McNemar exato."""
    linhas = []
    for conjunto in ("humanos", "bots"):
        if conjunto not in predicoes[0]["svm"]:
            continue
        so_svm = so_floresta = ambos = nenhum = 0
        for p in predicoes:
            rs, rf = p["svm"][conjunto] == -1, p["floresta"][conjunto] == -1
            so_svm += int(np.sum(rs & ~rf))
            so_floresta += int(np.sum(~rs & rf))
            ambos += int(np.sum(rs & rf))
            nenhum += int(np.sum(~rs & ~rf))
        discordantes = so_svm + so_floresta
        p_valor = binomtest(so_svm, discordantes, 0.5).pvalue if discordantes else np.nan
        linhas.append({"conjunto": conjunto, "rejeitado_por_ambos": ambos, "aceito_por_ambos": nenhum,
                       "rejeitado_so_pelo_svm": so_svm, "rejeitado_so_pela_floresta": so_floresta,
                       "p_valor_mcnemar": p_valor})
    return pd.DataFrame(linhas)


def ablacao(H, B, config, base: pd.DataFrame) -> pd.DataFrame:
    """Retira uma característica, depois um grupo, por vez; compara com o conjunto completo."""
    referencia = base.groupby("modelo").mean(numeric_only=True)
    metricas = [m for m in ("fpr", "deteccao", "auc") if m in referencia.columns]
    casos = [(f"sem {c}", "caracteristica", [x for x in CARACTERISTICAS if x != c]) for c in CARACTERISTICAS]
    casos += [(f"sem grupo {g}", "grupo", [x for x in CARACTERISTICAS if x not in cols]) for g, cols in GRUPOS.items()]
    linhas = []
    for i, (rotulo, tipo, colunas) in enumerate(casos, 1):
        print(f"  ablação {i}/{len(casos)}: {rotulo}", end="\r")
        media = avaliar(H, B, config, colunas)[0].groupby("modelo").mean(numeric_only=True)
        for nome in MODELOS:
            linha = {"caso": rotulo, "tipo": tipo, "modelo": nome}
            for m in metricas:
                linha[m] = media.loc[nome, m]
                linha[f"delta_{m}"] = media.loc[nome, m] - referencia.loc[nome, m]
            linhas.append(linha)
    print()
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------
def principal():
    config = carregar_config()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--humanos", default=str(caminho(config, "humanos")))
    parser.add_argument("--bots", default=str(caminho(config, "bots_sessoes")))
    parser.add_argument("--saida", default=str(caminho(config, "tabelas")))
    parser.add_argument("--modelos", default=str(caminho(config, "modelos")))
    parser.add_argument("--recalcular", action="store_true", help="refaz a extração em vez de usar o cache CSV")
    parser.add_argument("--sem-ablacao", action="store_true")
    args = parser.parse_args()

    inicio = time.time()
    saida = Path(args.saida)
    extrator = Caracteristicas.de_config(config)

    print("Extraindo características dos humanos...")
    H = extrair_com_cache(Path(args.humanos), saida / "caracteristicas_humanos.csv", extrator, args.recalcular)
    pasta_bots = Path(args.bots)
    if pasta_bots.is_dir() and any(pasta_bots.glob("*.json")):
        print("Extraindo características dos bots...")
        B = extrair_com_cache(pasta_bots, saida / "caracteristicas_bots.csv", extrator, args.recalcular, com_familia=True)
    else:
        B = None
        print(f"Sem sessões de bots em {pasta_bots}: avaliação só com humanos (FPR).")
    print(f"Humanos: {len(H)} sessões" + (f" | bots: {len(B)} sessões, {B['familia'].nunique()} famílias" if B is not None else ""))

    print(f"Validação cruzada ({config['k_folds']} partições)...")
    particoes, predicoes = avaliar(H, B, config, guardar_predicoes=True)
    particoes.to_csv(saida / "avaliacao_particoes.csv", index=False)
    resumo = resumir(particoes)
    resumo.to_csv(saida / "resumo_modelos.csv")
    if B is not None:
        cols = [c for c in particoes.columns if c.startswith("det:")]
        fam = particoes.groupby("modelo")[cols].mean().T
        fam.index = [c[4:] for c in fam.index]
        fam.index.name = "familia"
        fam.to_csv(saida / "deteccao_familias.csv")

    div = divergencias(predicoes)
    div.to_csv(saida / "divergencias.csv", index=False)

    if not args.sem_ablacao:
        print("Ablação...")
        ablacao(H, B, config, particoes).to_csv(saida / "ablacao.csv", index=False)

    destino_modelos = Path(args.modelos)
    destino_modelos.mkdir(parents=True, exist_ok=True)
    for nome in MODELOS:
        final = montar_modelo(nome, config).fit(H[COLUNAS_EXTRAIDAS])
        limiar = calibrar_limiar(final, H[COLUNAS_EXTRAIDAS], config["fpr_alvo"])
        joblib.dump({"modelo": final, "limiar": limiar}, destino_modelos / f"{nome}_f.joblib")

    print("\nResumo (média ± desvio entre partições):")
    for nome, rotulo in MODELOS.items():
        r = resumo.loc[nome]
        texto = (f"  {rotulo}: FPR {r['fpr_mean']:.3f} ± {r['fpr_std']:.3f}"
                 f" (limiar padrão: {r['fpr_limiar_padrao_mean']:.3f})")
        if B is not None:
            texto += (f" | detecção {r['deteccao_mean']:.3f} ± {r['deteccao_std']:.3f}"
                      f" | AUC {r['auc_mean']:.3f} ± {r['auc_std']:.3f}")
        print(texto)
    print("\nDivergências entre os modelos:")
    print(div.to_string(index=False))
    print(f"\nTabelas em {saida} | modelos em {destino_modelos} | {time.time() - inicio:.0f} s")


if __name__ == "__main__":
    principal()
