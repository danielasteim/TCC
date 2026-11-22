"""
05_avaliar_f.py

Avalia o SVM de Uma Classe e a Floresta de Isolação em validação cruzada de k
partições sobre as sessões humanas (seção "Avaliação dos modelos" da
Metodologia). Roda depois do 02_treinar_f.py (cache de características
humanas) e do 03_bot_replay_f.py (sessões de bots).

Convenção: bot = classe positiva (anômala); humano = classe negativa (normal).
  VP = bot rejeitado      FN = bot aceito
  FP = humano rejeitado   VN = humano aceito

Em cada partição, os dois modelos (mesmo Pipeline do treino, modelos_f.py)
são ajustados só com os humanos de treino. O limiar de decisão é calibrado
para rejeitar config["fpr_alvo"] dos humanos de TREINO (mesmo ponto de
operação nos dois). O teste de cada partição reúne os humanos de teste e
TODAS as sessões de bots (que nunca entram no treino).

Análises:
  0. comparação entre os níveis de tempo T0 e T1 (métricas e curva ROC);
  1. desempenho por modelo (média e desvio-padrão entre partições):
       no ponto de operação: VP, FN, FP, VN, TPR (deteccao), FPR, precisão, F1, F2
       sem limiar: AUC-ROC e curva ROC média (média vertical entre partições)
       custo: tempo de treino, tempo de inferência por sessão, tamanho do
              modelo serializado e sua complexidade (vetores de suporte / nós)
  2. por família de bots: TPR e AUC (bots da família x humanos de teste), e os
     mesmos valores agrupados por forma e por perfil;
  3. variação da quantidade de bots: reamostra os bots do teste para várias
     prevalências (config["variacao_bots"]) e recalcula as métricas;
  4. ablação: retira uma característica (e um grupo) por vez e refaz o k-fold;
     compara AUC e TPR (geral e por família) com o conjunto completo.
As análises 1 a 4 usam só os bots do nível principal.

Níveis de tempo: os bots T0 e T1 são comparados UMA vez (análise 0), com os
mesmos modelos e os mesmos humanos de teste; como os dois níveis têm a mesma
quantidade de bots, a prevalência é igual e todas as métricas são comparáveis.
Todas as demais análises usam só os bots do nível config["nivel_principal"]
(T1), que é o conjunto de teste do trabalho.

Contagens: FP e VN podem ser somados nas partições (cada humano é testado uma
vez); VP e FN não, porque os mesmos bots são avaliados nas k partições.

As sessões de bots são lidas de config['caminhos']['bots_sessoes'] (mesmo
formato JSON do coletor). A família de cada bot vem do campo "bot_familia" da
sessão ou, se ausente, do padrão "shape=A_vp=B" no nome do arquivo.

Saídas em trabalho_f/resultados/tabelas/:
  caracteristicas_humanos.csv   (cache; gerado aqui se o 02 ainda não rodou)
  caracteristicas_bots.csv      (cache; use --recalcular para refazer)
  avaliacao_particoes.csv       uma linha por modelo x partição
  resumo_modelos.csv            média ± desvio por modelo
  deteccao_familias.csv         TPR e AUC por modelo x família (se houver bots)
  deteccao_fatores.csv          TPR e AUC por modelo x forma e x perfil
  comparacao_niveis_particoes.csv  métricas por modelo x partição x nível (T0, T1)
  comparacao_niveis.csv         média ± desvio por modelo x nível
  curva_roc_niveis.csv          curva ROC média por modelo x nível (T0, T1)
  curva_roc.csv                 TPR médio ± desvio por modelo numa grade de FPR (nível principal)
  variacao_bots.csv             métricas por modelo x prevalência de bots (nível principal)
  ablacao.csv                   variação de AUC, TPR e FPR ao retirar cada característica/grupo
  ablacao_familias.csv          variação do TPR por família em cada caso da ablação

Uso:
    python trabalho_f/scripts/05_avaliar_f.py
    python trabalho_f/scripts/05_avaliar_f.py --recalcular
    python trabalho_f/scripts/05_avaliar_f.py --sem-ablacao
    python trabalho_f/scripts/05_avaliar_f.py --bots caminho/das/sessoes_de_bots
"""

import argparse
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold

from caracteristicas_f import CARACTERISTICAS, COLUNAS_EXTRAIDAS, GRUPOS, Caracteristicas
from common_f import FAMILIAS_FORMA, FAMILIAS_PERFIL, caminho, carregar_config
from modelos_f import MODELOS, calibrar_limiar, extrair_com_cache, montar_modelo, selecionar_humanos

GRADE_FPR = np.linspace(0.0, 1.0, 101)


# ----------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------
def _razao(numerador: float, denominador: float) -> float:
    return float(numerador / denominador) if denominador else float("nan")


def f_beta(precisao: float, revocacao: float, beta: float) -> float:
    """F_beta = (1 + beta²)·P·R / (beta²·P + R) (Manning, Raghavan e Schütze, 2008)."""
    if np.isnan(precisao) or np.isnan(revocacao):
        return float("nan")
    return _razao((1 + beta ** 2) * precisao * revocacao, beta ** 2 * precisao + revocacao)


def metricas_ponto_operacao(humano_rejeitado: np.ndarray, bot_rejeitado: np.ndarray) -> dict:
    """Métricas da matriz de confusão, com bot como classe positiva."""
    vp, fn = int(bot_rejeitado.sum()), int((~bot_rejeitado).sum())
    fp, vn = int(humano_rejeitado.sum()), int((~humano_rejeitado).sum())
    revocacao = _razao(vp, vp + fn)
    precisao = _razao(vp, vp + fp)
    return {
        "vp": vp, "fn": fn, "fp": fp, "vn": vn,
        "deteccao": revocacao,                       # revocação (TPR)
        "precisao": precisao,
        "f1": f_beta(precisao, revocacao, 1.0),
        "f2": f_beta(precisao, revocacao, 2.0),
    }


def auc_roc(fh: np.ndarray, fb: np.ndarray) -> float:
    """
    AUC entre humanos de teste e bots. decision_function é maior para amostras
    normais; o sinal é invertido para que maior signifique mais anômalo.
    """
    rotulos = np.r_[np.zeros(len(fh)), np.ones(len(fb))]
    return float(roc_auc_score(rotulos, -np.r_[fh, fb]))


def memoria_modelo(modelo) -> dict:
    """
    Tamanho do Pipeline ajustado quando serializado (o que precisa ficar em
    memória ou em disco para classificar) e o que determina esse tamanho:
    vetores de suporte (SVM) ou nós das árvores (Floresta). Não mede o pico
    de memória durante o treino.
    """
    estimador = modelo[-1]
    return {
        "tamanho_modelo_kb": len(pickle.dumps(modelo)) / 1024.0,
        "n_vetores_suporte": (len(estimador.support_) if hasattr(estimador, "support_") else np.nan),
        "n_nos_arvores": (sum(a.tree_.node_count for a in estimador.estimators_)
                          if hasattr(estimador, "estimators_") else np.nan),
    }


# ----------------------------------------------------------------------
# Avaliação em k partições
# ----------------------------------------------------------------------
def avaliar(H: pd.DataFrame, B: pd.DataFrame, config: dict, colunas=None, guardar_escores=False):
    """
    Devolve (linhas por modelo x partição, escores) para um subconjunto de
    colunas. `escores` é uma lista por partição de
    {modelo: {"limiar", "fh", "fb", "indices_humanos"}}; serve à curva ROC
    e à variação da quantidade de bots.
    """
    kf = KFold(config["k_folds"], shuffle=True, random_state=config["semente"])
    linhas, escores = [], []
    XB = B[COLUNAS_EXTRAIDAS] if B is not None and len(B) else None
    for particao, (tr, te) in enumerate(kf.split(H)):
        escores_particao = {}
        Xtr, Xte = H.iloc[tr][COLUNAS_EXTRAIDAS], H.iloc[te][COLUNAS_EXTRAIDAS]
        for nome in MODELOS:
            t0 = time.perf_counter()
            modelo = montar_modelo(nome, config, colunas).fit(Xtr)
            tempo_treino = time.perf_counter() - t0
            limiar = calibrar_limiar(modelo, Xtr, config["fpr_alvo"])

            t0 = time.perf_counter()
            fh = modelo.decision_function(Xte)
            fb = modelo.decision_function(XB) if XB is not None else None
            tempo_inferencia = time.perf_counter() - t0
            n_inferidas = len(fh) + (len(fb) if fb is not None else 0)

            humano_rejeitado = fh < limiar
            linha = {"modelo": nome, "particao": particao, "limiar": limiar,
                     "n_humanos_teste": len(fh),
                     "fpr": float(np.mean(humano_rejeitado)),
                     "fpr_limiar_padrao": float(np.mean(fh < 0)),
                     "tempo_treino_s": tempo_treino,
                     "tempo_inferencia_ms_por_sessao": 1000.0 * tempo_inferencia / n_inferidas,
                     **memoria_modelo(modelo)}
            if fb is not None:
                bot_rejeitado = fb < limiar
                linha["n_bots_teste"] = len(fb)
                linha["prevalencia_bots"] = len(fb) / (len(fb) + len(fh))
                linha.update(metricas_ponto_operacao(humano_rejeitado, bot_rejeitado))
                linha["deteccao_limiar_padrao"] = float(np.mean(fb < 0))
                linha["auc"] = auc_roc(fh, fb)
                for familia, idx in B.groupby("familia").indices.items():
                    linha[f"det:{familia}"] = float(np.mean(bot_rejeitado[idx]))
                    # AUC da família: só os bots dela contra todos os humanos de teste
                    linha[f"auc:{familia}"] = auc_roc(fh, fb[idx])
            linhas.append(linha)
            escores_particao[nome] = {"limiar": limiar, "fh": fh, "fb": fb, "indices_humanos": te}
        if guardar_escores:
            escores.append(escores_particao)
    return pd.DataFrame(linhas), escores


def resumir(df: pd.DataFrame) -> pd.DataFrame:
    metricas = [c for c in df.columns if c not in ("modelo", "particao")]
    agg = df.groupby("modelo")[metricas].agg(["mean", "std"])
    agg.columns = [f"{m}_{e}" for m, e in agg.columns]
    return agg


def curva_roc_media(escores, cenarios: dict) -> pd.DataFrame:
    """
    Média vertical das curvas ROC das partições (Fawcett, 2006, seção 8.1),
    por cenário: o TPR de cada partição é interpolado numa grade fixa de FPR.
    """
    linhas = []
    for cenario, idx in cenarios.items():
        for nome in MODELOS:
            tprs = []
            for p in escores:
                fh, fb = p[nome]["fh"], p[nome]["fb"][idx]
                rotulos = np.r_[np.zeros(len(fh)), np.ones(len(fb))]
                fpr, tpr, _ = roc_curve(rotulos, -np.r_[fh, fb])
                tprs.append(np.interp(GRADE_FPR, fpr, tpr))
            tprs = np.array(tprs)
            for x, media, desvio in zip(GRADE_FPR, tprs.mean(axis=0), tprs.std(axis=0, ddof=1)):
                linhas.append({"cenario": cenario, "modelo": nome, "fpr": x, "tpr_mean": media, "tpr_std": desvio})
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------
# Detecção por família de bots
# ----------------------------------------------------------------------
PADRAO_FAMILIA = re.compile(r"shape=(\d+)_vp=(\d+)(?:_(T\d+))?")


def fatores_da_familia(familia: str) -> dict:
    """'shape=1_vp=2_T1' -> forma, perfil e nível de tempo (rótulos de common_f)."""
    m = PADRAO_FAMILIA.fullmatch(familia)
    if not m:
        return {"forma": familia, "perfil": "", "nivel": ""}
    return {"forma": FAMILIAS_FORMA[int(m.group(1))], "perfil": FAMILIAS_PERFIL[int(m.group(2))],
            "nivel": m.group(3) or ""}


def deteccao_por_familia(particoes: pd.DataFrame) -> pd.DataFrame:
    """TPR e AUC de cada família: média e desvio entre partições, por modelo (formato longo)."""
    familias = [c[4:] for c in particoes.columns if c.startswith("det:")]
    linhas = []
    for familia in familias:
        for nome, grupo in particoes.groupby("modelo"):
            linhas.append({"familia": familia, **fatores_da_familia(familia), "modelo": nome,
                           "tpr_mean": grupo[f"det:{familia}"].mean(), "tpr_std": grupo[f"det:{familia}"].std(),
                           "auc_mean": grupo[f"auc:{familia}"].mean(), "auc_std": grupo[f"auc:{familia}"].std()})
    return pd.DataFrame(linhas)


def deteccao_por_fator(escores, B: pd.DataFrame) -> pd.DataFrame:
    """
    TPR e AUC agrupando os bots por forma, por perfil e por nível de tempo
    (cada fator isoladamente). Média e desvio entre partições.
    """
    fatores = pd.DataFrame([fatores_da_familia(f) for f in B["familia"]])
    linhas = []
    for fator in ("forma", "perfil", "nivel"):
        if fatores[fator].nunique() < 2:   # ex.: só o nível principal está no teste
            continue
        for valor, idx in fatores.groupby(fator).indices.items():
            if valor == "":
                continue
            for nome in MODELOS:
                tprs = [float(np.mean(p[nome]["fb"][idx] < p[nome]["limiar"])) for p in escores]
                aucs = [auc_roc(p[nome]["fh"], p[nome]["fb"][idx]) for p in escores]
                linhas.append({"fator": fator, "valor": valor, "n_bots": len(idx), "modelo": nome,
                               "tpr_mean": np.mean(tprs), "tpr_std": np.std(tprs, ddof=1),
                               "auc_mean": np.mean(aucs), "auc_std": np.std(aucs, ddof=1)})
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------
# Cenários de teste: todos os bots, só T0, só T1
# ----------------------------------------------------------------------
METRICAS_CENARIO = ["deteccao", "fpr", "precisao", "f1", "f2", "auc"]


def nivel_dos_bots(B: pd.DataFrame) -> np.ndarray:
    """Nível de tempo de cada bot ('T0', 'T1' ou '' se a família não informa)."""
    return np.array([fatores_da_familia(f)["nivel"] for f in B["familia"]])


def cenarios_de_teste(B: pd.DataFrame) -> dict:
    """{'T0': índices dos bots T0, 'T1': ...}; sem níveis, {'todos': todos os bots}."""
    niveis = nivel_dos_bots(B)
    cenarios = {nivel: np.flatnonzero(niveis == nivel) for nivel in sorted(set(niveis) - {""})}
    return cenarios or {"todos": np.arange(len(B))}


def metricas_por_cenario(escores, cenarios: dict) -> pd.DataFrame:
    """Uma linha por modelo x partição x cenário, com os humanos de teste da partição."""
    linhas = []
    for particao, p in enumerate(escores):
        for cenario, idx in cenarios.items():
            for nome in MODELOS:
                fh, fb, limiar = p[nome]["fh"], p[nome]["fb"][idx], p[nome]["limiar"]
                linha = {"modelo": nome, "cenario": cenario, "particao": particao,
                         "n_humanos_teste": len(fh), "n_bots_teste": len(fb),
                         "prevalencia_bots": len(fb) / (len(fb) + len(fh)),
                         "fpr": float(np.mean(fh < limiar))}
                linha.update(metricas_ponto_operacao(fh < limiar, fb < limiar))
                linha["auc"] = auc_roc(fh, fb)
                linhas.append(linha)
    return pd.DataFrame(linhas)


def resumir_cenarios(df: pd.DataFrame) -> pd.DataFrame:
    metricas = [c for c in df.columns if c not in ("modelo", "cenario", "particao")]
    agg = df.groupby(["cenario", "modelo"])[metricas].agg(["mean", "std"])
    agg.columns = [f"{m}_{e}" for m, e in agg.columns]
    return agg.reset_index()


# ----------------------------------------------------------------------
# Variação da quantidade de bots
# ----------------------------------------------------------------------
def variacao_bots(escores, config: dict, cenarios: dict) -> pd.DataFrame:
    """
    Em cada cenário, reamostra os bots do teste, sem reposição, para que
    representem cada prevalência pedida (bots / (bots + humanos de teste)), e
    recalcula as métricas com os escores já obtidos (os modelos não são
    retreinados). A mesma amostra de bots é usada nos dois modelos. Média e
    desvio sobre partições x repetições. Prevalências que exigem mais bots do
    que o cenário tem são ignoradas.
    """
    c = config["variacao_bots"]
    rng = np.random.default_rng(config["semente"])
    linhas = []
    for cenario, idx_cenario in cenarios.items():
        for prevalencia in c["prevalencias"]:
            for particao, p in enumerate(escores):
                n_h = len(p["svm"]["fh"])
                n_b = int(round(prevalencia / (1 - prevalencia) * n_h))
                if not 1 <= n_b <= len(idx_cenario):
                    print(f"  {cenario}, prevalência {prevalencia}: exigiria {n_b} bots "
                          f"(há {len(idx_cenario)}); ignorada")
                    break
                for repeticao in range(c["repeticoes"]):
                    idx = rng.choice(idx_cenario, n_b, replace=False)
                    for nome in MODELOS:
                        fh, fb, limiar = p[nome]["fh"], p[nome]["fb"][idx], p[nome]["limiar"]
                        linha = {"cenario": cenario, "modelo": nome, "prevalencia": prevalencia, "n_bots": n_b,
                                 "particao": particao, "repeticao": repeticao,
                                 "fpr": float(np.mean(fh < limiar)), "auc": auc_roc(fh, fb)}
                        linha.update(metricas_ponto_operacao(fh < limiar, fb < limiar))
                        linhas.append(linha)
    if not linhas:
        return pd.DataFrame()
    df = pd.DataFrame(linhas)
    grupos = df.groupby(["cenario", "modelo", "prevalencia"])
    agg = grupos[METRICAS_CENARIO].agg(["mean", "std"])
    agg.columns = [f"{m}_{e}" for m, e in agg.columns]
    # n_bots varia em 1 entre partições quando o nº de humanos de teste difere
    agg.insert(0, "n_bots", grupos["n_bots"].mean().round().astype(int))
    return agg.reset_index()


# ----------------------------------------------------------------------
# Ablação
# ----------------------------------------------------------------------
def ablacao(H, B, config, base: pd.DataFrame, base_cenarios: pd.DataFrame = None, cenarios: dict = None):
    """
    Retira uma característica, depois um grupo, por vez, e compara com o
    conjunto completo. Devolve (geral, por família, por cenário):
      geral:       AUC, TPR e FPR e suas variações (o FPR fica perto de fpr_alvo
                   por construção, pois o limiar é recalibrado em cada caso);
      por família: variação do TPR de cada família;
      por cenário: variação de AUC e TPR nos testes "todos", T0 e T1.
    """
    referencia = base.groupby("modelo").mean(numeric_only=True)
    metricas = [m for m in ("auc", "deteccao", "fpr") if m in referencia.columns]
    familias = [c for c in referencia.columns if c.startswith("det:")]
    ref_cenarios = (base_cenarios.groupby(["cenario", "modelo"])[["auc", "deteccao"]].mean()
                    if base_cenarios is not None else None)
    casos = [(f"sem {c}", "caracteristica", [x for x in CARACTERISTICAS if x != c]) for c in CARACTERISTICAS]
    casos += [(f"sem grupo {g}", "grupo", [x for x in CARACTERISTICAS if x not in cols]) for g, cols in GRUPOS.items()]
    linhas, linhas_familias, linhas_cenarios = [], [], []
    for i, (rotulo, tipo, colunas) in enumerate(casos, 1):
        print(f"  ablação {i}/{len(casos)}: {rotulo}", end="\r")
        particoes, escores = avaliar(H, B, config, colunas, guardar_escores=ref_cenarios is not None)
        media = particoes.groupby("modelo").mean(numeric_only=True)
        for nome in MODELOS:
            linha = {"caso": rotulo, "tipo": tipo, "modelo": nome}
            for m in metricas:
                linha[m] = media.loc[nome, m]
                linha[f"delta_{m}"] = media.loc[nome, m] - referencia.loc[nome, m]
            linhas.append(linha)
            for coluna in familias:
                linhas_familias.append({"caso": rotulo, "tipo": tipo, "modelo": nome, "familia": coluna[4:],
                                        "tpr": media.loc[nome, coluna],
                                        "delta_tpr": media.loc[nome, coluna] - referencia.loc[nome, coluna]})
        if ref_cenarios is not None:
            media_c = metricas_por_cenario(escores, cenarios).groupby(["cenario", "modelo"])[["auc", "deteccao"]].mean()
            for (cenario, nome), valores in media_c.iterrows():
                ref = ref_cenarios.loc[(cenario, nome)]
                linhas_cenarios.append({"caso": rotulo, "tipo": tipo, "cenario": cenario, "modelo": nome,
                                        "auc": valores["auc"], "delta_auc": valores["auc"] - ref["auc"],
                                        "tpr": valores["deteccao"], "delta_tpr": valores["deteccao"] - ref["deteccao"]})
    print()
    return pd.DataFrame(linhas), pd.DataFrame(linhas_familias), pd.DataFrame(linhas_cenarios)


# ----------------------------------------------------------------------
def principal():
    config = carregar_config()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--humanos", default=str(caminho(config, "humanos")))
    parser.add_argument("--bots", default=str(caminho(config, "bots_sessoes")))
    parser.add_argument("--saida", default=str(caminho(config, "tabelas")))
    parser.add_argument("--recalcular", action="store_true", help="refaz a extração em vez de usar o cache CSV")
    parser.add_argument("--sem-ablacao", action="store_true")
    args = parser.parse_args()

    inicio = time.time()
    saida = Path(args.saida)
    extrator = Caracteristicas.de_config(config)

    print("Características dos humanos...")
    H = selecionar_humanos(
        extrair_com_cache(Path(args.humanos), saida / "caracteristicas_humanos.csv", extrator, args.recalcular), config)
    pasta_bots = Path(args.bots)
    if pasta_bots.is_dir() and any(pasta_bots.glob("*.json")):
        print("Características dos bots...")
        B = extrair_com_cache(pasta_bots, saida / "caracteristicas_bots.csv", extrator, args.recalcular, com_familia=True)
        n_arquivos = sum(1 for _ in pasta_bots.glob("*.json"))
        if len(B) < 0.9 * n_arquivos:
            print(f"  AVISO: o cache tem {len(B)} bots, mas a pasta tem {n_arquivos} sessões. "
                  f"Rode com --recalcular para refazer a extração.")
    else:
        B = None
        print(f"Sem sessões de bots em {pasta_bots}: avaliação só com humanos (FPR).")
    print(f"Humanos: {len(H)} sessões" + (f" | bots: {len(B)} sessões, {B['familia'].nunique()} famílias" if B is not None else ""))

    # Análise 0: T0 x T1, uma única vez, com todos os bots. Depois, só o nível principal.
    comparacao = None
    principal_nivel = config.get("nivel_principal")
    if B is not None and len(set(nivel_dos_bots(B)) - {""}) > 1:
        print("Comparação entre os níveis de tempo...")
        _, escores_todos = avaliar(H, B, config, guardar_escores=True)
        niveis = cenarios_de_teste(B)
        por_particao = metricas_por_cenario(escores_todos, niveis)
        por_particao.to_csv(saida / "comparacao_niveis_particoes.csv", index=False)
        comparacao = resumir_cenarios(por_particao)
        comparacao.to_csv(saida / "comparacao_niveis.csv", index=False)
        curva_roc_media(escores_todos, niveis).to_csv(saida / "curva_roc_niveis.csv", index=False)
        if principal_nivel:
            B = B[nivel_dos_bots(B) == principal_nivel]
            if not len(B):
                raise SystemExit(f"Nenhum bot do nível principal '{principal_nivel}'.")
            print(f"Demais análises: só bots {principal_nivel} ({len(B)} sessões, {B['familia'].nunique()} famílias)")

    print(f"Validação cruzada ({config['k_folds']} partições)...")
    particoes, escores = avaliar(H, B, config, guardar_escores=True)
    particoes.to_csv(saida / "avaliacao_particoes.csv", index=False)
    resumo = resumir(particoes)
    resumo.to_csv(saida / "resumo_modelos.csv")
    cenarios = None
    if B is not None:
        cenarios = {principal_nivel or "todos": np.arange(len(B))}
        deteccao_por_familia(particoes).to_csv(saida / "deteccao_familias.csv", index=False)
        deteccao_por_fator(escores, B).to_csv(saida / "deteccao_fatores.csv", index=False)
        curva_roc_media(escores, cenarios).to_csv(saida / "curva_roc.csv", index=False)

    if B is not None and "variacao_bots" in config:
        print("Variação da quantidade de bots...")
        variacao_bots(escores, config, cenarios).to_csv(saida / "variacao_bots.csv", index=False)

    if not args.sem_ablacao:
        print("Ablação...")
        geral, por_familia, _ = ablacao(H, B, config, particoes)
        geral.to_csv(saida / "ablacao.csv", index=False)
        if len(por_familia):
            por_familia.to_csv(saida / "ablacao_familias.csv", index=False)

    print(f"\nResumo{' — bots ' + principal_nivel if principal_nivel and B is not None else ''}"
          f" (média ± desvio entre partições):")
    for nome, rotulo in MODELOS.items():
        r = resumo.loc[nome]
        print(f"  {rotulo}:")
        print(f"    FPR {r['fpr_mean']:.3f} ± {r['fpr_std']:.3f} (limiar padrão: {r['fpr_limiar_padrao_mean']:.3f})"
              f" | treino {r['tempo_treino_s_mean']:.2f} s"
              f" | inferência {r['tempo_inferencia_ms_por_sessao_mean']:.3f} ms/sessão"
              f" | modelo {r['tamanho_modelo_kb_mean']:.0f} kB")
        if B is not None:
            print("    " + " | ".join(f"{rot} {r[m + '_mean']:.3f} ± {r[m + '_std']:.3f}" for rot, m in
                                      [("TPR", "deteccao"), ("precisão", "precisao"),
                                       ("F1", "f1"), ("F2", "f2"), ("AUC", "auc")]))
            print(f"    prevalência de bots no teste: {r['prevalencia_bots_mean']:.1%}")
    if comparacao is not None:
        print("\nComparação entre os níveis de tempo (média entre partições):")
        colunas = ["cenario", "modelo", "n_bots_teste_mean", "prevalencia_bots_mean"] + \
                  [f"{m}_mean" for m in METRICAS_CENARIO]
        print(comparacao[colunas].round(3).to_string(index=False))
    print(f"\nTabelas em {saida} | {time.time() - inicio:.0f} s")


if __name__ == "__main__":
    principal()
