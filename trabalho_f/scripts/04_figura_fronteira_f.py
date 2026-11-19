"""
04_figura_fronteira_f.py

Figura ilustrativa da fronteira de decisão do SVM de Uma Classe e da Floresta
de Isolação. Como os modelos do trabalho usam 12 características, a fronteira
real não cabe num plano; aqui os dois modelos são treinados SÓ com duas
características (padrão: razao_trajetoria x tempo_parado), com os mesmos
hiperparâmetros, a mesma winsorização/padronização e o mesmo limiar
calibrado (fpr_alvo) do 02_treinar_avaliar_f.py. É um modelo 2D de
ilustração, não o modelo avaliado.

Os humanos são divididos em treino (80 %) e teste (20 %):
  brancos   humanos de treino (amostra)
  verdes    humanos de teste aceitos
  vermelhos humanos de teste rejeitados
O fundo é o escore de decisão f(x) (mais claro = mais "normal") e a linha
vermelha é o limiar calibrado.

Lê o cache trabalho_f/resultados/tabelas/caracteristicas_humanos.csv
(gerado pelo 02). Saída, uma figura por modelo:
trabalho_f/resultados/figuras/fronteira_svm_f.png/.pdf e fronteira_floresta_f.png/.pdf

Uso:
    python trabalho_f/scripts/04_figura_fronteira_f.py
    python trabalho_f/scripts/04_figura_fronteira_f.py --x desvio_maximo --y tempo_parado
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from caracteristicas_f import WinsorizadorF
from common_f import caminho, carregar_config

ROTULOS = {
    "razao_trajetoria": "Razão da trajetória",
    "desvio_maximo": "Desvio máximo (fração de L)",
    "curvatura_media": "Curvatura média (rad)",
    "jerk_adimensional": "Jerk adimensional",
    "razao_pico_media": "Razão pico/média",
    "entropia_velocidade": "Entropia da velocidade (bits)",
    "instante_pico": "Instante do pico",
    "desaceleracao_final": "Desaceleração final",
    "dist_clique": "Distância do clique",
    "tempo_parado": "Tempo parado antes do clique (s)",
}


def montar(nome, config):
    if nome == "svm":
        c = config["svm"]
        # gamma='auto' = 1/n_características: com 2 colunas vale 0,5 (e não 1/12)
        modelo = OneClassSVM(kernel=c["kernel"], nu=c["nu"], gamma=c["gamma"])
    else:
        c = config["floresta"]
        modelo = IsolationForest(n_estimators=c["n_estimators"], max_samples=c["max_samples"],
                                 contamination=c["contamination"], random_state=config["semente"])
    inf, sup = config["winsorizacao"]
    return Pipeline([("winsor", WinsorizadorF(inf, sup)), ("padronizacao", StandardScaler()), ("modelo", modelo)])


def principal():
    config = carregar_config()
    parser = argparse.ArgumentParser(description="Figura da fronteira de decisão em 2 características.")
    parser.add_argument("--x", default="razao_trajetoria")
    parser.add_argument("--y", default="tempo_parado")
    parser.add_argument("--amostra", type=int, default=400, help="pontos de treino e de teste exibidos")
    args = parser.parse_args()

    H = pd.read_csv(caminho(config, "tabelas") / "caracteristicas_humanos.csv", index_col="arquivo")
    X = H[[args.x, args.y]]
    Xtr, Xte = train_test_split(X, test_size=0.2, random_state=config["semente"])
    rng = np.random.default_rng(config["semente"])

    # grade dentro dos limites de winsorização do treino (fora deles o modelo vê o valor limitado)
    inf, sup = config["winsorizacao"]
    lim_inf, lim_sup = Xtr.quantile(inf), Xtr.quantile(sup)
    folga = 0.08 * (lim_sup - lim_inf)
    xs = np.linspace(lim_inf[args.x] - folga[args.x], lim_sup[args.x] + folga[args.x], 300)
    ys = np.linspace(lim_inf[args.y] - folga[args.y], lim_sup[args.y] + folga[args.y], 300)
    gx, gy = np.meshgrid(xs, ys)
    grade = pd.DataFrame({args.x: gx.ravel(), args.y: gy.ravel()})

    destino = caminho(config, "figuras")
    destino.mkdir(parents=True, exist_ok=True)
    # a mesma amostra de pontos nas duas figuras, para comparar só a fronteira
    itr = rng.choice(len(Xtr), min(args.amostra, len(Xtr)), replace=False)
    ite = rng.choice(len(Xte), min(args.amostra, len(Xte)), replace=False)

    titulos = {"svm": "SVM de Uma Classe", "floresta": "Floresta de Isolação"}
    for nome in ("svm", "floresta"):
        modelo = montar(nome, config).fit(Xtr)
        limiar = float(np.quantile(modelo.decision_function(Xtr), config["fpr_alvo"]))
        f = modelo.decision_function(grade).reshape(gx.shape)
        rejeitado = modelo.decision_function(Xte) < limiar

        fig, eixo = plt.subplots(figsize=(7.5, 5.5))
        # fundo: escore de decisão; região aceita (f >= limiar) destacada em amarelo
        fundo = eixo.contourf(gx, gy, f, levels=20, cmap="Blues_r")
        eixo.contourf(gx, gy, f, levels=[limiar, f.max()], colors="gold", alpha=0.35)
        eixo.contour(gx, gy, f, levels=[limiar], colors="darkred", linewidths=2, linestyles="solid")
        barra = fig.colorbar(fundo, ax=eixo)
        barra.set_label("escore de decisão f(x)  (maior = mais semelhante aos humanos)")

        aceitos, rejeitados = ite[~rejeitado[ite]], ite[rejeitado[ite]]
        kw = dict(s=18, edgecolors="k", linewidths=0.5)
        eixo.scatter(Xtr.iloc[itr][args.x], Xtr.iloc[itr][args.y], c="white", label="humanos de treino", **kw)
        eixo.scatter(Xte.iloc[aceitos][args.x], Xte.iloc[aceitos][args.y], c="green", label="humanos de teste aceitos", **kw)
        eixo.scatter(Xte.iloc[rejeitados][args.x], Xte.iloc[rejeitados][args.y], c="red", label="humanos de teste rejeitados", **kw)
        eixo.fill([], [], color="gold", alpha=0.35, label="região aceita")
        eixo.plot([], [], color="darkred", lw=2, label="limiar de decisão")

        eixo.set_xlim(xs[0], xs[-1])
        eixo.set_ylim(ys[0], ys[-1])
        eixo.set_title(titulos[nome])
        eixo.set_xlabel(ROTULOS.get(args.x, args.x))
        eixo.set_ylabel(ROTULOS.get(args.y, args.y))
        eixo.legend(loc="upper right", fontsize=8, framealpha=0.9)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(destino / f"fronteira_{nome}_f.{ext}", dpi=200)
        plt.close(fig)
        print(f"{titulos[nome]}: limiar {limiar:.4f}, FPR no teste {rejeitado.mean():.3f} "
              f"-> {destino / f'fronteira_{nome}_f.png'}")


if __name__ == "__main__":
    principal()
