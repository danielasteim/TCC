"""
02_treinar_f.py

Treina o SVM de Uma Classe e a Floresta de Isolação com as 12 características
(caracteristicas_f.py) usando todas as sessões humanas.

Cada modelo é um Pipeline ResiduoFitts -> WinsorizadorF -> StandardScaler ->
modelo (modelos_f.py). O limiar de decisão é calibrado para rejeitar
config["fpr_alvo"] dos humanos de treino, igualando o ponto de operação dos
dois modelos.

Saídas:
  trabalho_f/resultados/tabelas/caracteristicas_humanos.csv
      cache da extração (use --recalcular para refazer); lido também pelo
      05_avaliar_f.py e pelo 04_figura_fronteira_f.py
  trabalho_f/modelos/svm_f.joblib, floresta_f.joblib
      {"modelo": Pipeline ajustado, "limiar": float}

A avaliação (validação cruzada, ablação, comparação entre modelos) fica no
05_avaliar_f.py.

Uso:
    python trabalho_f/scripts/02_treinar_f.py
    python trabalho_f/scripts/02_treinar_f.py --recalcular
"""

import argparse
import time
from pathlib import Path

import joblib

from caracteristicas_f import COLUNAS_EXTRAIDAS, Caracteristicas
from common_f import caminho, carregar_config
from modelos_f import MODELOS, calibrar_limiar, extrair_com_cache, montar_modelo, selecionar_humanos


def principal():
    config = carregar_config()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--humanos", default=str(caminho(config, "humanos")))
    parser.add_argument("--saida", default=str(caminho(config, "tabelas")),
                        help="pasta do cache de características")
    parser.add_argument("--modelos", default=str(caminho(config, "modelos")))
    parser.add_argument("--recalcular", action="store_true", help="refaz a extração em vez de usar o cache CSV")
    args = parser.parse_args()

    inicio = time.time()
    extrator = Caracteristicas.de_config(config)

    print("Extraindo características dos humanos...")
    H = selecionar_humanos(
        extrair_com_cache(Path(args.humanos), Path(args.saida) / "caracteristicas_humanos.csv",
                          extrator, args.recalcular), config)
    X = H[COLUNAS_EXTRAIDAS]
    print(f"Humanos: {len(H)} sessões")

    destino = Path(args.modelos)
    destino.mkdir(parents=True, exist_ok=True)
    for nome, rotulo in MODELOS.items():
        modelo = montar_modelo(nome, config).fit(X)
        limiar = calibrar_limiar(modelo, X, config["fpr_alvo"])
        joblib.dump({"modelo": modelo, "limiar": limiar}, destino / f"{nome}_f.joblib")
        print(f"  {rotulo}: limiar {limiar:.4f}")

    print(f"\nModelos em {destino} | {time.time() - inicio:.0f} s")


if __name__ == "__main__":
    principal()
