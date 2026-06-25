"""
train_models.py

Treina os dois modelos de classificação de uma classe descritos no
capítulo de Metodologia do TCC: SVM de Uma Classe e Floresta de
Isolação. Ambos são treinados exclusivamente com sessões humanas.

Pipeline de pré-processamento (nessa ordem):
    1. Extração das 12 features oficiais via Features (feature_processor.py)
    2. Winsorização (limita outliers extremos aos percentis 0.5%/99.5%
       aprendidos no próprio conjunto de treino — etapa adicional não
       descrita no capítulo, necessária por causa da cauda pesada de
       algumas features como razao_trajetoria e jerk_medio)
    3. Escalonamento via StandardScaler (também uma etapa adicional —
       necessária porque o SVM de Uma Classe com kernel RBF é sensível
       à escala das features)

Uso:
    python train_models.py --dir-dados captcha_data/ --dir-saida modelos/
"""

import argparse
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from features import Features, Winsorizador


COLUNAS_FEATURES = [
    "jerk_medio", "jerk_dp", "razao_trajetoria", "entropia_velocidade",
    "curvatura_media", "curvatura_dp", "taxa_retrocesso",
    "tempo_reacao", "dist_clique", "vel_aproximacao",
    "dp_temporal", "entropia_temporal",
]

SEMENTE_ALEATORIA = 42  # fixada para reprodutibilidade (relevante p/ a Floresta de Isolação)


def construir_dataset(dir_dados: str) -> pd.DataFrame:
    """Extrai as 12 features oficiais de todas as sessões em dir_dados."""
    extrator = Features()
    return extrator.transform_batch_from_dir(dir_dados)


def treinar(dir_dados: str, dir_saida: str) -> None:
    caminho_saida = Path(dir_saida)
    caminho_saida.mkdir(parents=True, exist_ok=True)

    print(f"Extraindo features de '{dir_dados}'...")
    t0 = time.time()
    df = construir_dataset(dir_dados)
    print(f"  {len(df)} sessões processadas em {time.time() - t0:.1f}s")

    X = df[COLUNAS_FEATURES]

    # --- Pré-processamento: winsorização + escalonamento ---
    print("Ajustando Winsorizador (percentis 0.5% / 99.5%)...")
    winsorizador = Winsorizador(percentil_inferior=0.005, percentil_superior=0.995)
    X_winsorizado = winsorizador.ajustar_transformar(X)

    print("Ajustando StandardScaler...")
    escalonador = StandardScaler()
    X_escalonado = escalonador.fit_transform(X_winsorizado)

    # --- SVM de Uma Classe ---
    # kernel='rbf'; nu=0.05
    # gamma='auto'.
    print("Treinando SVM de Uma Classe (kernel='rbf', nu=0.05, gamma='auto')...")
    t0 = time.time()
    svm_uma_classe = OneClassSVM(kernel="rbf", nu=0.05, gamma="auto")
    svm_uma_classe.fit(X_escalonado)
    tempo_svm = time.time() - t0
    print(f"  treinado em {tempo_svm:.2f}s")

    # --- Floresta de Isolação ---
    # n_estimators=250; max_samples=1.0; contamination='auto'.
    # Ver seção 3.5 do capítulo de Metodologia.
    print("Treinando Floresta de Isolação (n_estimators=250, max_samples=1.0, contamination='auto')...")
    t0 = time.time()
    floresta_isolacao = IsolationForest(
        n_estimators=250,
        max_samples=1.0,
        contamination="auto",
        random_state=SEMENTE_ALEATORIA,
    )
    floresta_isolacao.fit(X_escalonado)
    tempo_floresta = time.time() - t0
    print(f"  treinado em {tempo_floresta:.2f}s")

    # --- Serialização (joblib, conforme seção 3.5) ---
    winsorizador.salvar(caminho_saida / "winsorizador.joblib")
    joblib.dump(escalonador, caminho_saida / "escalonador.joblib")
    joblib.dump(svm_uma_classe, caminho_saida / "svm_uma_classe.joblib")
    joblib.dump(floresta_isolacao, caminho_saida / "floresta_isolacao.joblib")
    joblib.dump(COLUNAS_FEATURES, caminho_saida / "colunas_features.joblib")

    print()
    print(f"Modelos salvos em '{caminho_saida}/':")
    for nome_arquivo in ["winsorizador.joblib", "escalonador.joblib", "svm_uma_classe.joblib",
                          "floresta_isolacao.joblib", "colunas_features.joblib"]:
        tamanho_kb = (caminho_saida / nome_arquivo).stat().st_size / 1024
        print(f"  {nome_arquivo:<25} {tamanho_kb:>8.1f} KB")

    print()
    print(f"Sessões usadas no treino: {len(df)}")
    print(f"Features utilizadas: {len(COLUNAS_FEATURES)}")
    print(f"Tempo de treino - SVM de Uma Classe: {tempo_svm:.2f}s")
    print(f"Tempo de treino - Floresta de Isolação: {tempo_floresta:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina os modelos SVM de Uma Classe e Floresta de Isolação.")
    parser.add_argument("--dir-dados", required=True, help="Diretório com as sessões .json válidas")
    parser.add_argument("--dir-saida", default="modelos", help="Diretório de saída dos modelos treinados")
    args = parser.parse_args()

    treinar(args.dir_dados, args.dir_saida)
