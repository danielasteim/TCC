"""
avaliacao_modelos.py

Classe que reúne os critérios de avaliação e comparação dos modelos
(seção 3.6 do capítulo de Metodologia). Carrega o dataset uma única
vez no construtor, e cada critério da seção 3.6 é implementado como um
método — adicionados um por vez, na mesma lógica da classe Features.

Métodos disponíveis até agora:
    - validacao_cruzada()  : seção 3.6.1, k-fold (k=5)

Uso:
    from avaliacao_modelos import AvaliacaoModelos

    av = AvaliacaoModelos(dir_dados="captcha_data/")
    av.validacao_cruzada(dir_saida="resultados/")
"""

from pathlib import Path
import time
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from features import Features, Winsorizador


class AvaliacaoModelos:
    # As 12 features oficiais do capítulo de Metodologia.
    COLUNAS_FEATURES = [
        "jerk_medio", "jerk_dp", "razao_trajetoria", "entropia_velocidade",
        "curvatura_media", "curvatura_dp", "taxa_retrocesso",
        "tempo_reacao", "dist_clique", "vel_aproximacao",
        "dp_temporal", "entropia_temporal",
    ]

    SEMENTE_ALEATORIA = 42
    K_FOLDS = 5

    def __init__(self, dir_dados: str):
        """Extrai as features de todas as sessões em dir_dados uma única vez."""
        self.dir_dados = dir_dados
        self.X = self._construir_dataset(dir_dados)
        # Guarda o resultado de cada método de avaliação já executado,
        # pra reaproveitar (ex.: montar um relatório final) sem precisar
        # rodar tudo de novo.
        self.resultados = {}

    def _construir_dataset(self, dir_dados: str) -> pd.DataFrame:
        extrator = Features()
        df = extrator.transform_batch_from_dir(dir_dados)
        return df[self.COLUNAS_FEATURES].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Helpers internos (não são critérios de avaliação em si)
    # ------------------------------------------------------------------
    def _treinar_pipeline(self, X_treino: pd.DataFrame):
        """Ajusta winsorizador + escalonador + SVM + Floresta usando só os dados recebidos."""
        winsorizador = Winsorizador(percentil_inferior=0.005, percentil_superior=0.995)
        X_w = winsorizador.ajustar_transformar(X_treino)

        escalonador = StandardScaler()
        X_s = escalonador.fit_transform(X_w)

        svm = OneClassSVM(kernel="rbf", nu=0.05, gamma="auto")
        svm.fit(X_s)

        floresta = IsolationForest(
            n_estimators=250,
            max_samples=1.0,
            contamination="auto",
            random_state=self.SEMENTE_ALEATORIA,
        )
        floresta.fit(X_s)

        return winsorizador, escalonador, svm, floresta

    def _avaliar_pipeline(self,
                      winsorizador,
                      escalonador,
                      modelo,
                      X_val: pd.DataFrame,
                      y_val: pd.Series = None):

        X_w = winsorizador.transformar(X_val)
        X_s = escalonador.transform(X_w)

        scores = modelo.decision_function(X_s)
        predicoes = modelo.predict(X_s)

        n_testadas = len(scores)
        n_rejeitadas = int(np.sum(predicoes == -1))
        fpr = n_rejeitadas / n_testadas

        resultados = pd.DataFrame({
            "score": scores,
            "predicao": predicoes
        })

        if y_val is not None:
            resultados["classe_real"] = y_val.values

        return resultados, fpr, n_testadas, n_rejeitadas

    # ------------------------------------------------------------------
    # 3.6.1 — Validação cruzada k-fold
    # ------------------------------------------------------------------
    def validacao_cruzada(self, dir_saida: str =None) -> dict:

        X = self.X

        kf = KFold(
            n_splits=self.K_FOLDS,
            shuffle=True,
            random_state=self.SEMENTE_ALEATORIA
        )

        resultados = {"svm": [], "floresta": []}
        scores_por_fold = {"svm": [], "floresta": []}

        print(f"Validação cruzada k-fold (k={self.K_FOLDS}) — {len(X)} sessões\n")

        for fold_idx, (idx_treino, idx_val) in enumerate(kf.split(X), start=1):

            X_treino = X.iloc[idx_treino]
            X_val = X.iloc[idx_val]

            winsorizador, escalonador, svm, floresta = self._treinar_pipeline(X_treino)

            modelos = {
                "svm": svm,
                "floresta": floresta
            }

            print(f"Fold {fold_idx}/{self.K_FOLDS} — treino: {len(idx_treino)}, validação: {len(idx_val)}")

            for nome_modelo, modelo in modelos.items():

                X_w = winsorizador.transformar(X_val)
                X_s = escalonador.transform(X_w)

                scores = modelo.decision_function(X_s)
                predicoes = modelo.predict(X_s)

                n_testadas = len(scores)
                n_rejeitadas = int(np.sum(predicoes == -1))
                fpr = n_rejeitadas / n_testadas

                q1, q3 = np.percentile(scores, [25, 75])

                resultados[nome_modelo].append({
                    "fold": fold_idx,
                    "n_testadas": n_testadas,
                    "n_rejeitadas": n_rejeitadas,
                    "fpr": fpr,
                    "mediana": float(np.median(scores)),
                    "q1": float(q1),
                    "q3": float(q3),
                    "min": float(scores.min()),
                    "max": float(scores.max()),
                })

                dados_fold = pd.DataFrame({
                    "fold": fold_idx,
                    "indice_original": idx_val,
                    "score": scores,
                    "predicao": predicoes
                })

                scores_por_fold[nome_modelo].append(dados_fold)

                print(
                    f"  {nome_modelo:<10}"
                    f"FPR={fpr:.4f}  "
                    f"Rejeitadas={n_rejeitadas}/{n_testadas}  "
                    f"Mediana={np.median(scores):.4f}  "
                    f"IQR=[{q1:.4f}, {q3:.4f}]"
                )

            print()

        print("=" * 78)
        print("RESUMO DA VALIDAÇÃO CRUZADA")
        print("=" * 78)

        resumo = {}

        for nome_modelo in ["svm", "floresta"]:

            df_resultados = pd.DataFrame(resultados[nome_modelo])

            scores_por_fold[nome_modelo] = pd.concat(
                scores_por_fold[nome_modelo],
                ignore_index=True
            )

            std_medianas = df_resultados["mediana"].std()
            fpr_medio = df_resultados["fpr"].mean()
            fpr_std = df_resultados["fpr"].std()

            sobreposicoes = []

            for i in range(len(df_resultados) - 1):

                a_q1, a_q3 = df_resultados.iloc[i][["q1", "q3"]]
                b_q1, b_q3 = df_resultados.iloc[i + 1][["q1", "q3"]]

                sobreposicoes.append(
                    (a_q1 <= b_q3) and (b_q1 <= a_q3)
                )

            print(f"\n{nome_modelo.upper()}:")
            print(df_resultados.to_string(index=False))
            print(f"Desvio padrão das medianas: {std_medianas:.5f}")
            print(f"FPR médio: {fpr_medio:.4f} ± {fpr_std:.4f}")
            print(f"IQRs sobrepostos: {sum(sobreposicoes)}/{len(sobreposicoes)}")

            resumo[nome_modelo] = {
                "por_fold": df_resultados,
                "scores": scores_por_fold[nome_modelo],
                "std_medianas": std_medianas,
                "fpr_medio": fpr_medio,
                "fpr_std": fpr_std,
                "sobreposicoes_iqr": sobreposicoes,
            }

            if dir_saida is not None:

                os.makedirs(dir_saida, exist_ok=True)

                df_resultados.to_csv(
                    os.path.join(dir_saida, f"{nome_modelo}_resumo.csv"),
                    index=False
                )

                scores_por_fold[nome_modelo].to_csv(
                    os.path.join(dir_saida, f"{nome_modelo}_scores.csv"),
                    index=False
                )

        return resumo

    def benchmark_inferencia(self, n_sessoes: int = 30, repeticoes: int = 100):
        """
        Mede o tempo médio de inferência do One-Class SVM e da Floresta de
        Isolação utilizando um conjunto fixo de sessões.

        O tempo medido inclui:
            - Winsorização
            - Escalonamento
            - Predição (decision_function)

        Não inclui o treinamento.
        """

        X_teste = self.X.iloc[:n_sessoes]

        winsorizador, escalonador, svm, floresta = self._treinar_pipeline(self.X)

        print("=" * 70)
        print(f"BENCHMARK DE INFERÊNCIA ({n_sessoes} sessões)")
        print("=" * 70)

        for nome, modelo in [
            ("One-Class SVM", svm),
            ("Isolation Forest", floresta),
        ]:

            tempos = []

            for _ in range(repeticoes):

                inicio = time.perf_counter()

                X_w = winsorizador.transformar(X_teste)
                X_s = escalonador.transform(X_w)

                _ = modelo.decision_function(X_s)

                fim = time.perf_counter()

                tempos.append((fim - inicio) * 1000)

            tempos = np.array(tempos)

            print(f"\n{nome}")
            print(f"Tempo médio : {tempos.mean():.3f} ms")
            print(f"Desvio padr.: {tempos.std():.3f} ms")
            print(f"Mínimo      : {tempos.min():.3f} ms")
            print(f"Máximo      : {tempos.max():.3f} ms")
            print(f"Tempo/sessão: {tempos.mean()/n_sessoes:.4f} ms")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Avaliação e comparação dos modelos treinados.")
    parser.add_argument("--dir-dados", required=True, help="Diretório com as sessões .json válidas")
    parser.add_argument("--dir-saida", default="resultados_avaliacao", help="Diretório de saída dos resultados")
    args = parser.parse_args()

    av = AvaliacaoModelos(dir_dados=args.dir_dados)
    av.benchmark_inferencia(n_sessoes=30, repeticoes=100)