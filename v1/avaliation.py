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

    def _avaliar_pipeline(self, winsorizador, escalonador, modelo, X_val: pd.DataFrame):
        X_w = winsorizador.transformar(X_val)
        X_s = escalonador.transform(X_w)
        scores = modelo.decision_function(X_s)
        predicoes = modelo.predict(X_s)
        fpr = float((predicoes == -1).mean())
        return scores, fpr

    # ------------------------------------------------------------------
    # 3.6.1 — Validação cruzada k-fold
    # ------------------------------------------------------------------
    def validacao_cruzada(self, dir_saida: str = None) -> dict:
        """
        Validação cruzada k-fold (k=5). Em cada fold, treina o pipeline
        completo (winsorizador + escalonador + modelo) do zero usando
        4 partições, e avalia na partição restante (held-out).

        Registra, por fold: FPR, mediana e IQR do score de decisão.
        Mede a estabilidade entre folds pelo desvio padrão das medianas
        e pela sobreposição dos IQRs entre folds consecutivos.

        Se dir_saida for informado, salva um .csv por modelo e os
        scores brutos de cada fold (scores_por_fold.joblib, reutilizável
        depois nos histogramas/boxplots da seção 3.6.1).
        """
        X = self.X
        kf = KFold(n_splits=self.K_FOLDS, shuffle=True, random_state=self.SEMENTE_ALEATORIA)

        resultados = {"svm": [], "floresta": []}
        scores_por_fold = {"svm": [], "floresta": []}

        print(f"Validação cruzada k-fold (k={self.K_FOLDS}) — {len(X)} sessões\n")

        for fold_idx, (idx_treino, idx_val) in enumerate(kf.split(X), start=1):
            X_treino = X.iloc[idx_treino]
            X_val = X.iloc[idx_val]

            winsorizador, escalonador, svm, floresta = self._treinar_pipeline(X_treino)

            scores_svm, fpr_svm = self._avaliar_pipeline(winsorizador, escalonador, svm, X_val)
            scores_floresta, fpr_floresta = self._avaliar_pipeline(winsorizador, escalonador, floresta, X_val)

            scores_por_fold["svm"].append(scores_svm)
            scores_por_fold["floresta"].append(scores_floresta)

            print(f"Fold {fold_idx}/{self.K_FOLDS} — treino: {len(idx_treino)}, validação: {len(idx_val)}")
            for nome_modelo, scores, fpr in [("svm", scores_svm, fpr_svm), ("floresta", scores_floresta, fpr_floresta)]:
                q1, q3 = np.percentile(scores, [25, 75])
                resultados[nome_modelo].append({
                    "fold": fold_idx,
                    "fpr": fpr,
                    "mediana": float(np.median(scores)),
                    "q1": float(q1),
                    "q3": float(q3),
                    "min": float(scores.min()),
                    "max": float(scores.max()),
                })
                r = resultados[nome_modelo][-1]
                print(f"  {nome_modelo:<10} -> FPR={r['fpr']:.4f}  mediana={r['mediana']:.4f}  "
                      f"IQR=[{r['q1']:.4f}, {r['q3']:.4f}]")
            print()

        print("=" * 78)
        print("RESUMO DA VALIDAÇÃO CRUZADA (k=5)")
        print("=" * 78)

        resumo = {}
        for nome_modelo in ["svm", "floresta"]:
            df_resultados = pd.DataFrame(resultados[nome_modelo])
            std_medianas = df_resultados["mediana"].std()
            fpr_medio = df_resultados["fpr"].mean()
            fpr_std = df_resultados["fpr"].std()

            sobreposicoes = []
            for i in range(len(df_resultados) - 1):
                a_q1, a_q3 = df_resultados.iloc[i][["q1", "q3"]]
                b_q1, b_q3 = df_resultados.iloc[i + 1][["q1", "q3"]]
                sobreposicoes.append((a_q1 <= b_q3) and (b_q1 <= a_q3))

            print(f"\n{nome_modelo.upper()}:")
            print(df_resultados.to_string(index=False))
            print(f"  Desvio padrão das medianas entre folds: {std_medianas:.5f}")
            print(f"  FPR médio: {fpr_medio:.4f} (desvio padrão: {fpr_std:.4f})")
            print(f"  IQRs com sobreposição entre folds consecutivos: {sum(sobreposicoes)}/{len(sobreposicoes)}")

            resumo[nome_modelo] = {
                "por_fold": df_resultados,
                "std_medianas": std_medianas,
                "fpr_medio": fpr_medio,
                "fpr_std": fpr_std,
                "sobreposicoes_iqr": sobreposicoes,
            }

            if dir_saida:
                caminho_saida = Path(dir_saida)
                caminho_saida.mkdir(parents=True, exist_ok=True)
                df_resultados.to_csv(caminho_saida / f"validacao_cruzada_{nome_modelo}.csv", index=False)

        if dir_saida:
            joblib.dump(scores_por_fold, Path(dir_saida) / "scores_por_fold.joblib")
            print(f"\nResultados salvos em '{dir_saida}/'")

        resumo["scores_por_fold"] = scores_por_fold
        self.resultados["validacao_cruzada"] = resumo
        return resumo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Avaliação e comparação dos modelos treinados.")
    parser.add_argument("--dir-dados", required=True, help="Diretório com as sessões .json válidas")
    parser.add_argument("--dir-saida", default="resultados_avaliacao", help="Diretório de saída dos resultados")
    args = parser.parse_args()

    av = AvaliacaoModelos(dir_dados=args.dir_dados)
    av.validacao_cruzada(dir_saida=args.dir_saida)
