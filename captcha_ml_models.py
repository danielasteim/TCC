"""
captcha_ml_models.py
=====================
Classificador comportamental de CAPTCHA usando One-Class Learning.

Lê o dataset.csv gerado pelo captcha_feature_pipeline.py e treina
dois modelos: One-Class SVM e Isolation Forest.

ESTRATÉGIA: todos os dados de treino são humanos (classe positiva).
Os modelos aprendem "como humanos se comportam" e rejeitam padrões
que fogem significativamente desse perfil (bots).

Uso:
    python captcha_ml_models.py

Dependências:
    pip install numpy pandas scikit-learn matplotlib seaborn joblib
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ──────────────────────────────────────────────────────────────────────────────

DATASET_PATH  = "dataset.csv"
MODELS_DIR    = "models"
PLOTS_DIR     = "plots"

# Colunas que não são features — não entram no modelo
META_COLS = {"session_id", "user", "win_x", "win_y", "win_diag"}

# Features selecionadas para o modelo (subconjunto das 37 disponíveis)
# Remova colunas desta lista se quiser experimentar subconjuntos
FEATURE_COLS = [
    # Jerk
    "jerk_mean", "jerk_std", "jerk_max", "jerk_skew",
    # Distância / trajetória
    "dist_euclidiana", "dist_caminho", "ratio_dist_desl",
    # Curvatura
    "curvatura_mean", "curvatura_std", "curvatura_max", "n_mudancas_bruscas",
    # Retrocessos
    "retrocessos_x", "retrocessos_y", "n_retrocessos", "taxa_retrocesso",
    # Clique
    "time_to_click", "n_movimentos", "densidade_amostral", "passo_medio", "cv_passos",
    # Centro do clique
    "d_centro_clique", "offset_x_clique", "offset_y_clique",
    # Intervalos temporais
    "mean_dt", "var_dt", "cv_temporal", "std_dt", "skew_dt",
    "p25_dt", "p75_dt", "p95_dt",
    # Aproximação e desvio
    "vel_aproximacao", "ratio_desaceleracao",
    "desvio_lateral_max", "desvio_lateral_mean",
    # Entropia
    "entropia_vel", "entropia_vel_norm",
]


# ──────────────────────────────────────────────────────────────────────────────
# 1. CARREGAMENTO DO DATASET
# ──────────────────────────────────────────────────────────────────────────────

class BehavioralCaptchaClassifier:
    """
    Classificador comportamental de CAPTCHA usando One-Class Learning.

    Treina com dados exclusivamente humanos e rejeita sessões que
    desviam significativamente do padrão aprendido (bots).
    """

    def __init__(self):
        self.scaler               = RobustScaler()
        self.ocsvm_model          = None
        self.isolation_forest_model = None
        self.feature_names        = []

    # ── Carregamento ──────────────────────────────────────────────────────────

    def load_dataset(self, path: str = DATASET_PATH) -> pd.DataFrame:
        """
        Carrega o dataset.csv gerado pelo captcha_feature_pipeline.py.
        Remove colunas de metadados e colunas ausentes, trata NaN.
        """
        print("=" * 70)
        print("CARREGANDO DATASET")
        print("=" * 70)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Arquivo '{path}' não encontrado.\n"
                "Execute captcha_feature_pipeline.py primeiro para gerar o dataset."
            )

        df = pd.read_csv(path)
        print(f"  Linhas carregadas : {len(df)}")
        print(f"  Colunas totais    : {len(df.columns)}")

        # Seleciona apenas as colunas de feature definidas em FEATURE_COLS
        available = [c for c in FEATURE_COLS if c in df.columns]
        missing   = [c for c in FEATURE_COLS if c not in df.columns]

        if missing:
            print(f"\n  [AVISO] Colunas ausentes no dataset (serão ignoradas): {missing}")

        df_feat = df[available].copy()

        # Trata valores infinitos e NaN
        df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
        n_nan = df_feat.isna().sum().sum()
        if n_nan > 0:
            print(f"  [INFO] {n_nan} valores NaN encontrados — preenchidos com mediana da coluna.")
            df_feat.fillna(df_feat.median(), inplace=True)

        self.feature_names = df_feat.columns.tolist()

        print(f"\n  Features para treino : {len(self.feature_names)}")
        print(f"  Sessões válidas      : {len(df_feat)}")

        return df_feat

    # ── Análise da diversidade humana ─────────────────────────────────────────

    def analyze_human_diversity(self, df: pd.DataFrame):
        """
        Exibe estatísticas das features para entender a variância natural
        dos dados humanos coletados.
        """
        print("\n" + "=" * 70)
        print("ANÁLISE DE DIVERSIDADE COMPORTAMENTAL HUMANA")
        print("=" * 70)
        print("Alta variância = usuários diversos (positivo!), não 'outliers'.\n")

        key = [
            "time_to_click", "n_movimentos", "ratio_dist_desl",
            "cv_temporal",   "taxa_retrocesso", "entropia_vel_norm",
            "jerk_std",      "curvatura_mean",  "ratio_desaceleracao",
        ]
        cols = [c for c in key if c in df.columns]

        print(f"  {'Feature':<30} {'Mín':>8} {'Média':>8} {'Máx':>8} {'Std':>8}  CV")
        print("  " + "-" * 68)

        for col in cols:
            mn  = df[col].min()
            me  = df[col].mean()
            mx  = df[col].max()
            std = df[col].std()
            cv  = std / me if me != 0 else 0
            nivel = "Baixa" if cv < 0.3 else "Média" if cv < 0.7 else "Alta"
            print(f"  {col:<30} {mn:>8.3f} {me:>8.3f} {mx:>8.3f} {std:>8.3f}  {cv:.2f} ({nivel})")

        print("\n  💡 Diversidade alta = capturou diferentes perfis de usuário.")

    # ── Treinamento ───────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame):
        """
        Treina One-Class SVM e Isolation Forest sobre o dataset humano.

        Testa diferentes valores de nu / contamination e seleciona
        automaticamente o que gera menor taxa de falso positivo
        mantendo robustez (≤ 2% de falsos positivos no treino).
        """
        print("\n" + "=" * 70)
        print("TREINAMENTO DOS MODELOS ONE-CLASS")
        print("=" * 70)
        print("Todos os dados são humanos — modelos aprendem o perfil humano completo.\n")

        X = pd.DataFrame(self.scaler.fit_transform(df), columns=df.columns)

        # ── One-Class SVM ─────────────────────────────────────────────────────
        print("-" * 70)
        print("One-Class SVM")
        print("-" * 70)
        print(f"  {'nu':<8} {'Anomalias':>10} {'Taxa':>8}")

        # Testa nu em faixa ampla — nu define a fração máxima de suporte vetores
        # (quanto maior o nu, mais apertada a fronteira e mais bots detectados,
        # mas também mais falsos positivos em humanos).
        # gamma='scale' é obrigatório com muitas features (evita kernel colapsado).
        nu_values = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
        best_nu   = 0.05   # padrão conservador
        best_nu_rate = 1.0

        for nu in nu_values:
            m    = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
            m.fit(X)
            preds     = m.predict(X)
            anomalies = (preds == -1).sum()
            rate      = anomalies / len(preds)
            print(f"  {nu:<8.3f} {anomalies:>10}    {rate:>7.1%}")
            # Seleciona o maior nu que mantém falsos positivos ≤ 5%
            if rate <= 0.05 and nu > best_nu:
                best_nu      = nu
                best_nu_rate = rate

        print(f"\n  ✅ Selecionado nu={best_nu} (taxa de falso positivo: {best_nu_rate:.1%})")
        self.ocsvm_model = OneClassSVM(nu=best_nu, kernel='rbf', gamma='scale')
        self.ocsvm_model.fit(X)

        # ── Isolation Forest ──────────────────────────────────────────────────
        print("\n" + "-" * 70)
        print("Isolation Forest")
        print("-" * 70)
        print(f"  {'contam.':<8} {'Anomalias':>10} {'Taxa':>8}")

        cont_values = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
        best_cont      = 0.05
        best_cont_rate = 1.0

        for cont in cont_values:
            m    = IsolationForest(contamination=cont, n_estimators=200, random_state=42)
            m.fit(X)
            preds     = m.predict(X)
            anomalies = (preds == -1).sum()
            rate      = anomalies / len(preds)
            print(f"  {cont:<8.3f} {anomalies:>10}    {rate:>7.1%}")
            # Seleciona o maior contamination que mantém falsos positivos ≤ 5%
            if rate <= 0.05 and cont > best_cont:
                best_cont      = cont
                best_cont_rate = rate

        print(f"\n  ✅ Selecionado contamination={best_cont} (taxa de falso positivo: {best_cont_rate:.1%})")
        self.isolation_forest_model = IsolationForest(
            contamination=best_cont, n_estimators=200, random_state=42
        )
        self.isolation_forest_model.fit(X)

        # ── Avaliação final no treino ─────────────────────────────────────────
        print("\n" + "=" * 70)
        print("AVALIAÇÃO FINAL NO DATASET DE TREINO")
        print("=" * 70)

        for nome, modelo in [("One-Class SVM", self.ocsvm_model),
                              ("Isolation Forest", self.isolation_forest_model)]:
            preds     = modelo.predict(X)
            anomalies = (preds == -1).sum()
            rate      = anomalies / len(preds)
            status    = "✅" if rate <= 0.05 else "⚠️ "
            print(f"  {status} {nome:<22}: {anomalies}/{len(df)} anomalias ({rate:.1%} falso positivo)")

        return X

    # ── Predição ──────────────────────────────────────────────────────────────

    def predict_from_row(self, row: pd.Series) -> dict:
        """
        Faz predição para uma linha do dataset (pd.Series com as features).
        Retorna dict com resultado de cada modelo.
        """
        X = self.scaler.transform(
            pd.DataFrame([row[self.feature_names].values], columns=self.feature_names)
        )
        results = {}

        if self.ocsvm_model:
            pred  = self.ocsvm_model.predict(X)[0]
            score = self.ocsvm_model.score_samples(X)[0]
            results["ocsvm"] = {
                "predicao":    "Humano" if pred == 1 else "Bot/Anomalia",
                "score":       score,
                "confianca":   abs(score),
            }

        if self.isolation_forest_model:
            pred  = self.isolation_forest_model.predict(X)[0]
            score = self.isolation_forest_model.score_samples(X)[0]
            results["isolation_forest"] = {
                "predicao":    "Humano" if pred == 1 else "Bot/Anomalia",
                "score":       score,
                "confianca":   abs(score),
            }

        return results

    def test_on_samples(self, df: pd.DataFrame, n: int = 10):
        """
        Testa os modelos em n amostras aleatórias do dataset de treino.
        Todas devem ser classificadas como humano — erros indicam falso positivo.
        """
        print("\n" + "=" * 70)
        print(f"TESTE EM {n} AMOSTRAS ALEATÓRIAS DO TREINO")
        print("=" * 70)
        print("Todas são humanas — modelos devem aceitar a maioria.\n")

        sample = df.sample(min(n, len(df)), random_state=42)
        ocsvm_rejeicoes = 0
        if_rejeicoes    = 0

        for i, (_, row) in enumerate(sample.iterrows(), 1):
            res = self.predict_from_row(row)
            ocsvm_r = res["ocsvm"]["predicao"]
            if_r    = res["isolation_forest"]["predicao"]

            print(f"  Amostra {i:>2}:  SVM → {ocsvm_r:<15}  IF → {if_r}")

            if ocsvm_r != "Humano": ocsvm_rejeicoes += 1
            if if_r    != "Humano": if_rejeicoes    += 1

        print("\n" + "-" * 70)
        print(f"  One-Class SVM    rejeitou: {ocsvm_rejeicoes}/{n} ({ocsvm_rejeicoes/n:.0%} falso positivo)")
        print(f"  Isolation Forest rejeitou: {if_rejeicoes}/{n}  ({if_rejeicoes/n:.0%} falso positivo)")

        if ocsvm_rejeicoes / n > 0.05:
            print("\n  ⚠️  SVM com alta taxa de falso positivo — considere reduzir nu.")
        if if_rejeicoes / n > 0.05:
            print("  ⚠️  IF com alta taxa de falso positivo — considere reduzir contamination.")

    # ── Gráficos ──────────────────────────────────────────────────────────────

    def create_plots(self, df: pd.DataFrame):
        """
        Gera visualizações das distribuições das features principais
        e salva em plots/human_behavior_analysis.png.
        """
        os.makedirs(PLOTS_DIR, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle("Análise de Comportamento Humano — Dataset CAPTCHA", fontsize=14)

        plots = [
            ("time_to_click",       "Tempo até o clique (s)",         "steelblue"),
            ("ratio_dist_desl",     "Razão distância / deslocamento",  "seagreen"),
            ("cv_temporal",         "CV dos intervalos temporais",     "darkorange"),
            ("taxa_retrocesso",     "Taxa de retrocessos",             "crimson"),
            ("entropia_vel_norm",   "Entropia da velocidade (norm.)",  "mediumpurple"),
            ("jerk_std",            "Jerk std (variação da aceleração)","saddlebrown"),
        ]

        for ax, (col, label, color) in zip(axes.flat, plots):
            if col not in df.columns:
                ax.set_visible(False)
                continue
            data = df[col].dropna()
            ax.hist(data, bins=60, color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
            ax.axvline(data.mean(),   color="black", linestyle="--", linewidth=1.2,
                       label=f"Média: {data.mean():.3f}")
            ax.axvline(data.median(), color="gray",  linestyle=":",  linewidth=1.0,
                       label=f"Mediana: {data.median():.3f}")
            ax.set_xlabel(label, fontsize=10)
            ax.set_ylabel("Frequência", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "human_behavior_analysis.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n  📊 Gráfico salvo em '{path}'")
        plt.show()

    def create_correlation_plot(self, df: pd.DataFrame):
        """
        Gera mapa de correlação entre features e salva em plots/correlation.png.
        """
        os.makedirs(PLOTS_DIR, exist_ok=True)

        # Usa subconjunto das features mais importantes para legibilidade
        cols = [c for c in [
            "jerk_std", "ratio_dist_desl", "curvatura_mean", "taxa_retrocesso",
            "time_to_click", "cv_temporal", "std_dt", "d_centro_clique",
            "entropia_vel_norm", "ratio_desaceleracao", "desvio_lateral_max",
        ] if c in df.columns]

        corr = df[cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, square=True, linewidths=0.4, ax=ax,
                    annot_kws={"size": 8})
        ax.set_title("Correlação entre Features — Dados Humanos", fontsize=13)
        plt.tight_layout()

        path = os.path.join(PLOTS_DIR, "correlation.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  📊 Mapa de correlação salvo em '{path}'")
        plt.show()

    # ── Salvar / carregar ─────────────────────────────────────────────────────

    def save_models(self, directory: str = MODELS_DIR):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.ocsvm_model,             f"{directory}/ocsvm_model.pkl")
        joblib.dump(self.isolation_forest_model,  f"{directory}/isolation_forest_model.pkl")
        joblib.dump(self.scaler,                  f"{directory}/scaler.pkl")
        joblib.dump(self.feature_names,           f"{directory}/feature_names.pkl")
        print(f"\n  ✅ Modelos salvos em '{directory}/'")
        print(f"     ocsvm_model.pkl")
        print(f"     isolation_forest_model.pkl")
        print(f"     scaler.pkl")
        print(f"     feature_names.pkl")

    def load_models(self, directory: str = MODELS_DIR):
        self.ocsvm_model            = joblib.load(f"{directory}/ocsvm_model.pkl")
        self.isolation_forest_model = joblib.load(f"{directory}/isolation_forest_model.pkl")
        self.scaler                 = joblib.load(f"{directory}/scaler.pkl")
        self.feature_names          = joblib.load(f"{directory}/feature_names.pkl")
        print(f"  ✅ Modelos carregados de '{directory}/'")


# ──────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CAPTCHA BEHAVIORAL CLASSIFIER — ONE-CLASS LEARNING")
    print("=" * 70)
    print("\nEstratégia: aprender o perfil humano completo e rejeitar bots.")

    clf = BehavioralCaptchaClassifier()

    # 1. Carregar dataset gerado pelo pipeline
    df = clf.load_dataset(DATASET_PATH)

    if len(df) < 50:
        print(f"\n❌ Dados insuficientes: {len(df)} sessões. Mínimo: 50.")
        print("   Execute captcha_feature_pipeline.py para processar mais sessões.")
    else:
        # 2. Analisar diversidade dos dados humanos
        clf.analyze_human_diversity(df)

        # 3. Treinar modelos
        clf.train(df)

        # 4. Gráficos de análise
        clf.create_plots(df)
        clf.create_correlation_plot(df)

        # 5. Salvar modelos
        clf.save_models(MODELS_DIR)

        # 6. Teste rápido no treino (verificar falso positivo)
        clf.test_on_samples(df, n=15)

        print("\n" + "=" * 70)
        print("✅ TREINAMENTO CONCLUÍDO")
        print("=" * 70)
        print(f"\n  Modelos salvos em : {MODELS_DIR}/")
        print(f"  Gráficos salvos em: {PLOTS_DIR}/")
        print("\n  Próximos passos:")
        print("  1. Analise os gráficos para entender a diversidade dos seus dados")
        print("  2. Verifique a taxa de falso positivo nas amostras de teste acima")
        print("  3. Colete sessões de bot para validar a taxa de detecção")
