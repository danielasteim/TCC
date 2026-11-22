"""
app_captcha_f.py

Base dos apps de CAPTCHA comportamental em tempo real (app_svm_f.py e
app_floresta_f.py), equivalentes aos antigos svm_app.py e if_app.py (v1/).

Usa a mesma interface da coleta (data_collection.py: tela cheia, painel em
posição aleatória). Após o clique no checkbox, a sessão é salva, as
características são extraídas (caracteristicas_f.py) e o escore de decisão
do modelo treinado pelo 02_treinar_avaliar_f.py é comparado com o limiar
calibrado no treino: escore >= limiar -> humano; escore < limiar -> bot.

O resultado aparece no painel; Enter inicia uma nova tentativa e Esc encerra.
As sessões testadas são gravadas em trabalho_f/dados/app/<modelo>/ (fora do
git e separadas da base de treino).
"""

import json
import tkinter as tk

import joblib
import pandas as pd

from caracteristicas_f import COLUNAS_EXTRAIDAS, Caracteristicas
from common_f import PASTA_F, caminho, carregar_config

from data_collection import Captcha  # noqa: E402  (RAIZ está no sys.path via common_f)

NOMES = {"svm": "SVM de Uma Classe", "floresta": "Floresta de Isolação"}


def carregar_modelo(config: dict, nome: str) -> dict:
    arquivo = caminho(config, "modelos") / f"{nome}_f.joblib"
    if not arquivo.exists():
        raise SystemExit(f"Modelo não encontrado: {arquivo}. Rode antes o 02_treinar_avaliar_f.py.")
    return joblib.load(arquivo)  # {"modelo": Pipeline, "limiar": float}


def avaliar_sessao(sessao: dict, modelo: dict, extrator: Caracteristicas):
    """Devolve (escore, limiar, é_humano)."""
    X = pd.DataFrame([extrator.extrair(sessao)])[COLUNAS_EXTRAIDAS]
    if X.isna().any(axis=None):
        raise ValueError("trajetória curta demais para calcular as características")
    escore = float(modelo["modelo"].decision_function(X)[0])
    return escore, modelo["limiar"], escore >= modelo["limiar"]


class CaptchaAutenticacao(Captcha):
    """Interface de coleta que, em vez de seguir direto, mostra a decisão do modelo."""

    def __init__(self, root, nome: str, modelo: dict, extrator: Caracteristicas, pasta_saida):
        self.nome = nome
        self.modelo = modelo
        self.extrator = extrator
        self.resultado = None
        super().__init__(root, usuario="APP", pasta_saida=str(pasta_saida))
        self.root.title(f"Behavioral CAPTCHA - {NOMES[nome]}")

    def ao_salvar_sessao(self, arquivo):
        with open(arquivo, encoding="utf-8") as f:
            sessao = json.load(f)
        try:
            self.resultado = avaliar_sessao(sessao, self.modelo, self.extrator)
        except (ValueError, KeyError) as erro:
            self.resultado = str(erro)

    def reset_session(self):
        # O coletor chama reset_session logo após salvar; aqui a nova sessão
        # só começa depois que o resultado for mostrado e o usuário apertar Enter.
        if self.resultado is None:
            return super().reset_session()
        self._mostrar_resultado()
        self.root.bind("<Return>", self._continuar)

    def _mostrar_resultado(self):
        if isinstance(self.resultado, str):
            texto, cor = f"Não foi possível avaliar: {self.resultado}", "#ff6666"
        else:
            escore, limiar, humano = self.resultado
            texto = (f"{NOMES[self.nome]}: {'HUMANO' if humano else 'BOT'}\n"
                     f"escore {escore:.3f} (limiar {limiar:.3f})")
            cor = "#4CAF50" if humano else "#ff6666"
        print(texto.replace("\n", " | "))
        self.status_label.config(text=texto + "\n\nEnter: nova tentativa · Esc: sair", fg=cor)

    def _continuar(self, _evento=None):
        self.root.unbind("<Return>")
        self.resultado = None
        super().reset_session()


def executar(nome: str):
    config = carregar_config()
    modelo = carregar_modelo(config, nome)
    extrator = Caracteristicas.de_config(config)
    pasta_saida = PASTA_F / "dados" / "app" / nome

    root = tk.Tk()
    CaptchaAutenticacao(root, nome, modelo, extrator, pasta_saida)
    root.mainloop()
    print(f"Sessões testadas salvas em {pasta_saida}")
