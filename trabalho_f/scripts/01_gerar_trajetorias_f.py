"""
01_gerar_trajetorias_f.py

Gera as trajetórias sintéticas das 9 famílias (3 formas x 3 perfis de
velocidade) a partir de trabalho_f/resultados/params_humanos.json, e a figura de exemplos.

Saídas:
  trabalho_f/dados/bot/trajetorias/NNNN_shape=A_vp=B.txt   uma linha "x y" por ponto, em pixels
                                        da janela (mesma nomenclatura do
                                        BeCAPTCHA-Mouse)
  trabalho_f/dados/bot/trajetorias/indice.csv              parâmetros de cada trajetória (P0, Pc,
                                        centro do alvo, M, a, b, semente)
  trabalho_f/resultados/figuras/gerador_exemplos.pdf/.png

Cada trajetória guarda o centro do alvo usado; o bot de replay posiciona o
painel do checkbox nesse ponto antes de executar a trajetória.

Uso:
    python trabalho_f/scripts/01_gerar_trajetorias_f.py
    python trabalho_f/scripts/01_gerar_trajetorias_f.py --params trabalho_f/resultados/params_teste_v1.json --n 20
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common_f import (FAMILIAS_FORMA, FAMILIAS_PERFIL, RAIZ, caminho,  # noqa: E402
                      carregar_config, setor_direcao)
from gerador_f import AmostradorParametros, trajetoria  # noqa: E402


def _absoluto(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else RAIZ / p


def gerar_uma(amostrador, rng, forma, perfil, n_direcoes, tentativas):
    """Sorteia parâmetros e gera uma trajetória que caiba inteira na janela."""
    w, h = amostrador.largura, amostrador.altura
    for _ in range(tentativas):
        p0 = amostrador.p0(rng)
        alvo = amostrador.alvo(rng)
        pc = alvo + amostrador.deslocamento_clique(rng)
        L = float(np.linalg.norm(pc - p0))
        if L < 1:
            continue
        M = amostrador.M(rng, L, setor_direcao(p0, pc, n_direcoes))
        a, b = amostrador.coeficientes(rng, forma)
        pontos = trajetoria(p0, pc, forma, perfil, M, a, b)
        if pontos[:, 0].min() >= 0 and pontos[:, 1].min() >= 0 and pontos[:, 0].max() < w and pontos[:, 1].max() < h:
            return pontos, dict(p0=p0, pc=pc, alvo=alvo, M=M, a=a, b=b, L=L)
    return None, None


def figura_exemplos(exemplos, largura, altura, destino: Path):
    """3 exemplos por família (linhas = perfil, colunas = forma) + incrementos por perfil."""
    fig, eixos = plt.subplots(3, 4, figsize=(15, 9.5),
                              gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    cores = ["#1f77b4", "#d62728", "#2ca02c"]
    for perfil in range(3):
        for forma in range(3):
            ax = eixos[perfil, forma]
            for k, pontos in enumerate(exemplos[(forma, perfil)][:3]):
                ax.plot(pontos[:, 0], pontos[:, 1], "-o", ms=2.2, lw=0.9, color=cores[k])
                ax.plot(*pontos[0], "s", color=cores[k], ms=5)
                ax.plot(*pontos[-1], "*", color="black", ms=8)
            ax.set_xlim(0, largura)
            ax.set_ylim(altura, 0)  # y cresce para baixo, como na tela
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            if perfil == 0:
                ax.set_title(f"shape={forma} ({FAMILIAS_FORMA[forma]})", fontsize=10)
            if forma == 0:
                ax.set_ylabel(f"vp={perfil} ({FAMILIAS_PERFIL[perfil]})", fontsize=10)

        # incremento (distância entre pontos consecutivos) ao longo da trajetória
        ax = eixos[perfil, 3]
        for k, pontos in enumerate(exemplos[(1, perfil)][:3]):
            passo = np.linalg.norm(np.diff(pontos, axis=0), axis=1)
            ax.plot(np.linspace(0, 1, passo.size), passo, color=cores[k], lw=1.2)
        ax.set_xlabel("posição relativa na trajetória", fontsize=8)
        ax.set_ylabel("distância entre pontos (px)", fontsize=8)
        ax.tick_params(labelsize=7)
        if perfil == 0:
            ax.set_title("perfil de incrementos", fontsize=10)

    fig.suptitle("Trajetórias sintéticas: 3 exemplos por família "
                 "(■ início, ★ clique)", fontsize=12)
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino.with_suffix(".pdf"))
    fig.savefig(destino.with_suffix(".png"), dpi=130)
    plt.close(fig)


def principal():
    config = carregar_config()
    parser = argparse.ArgumentParser(description="Gera as trajetórias sintéticas das 9 famílias.")
    parser.add_argument("--params", default=str(caminho(config, "resultados") / "params_humanos.json"))
    parser.add_argument("--n", type=int, default=config["gerador"]["trajetorias_por_familia"],
                        help="Trajetórias por família")
    parser.add_argument("--saida", default=str(caminho(config, "bots_trajetorias")))
    parser.add_argument("--figura", default=str(caminho(config, "figuras") / "gerador_exemplos"))
    args = parser.parse_args()

    with open(_absoluto(args.params), "r", encoding="utf-8") as f:
        params = json.load(f)
    amostrador = AmostradorParametros(params, M_minimo=config["gerador"]["M_minimo"])
    n_direcoes = config["estimacao"]["n_direcoes"]
    tentativas = config["gerador"]["tentativas_por_trajetoria"]

    saida = _absoluto(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    for antigo in saida.glob("*_shape=*_vp=*.txt"):
        antigo.unlink()

    linhas_indice, exemplos, falhas = [], {}, 0
    numero = 0
    for forma in range(3):
        for perfil in range(3):
            exemplos[(forma, perfil)] = []
            for i in range(args.n):
                semente = [config["semente"], forma, perfil, i]
                rng = np.random.default_rng(semente)
                pontos, dados = gerar_uma(amostrador, rng, forma, perfil, n_direcoes, tentativas)
                if pontos is None:
                    falhas += 1
                    continue
                numero += 1
                nome = f"{numero:04d}_shape={forma}_vp={perfil}.txt"
                np.savetxt(saida / nome, pontos, fmt="%.2f")
                exemplos[(forma, perfil)].append(pontos)
                linhas_indice.append({
                    "arquivo": nome, "forma": forma, "perfil": perfil, "M": dados["M"],
                    "a": round(dados["a"], 6), "b": round(dados["b"], 6), "L": round(dados["L"], 2),
                    "p0_x": round(dados["p0"][0], 2), "p0_y": round(dados["p0"][1], 2),
                    "pc_x": round(dados["pc"][0], 2), "pc_y": round(dados["pc"][1], 2),
                    "alvo_x": round(dados["alvo"][0], 2), "alvo_y": round(dados["alvo"][1], 2),
                    "semente": "-".join(map(str, semente)),
                })

    with open(saida / "indice.csv", "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(linhas_indice[0].keys()))
        escritor.writeheader()
        escritor.writerows(linhas_indice)

    figura_exemplos(exemplos, amostrador.largura, amostrador.altura, _absoluto(args.figura))

    print(f"{numero} trajetórias em {saida} ({falhas} não couberam na janela após {tentativas} tentativas)")
    Ms = [l["M"] for l in linhas_indice]
    print(f"M: média {np.mean(Ms):.1f}, min {min(Ms)}, max {max(Ms)}")
    print(f"Figura: {_absoluto(args.figura)}.pdf / .png")


if __name__ == "__main__":
    principal()
