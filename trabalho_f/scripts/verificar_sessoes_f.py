"""
verificar_sessoes_f.py

Verifica a integridade das sessões de uma pasta (humanos ou bots) e
resume as distribuições principais. Serve para conferir a coleta antes de
usar os dados (checklist: "verificar integridade").

Verificações por sessão:
  - versão do coletor e campos obrigatórios
  - número de eventos (mínimo) e tamanho de mouse_movements == timestamps
  - timestamps em ordem crescente; clique depois do último evento
  - clique dentro do retângulo do alvo
  - pontos fora da janela
  - velocidade máxima entre eventos consecutivos (saltos suspeitos)

Resumo: eventos, duração, Δt, velocidade, distância P0 -> alvo, e a
distância entre o P0 de cada sessão e o clique da sessão anterior (no fluxo
automático do coletor v3, a trajetória deve começar perto do clique anterior).

Uso:
    python trabalho_f/scripts/verificar_sessoes_f.py
    python trabalho_f/scripts/verificar_sessoes_f.py --dir captcha_data_v2
"""

import argparse

import numpy as np

from common_f import caminho, carregar_config, carregar_sessoes, centro_alvo, pontos_e_tempos, tamanho_janela

CAMPOS_OBRIGATORIOS = ["session_id", "mouse_movements", "timestamps", "click_data",
                       "window_size", "target"]


def _percentis(nome, valores, unidade="", casas=1):
    v = np.asarray(valores, dtype=float)
    p = np.percentile(v, [0, 5, 50, 95, 100])
    fmt = f"{{:.{casas}f}}"
    print(f"  {nome:<34} min {fmt.format(p[0])}  p5 {fmt.format(p[1])}  mediana {fmt.format(p[2])}  "
          f"p95 {fmt.format(p[3])}  max {fmt.format(p[4])} {unidade}")


def verificar(diretorio, versao_esperada, eventos_minimos, velocidade_maxima):
    problemas = []  # (arquivo, descrição)
    resumo = {"eventos": [], "duracao": [], "dt_ms": [], "vel_max": [], "dist_alvo": [],
              "p0_vs_clique_anterior": [], "dt_sub_1ms": 0, "dt_total": 0}
    clique_anterior = None

    sessoes = list(carregar_sessoes(diretorio))
    for nome, s in sessoes:
        faltando = [c for c in CAMPOS_OBRIGATORIOS if c not in s]
        if faltando:
            problemas.append((nome, f"campos ausentes: {faltando}"))
            clique_anterior = None
            continue
        if versao_esperada is not None and s.get("collector_version") != versao_esperada:
            problemas.append((nome, f"versão {s.get('collector_version')} (esperado {versao_esperada})"))

        pontos, tempos = pontos_e_tempos(s)
        if len(pontos) != len(tempos):
            problemas.append((nome, "mouse_movements e timestamps com tamanhos diferentes"))
            clique_anterior = None
            continue
        if len(pontos) < eventos_minimos:
            problemas.append((nome, f"só {len(pontos)} eventos"))

        dt = np.diff(tempos)
        if np.any(dt < 0):
            problemas.append((nome, "timestamps fora de ordem"))

        clique = s["click_data"].get("click_position")
        instante_clique = s["click_data"].get("click_timestamp")
        if clique is None or instante_clique is None:
            problemas.append((nome, "clique não registrado"))
            clique_anterior = None
            continue
        if len(tempos) and instante_clique < tempos[-1]:
            problemas.append((nome, "eventos após o clique (cortados na extração de features)"))

        alvo = s["target"]
        if not (alvo["x"] <= clique[0] <= alvo["x"] + alvo["width"] and
                alvo["y"] <= clique[1] <= alvo["y"] + alvo["height"]):
            problemas.append((nome, f"clique {clique} fora do alvo"))

        largura, altura = tamanho_janela(s)
        fora = np.sum((pontos[:, 0] < 0) | (pontos[:, 1] < 0) |
                      (pontos[:, 0] >= largura) | (pontos[:, 1] >= altura))
        if fora:
            problemas.append((nome, f"{fora} pontos fora da janela"))

        passos = np.linalg.norm(np.diff(pontos, axis=0), axis=1)
        validos = dt >= 0.001
        vel = passos[validos] / dt[validos] if np.any(validos) else np.array([0.0])
        if vel.size and vel.max() > velocidade_maxima:
            problemas.append((nome, f"velocidade de {vel.max():.0f} px/s entre eventos"))

        resumo["eventos"].append(len(pontos))
        resumo["duracao"].append(s.get("total_time", 0))
        resumo["dt_ms"].extend((dt * 1000).tolist())
        resumo["dt_sub_1ms"] += int(np.sum(dt < 0.001))
        resumo["dt_total"] += dt.size
        resumo["vel_max"].append(float(vel.max()) if vel.size else 0.0)
        resumo["dist_alvo"].append(float(np.linalg.norm(centro_alvo(s) - pontos[0])))
        if clique_anterior is not None:
            resumo["p0_vs_clique_anterior"].append(float(np.linalg.norm(pontos[0] - clique_anterior)))
        clique_anterior = np.asarray(clique, dtype=float)

    return sessoes, problemas, resumo


def principal():
    config = carregar_config()
    parser = argparse.ArgumentParser(description="Verifica a integridade das sessões de uma pasta.")
    parser.add_argument("--dir", default=str(caminho(config, "humanos")))
    parser.add_argument("--versao", type=int, default=3, help="Versão esperada do coletor (0 = não verificar)")
    parser.add_argument("--eventos-minimos", type=int, default=10)
    parser.add_argument("--velocidade-maxima", type=float, default=20000, help="px/s entre eventos")
    args = parser.parse_args()

    sessoes, problemas, r = verificar(args.dir, args.versao or None, args.eventos_minimos, args.velocidade_maxima)
    usuarios = sorted({u for _, s in sessoes for u in s.get("session_user", [])})

    print(f"Pasta: {args.dir}")
    print(f"Sessões: {len(sessoes)}  |  usuários: {usuarios}")
    print(f"Sessões com problema: {len({p[0] for p in problemas})}")
    for nome, descricao in problemas[:30]:
        print(f"  - {nome}: {descricao}")
    if len(problemas) > 30:
        print(f"  ... e mais {len(problemas) - 30}")

    if not r["eventos"]:
        return
    print("\nDistribuições:")
    _percentis("eventos por sessão", r["eventos"], casas=0)
    _percentis("duração até o clique", r["duracao"], "s", casas=2)
    _percentis("Δt entre eventos", r["dt_ms"], "ms")
    print(f"  {'Δt < 1 ms':<34} {r['dt_sub_1ms']} de {r['dt_total']} intervalos "
          f"({100 * r['dt_sub_1ms'] / max(r['dt_total'], 1):.2f}%)")
    _percentis("velocidade máxima por sessão", r["vel_max"], "px/s", casas=0)
    _percentis("distância P0 -> centro do alvo", r["dist_alvo"], "px", casas=0)
    if r["p0_vs_clique_anterior"]:
        _percentis("P0 vs clique da sessão anterior", r["p0_vs_clique_anterior"], "px", casas=0)


if __name__ == "__main__":
    principal()
