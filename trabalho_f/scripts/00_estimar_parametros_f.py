"""
00_estimar_parametros_f.py

Estima, a partir das sessões humanas, os parâmetros livres do gerador de
trajetórias (seção 3.5.2) e salva em trabalho_f/resultados/params_humanos.json:

  - P0: posição do primeiro evento, relativa ao tamanho da janela
  - alvo: centro do checkbox, relativo ao tamanho da janela
  - deslocamento do clique em relação ao centro do alvo (px)
  - M (número de eventos): média e desvio por direção (8 setores, como em
    Acien et al., 2022) x faixa de comprimento L (tercis)
  - Δt entre eventos: função quantil empírica (usada no nível T1 do replay)
  - forma quadrática: distribuição de a
  - forma exponencial: distribuição de a e b (e os pares (a, b))

Os coeficientes de forma são adimensionais (sistema local s = u/L, z = w/L),
então valem para qualquer comprimento de trajetória.

Uso:
    python trabalho_f/scripts/00_estimar_parametros_f.py
    python trabalho_f/scripts/00_estimar_parametros_f.py --dir captcha_data_v2 --limite 1000   # teste rápido
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common_f import (RAIZ, caminho, carregar_config, carregar_sessoes, centro_alvo,
                      para_sistema_local, pontos_e_tempos, setor_direcao, tamanho_janela)
from gerador_f import ajustar_exponencial, ajustar_quadratica


def _resumo(valores, p_inf, p_sup, n_quantis=101):
    """Percentis e função quantil restrita a [p_inf, p_sup] (usada para amostrar)."""
    v = np.asarray(valores, dtype=float)
    return {
        "n": int(v.size),
        "p5": float(np.percentile(v, 5)),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
        "media": float(v.mean()),
        "desvio": float(v.std()),
        "quantis": np.percentile(v, np.linspace(p_inf, p_sup, n_quantis)).round(6).tolist(),
    }


def estimar(diretorio, config, limite=None):
    est = config["estimacao"]
    p_inf, p_sup = est["percentis_parametros"]
    grade_b = np.arange(est["grade_b"][0], est["grade_b"][1] + 1e-9, est["grade_b"][2])
    grade_b = grade_b[np.abs(grade_b) > 1e-9]

    p0_rel, alvo_rel, desloc, dts = [], [], [], []
    linhas_M = []  # (setor, L, M)
    a_quad, pares_exp, r2_quad, r2_exp = [], [], [], []
    larguras, alturas, versoes = [], [], set()
    n_sessoes = n_descartadas = 0

    for _, sessao in carregar_sessoes(diretorio, limite=limite, semente=config["semente"]):
        pontos, tempos = pontos_e_tempos(sessao)
        clique = sessao.get("click_data", {}).get("click_position")
        if len(pontos) < 2 or clique is None or len(tempos) != len(pontos):
            n_descartadas += 1
            continue
        n_sessoes += 1
        versoes.add(sessao.get("collector_version", 1))

        largura, altura = tamanho_janela(sessao)
        larguras.append(largura)
        alturas.append(altura)
        alvo = centro_alvo(sessao)
        p0 = pontos[0]
        pc = np.asarray(clique, dtype=float)

        p0_rel.append([p0[0] / largura, p0[1] / altura])
        alvo_rel.append([alvo[0] / largura, alvo[1] / altura])
        desloc.append((pc - alvo).tolist())
        dts.append(np.diff(tempos))

        L = float(np.linalg.norm(pc - p0))
        linhas_M.append((setor_direcao(p0, pc, est["n_direcoes"]), L, len(pontos)))

        if L < est["L_minimo_ajuste"]:
            continue
        s, z, _ = para_sistema_local(pontos, p0, pc)
        sst = float(z @ z)  # referência: trajetória reta (z = 0)
        a, sse_q = ajustar_quadratica(s, z)
        a_e, b_e, sse_e = ajustar_exponencial(s, z, grade_b)
        a_quad.append(a)
        pares_exp.append((a_e, b_e))
        if sst > 0:
            r2_quad.append(1 - sse_q / sst)
            r2_exp.append(1 - sse_e / sst)

    if n_sessoes == 0:
        raise SystemExit(f"Nenhuma sessão válida em '{diretorio}'.")

    # --- M por direção x faixa de L ---
    setores = np.array([l[0] for l in linhas_M])
    Ls = np.array([l[1] for l in linhas_M])
    Ms = np.array([l[2] for l in linhas_M])
    limites_L = np.percentile(Ls, np.linspace(0, 100, est["n_faixas_L"] + 1)[1:-1]).tolist()
    faixas = np.searchsorted(limites_L, Ls)
    celulas = {}
    for setor in range(est["n_direcoes"]):
        no_setor = setores == setor
        base = Ms[no_setor] if no_setor.sum() >= 2 else Ms
        celulas[str(setor)] = []
        for faixa in range(est["n_faixas_L"]):
            sel = no_setor & (faixas == faixa)
            amostra = Ms[sel] if sel.sum() >= est["n_minimo_celula"] else base
            celulas[str(setor)].append({
                "n": int(sel.sum()),
                "usa_fallback": bool(sel.sum() < est["n_minimo_celula"]),
                "media": float(amostra.mean()),
                "desvio": float(amostra.std()),
            })

    # --- exponencial: pares (a, b) dentro de p5..p95 de cada coeficiente ---
    pares = np.array(pares_exp)
    a_lim = np.percentile(pares[:, 0], [p_inf, p_sup])
    b_lim = np.percentile(pares[:, 1], [p_inf, p_sup])
    dentro = ((pares[:, 0] >= a_lim[0]) & (pares[:, 0] <= a_lim[1]) &
              (pares[:, 1] >= b_lim[0]) & (pares[:, 1] <= b_lim[1]))

    dts = np.concatenate(dts)
    return {
        "origem": {
            "diretorio": str(diretorio),
            "sessoes_usadas": n_sessoes,
            "sessoes_descartadas": n_descartadas,
            "versoes_coletor": sorted(versoes),
            "sessoes_no_ajuste_de_forma": len(a_quad),
            "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "janela": {"largura": int(np.median(larguras)), "altura": int(np.median(alturas))},
        "p0_relativo": np.round(p0_rel, 5).tolist(),
        "alvo_relativo": np.round(alvo_rel, 5).tolist(),
        "deslocamento_clique": np.round(desloc, 2).tolist(),
        "M": {
            "geral": {"media": float(Ms.mean()), "desvio": float(Ms.std()),
                      "min": int(Ms.min()), "max": int(Ms.max())},
            "limites_L": limites_L,
            "celulas": celulas,
        },
        "dt": {
            "geral": {"media_ms": float(dts.mean() * 1000), "mediana_ms": float(np.median(dts) * 1000)},
            "quantis_s": np.percentile(dts, np.linspace(0, 100, 1001)).round(6).tolist(),
        },
        "quadratica": {
            "a": _resumo(a_quad, p_inf, p_sup),
            "r2_mediano": float(np.median(r2_quad)) if r2_quad else None,
        },
        "exponencial": {
            "a": _resumo(pares[:, 0], p_inf, p_sup),
            "b": _resumo(pares[:, 1], p_inf, p_sup),
            "pares_ab": np.round(pares[dentro], 6).tolist(),
            "r2_mediano": float(np.median(r2_exp)) if r2_exp else None,
        },
    }


def principal():
    config = carregar_config()
    parser = argparse.ArgumentParser(description="Estima os parâmetros do gerador a partir dos humanos.")
    parser.add_argument("--dir", default=str(caminho(config, "humanos")), help="Diretório das sessões humanas")
    parser.add_argument("--saida", default=str(caminho(config, "resultados") / "params_humanos.json"))
    parser.add_argument("--limite", type=int, default=None, help="Usar só uma amostra de N sessões (teste)")
    args = parser.parse_args()

    t0 = time.time()
    params = estimar(args.dir, config, args.limite)

    saida = Path(args.saida)
    if not saida.is_absolute():
        saida = RAIZ / saida
    saida.parent.mkdir(parents=True, exist_ok=True)
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=1)

    o, M, q, e = params["origem"], params["M"], params["quadratica"], params["exponencial"]
    print(f"Sessões: {o['sessoes_usadas']} usadas, {o['sessoes_descartadas']} descartadas "
          f"(coletor v{o['versoes_coletor']}) — {time.time() - t0:.1f}s")
    print(f"Janela: {params['janela']['largura']}x{params['janela']['altura']}")
    print(f"M: média {M['geral']['media']:.1f}, desvio {M['geral']['desvio']:.1f}, "
          f"faixas de L (px): {[round(x) for x in M['limites_L']]}")
    print(f"Δt: média {params['dt']['geral']['media_ms']:.1f} ms, mediana {params['dt']['geral']['mediana_ms']:.1f} ms")
    print(f"Quadrática  a: p5 {q['a']['p5']:.3f}  p50 {q['a']['p50']:.3f}  p95 {q['a']['p95']:.3f}  "
          f"(R² mediano {q['r2_mediano']:.2f}, n={q['a']['n']})")
    print(f"Exponencial a: p5 {e['a']['p5']:.3f}  p50 {e['a']['p50']:.3f}  p95 {e['a']['p95']:.3f}")
    print(f"Exponencial b: p5 {e['b']['p5']:.2f}  p50 {e['b']['p50']:.2f}  p95 {e['b']['p95']:.2f}  "
          f"(R² mediano {e['r2_mediano']:.2f}, pares mantidos {len(e['pares_ab'])})")
    print(f"Salvo em {saida}")


if __name__ == "__main__":
    principal()
