"""
bots_teste.py

Gera sessões artificiais simulando diferentes estratégias de bot, e
testa os dois modelos treinados (SVM de Uma Classe e Floresta de
Isolação) contra elas. Serve como sanity check rápido da efetividade
dos modelos ANTES da avaliação formal (k-fold, AUC-ROC etc. — seção
3.6 do capítulo de Metodologia): se os modelos não rejeitarem a
maioria desses bots óbvios, algo está errado no treinamento.

Estratégias de bot implementadas:
    1. linear        - linha reta perfeita, velocidade constante, Δt fixo
    2. teleporte      - poucos eventos, "pula" quase direto pro alvo
    3. ruido_uniforme - pontos aleatórios sem nenhuma estrutura temporal/espacial
    4. curva_suave    - curva de Bézier suave, velocidade constante (sem jerk)
    5. replay_ruido   - replica uma sessão humana real, com pequeno ruído
                        (simula um bot mais sofisticado tentando imitar humano)

Uso:
    python bots_teste.py --dir-modelos modelos/ --dir-dados-humanos captcha_data/
"""

import argparse
import math
import random
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import Features

LARG_JANELA, ALT_JANELA = 500, 400
CENTRO_CHECKBOX = (Features.CHECKBOX_CENTER_X, Features.CHECKBOX_CENTER_Y)


# ----------------------------------------------------------------------
# Helpers de construção de sessão
# ----------------------------------------------------------------------
def _nova_sessao(nome_bot: str) -> dict:
    return {
        'session_id': datetime.now().strftime('%Y%m%d_%H%M%S_%f'),
        'session_user': [f'BOT_{nome_bot.upper()}'],
        'mouse_movements': [],
        'timestamps': [],
        'velocities': [],
        'accelerations': [],
        'click_data': {},
        'total_time': 0,
        'distance_traveled': 0,
        'window_origin': {'x': random.randint(0, 800), 'y': random.randint(0, 600)},
    }

def _finalizar_sessao(sessao: dict, pontos: list, tempos: list, posicao_clique: tuple) -> dict:
    """Preenche mouse_movements/timestamps/distance_traveled/click_data a partir de pontos+tempos."""
    t0 = tempos[0]
    distancia_total = 0.0
    for i, ((x, y), t) in enumerate(zip(pontos, tempos)):
        sessao['mouse_movements'].append({'x': x, 'y': y, 'time_offset': t - t0})
        sessao['timestamps'].append(t)
        if i > 0:
            dx = x - pontos[i - 1][0]
            dy = y - pontos[i - 1][1]
            distancia_total += math.hypot(dx, dy)

    sessao['distance_traveled'] = distancia_total
    momento_clique = tempos[-1] + 0.05
    sessao['click_data'] = {
        'time_to_click': momento_clique - t0,
        'click_position': list(posicao_clique),
        'click_timestamp': momento_clique,
    }
    sessao['total_time'] = momento_clique - t0
    return sessao


# ----------------------------------------------------------------------
# Estratégias de bot
# ----------------------------------------------------------------------
def gerar_bot_linear(n_pontos: int = 30) -> dict:
    """Linha reta perfeita, velocidade constante, Δt fixo — o bot mais ingênuo possível."""
    sessao = _nova_sessao('linear')
    x0, y0 = random.randint(0, 100), random.randint(0, 100)
    xc, yc = CENTRO_CHECKBOX

    agora = time.time()
    dt_fixo = 0.02
    pontos, tempos = [], []
    for i in range(n_pontos):
        t = i / (n_pontos - 1)
        x = x0 + t * (xc - x0)
        y = y0 + t * (yc - y0)
        pontos.append((x, y))
        tempos.append(agora + i * dt_fixo)

    return _finalizar_sessao(sessao, pontos, tempos, CENTRO_CHECKBOX)


def gerar_bot_teleporte(n_pontos: int = 3) -> dict:
    """Poucos eventos, salto quase direto pro alvo — sem trajetória real."""
    sessao = _nova_sessao('teleporte')
    xc, yc = CENTRO_CHECKBOX
    agora = time.time()
    pontos = [(xc + random.uniform(-5, 5), yc + random.uniform(-5, 5)) for _ in range(n_pontos)]
    tempos = [agora + i * 0.01 for i in range(n_pontos)]
    return _finalizar_sessao(sessao, pontos, tempos, CENTRO_CHECKBOX)


def gerar_bot_ruido_uniforme(n_pontos: int = 25) -> dict:
    """Pontos aleatórios sem nenhuma estrutura espacial ou temporal."""
    sessao = _nova_sessao('ruido_uniforme')
    agora = time.time()
    pontos = [(random.uniform(0, LARG_JANELA), random.uniform(0, ALT_JANELA)) for _ in range(n_pontos)]
    tempos = sorted(agora + random.uniform(0, 2.0) for _ in range(n_pontos))
    xc, yc = CENTRO_CHECKBOX
    return _finalizar_sessao(sessao, pontos, tempos, (xc + random.uniform(-3, 3), yc + random.uniform(-3, 3)))


def gerar_bot_curva_suave(n_pontos: int = 35) -> dict:
    """
    Curva de Bézier quadrática suave, com velocidade aproximadamente
    constante e Δt fixo — tenta parecer "mais humano" evitando uma
    linha reta, mas ainda é matematicamente perfeita demais (jerk e
    curvatura quase sem variância).
    """
    sessao = _nova_sessao('curva_suave')
    x0, y0 = random.randint(0, 100), random.randint(0, 100)
    xc, yc = CENTRO_CHECKBOX
    # ponto de controle deslocado, pra criar uma curva
    xm = (x0 + xc) / 2 + random.uniform(-60, 60)
    ym = (y0 + yc) / 2 + random.uniform(-60, 60)

    agora = time.time()
    dt_fixo = 0.018
    pontos, tempos = [], []
    for i in range(n_pontos):
        t = i / (n_pontos - 1)
        x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * xm + t ** 2 * xc
        y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * ym + t ** 2 * yc
        pontos.append((x, y))
        tempos.append(agora + i * dt_fixo)

    return _finalizar_sessao(sessao, pontos, tempos, CENTRO_CHECKBOX)


def gerar_bot_replay_ruido(sessao_humana: dict, ruido_px: float = 3.0, ruido_t: float = 0.003) -> dict:
    """
    Replica uma sessão humana real, adicionando pequeno ruído gaussiano
    de posição e tempo — simula um bot mais sofisticado tentando
    imitar um humano (ataque de "replay" mencionado na seção de
    Limitações do capítulo).
    """
    sessao = _nova_sessao('replay_ruido')
    movimentos = sessao_humana.get('mouse_movements', [])
    timestamps = sessao_humana.get('timestamps', [])
    click_data = sessao_humana.get('click_data', {})

    if not movimentos or not timestamps:
        return None

    agora = time.time()
    t0_original = timestamps[0]
    pontos, tempos = [], []
    for m, t in zip(movimentos, timestamps):
        x = m['x'] + random.gauss(0, ruido_px)
        y = m['y'] + random.gauss(0, ruido_px)
        dt_relativo = (t - t0_original) + random.gauss(0, ruido_t)
        pontos.append((x, y))
        tempos.append(agora + max(dt_relativo, 0))

    tempos = sorted(tempos)  # garante ordem crescente mesmo com ruído no tempo

    click_pos_original = click_data.get('click_position', list(CENTRO_CHECKBOX))
    click_pos_ruidoso = (
        click_pos_original[0] + random.gauss(0, ruido_px),
        click_pos_original[1] + random.gauss(0, ruido_px),
    )
    return _finalizar_sessao(sessao, pontos, tempos, click_pos_ruidoso)


# ----------------------------------------------------------------------
# Avaliação contra os modelos treinados
# ----------------------------------------------------------------------
def carregar_pipeline(dir_modelos: Path):
    extrator = Features()
    winsorizador = joblib.load(dir_modelos / "winsorizador.joblib")
    escalonador = joblib.load(dir_modelos / "escalonador.joblib")
    svm = joblib.load(dir_modelos / "svm_uma_classe.joblib")
    floresta = joblib.load(dir_modelos / "floresta_isolacao.joblib")
    colunas = joblib.load(dir_modelos / "colunas_features.joblib")
    return extrator, winsorizador, escalonador, svm, floresta, colunas


def pontuar_sessao(sessao, extrator, winsorizador, escalonador, svm, floresta, colunas):
    vetor = extrator.transform(sessao)
    valores = vetor[colunas].values.reshape(1, -1)
    X = pd.DataFrame(valores, columns=colunas)
    X_w = winsorizador.transformar(X)
    X_s = escalonador.transform(X_w)

    pred_svm = svm.predict(X_s)[0]
    pred_floresta = floresta.predict(X_s)[0]
    return pred_svm, pred_floresta


def main():
    parser = argparse.ArgumentParser(description="Gera bots artificiais e testa os modelos treinados contra eles.")
    parser.add_argument("--dir-modelos", default="modelos", help="Diretório com os modelos treinados (.joblib)")
    parser.add_argument("--dir-dados-humanos", default="captcha_data",
                         help="Diretório com sessões humanas reais, usado pelo bot 'replay_ruido'")
    parser.add_argument("--n-por-tipo", type=int, default=50, help="Quantas sessões gerar por tipo de bot")
    args = parser.parse_args()

    dir_modelos = Path(args.dir_modelos)
    extrator, winsorizador, escalonador, svm, floresta, colunas = carregar_pipeline(dir_modelos)

    # Carrega algumas sessões humanas reais para o bot de replay (se existirem)
    sessoes_humanas = []
    dir_humanos = Path(args.dir_dados_humanos)
    if dir_humanos.exists():
        import json
        arquivos = list(dir_humanos.glob("*.json"))
        random.shuffle(arquivos)
        for caminho in arquivos[:args.n_por_tipo]:
            with open(caminho, "r", encoding="utf-8") as f:
                sessoes_humanas.append(json.load(f))

    geradores = {
        'linear': lambda: gerar_bot_linear(),
        'teleporte': lambda: gerar_bot_teleporte(),
        'ruido_uniforme': lambda: gerar_bot_ruido_uniforme(),
        'curva_suave': lambda: gerar_bot_curva_suave(),
    }
    if sessoes_humanas:
        geradores['replay_ruido'] = lambda: gerar_bot_replay_ruido(random.choice(sessoes_humanas))
    else:
        print(f"[aviso] Nenhuma sessão humana encontrada em '{dir_humanos}' — "
              f"pulando o bot 'replay_ruido'.")

    print(f"\nGerando {args.n_por_tipo} sessões por tipo de bot e pontuando com os modelos...\n")
    print(f"{'Tipo de bot':<18} {'N':>5}  {'SVM rejeitou':>14}  {'Floresta rejeitou':>18}")
    print("-" * 62)

    resultados = {}
    for nome, gerar in geradores.items():
        rejeitados_svm = 0
        rejeitados_floresta = 0
        total = 0
        for _ in range(args.n_por_tipo):
            sessao = gerar()
            if sessao is None or len(sessao['mouse_movements']) < 2:
                continue
            pred_svm, pred_floresta = pontuar_sessao(
                sessao, extrator, winsorizador, escalonador, svm, floresta, colunas
            )
            total += 1
            if pred_svm == -1:
                rejeitados_svm += 1
            if pred_floresta == -1:
                rejeitados_floresta += 1

        taxa_svm = 100 * rejeitados_svm / total if total else 0
        taxa_floresta = 100 * rejeitados_floresta / total if total else 0
        resultados[nome] = (taxa_svm, taxa_floresta)
        print(f"{nome:<18} {total:>5}  {taxa_svm:>12.1f}%  {taxa_floresta:>16.1f}%")

    print()
    print("Taxa de rejeição = % dos bots corretamente identificados como anômalos.")
    print("Valores altos (próximos de 100%) nos bots 'óbvios' (linear, teleporte, ruido_uniforme)")
    print("indicam que os modelos aprenderam o padrão humano corretamente.")
    print("Valores baixos no 'replay_ruido' são esperados — é o bot mais sofisticado e o mais")
    print("difícil de detectar, conforme discutido na seção de Limitações do capítulo.")


if __name__ == "__main__":
    main()
