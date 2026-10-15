"""
02_bot_replay_f.py

Bot de replay (seção 3.5.3): executa trajetórias sintéticas na MESMA
interface de coleta dos humanos (data_collection.py), movendo o cursor real
com pyautogui, para que humanos e bots passem pelo mesmo mecanismo de captura.

A cada sessão:
  1. a interface abre o painel em posição aleatória (igual à coleta humana);
  2. P0 = posição atual do cursor (o clique da sessão anterior, como nos
     humanos); Pc = centro do alvo + deslocamento de clique sorteado entre
     os humanos;
  3. M, a e b são sorteados de params_humanos.json e a trajetória é gerada
     pela família (forma, perfil) da vez (gerador_f.py);
  4. os pontos são executados com intervalo fixo (T0) ou sorteado da
     distribuição humana de Δt (T1), e o clique é feito no último ponto;
  5. a interface salva o JSON no formato humano, com os campos extras
     shape, vp, timing e bot_params.

As sessões são executadas em ordem aleatória (semente fixa) entre famílias e
níveis, para que nenhuma condição fique concentrada num trecho da execução.
Se for interrompido, retoma de onde parou (conta o que já está salvo).

Requer sessão gráfica X11 no Linux (pyautogui não move o cursor no Wayland).
Esc encerra; mover o mouse para um canto da tela também (failsafe do pyautogui).

Uso:
    python trabalho_f/scripts/02_bot_replay_f.py --teste          # 2 sessões por célula
    python trabalho_f/scripts/02_bot_replay_f.py                  # coleta completa
"""

import argparse
import csv
import json
import threading
import time
import tkinter as tk
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pyautogui

from common_f import RAIZ, FAMILIAS_FORMA, FAMILIAS_PERFIL, caminho, carregar_config, setor_direcao
from gerador_f import AmostradorParametros, trajetoria

from data_collection import Captcha  # noqa: E402  (RAIZ já está no sys.path via common_f)

pyautogui.PAUSE = 0  # sem a pausa padrão de 0,1 s depois de cada comando


def _esperar_ate(instante):
    """Dorme até `instante` (perf_counter) com precisão de ~0,1 ms."""
    restante = instante - time.perf_counter()
    if restante > 0.002:
        time.sleep(restante - 0.0015)
    while time.perf_counter() < instante:
        pass


def executar_trajetoria(pontos_tela, intervalos_s):
    """Move o cursor ponto a ponto no horário planejado e clica no último ponto."""
    instante = time.perf_counter()
    pyautogui.moveTo(*pontos_tela[0])
    for ponto, dt in zip(pontos_tela[1:], intervalos_s):
        instante += dt
        _esperar_ate(instante)
        pyautogui.moveTo(*ponto)
    pyautogui.click()


class BotReplay(Captcha):
    def __init__(self, root, plano, params, config, pasta_sessoes, pasta_trajetorias):
        # atributos usados por setup_ui, que o Captcha.__init__ já chama
        self.plano = deque(plano)
        self.params = params
        self.config = config
        self.replay = config["replay"]
        self.amostrador = AmostradorParametros(params, M_minimo=config["gerador"]["M_minimo"])
        self.quantis_dt = np.asarray(params["dt"]["quantis_s_min_1ms"])
        self.pasta_trajetorias = pasta_trajetorias
        self.registro = []      # uma linha por sessão salva
        self.falhas = 0
        self.atual = None
        self.numero_trajetoria = len(list(pasta_trajetorias.glob("*.txt")))
        super().__init__(root, usuario="BOT", pasta_saida=str(pasta_sessoes))

    # --- integração com a interface ------------------------------------
    def setup_ui(self):
        super().setup_ui()
        if not self.plano:
            self.root.after(300, self.root.quit)
            return
        self.root.after(self.replay["atraso_inicio_ms"], self._iniciar_sessao_bot)

    def ao_salvar_sessao(self, arquivo):
        forma, perfil, nivel, semente = self.atual
        n_registrado = len(self.session_data["mouse_movements"])
        self.registro.append({
            "arquivo": arquivo.split("/")[-1], "shape": forma, "vp": perfil, "timing": nivel,
            "M": self.session_data["bot_params"]["M"], "N": n_registrado, "semente": semente,
        })
        self.plano.popleft()
        if len(self.registro) % 25 == 0:
            print(f"  {len(self.registro)} sessões salvas, {len(self.plano)} restantes")

    # --- uma sessão de bot -----------------------------------------------
    def _sortear_intervalos(self, rng, n):
        if self.atual[2] == "T0":
            return np.full(n, self.replay["dt_T0_ms"] / 1000.0)
        u = rng.uniform(0, 1, n)
        return np.interp(u, np.linspace(0, 1, len(self.quantis_dt)), self.quantis_dt)

    def _iniciar_sessao_bot(self):
        forma, perfil, nivel, semente = self.atual = self.plano[0]
        rng = np.random.default_rng(semente)
        self.root.update_idletasks()
        origem = np.array([self.root.winfo_rootx(), self.root.winfo_rooty()], dtype=float)
        alvo = self.session_data["target"]

        # Pc: centro do alvo + deslocamento humano, mantido dentro do quadrado
        desloc = self.amostrador.deslocamento_clique(rng)
        pc = np.array([alvo["center_x"], alvo["center_y"]]) + desloc
        pc[0] = np.clip(pc[0], alvo["x"] + 1, alvo["x"] + alvo["width"] - 2)
        pc[1] = np.clip(pc[1], alvo["y"] + 1, alvo["y"] + alvo["height"] - 2)

        p0 = np.array(pyautogui.position(), dtype=float) - origem
        L = float(np.linalg.norm(pc - p0))
        M = self.amostrador.M(rng, L, setor_direcao(p0, pc, self.config["estimacao"]["n_direcoes"]))
        a, b = self.amostrador.coeficientes(rng, forma)
        pontos = np.rint(trajetoria(p0, pc, forma, perfil, M, a, b))
        intervalos = self._sortear_intervalos(rng, M - 1)

        self.session_data["session_user"] = ["BOT"]
        self.session_data["shape"] = forma
        self.session_data["vp"] = perfil
        self.session_data["timing"] = nivel
        self.session_data["bot_params"] = {
            "M": int(M), "a": float(a), "b": float(b), "L": L, "semente": list(semente),
            "p0": p0.tolist(), "pc": pc.tolist(),
            "dt_planejado_ms_media": float(intervalos.mean() * 1000),
        }

        self.numero_trajetoria += 1
        nome = f"{self.numero_trajetoria:04d}_shape={forma}_vp={perfil}_timing={nivel}.txt"
        tempos = np.concatenate([[0.0], np.cumsum(intervalos)])
        np.savetxt(self.pasta_trajetorias / nome, np.column_stack([pontos, tempos]),
                   fmt=["%d", "%d", "%.4f"], header="x y t_planejado_s")
        self.session_data["bot_params"]["trajetoria"] = nome

        pontos_tela = [tuple(int(v) for v in p + origem) for p in pontos]
        threading.Thread(target=executar_trajetoria, args=(pontos_tela, intervalos), daemon=True).start()

        sessao = self.session_data["session_id"]
        self.root.after(int(self.replay["limite_sessao_s"] * 1000), self._verificar_travamento, sessao)

    def _verificar_travamento(self, sessao):
        if self.session_data["session_id"] == sessao and self.plano and self.atual == self.plano[0]:
            self.falhas += 1
            print(f"  [aviso] sessão {sessao} não salvou em {self.replay['limite_sessao_s']} s; refazendo")
            self.reset_session()


def montar_plano(config, n_por_celula, pasta_sessoes):
    """Lista (forma, perfil, nível, semente) do que falta executar, em ordem aleatória."""
    feitas = Counter()
    for arquivo in pasta_sessoes.glob("*.json"):
        with open(arquivo, encoding="utf-8") as f:
            s = json.load(f)
        feitas[(s.get("shape"), s.get("vp"), s.get("timing"))] += 1

    plano = []
    for nivel in config["replay"]["niveis_tempo"]:
        for forma in FAMILIAS_FORMA:
            for perfil in FAMILIAS_PERFIL:
                for i in range(feitas[(forma, perfil, nivel)], n_por_celula):
                    plano.append((forma, perfil, nivel, (config["semente"], forma, perfil, int(nivel[1]), i)))
    rng = np.random.default_rng(config["semente"])
    ordem = rng.permutation(len(plano))
    return [plano[i] for i in ordem], sum(feitas.values())


def resumo(registro, falhas, caminho_csv):
    if not registro:
        print("Nenhuma sessão salva.")
        return
    existe = caminho_csv.exists()
    with open(caminho_csv, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(registro[0].keys()))
        if not existe:
            escritor.writeheader()
        escritor.writerows(registro)

    print(f"\nSessões salvas nesta execução: {len(registro)}  |  refeitas por travamento: {falhas}")
    print(f"{'nível':<6}{'shape':>6}{'vp':>4}{'sessões':>9}{'M médio':>9}{'N médio':>9}{'N/M':>7}")
    chaves = sorted({(r["timing"], r["shape"], r["vp"]) for r in registro})
    for nivel, forma, perfil in chaves:
        sel = [r for r in registro if (r["timing"], r["shape"], r["vp"]) == (nivel, forma, perfil)]
        M = np.mean([r["M"] for r in sel])
        N = np.mean([r["N"] for r in sel])
        print(f"{nivel:<6}{forma:>6}{perfil:>4}{len(sel):>9}{M:>9.1f}{N:>9.1f}{N / M:>7.2f}")
    print(f"Registro acumulado em {caminho_csv}")


def principal():
    config = carregar_config()
    parser = argparse.ArgumentParser(description="Bot de replay das trajetórias sintéticas.")
    parser.add_argument("--n", type=int, default=config["replay"]["sessoes_por_celula"],
                        help="Sessões por família e por nível de tempo")
    parser.add_argument("--teste", action="store_true",
                        help="Teste de fumaça: 2 sessões por célula, em pasta separada")
    parser.add_argument("--limite", type=int, default=None, help="Executar só as N primeiras sessões do plano")
    parser.add_argument("--params", default=str(caminho(config, "resultados") / "params_humanos.json"))
    args = parser.parse_args()

    pasta_sessoes = caminho(config, "bots_sessoes")
    pasta_trajetorias = caminho(config, "bots_trajetorias")
    n = args.n
    if args.teste:
        pasta_sessoes = pasta_sessoes.with_name(pasta_sessoes.name + "_teste")
        pasta_trajetorias = pasta_trajetorias.with_name(pasta_trajetorias.name + "_teste")
        n = 2
    pasta_sessoes.mkdir(parents=True, exist_ok=True)
    pasta_trajetorias.mkdir(parents=True, exist_ok=True)

    caminho_params = Path(args.params)
    if not caminho_params.is_absolute():
        caminho_params = RAIZ / caminho_params
    with open(caminho_params, encoding="utf-8") as f:
        params = json.load(f)

    plano, ja_feitas = montar_plano(config, n, pasta_sessoes)
    if args.limite is not None:
        plano = plano[:args.limite]
    print(f"Pasta: {pasta_sessoes}")
    print(f"Já salvas: {ja_feitas}  |  a executar: {len(plano)} "
          f"({n} por célula x 9 famílias x {len(config['replay']['niveis_tempo'])} níveis)")
    if not plano:
        return
    print(f"Tempo estimado: ~{len(plano) * 1.5 / 60:.0f} min. Não use o mouse durante a execução; Esc encerra.\n")

    root = tk.Tk()
    bot = BotReplay(root, plano, params, config, pasta_sessoes, pasta_trajetorias)
    try:
        root.mainloop()
    finally:
        resumo(bot.registro, bot.falhas, pasta_sessoes / "registro_execucao.csv")


if __name__ == "__main__":
    principal()
