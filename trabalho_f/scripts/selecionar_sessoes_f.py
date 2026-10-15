"""
selecionar_sessoes_f.py

Define quais sessões humanas entram nos experimentos, sem mover nem apagar
arquivos: grava trabalho_f/resultados/selecao_humanos.csv com uma linha por
sessão, os atributos usados na decisão, se foi incluída e o motivo.

Critérios (config_f.yaml, bloco "selecao"):
  - computador: tela e escala de DPI do computador principal, onde o bot
    também roda (sessões de outro computador, ou com a tela cheia ainda
    não aplicada, ex. 200x200, ficam de fora)
  - pausa entre sessões: lida do campo pause_between_sessions_ms ou, nas
    sessões antigas sem o campo, inferida pelo intervalo entre o clique da
    sessão anterior e o início desta
  - duração máxima (exclui distrações durante a sessão)
  - número mínimo de eventos

Uso:
    python trabalho_f/scripts/selecionar_sessoes_f.py
"""

import csv

import numpy as np

from common_f import caminho, carregar_config, carregar_sessoes

# Intervalo clique anterior -> início da sessão, para inferir a pausa nas
# sessões sem o campo. Inclui ~0,1 s entre pressionar e soltar o botão.
FAIXAS_PAUSA = [(0.0, 0.45, 100), (0.45, 0.9, 500), (0.9, 2.0, 1000)]


def _pausa_inferida(intervalo):
    if intervalo is None:
        return None
    for inicio, fim, pausa in FAIXAS_PAUSA:
        if inicio <= intervalo < fim:
            return pausa
    return None


def selecionar(diretorio, criterios):
    linhas = []
    clique_anterior = None
    for nome, s in carregar_sessoes(diretorio):  # ordem do nome = ordem cronológica
        movimentos = s.get("mouse_movements", [])
        timestamps = s.get("timestamps", [])
        janela = s.get("window_size", {})
        clique = s.get("click_data", {}).get("click_timestamp")

        inicio = timestamps[0] - movimentos[0]["time_offset"] if movimentos and timestamps else None
        intervalo = inicio - clique_anterior if (inicio is not None and clique_anterior is not None) else None
        pausa_campo = s.get("pause_between_sessions_ms")
        pausa = pausa_campo if pausa_campo is not None else _pausa_inferida(intervalo)
        clique_anterior = clique

        linha = {
            "arquivo": nome,
            "usuario": s.get("session_user", ["?"])[0].replace(" ", ""),
            "largura": janela.get("width"),
            "altura": janela.get("height"),
            "dpi": s.get("dpi_scale"),
            "pausa_ms": pausa,
            "pausa_origem": "campo" if pausa_campo is not None else ("inferida" if pausa else "desconhecida"),
            "duracao_s": round(float(s.get("total_time", 0)), 3),
            "eventos": len(movimentos),
        }

        motivos = []
        if (linha["largura"], linha["altura"]) != (criterios["largura"], criterios["altura"]):
            motivos.append("tela")
        if linha["dpi"] != criterios["dpi"]:
            motivos.append("dpi")
        if linha["pausa_ms"] != criterios["pausa_ms"]:
            motivos.append("pausa")
        if linha["duracao_s"] > criterios["duracao_max_s"]:
            motivos.append("duracao")
        if linha["eventos"] < criterios["eventos_min"]:
            motivos.append("eventos")
        linha["incluida"] = not motivos
        linha["motivo_exclusao"] = "+".join(motivos)
        linhas.append(linha)
    return linhas


def arquivos_incluidos(config) -> list:
    """Nomes dos arquivos incluídos, lidos de selecao_humanos.csv (para os outros scripts)."""
    with open(caminho(config, "selecao_humanos"), encoding="utf-8") as f:
        return [l["arquivo"] for l in csv.DictReader(f) if l["incluida"] == "True"]


def principal():
    config = carregar_config()
    criterios = config["selecao"]
    linhas = selecionar(caminho(config, "humanos"), criterios)

    saida = caminho(config, "selecao_humanos")
    saida.parent.mkdir(parents=True, exist_ok=True)
    with open(saida, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        escritor.writeheader()
        escritor.writerows(linhas)

    incluidas = [l for l in linhas if l["incluida"]]
    print(f"Sessões: {len(linhas)}  |  incluídas: {len(incluidas)}  |  excluídas: {len(linhas) - len(incluidas)}")
    print(f"Critérios: {criterios}")
    contagem = {}
    for l in linhas:
        if not l["incluida"]:
            contagem[l["motivo_exclusao"]] = contagem.get(l["motivo_exclusao"], 0) + 1
    print("Motivos de exclusão:")
    for motivo, n in sorted(contagem.items(), key=lambda kv: -kv[1]):
        print(f"  {motivo:<30} {n}")
    if incluidas:
        d = np.array([l["duracao_s"] for l in incluidas])
        e = np.array([l["eventos"] for l in incluidas])
        print(f"Incluídas: duração mediana {np.median(d):.2f} s (p95 {np.percentile(d, 95):.2f}); "
              f"eventos mediana {np.median(e):.0f} (p5 {np.percentile(e, 5):.0f}, p95 {np.percentile(e, 95):.0f})")
    print(f"Salvo em {saida}")


if __name__ == "__main__":
    principal()
