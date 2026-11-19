"""
05_gif_apresentacao_f.py

GIF da tela inicial da apresentação: imita o painel do CAPTCHA da coleta
(data_collection.py), ampliado para 1920x1080, com o título do trabalho no
lugar do texto original e o logo da UTFPR no lugar do ícone. Um cursor grande
(visível em projetor) passeia pela tela, chega perto do checkbox
"Começar apresentação", hesita e se afasta.

O percurso é uma spline cúbica PERIÓDICA no tempo (scipy CubicSpline com
bc_type="periodic"): posição e velocidade no fim do ciclo são iguais às do
início, então o loop não tem emenda visível.

Saída: trabalho_f/apresentacao_f/tela_inicial_f.gif (e um PNG estático do
primeiro quadro, para conferência).

Uso:
    python trabalho_f/scripts/05_gif_apresentacao_f.py
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import CubicSpline

RAIZ = Path(__file__).resolve().parents[2]
PASTA = RAIZ / "trabalho_f" / "apresentacao_f"
FONTES = Path("C:/Windows/Fonts")

LARGURA, ALTURA = 1920, 1080
DURACAO_S, QUADROS_POR_S = 8.0, 25
ALTURA_CURSOR = 64                   # px; o cursor padrão tem ~20 px numa tela 1080p

# cores da interface original (data_collection.py)
FUNDO, BRANCO = "#fefbf5", "#ffffff"
TEXTO_TITULO, TEXTO_STATUS, TEXTO_CONTADOR = "#333333", "#666666", "#999999"
BORDA_CLARA, BORDA_ESCURA = "#ffffff", "#a0a0a0"
BORDO = (128, 0, 32)                 # #800020, cursor e "Começar apresentação"


def fonte(arquivo, tamanho):
    return ImageFont.truetype(str(FONTES / arquivo), tamanho)


def texto_centralizado(desenho, cx, y, texto, f, cor):
    desenho.text((cx, y), texto, font=f, fill=cor, anchor="mt")


def ajustar_fonte(desenho, texto, arquivo, tamanho, largura_max):
    """Diminui a fonte até o texto caber em largura_max."""
    while tamanho > 10:
        f = fonte(arquivo, tamanho)
        if desenho.textlength(texto, font=f) <= largura_max:
            return f
        tamanho -= 1
    return fonte(arquivo, tamanho)


def borda_relevo(desenho, caixa, largura, elevada=True):
    """Borda RAISED (elevada) ou GROOVE (sulco) do Tkinter."""
    x0, y0, x1, y1 = caixa
    sup_esq, inf_dir = (BORDA_CLARA, BORDA_ESCURA) if elevada else (BORDA_ESCURA, BORDA_CLARA)
    for i in range(largura):
        desenho.line([(x0 + i, y1 - i), (x0 + i, y0 + i), (x1 - i, y0 + i)], fill=sup_esq)
        desenho.line([(x0 + i, y1 - i), (x1 - i, y1 - i), (x1 - i, y0 + i)], fill=inf_dir)


def desenhar_tela():
    """Fundo estático; devolve a imagem e o centro do checkbox."""
    tela = Image.new("RGB", (LARGURA, ALTURA), FUNDO)
    d = ImageDraw.Draw(tela)

    # painel (main_frame: branco, RAISED)
    lp, ap = 1080, 545
    px0, py0 = (LARGURA - lp) // 2, (ALTURA - ap) // 2
    d.rectangle([px0, py0, px0 + lp, py0 + ap], fill=BRANCO)
    borda_relevo(d, (px0, py0, px0 + lp, py0 + ap), 4, elevada=True)
    cx = px0 + lp // 2

    # título e subtítulo
    texto_centralizado(d, cx, py0 + 55, "CAPTCHA COMPORTAMENTAL", fonte("arialbd.ttf", 60), TEXTO_TITULO)
    sub = "APLICAÇÃO DE SVM DE UMA CLASSE E FLORESTA DE ISOLAMENTO"
    texto_centralizado(d, cx, py0 + 135, sub, ajustar_fonte(d, sub, "arial.ttf", 30, lp - 140), TEXTO_STATUS)

    # moldura do checkbox (GROOVE)
    fx0, fy0, fx1, fy1 = cx - 360, py0 + 215, cx + 360, py0 + 345
    borda_relevo(d, (fx0, fy0, fx1, fy1), 2, elevada=False)
    borda_relevo(d, (fx0 + 2, fy0 + 2, fx1 - 2, fy1 - 2), 2, elevada=True)
    meio_y = (fy0 + fy1) // 2

    # checkbox vazio (quadrado afundado, como o Checkbutton do Windows)
    lado = 36
    qx0, qy0 = fx0 + 30, meio_y - lado // 2
    d.rectangle([qx0, qy0, qx0 + lado, qy0 + lado], fill=BRANCO, outline="#333333", width=3)
    centro_checkbox = np.array([qx0 + lado / 2, qy0 + lado / 2])

    d.text((qx0 + lado + 18, meio_y), "Começar apresentação", font=fonte("arial.ttf", 34),
           fill=BORDO, anchor="lm")

    # logo da UTFPR no lugar do ícone
    logo = Image.open(PASTA / "logo_utfpr_f.png").convert("RGB")
    altura_logo = fy1 - fy0 - 30
    logo = logo.resize((round(logo.width * altura_logo / logo.height), altura_logo), Image.LANCZOS)
    tela.paste(logo, (fx1 - 24 - logo.width, meio_y - altura_logo // 2))

    # linhas de status e contador
    texto_centralizado(d, cx, py0 + 400, "Trabalho de Conclusão de Curso", fonte("arial.ttf", 38), TEXTO_STATUS)
    texto_centralizado(d, cx, py0 + 460, "Graduação em Engenharia da Computação", fonte("arial.ttf", 32), TEXTO_CONTADOR)
    return tela, centro_checkbox


def desenhar_cursor(altura):
    """Seta padrão bordô, com contorno bordô escuro e sombra leve, com antisserrilhado."""
    # contorno da seta clássica numa grade 12 x 19
    seta = [(0, 0), (0, 16), (4, 12.4), (7, 18.6), (9.4, 17.6), (6.4, 11.4), (11.6, 11.4)]
    s = 4                                  # superamostragem
    escala = altura / 19 * s
    margem = 6 * s
    pts = [(x * escala + margem, y * escala + margem) for x, y in seta]
    tam = (int(12 * escala + 3 * margem), int(19 * escala + 3 * margem))

    camada = Image.new("RGBA", tam, (0, 0, 0, 0))
    d = ImageDraw.Draw(camada)
    sombra = [(x + 3 * s, y + 3 * s) for x, y in pts]
    d.polygon(sombra, fill=(0, 0, 0, 70))
    d.polygon(pts, fill=BORDO + (255,), outline=(70, 0, 18, 255), width=int(1.6 * s * altura / 32))
    camada = camada.resize((tam[0] // s, tam[1] // s), Image.LANCZOS)
    return camada, margem // s             # ponta da seta fica em (margem, margem)


def percurso(centro):
    """Spline periódica (x(t), y(t)), t em segundos, passando perto do checkbox."""
    cx, cy = centro
    # (fração do ciclo, x, y)
    pontos = [
        (0.00, 1560, 930),
        (0.11, 1760, 560),
        (0.22, 1400, 150),
        (0.33, 820, 110),
        (0.43, 260, 330),
        (0.53, cx - 90, cy + 150),      # aproxima por baixo
        (0.61, cx + 30, cy + 34),       # desacelera ao lado do checkbox
        (0.67, cx + 20, cy + 24),       # hesita, quase clica
        (0.72, cx + 34, cy + 40),
        (0.79, cx + 260, cy + 260),     # desiste e se afasta
        (0.89, 1050, 1000),
    ]
    t = np.array([p[0] for p in pontos] + [1.0]) * DURACAO_S
    xy = np.array([p[1:] for p in pontos] + [pontos[0][1:]], dtype=float)
    return CubicSpline(t, xy, bc_type="periodic")


def principal():
    PASTA.mkdir(parents=True, exist_ok=True)
    tela, centro = desenhar_tela()
    cursor, ponta = desenhar_cursor(ALTURA_CURSOR)
    caminho = percurso(centro)

    # paleta única para todos os quadros (sem cintilação entre quadros). O
    # fundo e o branco do painel entram exatos na paleta, e cada pixel recebe
    # a cor mais próxima (a conversão do Pillow aproximaria tons parecidos).
    amostra = tela.copy()
    amostra.paste(cursor, (100, 100), cursor)
    medianas = amostra.quantize(colors=254, method=Image.Quantize.MEDIANCUT).getpalette()[:254 * 3]
    exatas = [int(FUNDO[i:i + 2], 16) for i in (1, 3, 5)] + [255, 255, 255]
    cores = exatas + medianas
    paleta = np.array(cores, dtype=np.int32).reshape(-1, 3)

    def indexar(rgb):
        """Índice da cor mais próxima na paleta para cada pixel (h x w x 3)."""
        compactado = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2]
        unicos, inverso = np.unique(compactado, return_inverse=True)
        cu = np.stack([unicos >> 16, (unicos >> 8) & 255, unicos & 255], axis=1)
        dist = ((cu[:, None, :] - paleta[None, :, :]) ** 2).sum(axis=2)
        return dist.argmin(axis=1).astype(np.uint8)[inverso].reshape(rgb.shape[:2])

    base = indexar(np.asarray(tela))
    larg_c, alt_c = cursor.size
    n = int(DURACAO_S * QUADROS_POR_S)
    quadros = []
    for k in range(n):                     # t = k/n do ciclo: o quadro n seria igual ao 0
        x, y = caminho(k / QUADROS_POR_S)
        x0, y0 = round(x) - ponta, round(y) - ponta
        q = tela.copy()
        q.paste(cursor, (x0, y0), cursor)
        # só a região do cursor muda em relação à tela de fundo
        caixa = (max(x0, 0), max(y0, 0), min(x0 + larg_c, LARGURA), min(y0 + alt_c, ALTURA))
        indices = base.copy()
        indices[caixa[1]:caixa[3], caixa[0]:caixa[2]] = indexar(np.asarray(q.crop(caixa)))
        quadro = Image.fromarray(indices, mode="P")
        quadro.putpalette(cores)
        quadros.append(quadro)

    destino = PASTA / "tela_inicial_f.gif"
    quadros[0].save(destino, save_all=True, append_images=quadros[1:], loop=0,
                    duration=round(1000 / QUADROS_POR_S), optimize=False, disposal=1)
    quadros[0].convert("RGB").save(PASTA / "tela_inicial_quadro_f.png")
    print(f"{n} quadros, {DURACAO_S:.0f} s em loop -> {destino} ({destino.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    principal()
