# Pendências de texto

Ajustes já identificados no texto do TCC. Marcar ao resolver no `.tex`.

## Do checklist original
- [ ] Tabela 2 → 5 (referência cruzada errada)
- [ ] Mediana 9,555 → 4,9555
- [ ] Placeholder BOXPLOT-IF
- [ ] Renumerar seções 4.0.1 / 4.0.2
- [ ] Padronizar "Floresta de Isolamento" (ou "Isolação") em todo o texto
- [ ] Consolidar referências duplicadas (Schölkopf 2001a/b; Acien 2020 / Niu 2021)
- [ ] Palavras-chave do resumo
- [ ] Conferir a contagem de divergências (169/78) e o sinal da diferença contra as FPRs

## Seção 3.3 — Interface e coleta
- [ ] Interface passa a ser **tela cheia**, com o painel do checkbox (400 × 250 px) em posição aleatória a cada sessão (no lugar da janela 500 × 400 em posição aleatória).
- [ ] Coordenadas de movimento e de clique no referencial da janela (`x_root − origem da janela`); clique registrado no pressionamento do botão sobre o checkbox.
- [ ] Primeiro evento de movimento incluído na sessão.
- [ ] Alvo = apenas o quadrado do checkbox (≈ 27 × 22 px); o texto "Autenticação" é um rótulo não clicável. Atualizar $W$ da Lei de Fitts, se citado.
- [ ] Fluxo automático: após o clique, a sessão é salva e a próxima começa 0,5 s depois, com o painel em nova posição, sem diálogos. A trajetória seguinte parte de onde o cursor parou (o ponto do clique anterior), o que define a distribuição de $P_0$.
- [ ] Versões da coleta: v1 (`captcha_data/`, erro de coordenadas, 10.175 sessões), v2 (piloto de 80 sessões com o texto clicável, descartado), v3 (`captcha_data_f/`, base final).
- [ ] O clique é o instante em que o botão é pressionado; movimentos entre pressionar e soltar (~100 ms, até ~12 px, dentro do alvo) não fazem parte da trajetória e são separados em `movements_after_click`.
- [ ] Taxa de amostragem do mouse: Δt mediano de 8 ms (≈ 125 Hz). Citar ao descrever o dispositivo e ao justificar o Δt do nível T0.
- [ ] Justificar a recoleta: o coletor v1 registrava as coordenadas no referencial do elemento sob o cursor (checkbox, moldura, painel), gerando saltos de 50–100 px sem movimento real em ~92% das sessões e invalidando `dist_clique`. Evidência: cursor em linha reta horizontal (y = 206) gravado como y ∈ {14, 31, 131}.
- [ ] Número de sessões e participantes da coleta v3 `[preencher]`.

## Seção 3.4 — Características
- [ ] Equação 24: $t_{\text{reação}} = t_c - t_0$ (o código já calcula assim). Remover o parágrafo órfão sobre $T$, $D$ e $W = 136$ px.
- [ ] 3.4.3: centro de $\text{dist}_{\text{clique}}$ = centro do checkbox, lido de cada sessão (o alvo muda de lugar).
- [ ] Declarar o pré-processamento: winsorização nos percentis 0,5 % e 99,5 % seguida de padronização (`StandardScaler`), ambas ajustadas só no treino de cada fold, aplicadas aos **dois** modelos (o texto reformulado diz que a Floresta recebe as características sem transformação — não é o que o código faz; decidir e alinhar).

## Seção 3.5 — Gerador
- [ ] Declarar a adaptação: Acien et al. definem as formas como $y = f(x)$ em coordenadas de tela; aqui são definidas no sistema local $P_0 \to P_c$, para valer em qualquer direção.
- [ ] Escrever as formas na versão adimensional ($s = u/L$, $z = w/L$): quadrática $z = a\,s(s-1)$; exponencial $z = a\,[(e^{bs}-1)/(e^b-1) - s]$. Assim $a$ e $b$ não dependem do comprimento da trajetória.
- [ ] Perfil gaussiano: centro em $i = M/2$ (centro do intervalo $i = 1..M-1$), não $(M-1)/2$.
- [ ] $M$: Acien et al. condicionam $M$ à **direção** (8 trajetórias); o gerador usa direção (8 setores) × faixa de comprimento (tercis de $L$). Descrever assim.
- [ ] $a$ e $b$ da exponencial são sorteados como par (da mesma trajetória humana), dentro dos percentis 5–95 de cada um.
- [ ] Perfil "constante" espaça os pontos uniformemente ao longo do segmento $P_0 \to P_c$ (como Acien, que espaça $x$), e não ao longo do arco: em formas muito curvas, a distância entre pontos varia. Decidir se mantém (fiel ao artigo) ou se reparametriza por comprimento de arco.
- [ ] $P_0$ dos humanos v3 é onde o cursor estava ao abrir a sessão (o ponto do clique anterior). Descrever ao apresentar a distribuição de $P_0$.
- [ ] Expoente $b$ da exponencial: a busca foi limitada a $|b| \le 10$. Mais de 5 % das trajetórias humanas têm o melhor ajuste no limite (mesmo ampliando para ±20), o que corresponde a desvio concentrado num extremo do movimento (correção final). Declarar o limite e essa interpretação.
- [ ] Base humana: notebook Linux (1366 × 768), mouse e touchpad sem rótulo por sessão; 5.060 sessões incluídas de 6.000 (critérios em `selecao_humanos.csv`: tela, pausa de 100 ms, duração ≤ 5 s, ≥ 10 eventos). O bot roda no mesmo notebook. Mistura de dispositivos de entrada vai para as limitações.
- [ ] Notebook: 4–7 % dos intervalos entre eventos < 1 ms (eventos entregues em rajadas ao programa); tratado pelo piso `MIN_DT`. Afeta humanos e bots igualmente, pois rodam no mesmo computador.
