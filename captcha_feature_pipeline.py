"""
captcha_feature_pipeline.py

Pipeline de extração e normalização de features comportamentais
de sessões de mouse coletadas em CAPTCHA.

Lê todos os arquivos .json da pasta  captcha_data/  e gera:
  - dataset.csv          → features normalizadas (entrada do modelo ML)

Uso:
    python captcha_feature_pipeline.py

    # Opcional: pasta e saída personalizadas
    python captcha_feature_pipeline.py --input outra_pasta/ --output resultado.csv

Dependências:
    pip install numpy pandas scipy
"""

import json
import math
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, entropy as scipy_entropy


# ──────────────────────────────────────────────────────────────────────────────
# 1. CARREGAMENTO E VALIDAÇÃO
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_KEYS = {"mouse_movements", "timestamps", "velocities", "accelerations", "click_data"}


def load_session(path: str) -> dict:
    """Carrega e valida um arquivo JSON de sessão."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Campos obrigatórios ausentes: {missing}")

    n_mov = len(data["mouse_movements"])
    n_ts  = len(data["timestamps"])
    n_vel = len(data["velocities"])
    n_acc = len(data["accelerations"])

    if not (n_mov == n_ts == n_vel):
        raise ValueError(
            f"Tamanhos inconsistentes: "
            f"mouse_movements={n_mov}, timestamps={n_ts}, velocities={n_vel}"
        )
    if n_acc != n_vel - 1 and n_acc != n_vel:
        raise ValueError(
            f"accelerations deve ter N-1 ou N entradas (tem {n_acc}, esperado ~{n_vel})"
        )
    if n_mov < 5:
        raise ValueError(f"Sessão muito curta ({n_mov} movimentos). Mínimo: 5.")

    return data


# ──────────────────────────────────────────────────────────────────────────────
# 2. PRÉ-PROCESSAMENTO  (inclui normalização pela janela)
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(data: dict) -> dict:
    """
    Converte dados brutos em arrays numpy, aplicando normalização pela janela
    em TODAS as grandezas com dimensão espacial:

      Posição:      P_norm   = P_px   / diagonal_janela
      Velocidade:   v_norm   = v_px/s / diagonal_janela        → unidade: 1/s
      Aceleração:   a_norm   = a_px/s²/ diagonal_janela        → unidade: 1/s²

    Usar a diagonal em vez de largura/altura individualmente garante
    invariância tanto para movimentos horizontais quanto verticais,
    independente da resolução ou proporção da tela.

    A normalização temporal (timestamps, Δt) NÃO é alterada — segundos
    são uma unidade absoluta e universal.
    """
    movements     = data["mouse_movements"]
    timestamps    = np.array(data["timestamps"],    dtype=np.float64)
    velocities    = np.array(data["velocities"],    dtype=np.float64)
    accelerations = np.array(data["accelerations"], dtype=np.float64)
    click         = data["click_data"]

    # ── Dimensões da janela ───────────────────────────────────────────────────
    win_origin = data.get("window_origin", {})
    win_x      = float(win_origin.get("x", 1920) or 1920)   # fallback: 1920px
    win_y      = float(win_origin.get("y", 1080) or 1080)   # fallback: 1080px
    win_diag   = math.sqrt(win_x**2 + win_y**2)             # diagonal em pixels

    # ── Posições normalizadas ─────────────────────────────────────────────────
    # Divide pela diagonal → valores adimensionais no intervalo [0, ~1]
    xs = np.array([m["x"] / win_diag for m in movements], dtype=np.float64)
    ys = np.array([m["y"] / win_diag for m in movements], dtype=np.float64)

    # ── Clique normalizado ────────────────────────────────────────────────────
    click_x = click["click_position"][0] / win_diag
    click_y = click["click_position"][1] / win_diag

    # ── Velocidades normalizadas ──────────────────────────────────────────────
    # v_px/s → v_norm = v_px/s / win_diag   (unidade: diagonal/s → adimensional/s)
    # Isso torna sessões de telas diferentes diretamente comparáveis.
    vel_norm = velocities / win_diag

    # ── Acelerações normalizadas ──────────────────────────────────────────────
    acc_norm = accelerations / win_diag

    # ── Intervalos de tempo ───────────────────────────────────────────────────
    dt = np.diff(timestamps)
    dt = np.where(dt < 1e-9, 1e-9, dt)   # floor para evitar divisão por zero

    # ── Deslocamentos normalizados ────────────────────────────────────────────
    dx         = np.diff(xs)
    dy         = np.diff(ys)
    step_sizes = np.sqrt(dx**2 + dy**2)

    return {
        # Posição
        "xs": xs, "ys": ys,
        # Clique
        "click_x": click_x, "click_y": click_y,
        "time_to_click":   float(click["time_to_click"]),
        "click_timestamp": float(click["click_timestamp"]),
        # Cinemática normalizada
        "velocities":     vel_norm,
        "accelerations":  acc_norm,
        # Deltas
        "dt": dt, "dx": dx, "dy": dy,
        "step_sizes": step_sizes,
        # Metadados de janela (guardados para auditoria)
        "win_x": win_x, "win_y": win_y, "win_diag": win_diag,
        # Metadados de sessão
        "session_id": data.get("session_id", "unknown"),
        "user": (data.get("session_user") or ["unknown"])[0],
        "n": len(movements),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. EXTRAÇÃO DE FEATURES
# ──────────────────────────────────────────────────────────────────────────────

def feat_jerk(p: dict) -> dict:
    """
    Jerk = derivada temporal da aceleração (já normalizada pela janela).
    j[i] = (a_norm[i] - a_norm[i-1]) / Δt[i]    unidade: (diag⁻¹·s⁻²) / s = diag⁻¹·s⁻³
    """
    acc = p["accelerations"]
    dt  = p["dt"]

    min_len = min(len(acc) - 1, len(dt))
    d_acc   = np.diff(acc[:min_len + 1])
    jerk    = d_acc[:min_len] / dt[:min_len]
    jerk_abs = np.abs(jerk)

    return {
        "jerk_mean": float(np.mean(jerk_abs)),
        "jerk_std":  float(np.std(jerk_abs)),
        "jerk_max":  float(np.max(jerk_abs)),
        "jerk_skew": float(skew(jerk_abs)) if len(jerk_abs) > 2 else 0.0,
    }


def feat_ratio_distance(p: dict) -> dict:
    """
    Razão distância percorrida / deslocamento euclidiano.
    Ambos já normalizados pela diagonal → ratio adimensional.
    ratio = d_caminho / d_euclidiana  (>= 1 por definição)
    """
    xs, ys = p["xs"], p["ys"]
    cx, cy = p["click_x"], p["click_y"]

    d_euclidiana = math.sqrt((cx - xs[0])**2 + (cy - ys[0])**2)
    d_caminho    = float(np.sum(p["step_sizes"]))
    ratio        = d_caminho / d_euclidiana if d_euclidiana > 1e-9 else 1.0

    return {
        "dist_euclidiana": d_euclidiana,
        "dist_caminho":    d_caminho,
        "ratio_dist_desl": ratio,
    }


def feat_curvatura(p: dict) -> dict:
    """
    Curvatura = ângulo de mudança de direção entre vetores consecutivos.
    θ[i] = arccos( (v1·v2) / (|v1|·|v2|) )   ∈ [0, π]
    """
    dx, dy  = p["dx"], p["dy"]
    thetas  = []

    for i in range(1, len(dx)):
        v1 = np.array([dx[i-1], dy[i-1]])
        v2 = np.array([dx[i],   dy[i]])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            continue
        cos_t = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        thetas.append(math.acos(cos_t))

    thetas    = np.array(thetas) if thetas else np.array([0.0])
    n_bruscas = int(np.sum(thetas > math.pi / 4))

    return {
        "curvatura_mean":      float(np.mean(thetas)),
        "curvatura_std":       float(np.std(thetas)),
        "curvatura_max":       float(np.max(thetas)),
        "n_mudancas_bruscas":  n_bruscas,
    }


def feat_retrocessos(p: dict) -> dict:
    """
    Retrocessos = inversões de sinal em dx ou dy (cursor volta na direção oposta).
    Filtro de ruído: ignora deslocamentos < 1e-6 (sub-pixel após normalização).
    """
    dx, dy = p["dx"], p["dy"]

    # Filtra passos sub-pixel para não contar ruído de quantização
    dx_f = np.where(np.abs(dx) < 1e-6, 0.0, dx)
    dy_f = np.where(np.abs(dy) < 1e-6, 0.0, dy)

    ret_x = int(np.sum(dx_f[1:] * dx_f[:-1] < 0))
    ret_y = int(np.sum(dy_f[1:] * dy_f[:-1] < 0))
    n_ret = ret_x + ret_y

    max_teorico   = 2 * max(len(dx) - 1, 1)
    taxa_retrocesso = n_ret / max_teorico

    return {
        "retrocessos_x":     ret_x,
        "retrocessos_y":     ret_y,
        "n_retrocessos":     n_ret,
        "taxa_retrocesso":   taxa_retrocesso,
    }


def feat_time_to_click(p: dict) -> dict:
    """TTC = t_clique - t[0]  (em segundos)."""
    return {"time_to_click": p["time_to_click"]}


def feat_n_movimentos(p: dict) -> dict:
    """
    Número de eventos e regularidade dos passos.
    CV dos passos = std(step_sizes) / mean(step_sizes)  — adimensional.
    """
    n           = p["n"]
    ttc         = p["time_to_click"]
    step_sizes  = p["step_sizes"]

    densidade  = n / ttc if ttc > 0 else 0.0
    mean_step  = float(np.mean(step_sizes)) if len(step_sizes) > 0 else 0.0
    cv_passos  = (float(np.std(step_sizes)) / mean_step) if mean_step > 1e-12 else 0.0

    return {
        "n_movimentos":       n,
        "densidade_amostral": densidade,
        "passo_medio":        mean_step,
        "cv_passos":          cv_passos,
    }


def feat_distancia_centro(p: dict, centro_alvo: tuple = None) -> dict:
    """
    Distância do clique ao centro do elemento alvo (normalizada pela diagonal).
    Se centro_alvo não fornecido, usa (win_x/2, win_y/2) — centro da janela.
    """
    win_diag = p["win_diag"]

    if centro_alvo is not None:
        cx_alvo = centro_alvo[0] / win_diag
        cy_alvo = centro_alvo[1] / win_diag
    else:
        # Centro da janela normalizado
        cx_alvo = (p["win_x"] / 2) / win_diag
        cy_alvo = (p["win_y"] / 2) / win_diag

    offset_x = p["click_x"] - cx_alvo
    offset_y = p["click_y"] - cy_alvo
    d_centro = math.sqrt(offset_x**2 + offset_y**2)

    return {
        "d_centro_clique": d_centro,
        "offset_x_clique": offset_x,
        "offset_y_clique": offset_y,
    }


def feat_variancia_temporal(p: dict) -> dict:
    """
    Variância e CV dos intervalos Δt entre eventos consecutivos.
    CV = std(Δt) / mean(Δt)  — adimensional, invariante à velocidade média.
    """
    dt      = p["dt"]
    mean_dt = float(np.mean(dt))
    std_dt  = float(np.std(dt))
    cv_dt   = std_dt / mean_dt if mean_dt > 1e-9 else 0.0

    return {
        "mean_dt":     mean_dt,
        "var_dt":      float(np.var(dt)),
        "cv_temporal": cv_dt,
    }


def feat_std_intervalos(p: dict) -> dict:
    """
    Desvio padrão, assimetria e percentis dos intervalos Δt.
    Assimetria positiva indica pausas ocasionais (típico humano).
    """
    dt = p["dt"]
    return {
        "std_dt":  float(np.std(dt)),
        "skew_dt": float(skew(dt)) if len(dt) > 2 else 0.0,
        "p25_dt":  float(np.percentile(dt, 25)),
        "p75_dt":  float(np.percentile(dt, 75)),
        "p95_dt":  float(np.percentile(dt, 95)),
    }


def feat_vel_aproximacao(p: dict, k: int = 5) -> dict:
    """
    Velocidade média nos últimos k eventos (normalizada pela janela).
    ratio_desaceleracao = v_aprox / v_global  < 1 para humanos (Lei de Fitts).
    """
    vel  = p["velocities"]
    k    = min(k, len(vel) - 1) or 1
    v_aprox  = float(np.mean(vel[-k:]))
    v_global = float(np.mean(vel))
    ratio    = v_aprox / v_global if v_global > 1e-12 else 1.0

    return {
        "vel_aproximacao":     v_aprox,
        "ratio_desaceleracao": ratio,
    }


def feat_desvio_lateral(p: dict) -> dict:
    """
    Desvio lateral máximo = maior distância perpendicular de qualquer ponto
    à reta origem → clique. Captura overshooting humano.
    Já em unidades normalizadas (diagonal).
    """
    xs, ys  = p["xs"], p["ys"]
    cx, cy  = p["click_x"], p["click_y"]

    vx = cx - xs[0]
    vy = cy - ys[0]
    comprimento = math.sqrt(vx**2 + vy**2)

    if comprimento < 1e-9:
        return {"desvio_lateral_max": 0.0, "desvio_lateral_mean": 0.0}

    d_perp = np.abs((xs - xs[0]) * vy - (ys - ys[0]) * vx) / comprimento

    return {
        "desvio_lateral_max":  float(np.max(d_perp)),
        "desvio_lateral_mean": float(np.mean(d_perp)),
    }


def feat_entropia_velocidade(p: dict, n_bins: int = 10) -> dict:
    """
    Entropia de Shannon das velocidades normalizadas.
    H_norm = H / log2(n_bins)  ∈ [0, 1]
    Alta entropia → movimento imprevisível → humano.
    """
    vel    = p["velocities"]
    counts, _ = np.histogram(vel, bins=n_bins)
    counts = counts[counts > 0]
    probs  = counts / counts.sum()

    h      = float(scipy_entropy(probs, base=2))
    h_norm = h / math.log2(n_bins) if n_bins > 1 else 0.0

    return {
        "entropia_vel":      h,
        "entropia_vel_norm": h_norm,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. PIPELINE POR SESSÃO
# ──────────────────────────────────────────────────────────────────────────────

def extract_features(data: dict, centro_alvo: tuple = None) -> dict:
    """Executa todos os extratores em uma sessão. Retorna dict plano de features."""
    p = preprocess(data)

    features = {
        "session_id": p["session_id"],
        "user":       p["user"],
        # Metadados de janela (auditoria — não entram no modelo)
        "win_x":      p["win_x"],
        "win_y":      p["win_y"],
        "win_diag":   p["win_diag"],
    }

    extractors = [
        feat_jerk,
        feat_ratio_distance,
        feat_curvatura,
        feat_retrocessos,
        feat_time_to_click,
        feat_n_movimentos,
        lambda pp: feat_distancia_centro(pp, centro_alvo),
        feat_variancia_temporal,
        feat_std_intervalos,
        feat_vel_aproximacao,
        feat_desvio_lateral,
        feat_entropia_velocidade,
    ]

    for extractor in extractors:
        features.update(extractor(p))

    return features


# ──────────────────────────────────────────────────────────────────────────────
# 5. EXECUÇÃO PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def collect_json_paths(input_path: str) -> list:
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        paths = sorted(p.glob("*.json"))
        if not paths:
            raise FileNotFoundError(f"Nenhum .json encontrado em: {input_path}")
        return [str(x) for x in paths]
    raise FileNotFoundError(f"Caminho não encontrado: {input_path}")


def run_pipeline(input_path: str = "captcha_data",
                 output_path: str = "dataset.csv",
                 centro_alvo: tuple = None):
    """
    Lê todos os JSONs, extrai features, normaliza e salva os CSVs.
    """
    paths = collect_json_paths(input_path)
    print(f"\n[INFO] {len(paths)} arquivo(s) encontrado(s) em '{input_path}'")

    rows, errors = [], []

    for path in paths:
        try:
            data  = load_session(path)
            feats = extract_features(data, centro_alvo=centro_alvo)
            rows.append(feats)
            print(f"  [OK] {Path(path).name}  ({feats['n_movimentos']} movimentos)")
        except Exception as e:
            errors.append((path, str(e)))
            print(f"  [WARN] {Path(path).name}: {e}")

    if not rows:
        print("\n[ERRO] Nenhuma sessão processada. Verifique os arquivos JSON.")
        sys.exit(1)

    df_raw = pd.DataFrame(rows)
    print(f"\n[INFO] {len(df_raw)} sessões OK  |  {len(errors)} ignoradas")
    print(f"[INFO] {len(df_raw.columns)} colunas no dataset")

    # ── Salvar arquivos ───────────────────────────────────────────────────────
    base     = output_path.replace(".csv", "")
    raw_path = f"{base}_raw.csv"

    df_raw.to_csv(output_path, index=False)

    print(f"\n[SALVO] {output_path}  ← dataset completo (entrada do modelo)")

    # ── Resumo ────────────────────────────────────────────────────────────────
    META_COLS = {"session_id", "user", "win_x", "win_y", "win_diag"}
    feat_cols = [c for c in df_raw.columns if c not in META_COLS]
    numeric   = df_raw[feat_cols].select_dtypes(include=[np.number])
    print("\n── Estatísticas das features (valores brutos, normalizados pela janela) ──")
    print(numeric.describe().round(6).to_string())

    if errors:
        print(f"\n── Arquivos ignorados ({len(errors)}) ──")
        for path, reason in errors:
            print(f"  {Path(path).name}: {reason}")

    return df_raw


# ──────────────────────────────────────────────────────────────────────────────
# 6. CLI (parâmetros opcionais — padrões já configurados para captcha_data/)
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline CAPTCHA — lê captcha_data/ e gera dataset.csv por padrão."
    )
    parser.add_argument(
        "--input", "-i", default="captcha_data",
        help="Pasta com os arquivos .json (padrão: captcha_data/)"
    )
    parser.add_argument(
        "--output", "-o", default="dataset.csv",
        help="CSV de saída normalizado (padrão: dataset.csv)"
    )
    parser.add_argument(
        "--centro-x", type=float, default=None,
        help="Coordenada X do centro do elemento CAPTCHA em pixels (opcional)"
    )
    parser.add_argument(
        "--centro-y", type=float, default=None,
        help="Coordenada Y do centro do elemento CAPTCHA em pixels (opcional)"
    )
    args = parser.parse_args()

    # ── Diagnóstico inicial ───────────────────────────────────────────────────
    import os
    print("=" * 60)
    print("CAPTCHA Feature Pipeline")
    print("=" * 60)
    print(f"[DIAG] Diretório atual : {os.getcwd()}")
    print(f"[DIAG] Pasta de entrada: {args.input}")
    print(f"[DIAG] Arquivo de saída: {args.output}")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"\n[ERRO] Pasta '{args.input}' nao encontrada em: {os.getcwd()}")
        print("       Crie a pasta e coloque os arquivos .json dentro dela.")
        print(f"       Ou passe o caminho correto com:  --input caminho/para/pasta")
        sys.exit(1)

    json_files = list(input_path.glob("*.json"))
    print(f"[DIAG] Arquivos .json encontrados: {len(json_files)}")
    for jf in json_files:
        print(f"         - {jf.name}")

    if not json_files:
        print(f"\n[ERRO] Nenhum arquivo .json encontrado em '{args.input}'.")
        print("       Verifique se os arquivos têm extensão .json (minúsculo).")
        sys.exit(1)

    print()

    # ── Execução ──────────────────────────────────────────────────────────────
    centro = None
    if args.centro_x is not None and args.centro_y is not None:
        centro = (args.centro_x, args.centro_y)

    try:
        run_pipeline(
            input_path=args.input,
            output_path=args.output,
            centro_alvo=centro,
        )
    except Exception as e:
        print(f"\n[ERRO INESPERADO] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
