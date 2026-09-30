# componentes/plotting.py
from __future__ import annotations
# --- Stdlib
import os
import glob
from typing import Optional, List
# --- Third-party
import pandas as pd
import numpy as np
import matplotlib
if os.environ.get("MPLBACKEND", "").lower() != "agg":
    matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from componentes import settings as CFG

# ------------------------------------------------------------
# Utilidades mínimas
# ------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _ensure_dir_is_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _auto_ext(path: str, fmt: Optional[str] = None) -> str:
    fmt = (fmt or getattr(CFG, "GRAFICOS_FORMAT", "png") or "png").lstrip(".")
    root, _ = os.path.splitext(path)
    return f"{root}.{fmt}"

def _with_timestamp(path: str) -> str:
    if not bool(getattr(CFG, "EXPORTAR_CON_TIMESTAMP", False)):
        return path
    root, ext = os.path.splitext(path)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{root}_{ts}{ext}"
'''
def _safe_savefig(fig: plt.Figure, out_path: str, fmt: Optional[str] = None, dpi: int = 150) -> str:
    out_path = _auto_ext(out_path, fmt)
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # copia con timestamp (si aplica)
    stamped = _with_timestamp(out_path)
    if stamped != out_path:
        try:
            with open(out_path, "rb") as src, open(stamped, "wb") as dst:
                dst.write(src.read())
        except Exception:
            pass
    return out_path
'''
def _safe_savefig(fig: plt.Figure, out_path: str, fmt: Optional[str] = None, dpi: int = 150) -> str:
    # Normaliza extensión
    out_path = _auto_ext(out_path, fmt)

    # Calcula el nombre final respetando el flag de timestamp:
    # - Si _with_timestamp(out_path) devuelve el mismo path, no hay timestamp.
    # - Si devuelve uno distinto, ese será el nombre final con timestamp.
    stamped = _with_timestamp(out_path)
    final_path = stamped  # siempre guardamos SOLO en este archivo

    _ensure_dir(final_path)
    fig.savefig(final_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Ya no copiamos ni generamos el archivo base -> no hay duplicado
    return final_path

def _read_csv_safe(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def _to_datetime(s: pd.Series, name_hint: str = "fecha") -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False).rename(name_hint)

def _load_serie() -> pd.DataFrame:
    df = _read_csv_safe(CFG.SERIE_LIMPIA)
    if df is None or df.empty:
        raise FileNotFoundError(f"No se pudo cargar la serie desde: {CFG.SERIE_LIMPIA}")
    cols = {c.lower(): c for c in df.columns}
    fecha_col = cols.get("fecha") or cols.get("date") or list(df.columns)[0]
    valor_col = cols.get("valor") or cols.get("value") or list(df.columns)[1]
    df = df[[fecha_col, valor_col]].copy()
    df.columns = ["fecha", "valor"]
    df["fecha"] = _to_datetime(df["fecha"])
    df = df.sort_values("fecha").dropna(subset=["fecha", "valor"])
    return df

def _read_anom_points(csv_path: str) -> Optional[pd.DataFrame]:
    """Lee puntos ['fecha','valor'] si existen en el CSV dado."""
    df = _read_csv_safe(csv_path)
    if df is None or df.empty:
        return None
    cols = {c.lower(): c for c in df.columns}
    if ("fecha" in cols or "date" in cols) and ("valor" in cols or "value" in cols):
        fcol = cols.get("fecha") or cols.get("date")
        vcol = cols.get("valor") or cols.get("value")
        out = df[[fcol, vcol]].copy()
        out.columns = ["fecha", "valor"]
        out["fecha"] = _to_datetime(out["fecha"])
        return out.dropna(subset=["fecha"])
    return None

def _read_anom_intervals(csv_path: str) -> Optional[pd.DataFrame]:
    """Lee tramos ['start_date','end_date'] si existen en el CSV dado."""
    df = _read_csv_safe(csv_path)
    if df is None or df.empty:
        return None
    cols = {c.lower(): c for c in df.columns}
    has_start = "start_date" in cols or "start" in cols or "inicio" in cols
    has_end = "end_date" in cols or "end" in cols or "fin" in cols
    if not (has_start and has_end):
        return None
    scol = cols.get("start_date") or cols.get("start") or cols.get("inicio")
    ecol = cols.get("end_date") or cols.get("end") or cols.get("fin")
    out = df[[scol, ecol]].copy()
    out.columns = ["start_date", "end_date"]
    out["start_date"] = _to_datetime(out["start_date"], "start_date")
    out["end_date"] = _to_datetime(out["end_date"], "end_date")
    out = out.dropna(subset=["start_date", "end_date"])
    # asegurar start <= end
    mask = out["start_date"] > out["end_date"]
    if mask.any():
        out.loc[mask, ["start_date", "end_date"]] = out.loc[mask, ["end_date", "start_date"]].values
    return out

# ------------------------------------------------------------
# API para Airflow
# ------------------------------------------------------------
#def grafico_serie_basico() -> str:
#    """Grafica la serie completa y guarda en CFG.GRAFICO_SERIE."""
#    df = _load_serie()
#    fig, ax = plt.subplots(figsize=(10, 4))
#    ax.plot(df["fecha"], df["valor"], linewidth=1.5)
#    ax.set_title("Serie temporal")
#    ax.set_xlabel("Fecha")
#    ax.set_ylabel("Valor")
#    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
#    return _safe_savefig(fig, CFG.GRAFICO_SERIE)

def plot_kde() -> str:
    """Histograma + suavizado simple; guarda en CFG.GRAFICO_KDE."""
    df = _load_serie()
    serie = pd.Series(df["valor"].dropna().values, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = min(50, max(10, int(np.sqrt(len(serie))))) if len(serie) else 10
    ax.hist(serie, bins=bins, alpha=0.6, density=True)

    # Suavizado simple del histograma (ventana triangular)
    counts, edges = np.histogram(serie, bins=bins, density=True)
    mids = (edges[:-1] + edges[1:]) / 2.0
    w = 5
    if len(counts) >= w:
        weights = np.arange(1, w + 1, dtype=float)
        weights = np.concatenate([weights, weights[::-1][1:]])
        if len(weights) % 2 == 0:
            weights = np.append(weights, 1.0)
        pad = len(weights) // 2
        padded = np.pad(counts, (pad, pad), mode="edge")
        smooth = np.convolve(padded, weights / weights.sum(), mode="valid")
        ax.plot(mids, smooth[: len(mids)], linewidth=1.5)

    ax.set_title("Distribución (KDE aproximada)")
    ax.set_xlabel("Valor")
    ax.set_ylabel("Densidad")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    return _safe_savefig(fig, CFG.GRAFICO_KDE)
'''
def plot_anomalies_by_method(modelo: str, fuente: str = "original") -> Optional[str]:
    """
    Grafica las anomalías de un método específico como puntos sobre la serie.
    Busca archivos exportados por detection.py en subcarpetas por 'fuente':
      RUTA_ANOMALIAS_METODO / <Original|Sintetica> / f"{MODELO}_puntos*.csv"
    Guarda la figura en la misma subcarpeta.
    """
    # Normalizar fuente y subcarpeta
    fuente_norm = (fuente or "original").strip().lower()
    subdir = "Original" if fuente_norm == "original" else "Sintetica"

    df = _load_serie()

    # Carpeta de entrada/salida según fuente
    base_dir = CFG.RUTA_ANOMALIAS_METODO
    in_dir = os.path.join(base_dir, subdir)
    modelo_up = modelo.upper()

    # Buscar CSVs del modelo en la subcarpeta correspondiente
    patrones = [
        os.path.join(in_dir, f"{modelo_up}_puntos.csv"),
        os.path.join(in_dir, f"{modelo_up}_puntos_*.csv"),
    ]
    archivos: List[str] = []
    for p in patrones:
        archivos.extend(glob.glob(p))
    archivos = sorted(set(archivos))

    # Si no hay CSVs, dibujar solo la serie
    if not archivos:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["fecha"], df["valor"], linewidth=1.5)
        ax.set_title(f"Serie temporal - {modelo_up} ({subdir}, sin anomalías)")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Valor")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        _ensure_dir_is_dir(in_dir)
        return _safe_savefig(fig, os.path.join(in_dir, f"{modelo_up}_serie"))

    # Unir puntos de todos los CSV del modelo
    frames = []
    for fp in archivos:
        dfi = _read_anom_points(fp)
        if dfi is not None and not dfi.empty:
            frames.append(dfi)
    if not frames:
        return None

    anom = (pd.concat(frames, ignore_index=True)
              .drop_duplicates(subset=["fecha", "valor"])
              .sort_values("fecha")
              .reset_index(drop=True))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["fecha"], df["valor"], linewidth=1.5)
    ax.scatter(anom["fecha"], anom["valor"], s=28, zorder=3, color="red", edgecolor="black")
    ax.set_title(f"Anomalías - {modelo_up} ({subdir})")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    _ensure_dir_is_dir(in_dir)
    return _safe_savefig(fig, os.path.join(in_dir, f"{modelo_up}_anomalias"))
'''
def plot_anomalies_by_method(modelo: str, fuente: str = "original", json_df: Optional[str] = None) -> Optional[str]:
    """
    Grafica las anomalías de un método específico como puntos sobre la serie.

    - Si 'json_df' viene, se usa ESA serie para el plot (p.ej. la sintética de 60 días).
      Si no viene, se cae al helper _load_serie() (serie completa).
    - Busca CSVs exportados por detection en varias rutas posibles:
        * RUTA_ANOMALIAS_METODO/<Original|Sintetica>/
        * RUTA_ANOMALIAS_METODO/**/  (búsqueda recursiva por si cambió la estructura)
    - Guarda la figura en la carpeta donde encontró los CSVs (si existen),
      o en RUTA_ANOMALIAS_METODO/<Original|Sintetica>/ como fallback.
    """
    import io, os, glob
    import pandas as pd
    import matplotlib.pyplot as plt

    # ── Normalizar fuente y derivar subcarpeta
    fuente_norm = (fuente or "original").strip().lower()
    subdir = "Original" if fuente_norm == "original" else "Sintetica"

    # ── Cargar la serie a graficar
    if json_df is not None:
        try:
            df = pd.read_json(io.StringIO(json_df))
        except Exception:
            df = pd.DataFrame(columns=["fecha", "valor"])

        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()]
            df["fecha"] = df.index

        # Priorizar valor crudo si existe
        if "valor" not in df.columns:
            if "valor_raw" in df.columns:
                df["valor"] = df["valor_raw"]
            elif "pred" in df.columns:
                df["valor"] = df["pred"]
        if "valor_raw" in df.columns:
            df["valor"] = df["valor_raw"]

        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        df = df.dropna(subset=["fecha", "valor"]).sort_values("fecha").reset_index(drop=True)
    else:
        # Fallback a la serie "global" (histórica completa)
        df = _load_serie()  # helper existente
        # Normalización mínima por si acaso
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()]
            df["fecha"] = df.index
        if "valor_raw" in df.columns:
            df["valor"] = df["valor_raw"]
        df["valor"] = pd.to_numeric(df.get("valor"), errors="coerce")
        df = df.dropna(subset=["fecha", "valor"]).sort_values("fecha").reset_index(drop=True)

    # ── Preparar búsqueda de CSVs de anomalías
    base_dir = CFG.RUTA_ANOMALIAS_METODO
    modelo_up = str(modelo).upper()

    patrones = [
        os.path.join(base_dir, subdir, f"{modelo_up}_puntos.csv"),
        os.path.join(base_dir, subdir, f"{modelo_up}_puntos_*.csv"),
        os.path.join(base_dir, "**", f"{modelo_up}_puntos.csv"),       # búsqueda recursiva
        os.path.join(base_dir, "**", f"{modelo_up}_puntos_*.csv"),
    ]

    archivos: List[str] = []
    for p in patrones:
        archivos.extend(glob.glob(p, recursive=True))
    archivos = sorted(set(archivos))  # únicos + ordenados

    # ── Determinar directorio de salida
    if archivos:
        out_dir = os.path.dirname(archivos[0])  # junto al primer CSV encontrado
    else:
        out_dir = os.path.join(base_dir, subdir)  # fallback estandarizado
    _ensure_dir_is_dir(out_dir)

    # ── Si no hay CSVs, graficar solo la serie y salir
    if not archivos:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["fecha"], df["valor"], linewidth=1.5)
        ax.set_title(f"Serie temporal - {modelo_up} ({subdir}, sin anomalías)")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Valor")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        return _safe_savefig(fig, os.path.join(out_dir, f"{modelo_up}_serie"))

    # ── Leer y unir puntos de anomalías de todos los CSVs del modelo
    frames = []
    for fp in archivos:
        dfi = _read_anom_points(fp)  # helper que devuelve columnas ['fecha','valor']
        if dfi is not None and not dfi.empty:
            frames.append(dfi)

    # Si no se pudieron leer puntos, graficar solo la serie
    if not frames:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["fecha"], df["valor"], linewidth=1.5)
        ax.set_title(f"Serie temporal - {modelo_up} ({subdir}, sin anomalías)")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Valor")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        return _safe_savefig(fig, os.path.join(out_dir, f"{modelo_up}_serie"))

    anom = (
        pd.concat(frames, ignore_index=True)
          .drop_duplicates(subset=["fecha", "valor"])
          .sort_values("fecha")
          .reset_index(drop=True)
    )

    # ── Plot final: serie + puntos anómalos alineados a la escala de la serie
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["fecha"], df["valor"], linewidth=1.5)

    if not anom.empty:
        # Alinear Y con la serie graficada (evita mezclar crudo vs normalizado)
        anom_merged = anom.merge(
            df[["fecha", "valor"]],
            on="fecha",
            how="inner",
            suffixes=("", "_serie")
        )

        if anom_merged.empty:
            # Si no hay intersección de fechas, caer al valor del CSV (mismo eje)
            x_points = anom["fecha"]
            y_points = anom["valor"]
        else:
            x_points = anom_merged["fecha"]
            y_points = anom_merged["valor"]  # valor de la SERIE graficada

        ax.scatter(x_points, y_points, s=28, zorder=3, color="red", edgecolor="black")

    ax.set_title(f"Anomalías - {modelo_up} ({subdir})")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    return _safe_savefig(fig, os.path.join(out_dir, f"{modelo_up}_anomalias"))


def plot_consolidated() -> str:
    """
    Lee CFG.ANOMALIAS_CONSOLIDADO (start_date, end_date, count, metodos),
    dibuja la serie de fondo, bandas por tramo (solo si duran >1 día)
    y puntos diarios de consenso.
    Guarda en: CFG.RUTA_RESULTADOS / "consolidado"
    """
    in_csv = CFG.ANOMALIAS_CONSOLIDADO
    out_path = os.path.join(CFG.RUTA_RESULTADOS, "consolidado")

    df = _load_serie()
    cons = _read_csv_safe(in_csv)

    # Si no hay consolidado, solo serie
    if cons is None or cons.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["fecha"], df["valor"], linewidth=1.5)
        ax.set_title("Anomalías consolidadas (sin datos)")
        ax.set_xlabel("Fecha"); ax.set_ylabel("Valor")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        return _safe_savefig(fig, out_path)

    seg = _read_anom_intervals(in_csv)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["fecha"], df["valor"], linewidth=1.2, label="Serie")

    if seg is not None and not seg.empty:
        seg = seg.dropna(subset=["start_date", "end_date"]).sort_values(["start_date", "end_date"])

        # Bandas por tramo (solo si end > start en al menos 1 día)
        for _, row in seg.iterrows():
            if (pd.to_datetime(row["end_date"]) - pd.to_datetime(row["start_date"])) >= pd.Timedelta(days=1):
                ax.axvspan(row["start_date"], row["end_date"], alpha=0.18)

        # Puntos diarios dentro de cada tramo (incluye tramos de 1 día)
        pts_list = [
            pd.DataFrame({"fecha": pd.date_range(r.start_date.normalize(),
                                                 r.end_date.normalize(), freq="D")})
            for r in seg.itertuples(index=False)
        ]
        if pts_list:
            pts = (pd.concat(pts_list, ignore_index=True)
                     .drop_duplicates()
                     .sort_values("fecha"))
            base = df[["fecha", "valor"]].sort_values("fecha")
            pts_vals = pd.merge_asof(
                pts, base, on="fecha", direction="nearest", tolerance=pd.Timedelta("1D")
            ).dropna()
            if not pts_vals.empty:
                ax.scatter(
                    pts_vals["fecha"], pts_vals["valor"],
                    s=26, zorder=3, color="red", edgecolor="black",
                    label="Consenso (puntos)"
                )

    ax.set_title("Anomalías consolidadas (tramos por consenso)")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Valor")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="best")
    return _safe_savefig(fig, out_path)

'''
def save_plot_anomalies(
    data,
    anomalies,
    date_column,
    target_column,
    days=15,
    mark_points=True,
    out_dir=None,
):
    """
    Genera un gráfico por anomalía recortando una ventana ±days alrededor del tramo
    [start_date, end_date] y opcionalmente marca el punto más cercano a start_date.
    Guarda como NNN_YYYY-MM-DD.png en out_dir y devuelve 'anomalies' con 'filename'.

    Requisitos en 'anomalies': columnas start_date, end_date; 'id' opcional.
    """
    import os
    from datetime import timedelta
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from componentes import settings as CFG

    # --- Asegurar tipos datetime y ordenar serie ---
    df = data.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column]).sort_values(date_column)
    if df.empty:
        return anomalies

    # --- Asegurar tipos y ordenar anomalías ---
    anom = anomalies.copy()
    anom["start_date"] = pd.to_datetime(anom.get("start_date"), errors="coerce")
    anom["end_date"]   = pd.to_datetime(anom.get("end_date"),   errors="coerce")
    anom = anom.dropna(subset=["start_date", "end_date"]).copy()

    # id: si no existe, crear; si viene string, convertir seguro
    if "id" not in anom.columns:
        anom["id"] = range(1, len(anom) + 1)
    else:
        def _to_int_safe(x):
            try:
                return int(x)
            except Exception:
                return None
        anom["id"] = anom["id"].apply(_to_int_safe)
        # re-asigna ids faltantes
        if anom["id"].isna().any():
            missing = anom["id"].isna()
            start_id = (anom["id"].dropna().max() or 0) + 1
            anom.loc[missing, "id"] = range(start_id, start_id + missing.sum())
        anom["id"] = anom["id"].astype(int)

    anom = anom.sort_values("id")

    if anom.empty:
        return anom

    # --- Directorio de salida ---
    out_dir = out_dir or getattr(CFG, "RUTA_GRAFICOS_CONSOLIDADOS",
                                 "/tmp/DatosSerie/Resultados/GraficosConsolidado/GraficosConDatos")
    os.makedirs(out_dir, exist_ok=True)

    serie_min, serie_max = df[date_column].min(), df[date_column].max()
    dpi = int(getattr(CFG, "GRAFICOS_DPI", 150))

    filenames = []
    for _, row in anom.iterrows():
        start_original = pd.to_datetime(row["start_date"])
        end_original   = pd.to_datetime(row["end_date"])

        # Ventana ±days alrededor del tramo anómalo
        win_start = max(start_original - timedelta(days=int(days)), serie_min)
        win_end   = min(end_original   + timedelta(days=int(days)), serie_max)

        segment = df[(df[date_column] >= win_start) & (df[date_column] <= win_end)]
        if segment.empty:
            filenames.append(None)
            continue

        # --- Figura ---
        fig, ax = plt.subplots(figsize=(6, 3), dpi=dpi)
        ax.plot(segment[date_column], segment[target_column], linewidth=1.0, label="Serie")

        # Sombrear el tramo anómalo completo
        ax.axvspan(start_original, end_original, alpha=0.15, color="tab:red", label="Tramo anómalo")

        # Punto más cercano a start_date (si se solicita)
        if mark_points:
            # índice del punto de tiempo más cercano a start_original dentro del segmento
            tseg = segment[date_column].values.astype("datetime64[ns]").astype("int64")
            t0 = np.int64(pd.to_datetime(start_original).value)
            idx = int(np.argmin(np.abs(tseg - t0)))
            px = segment[date_column].iloc[idx]
            py = segment[target_column].iloc[idx]
            ax.scatter(px, py, color="red", edgecolor="black", s=40, zorder=3, label="Punto cercano")

        # Estética
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_xlim(pd.to_datetime(win_start), pd.to_datetime(win_end))
        # Limitar ticks del eje X a [inicio, fin] para un look limpio
        ax.set_xticks([pd.to_datetime(win_start), pd.to_datetime(win_end)])
        ax.set_xticklabels([pd.to_datetime(win_start).strftime("%Y-%m-%d"),
                            pd.to_datetime(win_end).strftime("%Y-%m-%d")], fontsize=8)
        ax.tick_params(axis="y", labelsize=8)

        # Leyenda solo si se marcaron puntos
        if mark_points:
            ax.legend(loc="best", fontsize=8)

        fig.tight_layout()

        # Nombre de archivo
        anomaly_id = int(row["id"])
        fname = f"{anomaly_id:03d}_{start_original.date()}.png"
        fpath = os.path.join(out_dir, fname)

        fig.savefig(fpath, bbox_inches="tight")
        plt.close(fig)

        filenames.append(fname)

    anom["filename"] = filenames
    return anom
'''
def save_plot_anomalies(
    data,
    anomalies,
    date_column,
    target_column,
    days=15,
    mark_points=True,
    out_dir=None,
):
    """
    Igual que antes; esta versión solo agrega PRINTS para trazabilidad.
    """
    import os
    from datetime import timedelta
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from componentes import settings as CFG
    from pathlib import Path

    print("\n==================== SAVE_PLOT_ANOMALIES ====================")
    print(f"[INFO] Recibidos: data={len(data)} filas, anomalies={len(anomalies)} filas")
    print(f"[INFO] date_column='{date_column}', target_column='{target_column}'")

    # --- Asegurar tipos datetime y ordenar serie ---
    df = data.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column]).sort_values(date_column)

    print(f"[INFO] Serie ordenada: {len(df)} filas después de limpieza")
    if not df.empty:
        print(f"[INFO] Rango de serie: {df[date_column].min()}  →  {df[date_column].max()}")

    if df.empty:
        print("[WARN] Serie vacía, no se generará ninguna ventana")
        return anomalies

    # --- Asegurar tipos y ordenar anomalías ---
    anom = anomalies.copy()
    anom["start_date"] = pd.to_datetime(anom.get("start_date"), errors="coerce")
    anom["end_date"]   = pd.to_datetime(anom.get("end_date"),   errors="coerce")
    anom = anom.dropna(subset=["start_date", "end_date"]).copy()

    print(f"[INFO] Anomalías válidas tras limpieza: {len(anom)}")
    if anom.empty:
        print("[WARN] No hay anomalías para graficar")
        return anom

    # id: si no existe, crear
    if "id" not in anom.columns:
        print("[INFO] No existe columna 'id'; generando ids secuenciales")
        anom["id"] = range(1, len(anom) + 1)
    else:
        print("[INFO] Columna 'id' detectada, asegurando consistencia")

        def _to_int_safe(x):
            try:
                return int(x)
            except:
                return None

        anom["id"] = anom["id"].apply(_to_int_safe)
        if anom["id"].isna().any():
            missing = anom["id"].isna()
            start_id = (anom["id"].dropna().max() or 0) + 1
            print(f"[INFO] Reasignando {missing.sum()} IDs faltantes desde {start_id}")
            anom.loc[missing, "id"] = range(start_id, start_id + missing.sum())
        anom["id"] = anom["id"].astype(int)

    anom = anom.sort_values("id")

    # --- Directorio de salida ---
    out_dir = out_dir or getattr(
        CFG, "RUTA_GRAFICOS_CONSOLIDADOS",
        "/tmp/DatosSerie/Resultados/GraficosConsolidado/GraficosConDatos"
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Carpeta de salida PNG: {out_dir}")

    # Carpeta para CSV opcionales
    csv_dir = getattr(CFG, "RUTA_ANOMALIAS_LIMPIAS_CSV", None)
    if csv_dir:
        csv_dir = Path(csv_dir)
        csv_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Carpeta CSV habilitada: {csv_dir}")

    serie_min, serie_max = df[date_column].min(), df[date_column].max()
    dpi = int(getattr(CFG, "GRAFICOS_DPI", 150))

    filenames = []
    for _, row in anom.iterrows():
        anomaly_id = int(row["id"])
        start_original = pd.to_datetime(row["start_date"])
        end_original   = pd.to_datetime(row["end_date"])

        print("\n------------------------------------------------------------")
        print(f"[ANOM] ID={anomaly_id}")
        print(f"[ANOM] Fechas originales: {start_original} → {end_original}")

        # Ventana ±days
        win_start = max(start_original - timedelta(days=int(days)), serie_min)
        win_end   = min(end_original   + timedelta(days=int(days)), serie_max)
        print(f"[ANOM] Ventana extendida: {win_start} → {win_end}")

        segment = df[(df[date_column] >= win_start) & (df[date_column] <= win_end)]
        print(f"[ANOM] Segmento: {len(segment)} filas")

        if segment.empty:
            print(f"[WARN] Segmento vacío; NO se generará PNG para ID={anomaly_id}")
            filenames.append(None)
            continue

        # Exportar CSV ventana
        if csv_dir is not None:
            seg = segment[[date_column, target_column]].copy().sort_values(date_column)
            seg["id"] = 1
            seg[date_column] = pd.to_datetime(seg[date_column], errors="coerce").dt.strftime("%Y-%m-%d")
            seg = seg[["id", date_column, target_column]]

            csv_path = csv_dir / f"{anomaly_id:03d}__ORIGINAL.csv"
            seg.to_csv(csv_path, index=False)
            print(f"[CSV] Guardado: {csv_path} ({len(seg)} filas)")

        # Figura
        fig, ax = plt.subplots(figsize=(6, 3), dpi=dpi)
        ax.plot(segment[date_column], segment[target_column], linewidth=1.0, label="Serie")
        ax.axvspan(start_original, end_original, alpha=0.15, color="tab:red", label="Tramo anómalo")

        if mark_points:
            tseg = segment[date_column].values.astype("datetime64[ns]").astype("int64")
            t0 = np.int64(pd.to_datetime(start_original).value)
            idx = int(np.argmin(np.abs(tseg - t0)))
            px = segment[date_column].iloc[idx]
            py = segment[target_column].iloc[idx]
            ax.scatter(px, py, color="red", edgecolor="black", s=40, zorder=3, label="Punto cercano")

        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_xlim(pd.to_datetime(win_start), pd.to_datetime(win_end))
        ax.set_xticks([pd.to_datetime(win_start), pd.to_datetime(win_end)])
        ax.set_xticklabels([win_start.strftime("%Y-%m-%d"), win_end.strftime("%Y-%m-%d")], fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        if mark_points:
            ax.legend(loc="best", fontsize=8)

        fig.tight_layout()

        fname = f"{anomaly_id:03d}_{start_original.date()}.png"
        fpath = os.path.join(out_dir, fname)

        fig.savefig(fpath, bbox_inches="tight")
        plt.close(fig)

        print(f"[PNG] Guardado: {fpath}")

        filenames.append(fname)

    print(f"[INFO] Total PNG generados: {sum(1 for x in filenames if x)}")
    print("==================== END SAVE_PLOT_ANOMALIES ====================\n")

    anom["filename"] = filenames
    return anom


def plot_clean_anomaliesV1(data, anomalies, date_column, target_column, days=15):
    """
    Genera recortes de la serie alrededor de cada anomalía y guarda imágenes
    con línea azul y fondo blanco (sin sombreado ni elementos extra).
    Devuelve el DataFrame de anomalías con la columna 'filename' añadida.
    """
    import os
    from datetime import timedelta
    import pandas as pd
    import matplotlib.pyplot as plt
    from componentes import settings as CFG

    # --- Asegurar tipos datetime y ordenar ---
    data = data.copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data = data.dropna(subset=[date_column]).sort_values(date_column)

    anomalies = anomalies.copy()
    anomalies["start_date"] = pd.to_datetime(anomalies["start_date"], errors="coerce")
    anomalies["end_date"]   = pd.to_datetime(anomalies["end_date"],   errors="coerce")
    anomalies = anomalies.dropna(subset=["start_date", "end_date"]).sort_values(["id"])

    if data.empty or anomalies.empty:
        return anomalies  # nada que graficar

    # --- Directorio de salida ---
    out_dir = CFG.RUTA_GRAFICOS
    os.makedirs(out_dir, exist_ok=True)

    serie_min, serie_max = data[date_column].min(), data[date_column].max()

    filenames = []
    for _, row in anomalies.iterrows():
        start_original = row["start_date"]
        end_original   = row["end_date"]

        # Ventana de ±days
        start = max(start_original - timedelta(days=days), serie_min)
        end   = min(end_original   + timedelta(days=days), serie_max)

        # Recorte de la serie
        segment = data[(data[date_column] >= start) & (data[date_column] <= end)]
        if segment.empty:
            filenames.append(None)
            continue

        # --- Figura con fondo blanco y línea azul ---
        fig, ax = plt.subplots(figsize=(6, 3), dpi=100, facecolor="white")
        ax.set_facecolor("white")
        ax.plot(segment[date_column], segment[target_column],
                linewidth=1.2, color="blue", solid_capstyle="round", antialiased=True)

        # Sin ejes, ticks, grid, títulos
        ax.axis("off")

        # Guardar con fondo blanco
        anomaly_id = row["id"]
        try:
            fname = f"{int(anomaly_id):03d}_{start_original.date()}.png"
        except Exception:
            fname = f"{str(anomaly_id)}_{start_original.date()}.png"

        fpath = os.path.join(out_dir, fname)
        fig.savefig(fpath, bbox_inches="tight", transparent=False)
        plt.close(fig)

        filenames.append(fname)

    anomalies["filename"] = filenames
    return anomalies


def plot_clean_anomalies(data, anomalies, date_column, target_column, days=15):
    """
    Genera recortes de la serie alrededor de cada anomalía y guarda imágenes
    con línea azul y fondo blanco (sin sombreado ni elementos extra).
    Devuelve el DataFrame de anomalías con la columna 'filename' añadida.
    """
    import os
    from datetime import timedelta
    import pandas as pd
    import matplotlib.pyplot as plt
    from componentes import settings as CFG

    # --- Asegurar tipos datetime y ordenar ---
    data = data.copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data = data.dropna(subset=[date_column]).sort_values(date_column)

    anomalies = anomalies.copy()
    anomalies["start_date"] = pd.to_datetime(anomalies["start_date"], errors="coerce")
    anomalies["end_date"]   = pd.to_datetime(anomalies["end_date"],   errors="coerce")
    anomalies = anomalies.dropna(subset=["start_date", "end_date"]).sort_values(["id"])

    if data.empty or anomalies.empty:
        return anomalies  # nada que graficar

    # --- Directorio de salida ---
    out_dir = CFG.RUTA_GRAFICOS
    os.makedirs(out_dir, exist_ok=True)

    # --- Parámetros de salida (para igualar con sintéticas limpias) ---
    # Si existen en CFG, los usa; si no, usa defaults.
    figsize_clean = tuple(getattr(CFG, "PNG_LIMPIO_FIGSIZE", (6, 3.2)))
    dpi_clean     = int(getattr(CFG, "PNG_LIMPIO_DPI", 150))
    lw_clean      = float(getattr(CFG, "PNG_LIMPIO_LINEWIDTH", 1.5))
    pad_inches    = float(getattr(CFG, "PNG_LIMPIO_PAD_INCHES", 0.0))

    serie_min, serie_max = data[date_column].min(), data[date_column].max()

    filenames = []
    for _, row in anomalies.iterrows():
        start_original = row["start_date"]
        end_original   = row["end_date"]

        # Ventana de ±days
        start = max(start_original - timedelta(days=days), serie_min)
        end   = min(end_original   + timedelta(days=days), serie_max)

        # Recorte de la serie
        segment = data[(data[date_column] >= start) & (data[date_column] <= end)]
        if segment.empty:
            filenames.append(None)
            continue

        # --- Figura con fondo blanco y línea azul ---
        fig, ax = plt.subplots(figsize=figsize_clean, dpi=dpi_clean, facecolor="white")
        ax.set_facecolor("white")
        ax.plot(
            segment[date_column],
            segment[target_column],
            linewidth=lw_clean,
            color="blue",
            solid_capstyle="round",
            antialiased=True,
        )

        # Sin ejes, ticks, grid, títulos
        ax.axis("off")

        # Fijar el área del plot para que el "zoom" sea consistente (sin recortes variables)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # Guardar con fondo blanco (sin bbox_inches="tight" para no cambiar escala)
        anomaly_id = row["id"]
        try:
            fname = f"{int(anomaly_id):03d}_{start_original.date()}.png"
        except Exception:
            fname = f"{str(anomaly_id)}_{start_original.date()}.png"

        fpath = os.path.join(out_dir, fname)
        fig.savefig(fpath, transparent=False, bbox_inches=None, pad_inches=pad_inches)
        plt.close(fig)

        filenames.append(fname)

    anomalies["filename"] = filenames
    return anomalies


# --- Helper: encontrar anomalías para una ventana sintética por detector ---
def _find_anom_points_for_window(det_code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Busca recursivamente en RUTA_ANOMALIAS_METODO/Sintetica/** los CSV
    del detector (DET_puntos*.csv), concatena y filtra por rango [start, end].
    Devuelve DataFrame con columnas ['fecha','valor'] (únicos y ordenados).
    """
    import glob, os
    det_up = str(det_code).upper()
    base_dir = os.path.join(CFG.RUTA_ANOMALIAS_METODO, "Sintetica")

    print(f"[DEBUG] Buscando anomalías para detector={det_up}")
    print(f"[DEBUG] Carpeta base: {base_dir}")
    print(f"[DEBUG] Rango buscado: {start} → {end}")

    patrones = [
        os.path.join(base_dir, f"{det_up}_puntos.csv"),
        os.path.join(base_dir, f"{det_up}_puntos_*.csv"),
        os.path.join(base_dir, "**", f"{det_up}_puntos.csv"),
        os.path.join(base_dir, "**", f"{det_up}_puntos_*.csv"),
    ]
    files = []
    for p in patrones:
        encontrados = glob.glob(p, recursive=True)
        if encontrados:
            print(f"[DEBUG] Patrón {p} → encontrados: {len(encontrados)}")
            for f in encontrados:
                print(f"         - {f}")
        files.extend(encontrados)

    files = sorted(set(files))
    if not files:
        print("[DEBUG] No se encontró ningún archivo de anomalías.")
        return pd.DataFrame(columns=["fecha", "valor"])

    frames = []
    for fp in files:
        print(f"[DEBUG] Leyendo archivo: {fp}")
        dfi = _read_anom_points(fp)
        if dfi is None or dfi.empty:
            print("   → vacío o no válido")
            continue

        # recorte por ventana
        mask = (dfi["fecha"] >= start) & (dfi["fecha"] <= end)
        dfi = dfi.loc[mask]
        if dfi.empty:
            print("   → no hay puntos dentro del rango")
            continue

        print(f"   → puntos encontrados en rango: {len(dfi)}")
        print(dfi[["fecha", "valor"]].head())
        frames.append(dfi[["fecha", "valor"]])

    if not frames:
        print("[DEBUG] No hay anomalías en el rango dado.")
        return pd.DataFrame(columns=["fecha", "valor"])

    out = (pd.concat(frames, ignore_index=True)
             .drop_duplicates(subset=["fecha", "valor"])
             .sort_values("fecha")
             .reset_index(drop=True))

    print(f"[DEBUG] Total anomalías únicas devueltas: {len(out)}")
    return out



# --- Batch sintético: guarda PNG + CSV de ventanas usando VENTANA_CTX_DIAS ---


def plot_anomaly_padding():
    from componentes import settings as CFG
    import pandas as pd
    import os

    print("\n==================== PLOT_ANOMALY_PADDING ====================")

    # Cargar serie original
    print("[PADDING] Cargando serie original con _load_serie() ...")
    df = _load_serie()
    print(f"[PADDING] Serie cargada: {len(df)} filas")

    if df.empty:
        print("[WARN] Serie VACÍA – No se puede generar padding de anomalías")
        print("==================== END PADDING ====================\n")
        return None

    # Cargar anomalías consolidadas
    anom_path = CFG.ANOMALIAS_CONSOLIDADO
    print(f"[PADDING] Cargando anomalías desde: {anom_path}")

    if not os.path.exists(anom_path):
        print(f"[ERROR] No existe ANOMALIAS_CONSOLIDADO: {anom_path}")
        print("==================== END PADDING ====================\n")
        return None

    anom = pd.read_csv(anom_path)
    print(f"[PADDING] Anomalías cargadas: {len(anom)} filas")

    if anom.empty:
        print("[WARN] El archivo de anomalías consolidadas está vacío")
        print("==================== END PADDING ====================\n")
        return None

    # Mostrar primeras filas para depuración
    print("[PADDING] Primeras filas de anomalías:")
    print(anom.head())

    # Directorios de salida
    print(f"[PADDING] out_dir (original): {CFG.RUTA_GRAFICOS_CONSOLIDADOS}")

    # ==================== Gráfico INFORMAl ====================
    print("[PADDING] Llamando a save_plot_anomalies() ...")

    anom = save_plot_anomalies(
        data=df,
        anomalies=anom,
        date_column="fecha",
        target_column="valor",
        days=CFG.VENTANA_CTX_DIAS,
        mark_points=False,
        out_dir=CFG.RUTA_GRAFICOS_CONSOLIDADOS
    )

    print("[PADDING] save_plot_anomalies() completado")

    # ==================== Gráfico LIMPIO ====================
    print("[PADDING] Llamando a plot_clean_anomalies() ...")

    plot_clean_anomalies(
        data=df,
        anomalies=anom,
        date_column="fecha",
        target_column="valor",
        days=CFG.VENTANA_CTX_DIAS
    )

    print("[PADDING] plot_clean_anomalies() completado")
    print("[PADDING] Proceso de padding finalizado correctamente")
    print("==================== END PADDING ====================\n")

    return None


# --- Helpers de nombre/enum ---

def _next_seq_in_dir(out_dir: str) -> int:
    """
    Busca archivos existentes con prefijo numérico en la forma:
      <seq>_<PRED>_<DET>_<YYYYMMDD>.(png|csv)
    y devuelve el siguiente secuencial.
    """
    import os, re
    os.makedirs(out_dir, exist_ok=True)
    rx = re.compile(r"^(\d+)_([A-Z0-9]+)_([A-Z0-9]+)_\d{8}(?:\.[A-Za-z0-9]+)?$")
    maxseq = 0
    for nombre in os.listdir(out_dir):
        if rx.match(nombre):
            try:
                s = int(nombre.split("_", 1)[0])
                if s > maxseq:
                    maxseq = s
            except Exception:
                pass
    return maxseq + 1

def plot_synthetic_one(
    json_df,
    pred_code: str,
    det_code: str,
    out_dir: str = None,
    seq: int = None,
) -> dict:
    """
    Genera DOS archivos en 'out_dir' para una serie sintética y un detector:
      - <N>_<PRED>_<DET>_<YYYYMMDD>.png  (serie completa con puntos)
      - <N>_<PRED>_<DET>_<YYYYMMDD>.csv  (todas las ventanas ±VENTANA_CTX_DIAS)

    Donde:
      - N: si no se pasa, se calcula automáticamente escaneando 'out_dir'.
      - YYYYMMDD: fecha mínima de la serie sintética (inicio del tramo graficado).
      - PRED: método de predicción (ej. ARIMA, SVR_DIRECT).
      - DET : método de detección (ej. DIF, ARIMA, IFOREST).
    """
    import io, os, glob
    import pandas as pd
    import matplotlib.pyplot as plt
    from componentes import settings as CFG

    # ---- Parsear serie sintética desde JSON (con 'fecha' y 'valor'/'pred'/'valor_raw')
    try:
        df = pd.read_json(io.StringIO(json_df))
    except Exception:
        df = pd.DataFrame(columns=["fecha", "valor"])

    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        df["fecha"] = df.index

    if "valor" not in df.columns:
        if "valor_raw" in df.columns:
            df["valor"] = df["valor_raw"]
        elif "pred" in df.columns:
            df["valor"] = df["pred"]
    if "valor_raw" in df.columns:
        df["valor"] = df["valor_raw"]

    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.dropna(subset=["fecha", "valor"]).sort_values("fecha").reset_index(drop=True)
    if df.empty:
        return {"png": None, "csv": None, "stem": None}

    win_start, win_end = df["fecha"].min(), df["fecha"].max()
    date_tag = win_start.strftime("%Y%m%d")

    pred = str(pred_code).upper()
    det  = str(det_code).upper()

    # ---- Carpeta de salida
    out_root = out_dir or getattr(CFG, "RUTA_PREDICCION", "/tmp/DatosSerie/Resultados/Prediccion")
    _ensure_dir_is_dir(out_root)

    # ---- Secuencial y stem final (N_PRED_DET_YYYYMMDD)
    seq_val = int(seq) if seq is not None else _next_seq_in_dir(out_root)
    stem = f"{seq_val}_{pred}_{det}_{date_tag}"

    # ---- Recolectar puntos del detector que caen dentro de la ventana de la serie
    base_det = getattr(CFG, "RUTA_ANOMALIAS_METODO", "/tmp/DatosSerie/Resultados/DetectadoXMetodo")
    patrones = [
        os.path.join(base_det, f"{pred}_detect_{det}.csv"),     # convención nueva en mismo directorio del PRED
        os.path.join(base_det, f"{det}_puntos.csv"),            # compat. antigua
        os.path.join(base_det, f"{det}_puntos_*.csv"),          # compat. antigua con timestamp
        os.path.join(base_det, "**", f"{det}_puntos.csv"),
        os.path.join(base_det, "**", f"{det}_puntos_*.csv"),
    ]
    files = []
    for p in patrones:
        files.extend(glob.glob(p, recursive=True))
    files = sorted(set(files))

    frames = []
    for fp in files:
        dfi = _read_anom_points(fp)
        if dfi is None or dfi.empty:
            continue
        mask = (dfi["fecha"] >= win_start) & (dfi["fecha"] <= win_end)
        dfi = dfi.loc[mask, ["fecha", "valor"]]
        if not dfi.empty:
            frames.append(dfi)
    anom = pd.concat(frames, ignore_index=True).drop_duplicates().sort_values("fecha").reset_index(drop=True) if frames else pd.DataFrame(columns=["fecha","valor"])

    # ---- Plot (serie completa + puntos)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["fecha"], df["valor"], linewidth=1.5)

    if not anom.empty:
        anom_m = anom.merge(df[["fecha","valor"]], on="fecha", how="inner")
        if not anom_m.empty:
            ax.scatter(anom_m["fecha"], anom_m["valor"], s=28, zorder=3, edgecolor="black")

    ax.set_title(f"Anomalías - {pred} vs {det} (Sintética)")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Valor")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    png_path = _safe_savefig(fig, os.path.join(out_root, stem))

    # ---- CSV de ventanas ±VENTANA_CTX_DIAS por cada anomalía
    ventana = int(getattr(CFG, "VENTANA_CTX_DIAS", 15))
    rows = []
    if not anom.empty:
        anom_m = anom.merge(df[["fecha","valor"]], on="fecha", how="left").dropna(subset=["fecha"])
        for i, r in anom_m.reset_index(drop=True).iterrows():
            f_anom = r["fecha"]
            ini = max(f_anom - pd.Timedelta(days=ventana), win_start)
            fin = min(f_anom + pd.Timedelta(days=ventana), win_end)
            seg = df[(df["fecha"] >= ini) & (df["fecha"] <= fin)].copy()
            if seg.empty:
                continue
            seg.insert(0, "anom_id", i + 1)
            seg.insert(1, "fecha_anom", f_anom)
            seg.insert(2, "ventana_ini", ini)
            seg.insert(3, "ventana_fin", fin)
            seg["pred"] = pred
            seg["det"]  = det
            seg["es_anomalo"] = (seg["fecha"] == f_anom).astype(int)
            rows.append(seg)

    csv_path = os.path.join(out_root, f"{stem}.csv")
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(csv_path, index=False)
    else:
        pd.DataFrame(columns=["anom_id","fecha_anom","ventana_ini","ventana_fin","fecha","valor","pred","det","es_anomalo"]).to_csv(csv_path, index=False)

    return {"png": png_path, "csv": csv_path, "stem": stem}

def plot_synthetic_batch(items: list, out_dir: str) -> list:
    """
    Genera (por cada item) un PNG y un CSV en 'out_dir' con nombre:
      <N>_<PRED>_<DET>_<YYYYMMDD>.(png|csv)

    items = [{
        "seq": int | None,             # opcional; si None, se autocalcula
        "start_date": "YYYY-MM-DD",    # si no se pasa, se toma de la serie
        "pred_code": str,              # p.ej. "ARIMA", "SVR_DIRECT"
        "det_code": str,               # p.ej. "DIF", "ARIMA", "IFOREST"
        "json_df": str                 # serie sintética en JSON
    }, ...]
    """
    import io, os
    import pandas as pd
    import matplotlib.pyplot as plt
    from componentes import settings as CFG

    _ensure_dir_is_dir(out_dir)
    ventana = int(getattr(CFG, "VENTANA_CTX_DIAS", 15))
    outputs = []

    for it in items:
        pred = str(it["pred_code"]).upper()
        det  = str(it["det_code"]).upper()
        seq  = it.get("seq")

        # --- Parsear serie
        try:
            df = pd.read_json(io.StringIO(it["json_df"]))
        except Exception:
            df = pd.DataFrame(columns=["fecha","valor"])

        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()]
            df["fecha"] = df.index

        if "valor" not in df.columns:
            if "valor_raw" in df.columns:
                df["valor"] = df["valor_raw"]
            elif "pred" in df.columns:
                df["valor"] = df["pred"]
        if "valor_raw" in df.columns:
            df["valor"] = df["valor_raw"]

        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        df = df.dropna(subset=["fecha","valor"]).sort_values("fecha").reset_index(drop=True)
        if df.empty:
            continue

        win_start, win_end = df["fecha"].min(), df["fecha"].max()
        date_tag = (pd.to_datetime(it.get("start_date")) if it.get("start_date") else win_start).strftime("%Y%m%d")

        # --- Enumeración y nombres
        seq_val = int(seq) if seq is not None else _next_seq_in_dir(out_dir)
        stem = f"{seq_val}_{pred}_{det}_{date_tag}"
        png_path = os.path.join(out_dir, stem)
        csv_path = os.path.join(out_dir, stem + ".csv")

        # --- Cargar anomalías de ese DET dentro de la ventana
        anom = _find_anom_points_for_window(det, win_start, win_end)

        # --- Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["fecha"], df["valor"], linewidth=1.5)
        if not anom.empty:
            anom_m = anom.merge(df[["fecha","valor"]], on="fecha", how="inner")
            if not anom_m.empty:
                ax.scatter(anom_m["fecha"], anom_m["valor"], s=28, zorder=3, edgecolor="black")
        ax.set_title(f"Anomalías - {pred} vs {det} (Sintética)")
        ax.set_xlabel("Fecha"); ax.set_ylabel("Valor")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        _safe_savefig(fig, png_path)

        # --- CSV de ventanas ±VENTANA_CTX_DIAS
        rows = []
        if not anom.empty:
            anom_m = anom.merge(df[["fecha","valor"]], on="fecha", how="left").dropna(subset=["fecha"])
            for i, r in anom_m.reset_index(drop=True).iterrows():
                f_anom = r["fecha"]
                ini = max(f_anom - pd.Timedelta(days=ventana), win_start)
                fin = min(f_anom + pd.Timedelta(days=ventana), win_end)
                seg = df[(df["fecha"] >= ini) & (df["fecha"] <= fin)].copy()
                if seg.empty:
                    continue
                seg.insert(0, "anom_id", i + 1)
                seg.insert(1, "fecha_anom", f_anom)
                seg.insert(2, "ventana_ini", ini)
                seg.insert(3, "ventana_fin", fin)
                seg["pred"] = pred
                seg["det"]  = det
                seg["es_anomalo"] = (seg["fecha"] == f_anom).astype(int)
                rows.append(seg)

        if rows:
            pd.concat(rows, ignore_index=True).to_csv(csv_path, index=False)
        else:
            pd.DataFrame(columns=["anom_id","fecha_anom","ventana_ini","ventana_fin","fecha","valor","pred","det","es_anomalo"]).to_csv(csv_path, index=False)

        outputs.append(csv_path)

    return outputs

def plot_synthetic_windows_per_point(pred_code: str, det_code: str, json_df: str = None,
                                     days: int = CFG.VENTANA_CTX_DIAS,
                                     out_dir: str = CFG.RUTA_GRAFICOS_ANOMALIAS_SINTETICAS) -> int:
    """
    Genera un PNG por anomalía (sintética) con ventana ±days y lo guarda en out_dir.
    Retorna el conteo de imágenes creadas.
    """
    import pandas as pd, os
    os.makedirs(out_dir, exist_ok=True)

    # 1) Serie base: usa el JSON combinado (o el CSV que ya generas para sintética)
    if json_df:
        df = pd.read_json(json_df, orient="records")
    else:
        # fallback si no pasas el json_df: intenta cargar desde Prediccion/<PRED>/_pred_sintetica.csv
        csv_path = os.path.join(CFG.RUTA_PREDICCION, pred_code, f"{pred_code}_pred_sintetica.csv")
        df = pd.read_csv(csv_path)

    # Normaliza
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)

    # 2) Cargar puntos del detector en Sintética (fecha, valor)
    #    Reutiliza tu helper existente:
    anom = _find_anom_points_for_window(det_code, df["fecha"].min(), df["fecha"].max())
    if anom is None or anom.empty:
        return 0

    # Adaptar a la interfaz de plot_clean_anomalies: start_date / end_date por PUNTO
    anom = anom.copy()
    anom["start_date"] = pd.to_datetime(anom["fecha"], errors="coerce")
    anom["end_date"]   = anom["start_date"]

    # 3) Llamar al plot unitario por anomalía (usa ±days y guarda en out_dir)
    plot_clean_anomalies(
        data=df,
        anomalies=anom[["start_date","end_date"]].dropna(),
        date_column="fecha",
        target_column="valor",
        days=days,
        out_dir=out_dir
    )
    return len(anom)
