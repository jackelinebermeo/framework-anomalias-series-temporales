# componentes/resultados.py
from __future__ import annotations
import os
import glob
import pandas as pd
import numpy as np
import re

# --------- Config ---------
try:
    from componentes.settings import (
        # existentes (no tocar)
        RUTA_ANOMALIAS_METODO as _IN_DIR,
        RUTA_RESULTADOS as _OUT_DIR,
        CONSENSO_MIN_ALGOS as _MIN_MODELOS,  # p.ej., 3
        ANOMALIAS_CONSOLIDADO as _CSV_IN_DEF,      # path del consolidado base
        SERIE_ID as _SERIE_ID,                     # identificador de la serie
        RUTA_GRAFICOS_CONSOLIDADOS as _RUTA_GRAFICOS,
        SERIE_LIMPIA as _SERIE_LIMPIA,
        VENTANA_CTX_DIAS as _CTX_DIAS,
         SERIE as SERIE,
        FRECUENCIA as FRECUENCIA,
        UNIDAD as UNIDAD,
        DATOS_ONTOLOGIA as DATOS_ONTOLOGIA,
        IRI as IRI
    )
except Exception:
    # existentes (no tocar)
    _IN_DIR = "/tmp/DatosSerie/Resultados/DatosXMetodo"
    _OUT_DIR = "/tmp/DatosSerie/Resultados/"
    _MIN_MODELOS = 3
    _CSV_IN_DEF    = "/tmp/DatosSerie/Resultados/anomalias_consolidado.csv"
    _SERIE_ID      = "SERIE_DEF"
    _RUTA_GRAFICOS = "/tmp/DatosSerie/Resultados/GraficosConsolidados"
    _SERIE_LIMPIA  = "/tmp/DatosSerie/SerieOriginal.csv"
    _CTX_DIAS      = 15
    _SERIE= "Riesgo País"
    _FRECUENCIA="DIARIA"
    _UNIDAD="PUNTOS"
    _DATOS_ONTOLOGIA="/tmp/DatosSerie/Resultados/Ontologia"
    _IRI= "anom:series_riesgo_pais_1"

os.makedirs(_OUT_DIR, exist_ok=True)


# componentes/contexto.py
from pathlib import Path
import pandas as pd

# Cache de módulo (se mantiene mientras viva el proceso de Python)
_serie_name_CACHE: str | None = None

def obtener_nombre_serie_original(ruta_datos_serie: str) -> str:
    """
    Extrae el nombre canónico de la serie original desde DatosSerie.csv (catálogo).
    Soporta separador ';' y columnas en español.

    Prioridad:
      1) Columna 'nombre_es' / 'nombre_serie' / 'serie' / 'serie_name'
      2) Fallback: nombre del archivo sin extensión
    """
    global _serie_name_CACHE
    if _serie_name_CACHE and _serie_name_CACHE.strip():
        return _serie_name_CACHE.strip()

    try:
        # DatosSerie.csv es un catálogo, suele tener 1 fila por serie
        df = pd.read_csv(ruta_datos_serie, sep=";", dtype=str, nrows=10)

        # Normalizar nombres de columnas (por si vienen con espacios)
        df.columns = [c.strip() for c in df.columns]

        for col in ["nombre_es", "nombre_serie", "serie", "serie_name"]:
            if col in df.columns:
                s = df[col].dropna()
                if not s.empty:
                    val = str(s.iloc[0]).strip()
                    if val:
                        _serie_name_CACHE = val
                        return val
    except Exception:
        pass

    # Fallback: nombre del archivo
    _serie_name_CACHE = os.path.splitext(os.path.basename(ruta_datos_serie))[0]
    return _serie_name_CACHE


# --------- Lectura de puntos (simple) ---------
def _load_series_points() -> pd.DataFrame:
    """
    Lee CSV de puntos: *_puntos.csv y *_puntos_*.csv en _IN_DIR.
    Espera columnas: fecha, valor, modelo.
    """
    ruta_metodos = os.path.join(_IN_DIR, "Original")
   # print("[DEBUG] Usando _IN_DIR:", ruta_metodos)
    if not os.path.isdir(ruta_metodos):
        return pd.DataFrame(columns=["fecha", "valor", "modelo","fuente"])

    patrones = [
        os.path.join(ruta_metodos, "*_puntos.csv"),
        os.path.join(ruta_metodos, "*_puntos_*.csv"),
    ]
    archivos = []
    for p in patrones:
        archivos.extend(glob.glob(p))
    #print("[DEBUG] Archivos encontrados:", archivos)
    if not archivos:
        return pd.DataFrame(columns=["fecha", "valor", "modelo","fuente"])

    frames = []
    for fp in sorted(set(archivos)):
        try:
            df = pd.read_csv(fp)
            #print(f"[DEBUG] Leyendo archivo: {fp}")
          #  print(df.head(5).to_string())
            if {"fecha", "valor", "modelo"}.issubset(df.columns):
                df["fecha"] = pd.to_datetime(df
                ["fecha"], errors="coerce").dt.tz_localize(None)
                df = df.dropna(subset=["fecha"])
                if not df.empty:
                    frames.append(df[["fecha", "valor", "modelo","fuente"]])
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=["fecha", "valor", "modelo","fuente"])

    puntos = pd.concat(frames, ignore_index=True)
   # print("[DEBUG] Total de filas concatenadas:", len(puntos))
    #print(puntos.head(10).to_string())
    puntos = (puntos
              .drop_duplicates(subset=["fecha", "valor", "modelo","fuente"])
              .sort_values("fecha")
              .reset_index(drop=True))
    return puntos

# --------- Puntos -> tramos por modelo ---------
def _points_to_intervals_by_method(puntos: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte puntos (fecha, valor, modelo) a tramos consecutivos (diarios) por modelo.
    Salida: start_date, end_date, metodos (list[str])
    """
    if puntos.empty:
        return pd.DataFrame(columns=["start_date", "end_date", "metodos","fuente"])

    puntos = puntos.copy()
    puntos["fecha"] = pd.to_datetime(puntos["fecha"], errors="coerce").dt.tz_localize(None)
    puntos = puntos.dropna(subset=["fecha"]).sort_values(["modelo", "fecha"]).reset_index(drop=True)

    tramos = []
    for modelo, dfm in puntos.groupby("modelo"):
        fechas = dfm["fecha"].tolist()
        if not fechas:
            continue
        start = prev = fechas[0]
        for f in fechas[1:]:
            if (f - prev).days == 1:
                prev = f
            else:
                tramos.append({"start_date": start, "end_date": prev, "metodos": [modelo]})
                start = prev = f
        tramos.append({"start_date": start, "end_date": prev, "metodos": [modelo]})
    #print("[DEBUG] Primeros 10 tramos construidos:")
    #print(pd.DataFrame(tramos).head(10).to_string())
    return (pd.DataFrame(tramos, columns=["start_date", "end_date", "metodos"])
            if tramos else pd.DataFrame(columns=["start_date", "end_date", "metodos"]))

# --------- Consolidación por consenso ---------
def consolidate_segments(tramos: pd.DataFrame) -> pd.DataFrame:
    """
    - Une tramos idénticos (start_date, end_date) y suma métodos únicos.
    - Filtra por consenso (count >= _MIN_MODELOS).
    - Limpieza por inicio común: mismo start -> end máximo + unión de métodos.
    - Genera columnas:
        * method: métodos participantes, coma-separados (ARIMA,DIF)
        * values_method: valores por método con corchetes por método, p.ej. [786][786]
        * values: valores únicos del tramo, sin corchetes, coma-separados, p.ej. 786
        * num_values: número de valores únicos en 'values'
    - Escribe anomalias_consolidado.csv en _OUT_DIR y devuelve columnas base.
    """
    # columnas base de retorno (contrato “aguas abajo”)
    _cols_return = ["id", "start_date", "end_date", "count", "method"]

    # CSV de salida (siempre lo creamos, aunque vacío, para evitar FileNotFound)
    out_file = os.path.join(_OUT_DIR, "anomalias_consolidado.csv")
    os.makedirs(_OUT_DIR, exist_ok=True)

    if tramos.empty:
        vacio = pd.DataFrame(columns=_cols_return + ["values_method", "values", "num_values"])
        vacio.to_csv(out_file, index=False)
        return pd.DataFrame(columns=_cols_return)

    # Normalización de fechas
    t = tramos.copy()
    t["start_date"] = pd.to_datetime(t["start_date"], errors="coerce").dt.tz_localize(None)
    t["end_date"]   = pd.to_datetime(t["end_date"],   errors="coerce").dt.tz_localize(None)
    t = t.dropna(subset=["start_date", "end_date"])

    # Agrupar tramos idénticos y unificar métodos
    g = (
        t.groupby(["start_date", "end_date"], as_index=False)["metodos"]
          .apply(lambda s: sorted(set(m for lst in s for m in lst if m)))
    )
    g["count"] = g["metodos"].apply(len)

    # Filtro por consenso
    g = g[g["count"] >= int(_MIN_MODELOS)].reset_index(drop=True)
    if g.empty:
        vacio = pd.DataFrame(columns=_cols_return + ["values_method", "values", "num_values"])
        vacio.to_csv(out_file, index=False)
        return pd.DataFrame(columns=_cols_return)

    # Compactar por mismo start_date: end_date = max y métodos = unión
    def _comb(df_):
        end_max = df_["end_date"].max()
        met = sorted(set(m for lst in df_["metodos"] for m in lst))
        return pd.Series({"end_date": end_max, "metodos": met, "count": len(met)})

    limpio = g.groupby("start_date", as_index=False).apply(_comb)
    limpio = limpio.sort_values(["start_date", "end_date"]).reset_index(drop=True)

    # Agregar id incremental
    limpio.insert(0, "id", range(1, len(limpio) + 1))

    # ---------- Cargar puntos por método desde _IN_DIR ----------
    puntos_cache: dict[str, pd.DataFrame] = {}
    todos_metodos = sorted(set(m for lst in limpio["metodos"] for m in lst))
    ruta_metodos = os.path.join(_IN_DIR, "Original")
    for metodo in todos_metodos:
        patrones = [
            os.path.join(ruta_metodos, f"{metodo}_puntos.csv"),
            os.path.join(ruta_metodos, f"{metodo}_puntos_*.csv"),
        ]
        files = []
        for pat in patrones:
            files.extend(glob.glob(pat))

        frames = []
        for fp in sorted(set(files)):
            try:
                dfm = pd.read_csv(fp)
                # Normalizar columnas esperadas
                if "fecha" in dfm.columns:
                    dfm["fecha"] = pd.to_datetime(dfm["fecha"], errors="coerce", dayfirst=True)
                    dfm["fecha"] = dfm["fecha"].dt.tz_localize(None)
                else:
                    dfm["fecha"] = pd.NaT
                if "valor" in dfm.columns:
                    dfm["valor"] = pd.to_numeric(dfm["valor"], errors="coerce")
                else:
                    dfm["valor"] = pd.NA
                dfm = dfm.dropna(subset=["fecha"])
                if not dfm.empty:
                    frames.append(dfm[["fecha", "valor"]])
            except Exception:
                continue

        if frames:
            puntos_cache[metodo] = (
                pd.concat(frames, ignore_index=True)
                  .dropna(subset=["fecha"])
                  .sort_values("fecha")
                  .reset_index(drop=True)
            )
        else:
            puntos_cache[metodo] = pd.DataFrame(columns=["fecha", "valor"])

    # ---------- Construir method, values_method, values, num_values ----------
    method_txt, values_method_txt, values_txt, num_values_list = [], [], [], []

    for _, row in limpio.iterrows():
        start_i = pd.to_datetime(row["start_date"]).date()
        end_i   = pd.to_datetime(row["end_date"]).date()
        metodos = row["metodos"]

        # method (texto)
        method_str = ",".join(metodos)

        # Por método: valores con corchetes
        frag_por_metodo = []
        # Conjunto de valores únicos para 'values'
        unique_vals = set()

        for m in metodos:
            dfm = puntos_cache.get(m, pd.DataFrame(columns=["fecha", "valor"]))
            if dfm.empty:
                vals_m = []
            else:
                fechas = dfm["fecha"].dt.date
                mask = (fechas >= start_i) & (fechas <= end_i)
                vals_m = dfm.loc[mask].sort_values("fecha")["valor"].dropna().tolist()

            # Agregar al set global (deduplicado)
            for v in vals_m:
                unique_vals.add(v)

            # Formato por método: [v1,v2,...]
            frag_por_metodo.append(f"[{','.join(map(lambda x: str(x).rstrip('0').rstrip('.') if isinstance(x, float) else str(x), vals_m))}]")

        # values_method: concatenación de bloques por método
        values_method_str = "".join(frag_por_metodo)

        # values: únicos, ordenados, sin corchetes, coma-separados
        # Orden numérico estable (float->str limpio)
        vals_ordenados = sorted(unique_vals)
        def _fmt(x):
            if isinstance(x, float):
                s = f"{x}"
                # limpiar ceros y punto sobrante
                s = s.rstrip('0').rstrip('.') if '.' in s else s
                return s
            return str(x)
        values_str = ",".join(_fmt(v) for v in vals_ordenados)

        # num_values: cantidad de únicos
        num_vals = len(vals_ordenados)

        method_txt.append(method_str)
        values_method_txt.append(values_method_str)
        values_txt.append(values_str)
        num_values_list.append(num_vals)

    # Añadir columnas nuevas y renombrar según lo pedido
    limpio["method"]         = method_txt
    limpio["values_method"]  = values_method_txt
    limpio["values"]         = values_txt
    limpio["num_values"]     = num_values_list

    # Reordenar para guardar
    save_cols = ["id","start_date","end_date","count","method","values_method","values","num_values"]
    _save = limpio[save_cols].copy()

    # Persistir CSV
    _save.to_csv(out_file, index=False)
    print(f"Archivo consolidado guardado en: {out_file}")
    print("[DEBUG] Tramos agrupados (antes de consenso):")
    print(g.head(10).to_string())
    # Retornar contrato base (si tu pipeline downstream espera solo estas)
    return limpio[["id","start_date","end_date","count","method"]]



def crear_csv_enriquecido_v0(
    csv_in: str | None = None,
    csv_out: str | None = None,
    serie_id: str | None = None,
    ruta_graficos: str | None = None,
    serie_limpia: str | None = None,
    ctx_dias: int | None = None,
    norm_spec: str = "z-global",
) -> str:
    """
    Enriquecer el consolidado sin re-procesar tramos.
    Usa 'values' (valores únicos sin corchetes, separados por coma) para calcular:
      - min_value, max_value, amplitude (= max - min)
      - peak_value, peak_time (si num_values==1 usa start_date; si >1 usa end_date)
      - mean_value, std_value (si hay >= 2 valores)
    Además añade:
      - serie (SERIE_ID)
      - serie_id, ruta_graficos, serie_limpia, ctx_dias, norm_spec
      - duracion_dias, ctx_ini, ctx_fin
      - grafico_file: ruta sugerida NNN_YYYY-MM-DD.png en ruta_graficos
    """

    # Defaults desde settings (ya importadas en la cabecera del módulo)
    csv_in   = csv_in  or _CSV_IN_DEF
    csv_out  = csv_out or os.path.join(_OUT_DIR, "anomalias_consolidado_completo.csv")
    serie_id = _SERIE_ID if serie_id is None else serie_id
    ruta_graficos = _RUTA_GRAFICOS if ruta_graficos is None else ruta_graficos
    serie_limpia  = _SERIE_LIMPIA  if serie_limpia  is None else serie_limpia
    ctx_dias = int(_CTX_DIAS if ctx_dias is None else ctx_dias)

    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    cols_out = [
        "id","start_date","end_date","count","method","values_method","values","num_values",
        "serie","serie_id","ruta_graficos","serie_limpia","ctx_dias","norm_spec",
        "duracion_dias","ctx_ini","ctx_fin",
        "min_value","max_value","amplitude",
        "peak_value","peak_time",
        "mean_value","std_value",
        "grafico_file",
    ]

    # Si no existe el consolidado, escribe vacío con headers
    if not os.path.isfile(csv_in):
        pd.DataFrame(columns=cols_out).to_csv(csv_out, index=False)
        return csv_out

    df = pd.read_csv(csv_in, dtype=str)

    # Asegurar columnas base
    if "methods" in df.columns and "method" not in df.columns:
        df["method"] = df["methods"]
    for col in ["id","start_date","end_date","count","method","values_method","values","num_values"]:
        if col not in df.columns:
            df[col] = pd.NA

    start_parsed = pd.to_datetime(df["start_date"], errors="coerce")
    end_parsed   = pd.to_datetime(df["end_date"], errors="coerce")

    duracion = (end_parsed - start_parsed).dt.days.add(1)
    ctx_ini  = start_parsed - pd.to_timedelta(ctx_dias, unit="D")
    ctx_fin  = end_parsed   + pd.to_timedelta(ctx_dias, unit="D")

    def parse_values_plain(s: str) -> list[float]:
        if not isinstance(s, str):
            return []
        txt = s.strip()
        if not txt:
            return []
        txt = txt.replace(";", ",").replace("\n", " ")
        txt = ",".join(t for t in txt.replace(",", " , ").split() if t != "," or True)
        toks = [t.strip() for t in txt.split(",") if t.strip() != ""]
        vals: list[float] = []
        for tok in toks:
            if ("," in tok) and ("." not in tok):
                tok = tok.replace(",", ".")
            try:
                vals.append(float(tok))
            except Exception:
                continue
        return vals

    values_parsed = df["values"].apply(parse_values_plain)

    def to_int_safe(x):
        try:
            return int(x)
        except Exception:
            return None

    num_values_series = df["num_values"].apply(to_int_safe)

    min_list, max_list, amp_list = [], [], []
    peak_val_list, peak_time_list = [], []
    mean_list, std_list = [], []
    graf_paths = []

    os.makedirs(ruta_graficos, exist_ok=True)

    for i, vals in enumerate(values_parsed):
        if len(vals) >= 1:
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
            min_list.append(vmin)
            max_list.append(vmax)
            amp_list.append(vmax - vmin)
            peak_val_list.append(vmax)
            nuniq = num_values_series.iloc[i]
            pt = start_parsed.iloc[i] if nuniq == 1 else end_parsed.iloc[i]
            peak_time_list.append(pd.NA if pd.isna(pt) else pt.strftime("%Y-%m-%d %H:%M:%S"))
            if len(vals) >= 2:
                mean_list.append(float(np.nanmean(vals)))
                std_list.append(float(np.nanstd(vals, ddof=0)))
            else:
                mean_list.append(pd.NA)
                std_list.append(pd.NA)
        else:
            min_list.append(pd.NA); max_list.append(pd.NA); amp_list.append(pd.NA)
            peak_val_list.append(pd.NA); peak_time_list.append(pd.NA)
            mean_list.append(pd.NA); std_list.append(pd.NA)

        try:
            anom_id = int(df.loc[i, "id"]) if pd.notna(df.loc[i, "id"]) else i + 1
        except Exception:
            anom_id = i + 1
        sp = start_parsed.iloc[i]
        fecha_str = (sp.strftime("%Y-%m-%d") if not pd.isna(sp)
                     else (str(df.loc[i, "start_date"])[:10] if pd.notna(df.loc[i, "start_date"]) else "NA"))
        nombre_png = f"{anom_id:03d}_{fecha_str}.png"
        graf_paths.append(os.path.join(ruta_graficos, nombre_png))

    # Añadir columnas enriquecidas
    df["serie"]        = serie_id       # <<< nueva columna con SERIE_ID explícito
    df["serie_id"]     = serie_id
    df["ruta_graficos"] = ruta_graficos
    df["serie_limpia"]  = serie_limpia
    df["ctx_dias"]      = str(ctx_dias)
    df["norm_spec"]     = norm_spec
    df["duracion_dias"] = duracion.astype("Int64")
    df["ctx_ini"]       = ctx_ini.dt.strftime("%Y-%m-%d %H:%M:%S")
    df["ctx_fin"]       = ctx_fin.dt.strftime("%Y-%m-%d %H:%M:%S")

    df["min_value"]   = min_list
    df["max_value"]   = max_list
    df["amplitude"]   = amp_list
    df["peak_value"]  = peak_val_list
    df["peak_time"]   = peak_time_list
    df["mean_value"]  = mean_list
    df["std_value"]   = std_list
    df["grafico_file"] = graf_paths

    df.to_csv(csv_out, index=False)
    return csv_out

def crear_csv_enriquecido(
    csv_in: str | None = None,
    csv_out: str | None = None,
    serie_id: str | None = None,
    ruta_graficos: str | None = None,
    serie_limpia: str | None = None,
    ctx_dias: int | None = None,
    norm_spec: str = "z-global",
) -> str:
    """
    Enriquecer el consolidado sin re-procesar tramos.
    ...
    Además añade:
      - serie (SERIE_ID)
      - serie_id, ruta_graficos, serie_limpia, ctx_dias, norm_spec
      - serie_nombre (nombre canónico de la serie original) -> desde /tmp/DatosSerie.csv (catálogo)
      ...
    """

    csv_in   = csv_in  or _CSV_IN_DEF
    csv_out  = csv_out or os.path.join(_OUT_DIR, "anomalias_consolidado_completo.csv")
    serie_id = _SERIE_ID if serie_id is None else serie_id
    ruta_graficos = _RUTA_GRAFICOS if ruta_graficos is None else ruta_graficos
    serie_limpia  = _SERIE_LIMPIA  if serie_limpia  is None else serie_limpia
    ctx_dias = int(_CTX_DIAS if ctx_dias is None else ctx_dias)

    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    # ------------------------------------------------------------------
    # Nombre canónico de la serie original: SIEMPRE desde /tmp/DatosSerie.csv
    # Formato:
    #   serie_id;iri;nombre_es;frecuencia;unidad
    #   1;anom:series_riesgo_pais_1;Riesgo Pais;daily;points
    #
    # Nota: serie_id puede venir como "001" y en el catálogo como "1".
    # ------------------------------------------------------------------
    serie_nombre = str(serie_id)
    try:
        catalogo_series = "/tmp/DatosSerie.csv"
        if os.path.isfile(catalogo_series):
            df_cat = pd.read_csv(catalogo_series, sep=";", dtype=str)
            df_cat.columns = [c.strip() for c in df_cat.columns]

            # Normalizar serie_id: "001" -> "1" (si es numérico)
            sid_raw = "" if serie_id is None else str(serie_id).strip()
            sid_norm = str(int(sid_raw)) if sid_raw.isdigit() else sid_raw

            df_sel = df_cat
            if "serie_id" in df_cat.columns:
                df_f = df_cat[df_cat["serie_id"].astype(str).str.strip() == sid_norm]
                if not df_f.empty:
                    df_sel = df_f

            if "nombre_es" in df_sel.columns:
                s = df_sel["nombre_es"].dropna()
                if not s.empty:
                    v = str(s.iloc[0]).strip()
                    if v:
                        serie_nombre = v
    except Exception:
        # Mantener fallback a serie_id
        serie_nombre = str(serie_id)

    cols_out = [
        "id","start_date","end_date","count","method","values_method","values","num_values",
        "serie","serie_id","serie_nombre","ruta_graficos","serie_limpia","ctx_dias","norm_spec",
        "duracion_dias","ctx_ini","ctx_fin",
        "min_value","max_value","amplitude",
        "peak_value","peak_time",
        "mean_value","std_value",
        "grafico_file",
    ]

    if not os.path.isfile(csv_in):
        pd.DataFrame(columns=cols_out).to_csv(csv_out, index=False)
        return csv_out

    df = pd.read_csv(csv_in, dtype=str)

    if "methods" in df.columns and "method" not in df.columns:
        df["method"] = df["methods"]

    for col in ["id","start_date","end_date","count","method","values_method","values","num_values"]:
        if col not in df.columns:
            df[col] = pd.NA

    start_parsed = pd.to_datetime(df["start_date"], errors="coerce")
    end_parsed   = pd.to_datetime(df["end_date"], errors="coerce")

    duracion = (end_parsed - start_parsed).dt.days.add(1)
    ctx_ini  = start_parsed - pd.to_timedelta(ctx_dias, unit="D")
    ctx_fin  = end_parsed   + pd.to_timedelta(ctx_dias, unit="D")

    def parse_values_plain(s: str) -> list[float]:
        if not isinstance(s, str):
            return []
        txt = s.strip()
        if not txt:
            return []
        txt = txt.replace(";", ",").replace("\n", " ")
        toks = [t.strip() for t in txt.split(",") if t.strip() != ""]
        vals: list[float] = []
        for tok in toks:
            if ("," in tok) and ("." not in tok):
                tok = tok.replace(",", ".")
            try:
                vals.append(float(tok))
            except Exception:
                continue
        return vals

    values_parsed = df["values"].apply(parse_values_plain)

    def to_int_safe(x):
        try:
            return int(x)
        except Exception:
            return None

    num_values_series = df["num_values"].apply(to_int_safe)

    min_list, max_list, amp_list = [], [], []
    peak_val_list, peak_time_list = [], []
    mean_list, std_list = [], []
    graf_paths = []

    os.makedirs(ruta_graficos, exist_ok=True)

    for i, vals in enumerate(values_parsed):
        if len(vals) >= 1:
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
            min_list.append(vmin)
            max_list.append(vmax)
            amp_list.append(vmax - vmin)
            peak_val_list.append(vmax)

            nuniq = num_values_series.iloc[i]
            pt = start_parsed.iloc[i] if nuniq == 1 else end_parsed.iloc[i]
            peak_time_list.append(pd.NA if pd.isna(pt) else pt.strftime("%Y-%m-%d %H:%M:%S"))

            if len(vals) >= 2:
                mean_list.append(float(np.nanmean(vals)))
                std_list.append(float(np.nanstd(vals, ddof=0)))
            else:
                mean_list.append(pd.NA)
                std_list.append(pd.NA)
        else:
            min_list.append(pd.NA); max_list.append(pd.NA); amp_list.append(pd.NA)
            peak_val_list.append(pd.NA); peak_time_list.append(pd.NA)
            mean_list.append(pd.NA); std_list.append(pd.NA)

        try:
            anom_id = int(df.loc[i, "id"]) if pd.notna(df.loc[i, "id"]) else i + 1
        except Exception:
            anom_id = i + 1

        sp = start_parsed.iloc[i]
        fecha_str = (sp.strftime("%Y-%m-%d") if not pd.isna(sp)
                     else (str(df.loc[i, "start_date"])[:10] if pd.notna(df.loc[i, "start_date"]) else "NA"))
        nombre_png = f"{anom_id:03d}_{fecha_str}.png"
        graf_paths.append(os.path.join(ruta_graficos, nombre_png))

    # Añadir columnas enriquecidas
    df["serie"]         = serie_id
    df["serie_id"]      = serie_id
    df["serie_nombre"]  = serie_nombre
    df["ruta_graficos"] = ruta_graficos
    df["serie_limpia"]  = serie_limpia
    df["ctx_dias"]      = str(ctx_dias)
    df["norm_spec"]     = norm_spec
    df["duracion_dias"] = duracion.astype("Int64")
    df["ctx_ini"]       = ctx_ini.dt.strftime("%Y-%m-%d %H:%M:%S")
    df["ctx_fin"]       = ctx_fin.dt.strftime("%Y-%m-%d %H:%M:%S")

    df["min_value"]    = min_list
    df["max_value"]    = max_list
    df["amplitude"]    = amp_list
    df["peak_value"]   = peak_val_list
    df["peak_time"]    = peak_time_list
    df["mean_value"]   = mean_list
    df["std_value"]    = std_list
    df["grafico_file"] = graf_paths

    # Garantizar esquema y orden estable
    for c in cols_out:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[cols_out]

    df.to_csv(csv_out, index=False)
    return csv_out


def crear_csv_comparacion_numerica_V1(
    csv_enriquecido: str | None = None,
    csv_out: str | None = None,
) -> str:
    """
    A partir del CSV enriquecido (salida de `crear_csv_enriquecido`),
    genera un catálogo de anomalías listo para comparación numérica.

    - Toma una fila por anomalía.
    - Calcula `center_date` como punto medio entre start_date y end_date.
    - Conserva información numérica útil (min, max, amplitude, peak, mean, std, etc.).
    - Agrega columnas estándar para el pipeline:
        - id_anomalia, tipo='detectada', fuente='original'.

    Parámetros
    ----------
    csv_enriquecido : str | None
        Ruta del CSV enriquecido. Si es None, usa el mismo default que `crear_csv_enriquecido`.
    csv_out : str | None
        Ruta del CSV de salida para comparación numérica.

    Retorna
    -------
    str
        Ruta del CSV generado.
    """

    # Mismos defaults que en tu función base
    base_in  = os.path.join(_OUT_DIR, "anomalias_consolidado_completo.csv")
    base_out = os.path.join(_OUT_DIR, "anomalias_comparacion_numerica.csv")

    csv_enriquecido = csv_enriquecido or base_in
    csv_out = csv_out or base_out

    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    # Si no existe el enriquecido, escribe vacío con headers mínimos
    if not os.path.isfile(csv_enriquecido):
        cols_out = [
            "id_anomalia",
            "serie","serie_id",
            "start_date","end_date","center_date","duracion_dias",
            "method","num_values","count",
            "min_value","max_value","amplitude",
            "peak_value","peak_time",
            "mean_value","std_value",
            "grafico_file",
            "tipo","fuente",
        ]
        pd.DataFrame(columns=cols_out).to_csv(csv_out, index=False)
        return csv_out

    df = pd.read_csv(csv_enriquecido)

    # Asegurar que existan las columnas que vamos a usar
    for col in ["id","start_date","end_date","method","num_values","count",
                "min_value","max_value","amplitude",
                "peak_value","peak_time",
                "mean_value","std_value",
                "serie","serie_id",
                "grafico_file","duracion_dias"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Parseo de fechas para calcular center_date
    start_parsed = pd.to_datetime(df["start_date"], errors="coerce")
    end_parsed   = pd.to_datetime(df["end_date"], errors="coerce")

    # Si falta duracion_dias, la recalculamos
    if "duracion_dias" not in df.columns or df["duracion_dias"].isna().all():
        duracion = (end_parsed - start_parsed).dt.days.add(1)
    else:
        # Aseguramos tipo entero si viene de antes
        duracion = pd.to_numeric(df["duracion_dias"], errors="coerce")

    # Centro de la anomalía: si start=end, es puntual; si rango, punto medio
    center_dt = start_parsed + (end_parsed - start_parsed) / 2
    center_date_str = center_dt.dt.date.astype("string")  # YYYY-MM-DD

    # Construimos el catálogo de salida
    out = pd.DataFrame({
        "id_anomalia": df["id"],
        "serie": df["serie"],
        "serie_id": df["serie_id"],
        "start_date": start_parsed.dt.date.astype("string"),
        "end_date": end_parsed.dt.date.astype("string"),
        "center_date": center_date_str,
        "duracion_dias": duracion.astype("Int64"),

        "method": df["method"],
        "num_values": pd.to_numeric(df["num_values"], errors="coerce").astype("Int64"),
        "count": pd.to_numeric(df["count"], errors="coerce").astype("Int64"),

        "min_value": pd.to_numeric(df["min_value"], errors="coerce"),
        "max_value": pd.to_numeric(df["max_value"], errors="coerce"),
        "amplitude": pd.to_numeric(df["amplitude"], errors="coerce"),

        "peak_value": pd.to_numeric(df["peak_value"], errors="coerce"),
        "peak_time": df["peak_time"],

        "mean_value": pd.to_numeric(df["mean_value"], errors="coerce"),
        "std_value": pd.to_numeric(df["std_value"], errors="coerce"),

        "grafico_file": df["grafico_file"],

        # Metadatos estándar para tu pipeline
        "tipo": "detectada",
        "fuente": "original",
    })

    out.to_csv(csv_out, index=False)
    return csv_out

def crear_csv_comparacion_numerica_V(
    csv_enriquecido: str | None = None,
    csv_out: str | None = None,
) -> str:
    """
    A partir del CSV enriquecido (salida de `crear_csv_enriquecido`),
    genera un catálogo de anomalías listo para comparación numérica.

    - Toma una fila por anomalía.
    - Calcula `center_date` como punto medio entre start_date y end_date.
    - Conserva información numérica útil (min, max, amplitude, peak, mean, std, etc.).
    - Agrega columnas estándar para el pipeline:
        - id_anomalia, tipo='detectada', fuente='original'.
    - Asegura serie/serie_id con el nombre REAL de la serie original (desde DatosSerie).
    """

    # Mismos defaults que en tu función base
    base_in  = os.path.join(_OUT_DIR, "anomalias_consolidado_completo.csv")
    base_out = os.path.join(_OUT_DIR, "anomalias_comparacion_numerica.csv")

    csv_enriquecido = csv_enriquecido or base_in
    csv_out = csv_out or base_out

    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    # >>> NUEVO: nombre canónico de la serie original (desde DatosSerie)
    # Requiere que exista _RUTA_DATOS_SERIE en settings o en el módulo.
    nombre_serie = obtener_nombre_serie_original(_RUTA_DATOS_SERIE)

    # Si no existe el enriquecido, escribe vacío con headers mínimos
    if not os.path.isfile(csv_enriquecido):
        cols_out = [
            "id_anomalia",
            "serie","serie_id",
            "start_date","end_date","center_date","duracion_dias",
            "method","num_values","count",
            "min_value","max_value","amplitude",
            "peak_value","peak_time",
            "mean_value","std_value",
            "grafico_file",
            "tipo","fuente",
        ]
        df_empty = pd.DataFrame(columns=cols_out)
        # >>> NUEVO: si está vacío, al menos deja el nombre de serie como referencia (opcional)
        # (si prefieres que quede vacío, elimina estas 2 líneas)
        df_empty["serie"] = nombre_serie
        df_empty["serie_id"] = nombre_serie

        df_empty.to_csv(csv_out, index=False)
        return csv_out

    df = pd.read_csv(csv_enriquecido)

    # Asegurar que existan las columnas que vamos a usar
    for col in ["id","start_date","end_date","method","num_values","count",
                "min_value","max_value","amplitude",
                "peak_value","peak_time",
                "mean_value","std_value",
                "serie","serie_id",
                "grafico_file","duracion_dias"]:
        if col not in df.columns:
            df[col] = pd.NA

    # >>> NUEVO: asegurar serie/serie_id con nombre real de DatosSerie
    # Si viene todo vacío o parcialmente vacío, lo completamos.
    df["serie"] = df["serie"].where(df["serie"].notna() & (df["serie"].astype(str).str.strip() != ""), nombre_serie)
    df["serie_id"] = df["serie_id"].where(df["serie_id"].notna() & (df["serie_id"].astype(str).str.strip() != ""), nombre_serie)

    # Parseo de fechas para calcular center_date
    start_parsed = pd.to_datetime(df["start_date"], errors="coerce")
    end_parsed   = pd.to_datetime(df["end_date"], errors="coerce")

    # Si falta duracion_dias, la recalculamos
    if "duracion_dias" not in df.columns or df["duracion_dias"].isna().all():
        duracion = (end_parsed - start_parsed).dt.days.add(1)
    else:
        duracion = pd.to_numeric(df["duracion_dias"], errors="coerce")

    # Centro de la anomalía: si start=end, es puntual; si rango, punto medio
    center_dt = start_parsed + (end_parsed - start_parsed) / 2
    center_date_str = center_dt.dt.date.astype("string")  # YYYY-MM-DD

    # Construimos el catálogo de salida
    out = pd.DataFrame({
        "id_anomalia": df["id"],

        # >>> Serie real garantizada
        "serie": df["serie"],
        "serie_id": df["serie_id"],

        "start_date": start_parsed.dt.date.astype("string"),
        "end_date": end_parsed.dt.date.astype("string"),
        "center_date": center_date_str,
        "duracion_dias": duracion.astype("Int64"),

        "method": df["method"],
        "num_values": pd.to_numeric(df["num_values"], errors="coerce").astype("Int64"),
        "count": pd.to_numeric(df["count"], errors="coerce").astype("Int64"),

        "min_value": pd.to_numeric(df["min_value"], errors="coerce"),
        "max_value": pd.to_numeric(df["max_value"], errors="coerce"),
        "amplitude": pd.to_numeric(df["amplitude"], errors="coerce"),

        "peak_value": pd.to_numeric(df["peak_value"], errors="coerce"),
        "peak_time": df["peak_time"],

        "mean_value": pd.to_numeric(df["mean_value"], errors="coerce"),
        "std_value": pd.to_numeric(df["std_value"], errors="coerce"),

        "grafico_file": df["grafico_file"],

        # Metadatos estándar para tu pipeline
        "tipo": "detectada",
        "fuente": "original",
    })

    out.to_csv(csv_out, index=False)
    return csv_out



def crear_csv_comparacion_numerica(
    csv_enriquecido: str | None = None,
    csv_out: str | None = None,
) -> str:
    """
    A partir del CSV enriquecido (salida de `crear_csv_enriquecido`),
    genera un catálogo de anomalías listo para comparación numérica.

    Agrega:
      - serie_name (nombre de la serie original)
      - original_detection_method (método de detección de la anomalía original)
      - series_iri (IRI de la serie original, según convención v9)
      - anomaly_iri (IRI de la anomalía original, según convención v9)
      - point_iri (IRI de punto representativo; usa center_date)
    """
    import os
    import re
    import unicodedata
    import pandas as pd

    # Ajusta estos prefijos si los manejas de otra forma
    IRI_PREFIX = "anom:"
    SERIES_PREFIX = f"{IRI_PREFIX}series_"
    ANOM_PREFIX = f"{IRI_PREFIX}anomaly_"
    POINT_PREFIX = f"{IRI_PREFIX}pt_"

    def _slugify(s: str) -> str:
        if s is None or (isinstance(s, float) and pd.isna(s)):
            return "serie"
        s = str(s).strip().lower()
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
        return s or "serie"

    def _serie_num(serie, serie_id) -> int | None:
        """
        Preferimos 'serie' si es numérica; si no, derivamos de serie_id (ej. '001' -> 1).
        """
        # 1) Intentar con 'serie'
        try:
            if serie is not None and not (isinstance(serie, float) and pd.isna(serie)):
                v = int(str(serie).strip())
                return v
        except Exception:
            pass
        # 2) Fallback con 'serie_id'
        try:
            if serie_id is not None and not (isinstance(serie_id, float) and pd.isna(serie_id)):
                v = int(str(serie_id).strip())
                return v
        except Exception:
            pass
        return None

    def _serie_id_3(serie_id, serie_num) -> str:
        """
        Retorna serie_id en 3 dígitos. Si serie_id falta, usa serie_num.
        """
        sid = None
        if serie_id is not None and not (isinstance(serie_id, float) and pd.isna(serie_id)):
            sid = str(serie_id).strip()
        if sid:
            # si viene "001" lo respetamos; si viene "1" lo padded
            try:
                return f"{int(sid):03d}"
            except Exception:
                # si trae algo raro, lo dejamos pero intentamos recortar
                sid_digits = re.sub(r"\D+", "", sid)
                if sid_digits:
                    return f"{int(sid_digits):03d}"
                return sid
        if serie_num is None:
            return "000"
        return f"{int(serie_num):03d}"

    # Defaults
    base_in  = os.path.join(_OUT_DIR, "anomalias_consolidado_completo.csv")
    base_out = os.path.join(_OUT_DIR, "anomalias_comparacion_numerica.csv")

    csv_enriquecido = csv_enriquecido or base_in
    csv_out = csv_out or base_out

    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    # Si no existe el enriquecido, escribe vacío con headers mínimos
    if not os.path.isfile(csv_enriquecido):
        cols_out = [
            "id_anomalia",
            "serie", "serie_id",
            "serie_name",
            "series_iri",          # NUEVO
            "anomaly_iri",         # NUEVO
            "point_iri",           # NUEVO
            "start_date", "end_date", "center_date", "duracion_dias",
            "method",
            "original_detection_method",
            "num_values", "count",
            "min_value", "max_value", "amplitude",
            "peak_value", "peak_time",
            "mean_value", "std_value",
            "grafico_file",
            "tipo", "fuente",
        ]
        pd.DataFrame(columns=cols_out).to_csv(csv_out, index=False)
        return csv_out

    df = pd.read_csv(csv_enriquecido)

    # Columnas base requeridas
    needed = [
        "id", "start_date", "end_date",
        "method",
        "num_values", "count",
        "min_value", "max_value", "amplitude",
        "peak_value", "peak_time",
        "mean_value", "std_value",
        "serie", "serie_id",
        "serie_nombre",
        "grafico_file", "duracion_dias",
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = pd.NA

    # serie_name (label humano)
    if "serie_name" not in df.columns:
        # Preferimos serie_nombre si existe; fallback a serie
        df["serie_name"] = df["serie_nombre"].where(df["serie_nombre"].notna(), df["serie"])

    # original_detection_method
    if "original_detection_method" not in df.columns:
        if "metodos" in df.columns:
            df["original_detection_method"] = df["metodos"]
        elif "metodo" in df.columns:
            df["original_detection_method"] = df["metodo"]
        elif "detection_method" in df.columns:
            df["original_detection_method"] = df["detection_method"]
        elif "detected_by" in df.columns:
            df["original_detection_method"] = df["detected_by"]
        else:
            df["original_detection_method"] = df["method"]

    # Parseo de fechas
    start_parsed = pd.to_datetime(df["start_date"], errors="coerce")
    end_parsed   = pd.to_datetime(df["end_date"], errors="coerce")

    # duracion_dias
    if "duracion_dias" not in df.columns or df["duracion_dias"].isna().all():
        duracion = (end_parsed - start_parsed).dt.days.add(1)
    else:
        duracion = pd.to_numeric(df["duracion_dias"], errors="coerce")

    # center_date
    center_dt = start_parsed + (end_parsed - start_parsed) / 2
    center_date_str = center_dt.dt.date.astype("string")  # YYYY-MM-DD
    start_date_str = start_parsed.dt.date.astype("string")
    end_date_str = end_parsed.dt.date.astype("string")

    # Construcción de IRIs
    # - slug desde serie_name
    slug = df["serie_name"].apply(_slugify)
    # - serie_num desde serie o serie_id
    serie_num = [
        _serie_num(s, sid) for s, sid in zip(df["serie"], df["serie_id"])
    ]
    serie_num_s = pd.Series(serie_num, index=df.index, dtype="Int64")

    serie_id_3 = [
        _serie_id_3(sid, sn) for sid, sn in zip(df["serie_id"], serie_num_s)
    ]
    serie_id_3_s = pd.Series(serie_id_3, index=df.index, dtype="string")

    # series_iri: anom:series_<slug>_<serie_num>
    # si falta serie_num, dejamos NA para no inventar
    series_iri = []
    for sl, sn in zip(slug, serie_num_s):
        if pd.isna(sn):
            series_iri.append(pd.NA)
        else:
            series_iri.append(f"{SERIES_PREFIX}{sl}_{int(sn)}")
    series_iri_s = pd.Series(series_iri, index=df.index, dtype="string")

    # anomaly_iri: anom:anomaly_<serie_id_3>_<id>
    anomaly_iri_s = (
        serie_id_3_s.astype("string")
        + "_"
        + df["id"].astype("string")
    ).map(lambda x: f"{ANOM_PREFIX}{x}" if pd.notna(x) else pd.NA)

    # point_iri representativo: anom:pt_<slug>_<serie_num>_<YYYY_MM_DD> usando center_date
    point_iri = []
    for sl, sn, cd in zip(slug, serie_num_s, center_date_str):
        if pd.isna(sn) or pd.isna(cd):
            point_iri.append(pd.NA)
        else:
            yyyy_mm_dd = str(cd).replace("-", "_")
            point_iri.append(f"{POINT_PREFIX}{sl}_{int(sn)}_{yyyy_mm_dd}")
    point_iri_s = pd.Series(point_iri, index=df.index, dtype="string")

    # Construimos el catálogo de salida
    out = pd.DataFrame({
        "id_anomalia": df["id"],
        "serie": df["serie"],
        "serie_id": df["serie_id"],
        "serie_name": df["serie_name"],

        # IRIs
        "series_iri": series_iri_s,
        "anomaly_iri": anomaly_iri_s,
        "point_iri": point_iri_s,

        "start_date": start_date_str,
        "end_date": end_date_str,
        "center_date": center_date_str,
        "duracion_dias": duracion.astype("Int64"),

        "method": df["method"],
        "original_detection_method": df["original_detection_method"],

        "num_values": pd.to_numeric(df["num_values"], errors="coerce").astype("Int64"),
        "count": pd.to_numeric(df["count"], errors="coerce").astype("Int64"),

        "min_value": pd.to_numeric(df["min_value"], errors="coerce"),
        "max_value": pd.to_numeric(df["max_value"], errors="coerce"),
        "amplitude": pd.to_numeric(df["amplitude"], errors="coerce"),

        "peak_value": pd.to_numeric(df["peak_value"], errors="coerce"),
        "peak_time": df["peak_time"],

        "mean_value": pd.to_numeric(df["mean_value"], errors="coerce"),
        "std_value": pd.to_numeric(df["std_value"], errors="coerce"),

        "grafico_file": df["grafico_file"],

        "tipo": "detectada",
        "fuente": "original",
    })

    out.to_csv(csv_out, index=False)
    return csv_out


def comparar_anomalias_numericas_predichas(CFG, modo: str = "original_vs_sintetica") -> str:
    """
    Compara anomalías numéricamente usando DTW + features estadísticas.

    MODOS:
    - "original_vs_sintetica" (default): comportamiento original.
        A = anomalía ORIGINAL (anomalias_comparacion_numerica.csv)
        B = anomalía SINTÉTICA (anomalias_comparacion_numerica_sinteticas.csv)
        Vector sintético: leído desde /Prediccion/<MODELO>/..._sintetica.csv
        IRI sintético: construido con metodo_prediccion + metodo_deteccion

    - "historico_vs_reciente": nuevo modo sin predicción.
        Lee UN SOLO CSV (anomalias_comparacion_numerica.csv) y lo divide por fecha:
        A = anomalías con start_date < (max_date - NUM_DIAS_ANOMALIAS)  → históricas
        B = anomalías con start_date >= (max_date - NUM_DIAS_ANOMALIAS) → recientes
        Vector sintético (reciente): extraído de SerieOriginal.csv igual que el original
        IRI: reutiliza anomaly_iri que ya viene en el CSV

    En ambos modos el CSV de salida tiene las mismas columnas.

    DTW_MODE (CFG.DTW_MODE):
        - "segment":        compara solo el tramo anómalo (sin contexto).
        - "context_window": ventana fija 2*DTW_CTX_WINDOW centrada en el tramo.

    Score final = similarity_combined (si SIM_USE_COMBINED=True) o similarity_dtw.
    Solo se guardan pares con score_final >= CMP_MIN_SIM.
    """

    import os, math, time, logging, json, ast, glob, re
    import numpy as np
    import pandas as pd

    log = logging.getLogger(__name__)

    # -------------------------------------------------------------------------
    # Settings
    # -------------------------------------------------------------------------
    CMP_MIN_SIM      = float(getattr(CFG, "CMP_MIN_SIM", 0.50))
    DTW_MODE         = str(getattr(CFG, "DTW_MODE", "context_window")).lower().strip()
    if DTW_MODE not in ("segment", "context_window"):
        log.warning("[NUM] DTW_MODE=%r inválido, usando 'context_window'.", DTW_MODE)
        DTW_MODE = "context_window"

    SIM_USE_COMBINED = bool(getattr(CFG, "SIM_USE_COMBINED", True))
    SIM_COMB_METHOD  = str(getattr(CFG, "SIM_COMB_METHOD", "weighted_mean")).lower()
    SIM_ALPHA_DTW    = float(getattr(CFG, "SIM_ALPHA_DTW", 0.7))
    DTW_CTX_WINDOW   = int(getattr(CFG, "DTW_CTX_WINDOW", 10))
    DTW_BAND_RATIO   = float(getattr(CFG, "DTW_BAND_RATIO", 0.10))
    DTW_NORMALIZATION= str(getattr(CFG, "DTW_NORMALIZATION", "zscore")).lower()
    FEAT_USE         = bool(getattr(CFG, "FEAT_USE", True))
    FEAT_NORMALIZATION=str(getattr(CFG, "FEAT_NORMALIZATION", "zscore")).lower()
    FEAT_DISTANCE    = str(getattr(CFG, "FEAT_DISTANCE", "euclidean")).lower()

    # Ontología
    COMPUTED_BY_DTW  = "anom:SM_DTW"
    COMPUTED_BY_FEAT = "anom:SM_FEAT"
    COMPUTED_BY_COMB = "anom:SM_ENSEMBLE"
    FROM_RUN         = "anom:SR_NUM_AllPairs"
    HAS_SCALE        = "anom:Scale_0_1"

    if modo == "historico_vs_reciente":
        ruta_cmp = getattr(CFG, "RUTA_COMPARACION_HIST", 
                        os.path.join(getattr(CFG, "RUTA_RESULTADOS"), "ComparacionesHist"))
    else:
        ruta_cmp = getattr(CFG, "RUTA_COMPARACION")

    out_csv = os.path.join(ruta_cmp, "similaridades_dtw.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    
   # os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    pred_root = getattr(CFG, "RUTA_PREDICCION", None)
    if not pred_root:
        pred_root = os.path.join(getattr(CFG, "RUTA_RESULTADOS"), "Prediccion")

    log.info("[NUM] ===== Inicio comparar_anomalias_numericas_predichas | modo=%s =====", modo)
    log.info("[NUM] CMP_MIN_SIM=%.3f | DTW_MODE=%s", CMP_MIN_SIM, DTW_MODE)
    log.info("[NUM] SIM_USE_COMBINED=%s | SIM_COMB_METHOD=%s | SIM_ALPHA_DTW=%.3f",
             SIM_USE_COMBINED, SIM_COMB_METHOD, SIM_ALPHA_DTW)
    log.info("[NUM] DTW_CTX_WINDOW=%d | DTW_BAND_RATIO=%.3f | DTW_NORMALIZATION=%s",
             DTW_CTX_WINDOW, DTW_BAND_RATIO, DTW_NORMALIZATION)
    log.info("[NUM] FEAT_USE=%s | FEAT_NORMALIZATION=%s | FEAT_DISTANCE=%s",
             FEAT_USE, FEAT_NORMALIZATION, FEAT_DISTANCE)
    log.info("[NUM] pred_root=%s", pred_root)

    # -------------------------------------------------------------------------
    # Helpers: IO / parsing
    # -------------------------------------------------------------------------
    def _read_first_existing(paths):
        for p in paths:
            if p and os.path.isfile(p):
                return p
        return None

    def _coerce_series_df(df: pd.DataFrame, default_source: str = "original") -> pd.DataFrame:
        cols = {c.lower(): c for c in df.columns}
        fecha_col = cols.get("fecha") or cols.get("date") or cols.get("time")
        valor_col = cols.get("valor") or cols.get("value")
        fuente_col = cols.get("fuente") or cols.get("source")
        if fecha_col is None or valor_col is None:
            raise ValueError("Serie CSV debe tener columnas fecha y valor (y opcional fuente).")
        out = df.copy()
        out = out.rename(columns={fecha_col: "fecha", valor_col: "valor"})
        if fuente_col is None:
            out["fuente"] = default_source
        else:
            out = out.rename(columns={fuente_col: "fuente"})
        out["fecha"]  = pd.to_datetime(out["fecha"], errors="coerce")
        out["valor"]  = pd.to_numeric(out["valor"], errors="coerce")
        out["fuente"] = out["fuente"].astype(str)
        out = out.dropna(subset=["fecha", "valor"])
        out = out.sort_values("fecha").reset_index(drop=True)
        return out

    def _safe_to_date(s):
        try:
            return pd.to_datetime(s, errors="coerce").date()
        except Exception:
            return None

    def _parse_valores_field(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.array(v, dtype=np.float32)
            return arr if arr.size else None
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            for parser in ("json", "ast"):
                try:
                    obj = json.loads(s) if parser == "json" else ast.literal_eval(s)
                    if isinstance(obj, (list, tuple)):
                        arr = np.array(obj, dtype=np.float32)
                        return arr if arr.size else None
                except Exception:
                    pass
            try:
                parts = [p.strip() for p in s.split(",") if p.strip() != ""]
                arr = np.array([float(p) for p in parts], dtype=np.float32)
                return arr if arr.size else None
            except Exception:
                return None
        return None

    # -------------------------------------------------------------------------
    # Helpers: IRIs sintéticos (solo usados en modo original_vs_sintetica)
    # -------------------------------------------------------------------------
    def _fmt_ymd(d):
        dd = _safe_to_date(d)
        if dd is None:
            return None
        return f"{dd.year:04d}_{dd.month:02d}_{dd.day:02d}"

    def _build_syn_anomaly_iri(syn_row):
        serie_base  = syn_row.get("serie_base", None)
        metodo_pred = syn_row.get("metodo_prediccion", syn_row.get("modelo", None))
        metodo_det  = syn_row.get("metodo_deteccion", syn_row.get("method", None))
        s = _fmt_ymd(syn_row.get("start_date"))
        e = _fmt_ymd(syn_row.get("end_date"))
        if not serie_base or not metodo_pred or not metodo_det or not s or not e:
            return None
        dm = str(metodo_det).strip().lower()
        pm = str(metodo_pred).strip()
        return f"anom:anomaly_syn_{serie_base}__PM_{pm}_{s}_{e}__dm_{dm}"

    def _build_syn_series_iri(syn_row):
        serie_base  = syn_row.get("serie_base", None)
        metodo_pred = syn_row.get("metodo_prediccion", syn_row.get("modelo", None))
        if not serie_base or not metodo_pred:
            return None
        pm = str(metodo_pred).strip()
        return f"anom:series_{serie_base}__PM_{pm}"

    # -------------------------------------------------------------------------
    # Helpers: localizar series predichas en carpetas (solo original_vs_sintetica)
    # -------------------------------------------------------------------------
    syn_series_cache: dict = {}

    def _norm_name(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower())

    def _find_syn_series_file(syn_row):
        modelo   = syn_row.get("metodo_prediccion", syn_row.get("modelo", None))
        serie_id = syn_row.get("serie_id", None)
        if not modelo:
            return None
        modelo_str = str(modelo).strip()
        modelo_dir = os.path.join(pred_root, modelo_str)
        if not os.path.isdir(modelo_dir):
            want = _norm_name(modelo_str)
            for d in glob.glob(os.path.join(pred_root, "*")):
                if os.path.isdir(d) and _norm_name(os.path.basename(d)) == want:
                    modelo_dir = d
                    break
        if not os.path.isdir(modelo_dir):
            log.warning("[NUM][SYN] No existe carpeta de modelo: %s (modelo=%r)", modelo_dir, modelo_str)
            return None
        sid = None
        try:
            sid = int(serie_id) if pd.notna(serie_id) else None
        except Exception:
            sid = None
        if sid is None:
            sid = 1
        patterns = [
            os.path.join(modelo_dir, f"{sid}_*sintetica*.csv"),
            os.path.join(modelo_dir, f"{sid}_*Sintetica*.csv"),
            os.path.join(modelo_dir, f"{sid}_*SINTETICA*.csv"),
        ]
        cands = []
        for pat in patterns:
            cands.extend(glob.glob(pat))
        if not cands:
            patterns2 = [
                os.path.join(modelo_dir, f"*{sid}*sintetica*.csv"),
                os.path.join(modelo_dir, f"*{modelo_str}*sintetica*.csv"),
            ]
            for pat in patterns2:
                cands.extend(glob.glob(pat))
        if not cands:
            log.warning("[NUM][SYN] No encontré CSV de serie sintética en %s para serie_id=%s", modelo_dir, sid)
            return None
        cands = sorted(set(cands), key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))
        return cands[0]

    def _get_syn_series_df(syn_row):
        path = _find_syn_series_file(syn_row)
        if not path:
            return None
        if path in syn_series_cache:
            return syn_series_cache[path]
        if not os.path.isfile(path):
            syn_series_cache[path] = None
            return None
        try:
            sdf = _coerce_series_df(pd.read_csv(path), default_source="sintetica")
            sdf["fuente"] = "sintetica"
            syn_series_cache[path] = sdf
            return sdf
        except Exception as e:
            log.warning("[NUM][SYN] No pude leer serie sintética %s | error=%r", path, e)
            syn_series_cache[path] = None
            return None

    # -------------------------------------------------------------------------
    # Helpers: extracción de ventanas
    # -------------------------------------------------------------------------
    def _extract_segment_values(series_df: pd.DataFrame, fuente: str, start_date, end_date) -> np.ndarray:
        sdate = _safe_to_date(start_date)
        edate = _safe_to_date(end_date)
        if sdate is None or edate is None:
            return np.array([], dtype=np.float32)
        sdf = series_df[series_df["fuente"] == str(fuente)].copy()
        if sdf.empty:
            sdf = series_df.copy()
        sdf = sdf.sort_values("fecha").reset_index(drop=True)
        mask = (sdf["fecha"].dt.date >= sdate) & (sdf["fecha"].dt.date <= edate)
        idxs = np.flatnonzero(mask.to_numpy())
        if idxs.size == 0:
            return np.array([], dtype=np.float32)
        i0 = int(idxs[0]); i1 = int(idxs[-1])
        vals = sdf.loc[i0:i1, "valor"].to_numpy(dtype=np.float32)
        return vals if vals.size else np.array([], dtype=np.float32)

    def _extract_context_window(series_df: pd.DataFrame, fuente: str, start_date, end_date) -> np.ndarray:
        sdate = _safe_to_date(start_date)
        edate = _safe_to_date(end_date)
        if sdate is None or edate is None:
            return np.array([], dtype=np.float32)
        sdf = series_df[series_df["fuente"] == str(fuente)].copy()
        if sdf.empty:
            sdf = series_df.copy()
        sdf = sdf.sort_values("fecha").reset_index(drop=True)
        mask = (sdf["fecha"].dt.date >= sdate) & (sdf["fecha"].dt.date <= edate)
        idxs = np.flatnonzero(mask.to_numpy())
        if idxs.size == 0:
            return np.array([], dtype=np.float32)
        i0 = int(idxs[0]); i1 = int(idxs[-1])
        N = (i1 - i0 + 1)
        if N > 2 * DTW_CTX_WINDOW:
            return np.array([], dtype=np.float32)
        ctx_total = (2 * DTW_CTX_WINDOW) - N
        before = ctx_total // 2
        after  = ctx_total - before
        j0 = max(0, i0 - before)
        j1 = min(len(sdf) - 1, i1 + after)
        vals = sdf.loc[j0:j1, "valor"].to_numpy(dtype=np.float32)
        return vals if vals.size else np.array([], dtype=np.float32)

    # -------------------------------------------------------------------------
    # Helpers: normalización
    # -------------------------------------------------------------------------
    def _normalize_signal(x: np.ndarray, mode: str) -> np.ndarray:
        if x is None or x.size == 0:
            return x
        mode = (mode or "none").lower()
        if mode == "none":
            return x.astype(np.float32)
        if mode == "l2":
            v = x.astype(np.float32)
            n = float(np.linalg.norm(v))
            if not math.isfinite(n) or n <= 0:
                return v
            return (v / n).astype(np.float32)
        if mode == "zscore":
            v  = x.astype(np.float32)
            mu = float(np.mean(v))
            sd = float(np.std(v))
            if not math.isfinite(sd) or sd <= 1e-12:
                return (v - mu).astype(np.float32)
            return ((v - mu) / sd).astype(np.float32)
        return x.astype(np.float32)

    # -------------------------------------------------------------------------
    # Helpers: DTW
    # -------------------------------------------------------------------------
    def _dtw_distance(a: np.ndarray, b: np.ndarray, band=None) -> float:
        n = len(a); m = len(b)
        if n == 0 or m == 0:
            return float("inf")
        if band is None:
            band = max(1, int(DTW_BAND_RATIO * max(n, m)))
        band = max(band, abs(n - m))
        dp = np.full((n + 1, m + 1), float("inf"), dtype=np.float64)
        dp[0, 0] = 0.0
        for i in range(1, n + 1):
            j_start = max(1, i - band)
            j_end   = min(m, i + band)
            ai = a[i - 1]
            for j in range(j_start, j_end + 1):
                cost = abs(ai - b[j - 1])
                dp[i, j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
        return float(dp[n, m])

    def _dtw_similarity_0_1(a: np.ndarray, b: np.ndarray) -> float:
        dist = _dtw_distance(a, b, band=getattr(CFG, "DTW_BAND", None))
        if not math.isfinite(dist):
            return 0.0
        dist_norm = dist / max(1.0, float(len(a) + len(b)))
        sim = 1.0 / (1.0 + dist_norm)
        return float(max(0.0, min(1.0, sim)))

    # -------------------------------------------------------------------------
    # Helpers: Features
    # -------------------------------------------------------------------------
    def _feature_vector(x: np.ndarray) -> np.ndarray:
        if x is None or x.size == 0:
            return np.zeros((7,), dtype=np.float32)
        v  = x.astype(np.float32)
        mn = float(np.min(v)); mx = float(np.max(v))
        mu = float(np.mean(v)); sd = float(np.std(v))
        energy = float(np.mean(v * v))
        slope = 0.0
        if v.size >= 2:
            t    = np.arange(v.size, dtype=np.float32)
            t_mu = float(np.mean(t))
            denom = float(np.sum((t - t_mu) ** 2))
            if denom > 1e-12:
                slope = float(np.sum((t - t_mu) * (v - mu)) / denom)
        return np.array([float(v.size), mn, mx, mx - mn, mu, sd, energy + slope], dtype=np.float32)

    def _feat_similarity_0_1(fa: np.ndarray, fb: np.ndarray) -> float:
        a = _normalize_signal(fa.astype(np.float32), FEAT_NORMALIZATION)
        b = _normalize_signal(fb.astype(np.float32), FEAT_NORMALIZATION)
        if FEAT_DISTANCE == "cosine":
            na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
            if na <= 1e-12 or nb <= 1e-12:
                return 0.0
            cos = float(np.dot(a, b) / (na * nb))
            return float(max(0.0, min(1.0, (cos + 1.0) / 2.0)))
        dist = float(np.linalg.norm(a - b))
        return float(max(0.0, min(1.0, 1.0 / (1.0 + dist))))

    # -------------------------------------------------------------------------
    # Helpers: combinación de scores
    # -------------------------------------------------------------------------
    def _combine(sim_dtw: float, sim_feat: float) -> float:
        sim_dtw  = float(max(0.0, min(1.0, sim_dtw)))
        sim_feat = float(max(0.0, min(1.0, sim_feat)))
        if not SIM_USE_COMBINED:
            return sim_dtw
        if SIM_COMB_METHOD == "product":
            return float(max(0.0, min(1.0, sim_dtw * sim_feat)))
        if SIM_COMB_METHOD == "geometric":
            return float(max(0.0, min(1.0, math.sqrt(sim_dtw * sim_feat))))
        a = float(max(0.0, min(1.0, SIM_ALPHA_DTW)))
        return float(max(0.0, min(1.0, a * sim_dtw + (1.0 - a) * sim_feat)))

    # -------------------------------------------------------------------------
    # Carga de serie real (siempre necesaria para extraer vectores)
    # -------------------------------------------------------------------------
    serie_path = _read_first_existing([
        os.path.join(getattr(CFG, "RUTA_SALIDA"), "SerieOriginal.csv"),
        os.path.join(getattr(CFG, "RUTA_SALIDA"), "serie_original.csv"),
        getattr(CFG, "RUTA_SERIE_ORIGINAL", None),
    ])
    if not serie_path:
        raise FileNotFoundError("No se encontró SerieOriginal.csv en CFG.RUTA_SALIDA (o RUTA_SERIE_ORIGINAL).")

    series_real_df = _coerce_series_df(pd.read_csv(serie_path), default_source="original")
    series_real_df["fuente"] = "original"
    log.info("[NUM] serie_real=%s | filas=%d | rango=[%s..%s]",
             serie_path, len(series_real_df),
             series_real_df["fecha"].min(), series_real_df["fecha"].max())

    # -------------------------------------------------------------------------
    # Carga de anomalías según modo
    # -------------------------------------------------------------------------
    if modo == "historico_vs_reciente":
        num_dias = int(getattr(CFG, "NUM_DIAS_ANOMALIAS", 90))

        orig_path_used = _read_first_existing([
            os.path.join(getattr(CFG, "RUTA_RESULTADOS"), "anomalias_comparacion_numerica.csv"),
            os.path.join(getattr(CFG, "RUTA_RESULTADOS"), "anomalias_consolidado.csv"),
        ])
        if not orig_path_used:
            raise FileNotFoundError("No se encontró anomalias_comparacion_numerica.csv en CFG.RUTA_RESULTADOS.")

        df_all = pd.read_csv(orig_path_used)
        df_all["start_date"] = pd.to_datetime(df_all["start_date"], errors="coerce")
        df_all = df_all.dropna(subset=["start_date"]).sort_values("start_date").reset_index(drop=True)

        fecha_corte = df_all["start_date"].max() - pd.Timedelta(days=num_dias)
        df_orig = df_all[df_all["start_date"] <  fecha_corte].copy()
        df_syn  = df_all[df_all["start_date"] >= fecha_corte].copy()
        syn_path_used = orig_path_used

        if df_orig.empty:
            raise ValueError(f"[NUM] No hay anomalías históricas antes de {fecha_corte.date()}. "
                             f"Aumenta NUM_DIAS_ANOMALIAS (actual={num_dias}).")
        if df_syn.empty:
            raise ValueError(f"[NUM] No hay anomalías recientes desde {fecha_corte.date()}. "
                             f"Reduce NUM_DIAS_ANOMALIAS (actual={num_dias}).")

        log.info("[NUM] modo=historico_vs_reciente | corte=%s | historico=%d | reciente=%d",
                 fecha_corte.date(), len(df_orig), len(df_syn))

    else:
        # Comportamiento original: dos CSVs distintos
        orig_path_used = _read_first_existing([
            os.path.join(getattr(CFG, "RUTA_RESULTADOS"), "anomalias_comparacion_numerica.csv"),
            getattr(CFG, "RUTA_ANOMALIAS_ORIG_ONTO", None),
            os.path.join(getattr(CFG, "RUTA_RESULTADOS"), "anomalias_consolidado.csv"),
            getattr(CFG, "RUTA_ANOMALIAS_ORIG", None),
        ])
        if not orig_path_used:
            raise FileNotFoundError("No se encontró anomalias_comparacion_numerica.csv ni anomalias_consolidado.csv.")
        df_orig = pd.read_csv(orig_path_used)

        syn_path_used = _read_first_existing([
            os.path.join(getattr(CFG, "RUTA_RESULTADOS"), "Prediccion", "anomalias_sinteticas_csv",
                         "anomalias_comparacion_numerica_sinteticas.csv"),
            os.path.join(getattr(CFG, "RUTA_RESULTADOS"), "anomalias_comparacion_numerica_sinteticas.csv"),
            getattr(CFG, "RUTA_ANOMALIAS_SYN_ONTO", None),
            getattr(CFG, "RUTA_ANOMALIAS_SYN", None),
        ])
        if not syn_path_used:
            raise FileNotFoundError("No se encontró el catálogo de anomalías sintéticas.")
        df_syn = pd.read_csv(syn_path_used)

        log.info("[NUM] orig_path_used=%s | filas=%d", orig_path_used, len(df_orig))
        log.info("[NUM] syn_path_used=%s  | filas=%d", syn_path_used,  len(df_syn))

    # -------------------------------------------------------------------------
    # Normalización mínima de columnas
    # -------------------------------------------------------------------------
    for df_ in (df_orig, df_syn):
        if "id_anomalia" not in df_.columns and "id" in df_.columns:
            df_["id_anomalia"] = df_["id"]

    if "fuente" not in df_orig.columns:
        df_orig["fuente"] = "original"
    if "fuente" not in df_syn.columns:
        df_syn["fuente"] = "sintetica"

    for col in ["series_iri", "anomaly_iri", "point_iri", "serie_name", "serie", "serie_id",
                "method", "original_detection_method", "grafico_file", "tipo"]:
        if col not in df_orig.columns:
            df_orig[col] = pd.NA
        if col not in df_syn.columns:
            df_syn[col] = pd.NA

    # FIX: el filtro por fuente solo aplica en modo original_vs_sintetica.
    # En historico_vs_reciente ambos grupos tienen fuente="original" — filtrar los vaciaría.
    if modo != "historico_vs_reciente":
        df_orig = df_orig[df_orig["fuente"].astype(str).str.lower().eq("original")].copy()
        df_syn  = df_syn[~df_syn["fuente"].astype(str).str.lower().eq("original")].copy()

        if df_syn.empty:
            log.warning("[NUM] df_syn quedó vacío tras filtro por fuente; usando TODO el CSV sintético.")
            df_syn = pd.read_csv(syn_path_used)
            if "id_anomalia" not in df_syn.columns and "id" in df_syn.columns:
                df_syn["id_anomalia"] = df_syn["id"]
            if "fuente" not in df_syn.columns:
                df_syn["fuente"] = "sintetica"
            for col in ["series_iri", "anomaly_iri", "point_iri", "serie_name", "serie", "serie_id",
                        "method", "original_detection_method", "grafico_file", "tipo"]:
                if col not in df_syn.columns:
                    df_syn[col] = pd.NA

    # -------------------------------------------------------------------------
    # Pre-define extractores según DTW_MODE
    # -------------------------------------------------------------------------
    if DTW_MODE == "segment":
        extract_orig         = lambda row: _extract_segment_values(
            series_real_df, "original", row.get("start_date"), row.get("end_date"))
        extract_syn_from_series = lambda sdf, row: _extract_segment_values(
            sdf, "sintetica", row.get("start_date"), row.get("end_date"))
    else:
        extract_orig         = lambda row: _extract_context_window(
            series_real_df, "original", row.get("start_date"), row.get("end_date"))
        extract_syn_from_series = lambda sdf, row: _extract_context_window(
            sdf, "sintetica", row.get("start_date"), row.get("end_date"))

    # -------------------------------------------------------------------------
    # ALL_PAIRS
    # -------------------------------------------------------------------------
    run_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    results = []

    syn_total  = len(df_syn)
    orig_total = len(df_orig)
    log.info("[NUM] Totales: syn=%d | orig=%d | ALL_PAIRS=%d", syn_total, orig_total, syn_total * orig_total)

    syn_ok = syn_fail = orig_fail = 0
    pairs_attempted = pairs_with_vectors = pairs_pass = 0
    sims_dtw_all, sims_feat_all, sims_final_all = [], [], []

    for syn_idx, syn_row in df_syn.iterrows():
        syn_id = syn_row.get("id_anomalia", pd.NA)

        # --- IRI sintético según modo ---
        if modo == "historico_vs_reciente":
            # Reutiliza el anomaly_iri que ya viene en el CSV — no hay predicción
            syn_anom_iri   = syn_row.get("anomaly_iri", None)
            syn_series_iri = syn_row.get("series_iri", None)
            if not syn_anom_iri:
                syn_fail += 1
                log.warning("[NUM][RECIENTE] anomaly_iri vacío para syn_id=%r — fila saltada.", syn_id)
                continue
        else:
            # Construye IRI con metodo_prediccion + metodo_deteccion
            syn_anom_iri = _build_syn_anomaly_iri(syn_row)
            if not syn_anom_iri:
                syn_fail += 1
                if syn_fail <= 5:
                    log.warning("[NUM][SYN] No pude construir anomaly_iri (syn_id=%r). "
                                "Revisa: serie_base/metodo_prediccion/metodo_deteccion/start/end.", syn_id)
                continue
            syn_series_iri = _build_syn_series_iri(syn_row)

        # --- Vector sintético según modo ---
        syn_vals = None

        if modo == "historico_vs_reciente":
            # Extrae de SerieOriginal.csv — no hay serie predicha
            if DTW_MODE == "segment":
                syn_vals = _extract_segment_values(
                    series_real_df, "original", syn_row.get("start_date"), syn_row.get("end_date"))
            else:
                syn_vals = _extract_context_window(
                    series_real_df, "original", syn_row.get("start_date"), syn_row.get("end_date"))
        else:
            if DTW_MODE == "segment":
                if "valores" in df_syn.columns:
                    syn_vals = _parse_valores_field(syn_row.get("valores"))
                if syn_vals is None:
                    syn_series_df = _get_syn_series_df(syn_row)
                    if syn_series_df is not None:
                        syn_vals = extract_syn_from_series(syn_series_df, syn_row)
            else:
                syn_series_df = _get_syn_series_df(syn_row)
                if syn_series_df is None:
                    syn_fail += 1
                    if syn_fail <= 5:
                        log.warning("[NUM][SYN] No pude cargar serie predicha para syn_id=%r (modelo=%r).",
                                    syn_id, syn_row.get("metodo_prediccion", syn_row.get("modelo", None)))
                    continue
                syn_vals = extract_syn_from_series(syn_series_df, syn_row)

        if syn_vals is None or syn_vals.size == 0:
            syn_fail += 1
            continue
        syn_ok += 1

        syn_sig  = _normalize_signal(syn_vals.astype(np.float32), DTW_NORMALIZATION)
        syn_feat = _feature_vector(syn_vals.astype(np.float32)) if FEAT_USE else None

        local_rows = []

        for orig_idx, orig_row in df_orig.iterrows():
            pairs_attempted += 1

            orig_vals = extract_orig(orig_row)
            if orig_vals.size == 0:
                orig_fail += 1
                continue

            pairs_with_vectors += 1
            orig_sig = _normalize_signal(orig_vals.astype(np.float32), DTW_NORMALIZATION)

            sim_dtw = _dtw_similarity_0_1(orig_sig, syn_sig)
            sims_dtw_all.append(float(sim_dtw))

            sim_feat = 0.0
            if FEAT_USE:
                orig_feat = _feature_vector(orig_vals.astype(np.float32))
                sim_feat  = _feat_similarity_0_1(orig_feat, syn_feat)
                sims_feat_all.append(float(sim_feat))

            sim_comb  = _combine(sim_dtw, sim_feat if FEAT_USE else 0.0)
            sim_final = sim_comb if SIM_USE_COMBINED else sim_dtw
            sims_final_all.append(float(sim_final))

            if sim_final < CMP_MIN_SIM:
                continue

            pairs_pass += 1

            try:
                sim_iri = f"anom:sim_NUM_{int(orig_row.get('id_anomalia'))}_{int(syn_id)}"
            except Exception:
                sim_iri = f"anom:sim_NUM_{orig_idx}_{syn_idx}"

            local_rows.append({
                "similarity_iri":     sim_iri,
                "rdf_type":           "anom:Similarity",
                "betweenAnomalyA":    orig_row.get("anomaly_iri", pd.NA),
                "betweenAnomalyB":    syn_anom_iri,

                "computedBy_dtw":      COMPUTED_BY_DTW,
                "computedBy_feat":     COMPUTED_BY_FEAT if FEAT_USE else pd.NA,
                "computedBy_combined": COMPUTED_BY_COMB if SIM_USE_COMBINED else COMPUTED_BY_DTW,

                "fromRun":        FROM_RUN,
                "hasScale":       HAS_SCALE,
                "run_timestamp":  run_ts,
                "dtw_modo":       modo,  # campo adicional para trazabilidad

                "similarity_dtw":      float(sim_dtw),
                "similarity_feat":     float(sim_feat) if FEAT_USE else pd.NA,
                "similarity_combined": float(sim_comb),

                "sim_comb_method":  SIM_COMB_METHOD,
                "sim_alpha_dtw":    float(SIM_ALPHA_DTW),
                "passed_threshold": True,
                "threshold_used":   float(CMP_MIN_SIM),
                "score_final_used": "combined" if SIM_USE_COMBINED else "dtw",
                "dtw_mode":         DTW_MODE,

                "id_anomalia_original": orig_row.get("id_anomalia", pd.NA),
                "id_anomalia_sintetica": syn_id,

                "serie":       orig_row.get("serie", pd.NA),
                "serie_id":    orig_row.get("serie_id", pd.NA),
                "serie_name":  orig_row.get("serie_name", pd.NA),
                "series_iri_original":  orig_row.get("series_iri", pd.NA),
                "series_iri_sintetica": syn_series_iri if syn_series_iri else pd.NA,

                "anomaly_iri_original":  orig_row.get("anomaly_iri", pd.NA),
                "anomaly_iri_sintetica": syn_anom_iri,

                "point_iri_original":  orig_row.get("point_iri", pd.NA),
                "point_iri_sintetica": syn_row.get("point_iri", pd.NA),

                "start_date_original":  orig_row.get("start_date", pd.NA),
                "end_date_original":    orig_row.get("end_date", pd.NA),
                "start_date_sintetica": syn_row.get("start_date", pd.NA),
                "end_date_sintetica":   syn_row.get("end_date", pd.NA),

                "method_original":          orig_row.get("method", pd.NA),
                "method_sintetica":         syn_row.get("metodo_deteccion", syn_row.get("method", pd.NA)),
                "original_detection_method":orig_row.get("original_detection_method", orig_row.get("method", pd.NA)),

                "tipo_original":    orig_row.get("tipo", pd.NA),
                "tipo_sintetica":   syn_row.get("tipo", pd.NA),
                "fuente_original":  "original",
                "fuente_sintetica": syn_row.get("fuente", "sintetica"),

                "grafico_file_original":  orig_row.get("grafico_file", pd.NA),
                "grafico_file_sintetica": syn_row.get("grafico_file", pd.NA),

                "len_vector_original":  int(orig_vals.size),
                "len_vector_sintetica": int(syn_vals.size),

                "src_csv_originales":  orig_path_used,
                "src_csv_sinteticas":  syn_path_used,
                "src_serie_original":  serie_path,
                "src_pred_root":       pred_root,
                # FIX: en modo historico_vs_reciente no hay serie predicha en carpetas
                "src_serie_sintetica": None if modo == "historico_vs_reciente"
                                            else _find_syn_series_file(syn_row),
            })

        if local_rows:
            key_name = "similarity_combined" if SIM_USE_COMBINED else "similarity_dtw"
            local_rows.sort(key=lambda r: float(r.get(key_name, 0.0)), reverse=True)
            for rnk, row in enumerate(local_rows, start=1):
                row["rank"] = int(rnk)
                results.append(row)

        if syn_ok <= 3 or (syn_ok % 50 == 0):
            log.info("[NUM] syn_id=%s | vec_ok=%s | matches_pass=%d", syn_id, True, len(local_rows))

    # -------------------------------------------------------------------------
    # Resumen
    # -------------------------------------------------------------------------
    log.info("[NUM] ===== Resumen =====")
    log.info("[NUM] syn_total=%d | orig_total=%d", syn_total, orig_total)
    log.info("[NUM] syn_ok=%d | syn_fail=%d | orig_fail=%d", syn_ok, syn_fail, orig_fail)
    log.info("[NUM] pairs_attempted=%d | pairs_with_vectors=%d | pairs_pass(score>=%.3f)=%d",
             pairs_attempted, pairs_with_vectors, CMP_MIN_SIM, pairs_pass)

    if sims_dtw_all:
        log.info("[NUM] sim_dtw  stats: min=%.6f max=%.6f avg=%.6f",
                 min(sims_dtw_all), max(sims_dtw_all), sum(sims_dtw_all)/len(sims_dtw_all))
    else:
        log.warning("[NUM] No se calculó ninguna sim_dtw.")
    if FEAT_USE and sims_feat_all:
        log.info("[NUM] sim_feat stats: min=%.6f max=%.6f avg=%.6f",
                 min(sims_feat_all), max(sims_feat_all), sum(sims_feat_all)/len(sims_feat_all))
    if sims_final_all:
        log.info("[NUM] sim_final stats: min=%.6f max=%.6f avg=%.6f | final_used=%s",
                 min(sims_final_all), max(sims_final_all), sum(sims_final_all)/len(sims_final_all),
                 "combined" if SIM_USE_COMBINED else "dtw")
    else:
        log.warning("[NUM] No se calculó ninguna sim_final (no hubo pares con vectores).")

    # -------------------------------------------------------------------------
    # Output CSV
    # -------------------------------------------------------------------------
    df_out = pd.DataFrame(results)

    preferred_cols = [
        "similarity_iri", "rdf_type",
        "betweenAnomalyA", "betweenAnomalyB",
        "computedBy_dtw", "computedBy_feat", "computedBy_combined",
        "fromRun", "hasScale", "run_timestamp", "dtw_modo",
        "similarity_dtw", "similarity_feat", "similarity_combined",
        "score_final_used", "dtw_mode",
        "sim_comb_method", "sim_alpha_dtw",
        "rank", "passed_threshold", "threshold_used",
        "id_anomalia_original", "id_anomalia_sintetica",
        "serie", "serie_id", "serie_name",
        "series_iri_original", "series_iri_sintetica",
        "anomaly_iri_original", "anomaly_iri_sintetica",
        "point_iri_original", "point_iri_sintetica",
        "start_date_original", "end_date_original",
        "start_date_sintetica", "end_date_sintetica",
        "method_original", "method_sintetica", "original_detection_method",
        "tipo_original", "tipo_sintetica", "fuente_original", "fuente_sintetica",
        "grafico_file_original", "grafico_file_sintetica",
        "len_vector_original", "len_vector_sintetica",
        "src_csv_originales", "src_csv_sinteticas", "src_serie_original",
        "src_pred_root", "src_serie_sintetica",
    ]

    if not df_out.empty:
        cols = [c for c in preferred_cols if c in df_out.columns] + \
               [c for c in df_out.columns if c not in preferred_cols]
        df_out = df_out[cols].sort_values(
            ["id_anomalia_sintetica", "rank"], ascending=[True, True])
    else:
        df_out = pd.DataFrame(columns=preferred_cols)

    df_out.to_csv(out_csv, index=False)
    log.info("[NUM] CSV generado: %s | filas=%d", out_csv, len(df_out))
    return out_csv

def _export_series():
    """
    Exporta la definición de la serie en formato Turtle (.ttl) según la ontología.
    Utiliza los valores de configuración definidos en settings.py.
    """
    frequency_map = {
        "DIARIA": "daily",
        "SEMANAL": "weekly",
        "MENSUAL": "monthly"
    }

    unit_map = {
        "PUNTOS": "points",
        "PORCENTAJE": "percent"
    }

    freq_en = frequency_map.get(FRECUENCIA.upper(), FRECUENCIA.lower())
    unit_en = unit_map.get(UNIDAD.upper(), UNIDAD.lower())

    ttl = f"""@prefix anom: <http://w3id.org/anomaly-core#> .
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .

{IRI}
  rdf:type anom:Series ;
  anom:seriesName "{SERIE}"@es , "Country Risk"@en ;
  anom:frequency "{freq_en}"^^xsd:string ;
  anom:unit "{unit_en}"^^xsd:string .
"""

    os.makedirs(DATOS_ONTOLOGIA, exist_ok=True)
    output_path = os.path.join(DATOS_ONTOLOGIA, "1series.ttl")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ttl)




def generate_ttl():
    """
    Genera todos los archivos TTL para exportar la ontología de resultados.
    """
    _export_series()
    # Otros: _export_points(), _export_anomalies(), etc.


from datetime import timedelta


def _split_anomalies_by_cutoff(
    anomalias_csv: str,
    num_dias: int,
) -> tuple:
    """
    Lee el CSV de anomalías y las divide por fecha de corte.

    fecha_corte = max(start_date) - num_dias días

    Retorna:
        df_queries  : anomalías con start_date >= fecha_corte  (las "recortadas")
        df_corpus   : anomalías con start_date <  fecha_corte  (las históricas)
        fecha_corte : str YYYY-MM-DD
    """
    df = pd.read_csv(anomalias_csv)
    df["start_date"] = pd.to_datetime(df["start_date"])

    fecha_max   = df["start_date"].max()
    fecha_corte = fecha_max - timedelta(days=num_dias)

    df_queries = df[df["start_date"] >= fecha_corte].copy().reset_index(drop=True)
    df_corpus  = df[df["start_date"] <  fecha_corte].copy().reset_index(drop=True)

    print(f"[EMB_HIST][INFO] fecha_max   = {fecha_max.date()}")
    print(f"[EMB_HIST][INFO] fecha_corte = {fecha_corte.date()}  ({num_dias} días antes)")
    print(f"[EMB_HIST][INFO] queries (recortadas) = {len(df_queries)} anomalías")
    print(f"[EMB_HIST][INFO] corpus  (históricas) = {len(df_corpus)}  anomalías")

    return df_queries, df_corpus, str(fecha_corte.date())


def _find_png_for_anomaly(anomaly_id: int, start_date: str, graficos_dir: str) -> str | None:
    """
    Localiza el PNG de una anomalía en graficos_dir.
    Formato esperado: NNN_YYYY-MM-DD.png

    Estrategia:
      1) Busca por prefijo numérico exacto: {id:03d}_
      2) Fallback: busca por fecha en el nombre
    """
    base   = Path(graficos_dir)
    prefix = f"{int(anomaly_id):03d}_"

    for f in base.iterdir():
        if f.suffix.lower() == ".png" and f.name.startswith(prefix):
            return str(f.resolve())

    # Fallback por fecha
    date_str = str(start_date)[:10]
    for f in base.iterdir():
        if f.suffix.lower() == ".png" and date_str in f.name:
            return str(f.resolve())

    return None


def run_embeddings_comparison_hist(CFG) -> dict:
    """
    Orquesta la comparación visual histórica para el experimento de validación.

    Flujo:
      1) Separa anomalías en queries (recortadas) y corpus (históricas)
         usando CFG.NUM_DIAS_ANOMALIAS como ventana de corte.
      2) Calcula embeddings CLIP para ambos grupos leyendo los PNGs
         desde CFG.RUTA_GRAFICOS (formato NNN_YYYY-MM-DD.png).
      3) Para cada query busca las top-K históricas más similares.
      4) Guarda parquets, CSV de similitudes y láminas visuales
         en CFG.RUTA_COMPARACION_HIST / laminas/.

    Reutiliza internamente: _load_img_model, _cosine_topk,
    compare_synthetic_against_originals, generate_contact_sheets_per_synthetic.
    """
    import os

    anomalias_csv = CFG.ANOMALIAS_CONSOLIDADO
    graficos_dir  = CFG.RUTA_GRAFICOS
    out_root      = CFG.RUTA_COMPARACION_HIST
    num_dias      = CFG.NUM_DIAS_ANOMALIAS

    os.makedirs(out_root, exist_ok=True)

    print(f"[EMB_HIST][INFO] anomalias_csv = {anomalias_csv}")
    print(f"[EMB_HIST][INFO] graficos_dir  = {graficos_dir}")
    print(f"[EMB_HIST][INFO] out_root      = {out_root}")
    print(f"[EMB_HIST][INFO] num_dias      = {num_dias}")

    # ── 1) Separar anomalías ─────────────────────────────────────────────────
    df_queries, df_corpus, fecha_corte = _split_anomalies_by_cutoff(
        anomalias_csv = anomalias_csv,
        num_dias      = num_dias,
    )

    if df_queries.empty:
        print("[EMB_HIST][WARN] No hay anomalías en el periodo recortado → abortando.")
        return {}

    if df_corpus.empty:
        print("[EMB_HIST][WARN] No hay anomalías históricas anteriores al corte → abortando.")
        return {}

    # ── 2) Calcular embeddings y construir DataFrames ────────────────────────
    model = _load_img_model(CFG.CMP_MODEL_NAME)

    def _embed_anomaly_list(df_anom: pd.DataFrame, role: str) -> pd.DataFrame:
        rows = []
        missing = []
        for _, row in df_anom.iterrows():
            anom_id    = int(row["id"])
            start_date = str(row["start_date"])[:10]
            png_path   = _find_png_for_anomaly(anom_id, start_date, graficos_dir)

            if png_path is None:
                missing.append(anom_id)
                print(f"[EMB_HIST][WARN] PNG no encontrado id={anom_id} fecha={start_date}")
                continue

            img = Image.open(png_path).convert("RGB")
            emb = model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
            rows.append({
                "file_path": png_path,
                "file_name": Path(png_path).name,
                "folder":    str(Path(graficos_dir).resolve()),
                "vec":       emb.astype("float32").tolist(),
                # columnas extra para las láminas
                "anomaly_id":   anom_id,
                "start_date":   start_date,
            })

        print(f"[EMB_HIST][INFO] {role}: {len(rows)} indexados, {len(missing)} sin PNG")
        return pd.DataFrame(rows)

    df_q = _embed_anomaly_list(df_queries, "queries")
    df_c = _embed_anomaly_list(df_corpus,  "corpus")

    # ── 3) Guardar parquets ──────────────────────────────────────────────────
    q_parquet = os.path.join(out_root, "queries_hist.parquet")
    c_parquet = os.path.join(out_root, "corpus_hist.parquet")
    df_q.to_parquet(q_parquet, index=False)
    df_c.to_parquet(c_parquet, index=False)

    # ── 4) Comparar (reutiliza la función ya existente) ──────────────────────
    out_csv = os.path.join(out_root, "similaridades_embeddings_hist.csv")
    compare_synthetic_against_originals(
        queries_parquet     = q_parquet,
        corpus_parquet      = c_parquet,
        out_csv             = out_csv,
        cmp_min_sim         = CFG.CMP_MIN_SIM,
        topk_per_synthetic  = CFG.CMP_TOPK_PER_SINTETICA,
        model_name          = CFG.CMP_MODEL_NAME,
        model_weights       = getattr(CFG, "CMP_MODEL_WEIGHTS", ""),
    )

    # ── 5) Láminas (reutiliza la función ya existente) ───────────────────────
    laminas_dir = os.path.join(out_root, "laminas")
    generate_contact_sheets_per_synthetic(
        results_csv        = out_csv,
        output_dir         = laminas_dir,
        topk_per_synthetic = CFG.CMP_TOPK_PER_SINTETICA,
        gap_px             = CFG.CMP_CANVAS_GAP_PX,
        thumb_width_px     = CFG.CMP_THUMB_WIDTH_PX,
        show_labels        = CFG.CMP_SHOW_LABELS,
        show_percent       = CFG.CMP_SHOW_PERCENT,
    )

    return {
        "queries_parquet": q_parquet,
        "corpus_parquet":  c_parquet,
        "csv":             out_csv,
        "laminas_dir":     laminas_dir,
        "fecha_corte":     fecha_corte,
        "n_queries":       len(df_q),
        "n_corpus":        len(df_c),
    }


def comparar_anomalias_numericas_historicas_dtw(CFG) -> str:
    """
    Comparación numérica DTW para el experimento de VALIDACIÓN histórica.
    
    Divide anomalias_consolidado.csv en dos grupos por fecha de corte:
      - Históricas : start_date <  (max_date - NUM_DIAS_ANOMALIAS)  → corpus
      - Recientes  : start_date >= (max_date - NUM_DIAS_ANOMALIAS)  → queries
    
    Guarda resultados en CFG.RUTA_COMPARACION_HIST/similaridades_dtw.csv
    
    NO toca comparar_anomalias_numericas_predichas ni su ruta de salida.
    """
    import os
    import logging
    import pandas as pd

    log = logging.getLogger(__name__)

    # ── Rutas ─────────────────────────────────────────────────────────────────
    num_dias  = int(getattr(CFG, "NUM_DIAS_ANOMALIAS", 270))
    ruta_cmp  = getattr(CFG, "RUTA_COMPARACION_HIST",
                        os.path.join(CFG.RUTA_RESULTADOS, "ComparacionesHist"))
    
    # Entrada: anomalias_consolidado.csv (no el enriquecido ni el de predicción)
    ruta_anom = getattr(CFG, "ANOMALIAS_CONSOLIDADO",
                        os.path.join(CFG.RUTA_RESULTADOS, "anomalias_consolidado.csv"))
    ruta_serie = os.path.join(CFG.RUTA_SALIDA, "SerieOriginal.csv")

    os.makedirs(ruta_cmp, exist_ok=True)

    log.info("[HIST-DTW] ========== Configuración ==========")
    log.info("[HIST-DTW] NUM_DIAS_ANOMALIAS   : %d", num_dias)
    log.info("[HIST-DTW] RUTA_COMPARACION_HIST: %s", ruta_cmp)
    log.info("[HIST-DTW] anomalias_consolidado: %s | existe=%s", 
             ruta_anom, os.path.isfile(ruta_anom))
    log.info("[HIST-DTW] SerieOriginal.csv     : %s | existe=%s",
             ruta_serie, os.path.isfile(ruta_serie))

    if not os.path.isfile(ruta_anom):
        raise FileNotFoundError(f"[HIST-DTW] No encontrado: {ruta_anom}")
    if not os.path.isfile(ruta_serie):
        raise FileNotFoundError(f"[HIST-DTW] No encontrado: {ruta_serie}")

    # ── Leer y dividir anomalías ───────────────────────────────────────────────
    df_all = pd.read_csv(ruta_anom)
    df_all["start_date"] = pd.to_datetime(df_all["start_date"], errors="coerce")
    df_all = df_all.dropna(subset=["start_date"]).sort_values("start_date").reset_index(drop=True)

    fecha_max   = df_all["start_date"].max()
    fecha_corte = fecha_max - pd.Timedelta(days=num_dias)

    df_historico = df_all[df_all["start_date"] <  fecha_corte].copy()
    df_reciente  = df_all[df_all["start_date"] >= fecha_corte].copy()

    log.info("[HIST-DTW] fecha_max    = %s", fecha_max.date())
    log.info("[HIST-DTW] fecha_corte  = %s", fecha_corte.date())
    log.info("[HIST-DTW] Histórico    : %d anomalías", len(df_historico))
    log.info("[HIST-DTW] Reciente     : %d anomalías", len(df_reciente))

    print(f"[HIST-DTW] fecha_corte = {fecha_corte.date()} | "
          f"histórico={len(df_historico)} | reciente={len(df_reciente)}")

    if df_historico.empty:
        raise ValueError(f"[HIST-DTW] No hay anomalías históricas antes de {fecha_corte.date()}. "
                         f"Aumenta NUM_DIAS_ANOMALIAS (actual={num_dias}).")
    if df_reciente.empty:
        raise ValueError(f"[HIST-DTW] No hay anomalías recientes desde {fecha_corte.date()}. "
                         f"Reduce NUM_DIAS_ANOMALIAS (actual={num_dias}).")

    # ── Enriquecer con columnas que espera el motor DTW ───────────────────────
    # anomalias_consolidado.csv tiene: id, start_date, end_date, count, method
    # El motor DTW necesita: id_anomalia, anomaly_iri, fuente
    for df_, fuente in [(df_historico, "original"), (df_reciente, "original")]:
        if "id_anomalia" not in df_.columns and "id" in df_.columns:
            df_["id_anomalia"] = df_["id"]
        if "fuente" not in df_.columns:
            df_["fuente"] = fuente
        # anomaly_iri mínimo para que el motor no falle
        if "anomaly_iri" not in df_.columns:
            df_["anomaly_iri"] = df_["id_anomalia"].apply(
                lambda x: f"anom:anomaly_hist_{int(x)}" if pd.notna(x) else pd.NA
            )
        for col in ["series_iri", "point_iri", "serie_name", "serie",
                    "serie_id", "original_detection_method", "grafico_file", "tipo"]:
            if col not in df_.columns:
                df_[col] = pd.NA

    # ── Crear CFG parcheado que apunte a RUTA_COMPARACION_HIST ───────────────
    # Se crea una copia liviana del CFG con RUTA_COMPARACION sobreescrita,
    # así el motor DTW escribe en la ruta correcta sin tocar el CFG real.
    class _CFGPatch:
        def __init__(self, base, ruta_cmp_override):
            self._base = base
            self._override = ruta_cmp_override
        def __getattr__(self, name):
            if name == "RUTA_COMPARACION":
                return self._override
            return getattr(self._base, name)

    cfg_patch = _CFGPatch(CFG, ruta_cmp)

    # ── Llamar al motor DTW con modo historico_vs_reciente ────────────────────
    # Se reutiliza la lógica interna sin modificar la función de predicción.
    out_csv = comparar_anomalias_numericas_predichas(
        cfg_patch,
        modo="historico_vs_reciente"
    )

    log.info("[HIST-DTW] CSV generado : %s | existe=%s", out_csv, os.path.isfile(out_csv))
    print(f"[HIST-DTW] CSV generado : {out_csv}")

    return out_csv