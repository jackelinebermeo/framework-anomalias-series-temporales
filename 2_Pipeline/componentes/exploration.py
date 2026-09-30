# componentes/exploracion.py
# --------------------------------------------------------
# Clase encargada del preprocesamiento y análisis exploratorio de la serie temporal.
# Incluye: lectura, limpieza, imputación, visualización y análisis estacional.
# --------------------------------------------------------

import os
import pandas as pd
import numpy as np
import json 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from componentes.settings import (
    RUTA_ARCHIVO,
    SERIE_LIMPIA,
    GRAFICO_KDE,
    GRAFICO_SERIE,
    RUTA_ANOMALIAS_METODO
)

class ExploracionSerie:

    def read_file(self) -> str:
        import os
        import numpy as np
        import pandas as pd
        from io import StringIO

        # ✅ LOGS DE DIAGNÓSTICO
        print(f"[READ] RUTA_ARCHIVO: {RUTA_ARCHIVO}")
        print(f"[READ] Archivo existe: {os.path.isfile(RUTA_ARCHIVO)}")
        print(f"[READ] Tamaño archivo: {os.path.getsize(RUTA_ARCHIVO) if os.path.isfile(RUTA_ARCHIVO) else 'N/A'} bytes")
        print(f"[READ] SERIE_LIMPIA: {SERIE_LIMPIA}")
        print(f"[READ] Carpeta salida existe: {os.path.isdir(os.path.dirname(SERIE_LIMPIA))}")

        os.makedirs(os.path.dirname(SERIE_LIMPIA), exist_ok=True)

        print(f"[READ] Iniciando lectura del Excel...")
        df = pd.read_excel(RUTA_ARCHIVO, engine="openpyxl")
        print(f"[READ] Excel leído OK: {len(df)} filas, columnas: {list(df.columns)}")


        # 1) ✅ engine openpyxl explícito — más rápido y no se cuelga
        df = pd.read_excel(RUTA_ARCHIVO, engine="openpyxl")
        df.columns = [c.strip().lower() for c in df.columns]

        if "fecha" not in df.columns or "valor" not in df.columns:
            raise ValueError("El archivo debe contener columnas 'fecha' y 'valor'.")

        # 2) Parseo de fecha
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = df.dropna(subset=["fecha"]).sort_values("fecha")
        print(f"✅ Archivo leído con {len(df)} filas")

        # 3) ✅ Conversión vectorizada en vez de apply fila por fila
        valor = df["valor"].astype(str).str.strip()
        # Manejo '1.234,56' -> '1234.56'
        mask_ambos = valor.str.contains(",") & valor.str.contains("\\.")
        valor = valor.where(~mask_ambos, valor.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
        valor = valor.str.replace(",", ".", regex=False)
        # Dejar solo caracteres numéricos válidos
        valor = valor.str.replace(r"[^\d.\-+eE]", "", regex=True)
        df["valor"] = pd.to_numeric(valor, errors="coerce")

        # 4) Indexar diario
        df = df.set_index("fecha").asfreq("D")
        df["fuente"] = df.get("fuente", "original")
        df["fuente"] = df["fuente"].fillna("original")

        # 5) Saneos
        df.loc[~np.isfinite(df["valor"]), "valor"] = np.nan
        df.loc[df["valor"].abs() > 1e10, "valor"] = np.nan

        # 6) Imputación
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        df.interpolate(method="linear", inplace=True)
        df.dropna(subset=["valor"], inplace=True)
        df["valor"] = df["valor"].astype(int)

        # 7) Salida
        df.to_csv(SERIE_LIMPIA)
        return df.to_json(date_format="iso")
    def describe(self, json_df: str):
        from io import StringIO
        import pandas as pd
        df = pd.read_json(StringIO(json_df))
        print(df.describe())
            
    def plot_kde(self, json_df: str):
        """Genera el gráfico de distribución KDE de los valores."""
        df = pd.read_json(json_df)
        sns.kdeplot(df['valor'], fill=True)
        plt.axvline(df['valor'].mean(), color='red', linestyle=':', label='Media')
        plt.title("Distribución de valores")
        plt.savefig(GRAFICO_KDE)
        plt.close()

    def plot_series(self, json_df: str):
        """Genera un gráfico de línea de la serie temporal."""
        df = pd.read_json(json_df)
        df.index = pd.to_datetime(df.index)
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['valor'])
        plt.title("Serie Temporal")
        plt.savefig(GRAFICO_SERIE)
        plt.close()


    def analyze_seasonality(self, json_df: str) -> str:
        df = pd.read_json(json_df)
        df.index = pd.to_datetime(df.index)
        y = df["valor"]

        if y.isna().any():
            raise ValueError("❌ No se puede analizar estacionalidad con valores nulos.")

        # --- Inferir frecuencia y definir candidatos ---
        # Intento 1: freq explícita en el índice
        freq = pd.infer_freq(y.index) or y.index.inferred_freq
        # Paso simple: usar la mediana del paso si no se infiere
        if not freq:
            step = (y.index.to_series().diff().median()).total_seconds()
            # heurística grosera
            if step <= 3600:          # ~1h
                freq = "H"
            elif step <= 86400:       # ~1d
                freq = "D"
            else:
                freq = "M"            # ~mensual/irregular

        if freq.startswith("H"):
            candidatos = [24, 24*7]           # diario y semanal en horas
        elif freq.startswith("D"):
            candidatos = [7, 30, 365]         # semanal, mensual aprox, anual
        elif freq.startswith(("W", "2W")):
            candidatos = [52]                 # anual en semanas
        elif freq.startswith(("M", "MS")):
            candidatos = [12]                 # anual en meses
        elif freq.startswith(("Q", "QS")):
            candidatos = [4]                  # anual en trimestres
        elif freq.startswith(("A","Y")):
            candidatos = [1]                  # sin estacionalidad útil
        else:
            # Irregular: re-muestrear suavemente a diario para testear 7/30/365
            y = y.asfreq("D").interpolate("time")
            candidatos = [7, 30, 365]
            freq = "D"

        # --- Métrica de fuerza (Hyndman) ---
        def fuerza_estacionalidad(resid, seasonal):
            r = pd.Series(resid).dropna().to_numpy()
            s = pd.Series(seasonal).dropna().to_numpy()
            if r.size < 2 or s.size < 2:
                return 0.0
            var_r = float(np.nanvar(r, ddof=1))
            var_sum = float(np.nanvar(r + s, ddof=1))
            if var_sum <= 0:
                return 0.0
            return max(0.0, 1.0 - var_r / var_sum)

        # --- Evaluar candidatos (aditivo y multiplicativo) ---
        evals = []
        mejor = {"periodo": None, "fuerza": -1.0, "modelo": None, "result": None}

        for p in candidatos:
            for modelo in ("additive", "multiplicative"):
                try:
                    # STL es más robusto; si falla, intento seasonal_decompose
                    try:
                        res = STL(y, period=p, robust=True).fit()
                        seasonal, resid, trend = res.seasonal, res.resid, res.trend
                    except Exception:
                        decomp = seasonal_decompose(y, model=modelo, period=p, extrapolate_trend="freq")
                        seasonal, resid, trend = decomp.seasonal, decomp.resid, decomp.trend

                    f = fuerza_estacionalidad(resid, seasonal)
                    evals.append({"periodo": p, "modelo": modelo, "fuerza": float(round(f, 4))})

                    if f > mejor["fuerza"]:
                        mejor = {"periodo": p, "fuerza": f, "modelo": modelo,
                                "result": (seasonal, resid, trend)}
                except Exception:
                    evals.append({"periodo": p, "modelo": modelo, "fuerza": 0.0})

        # Si nada funcionó, salgo limpio
        if mejor["result"] is None:
            out_dir = os.path.dirname(GRAFICO_SERIE)
            os.makedirs(out_dir, exist_ok=True)
            resumen = {
                "estacionalidad_detectada": False,
                "periodo_sugerido": None,
                "modelo": None,
                "fuerza_estacionalidad": 0.0,
                "evaluaciones": evals,
                "frecuencia_inferida": freq,
                "grafico_path": None
            }
            path_json = os.path.join(out_dir, "estacionalidad.json")
            with open(path_json, "w", encoding="utf-8") as f:
                json.dump(resumen, f, ensure_ascii=False, indent=2)
            return json.dumps(resumen, ensure_ascii=False)

        # --- Graficar grande y guardar ---
        seasonal, resid, trend = mejor["result"]
        out_dir = os.path.dirname(GRAFICO_SERIE)
        os.makedirs(out_dir, exist_ok=True)

        fig, axes = plt.subplots(4, 1, figsize=(12.5, 9.0), sharex=True)
        axes[0].plot(y.index, y.values); axes[0].set_title("Valor")
        axes[1].plot(trend.index, trend.values); axes[1].set_title("Tendencia")
        axes[2].plot(seasonal.index, seasonal.values); axes[2].set_title("Estacional")
        axes[3].plot(resid.index, resid.values); axes[3].set_title("Residuo")
        plt.suptitle(f"Descomposición (freq={freq}, periodo={mejor['periodo']}, modelo={mejor['modelo']})", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        path_png = os.path.join(out_dir, "descomposicion.png")
        plt.savefig(path_png, dpi=130)
        plt.close(fig)

        # --- Resumen JSON ---
        fuerza = float(round(mejor["fuerza"], 4))
        resumen = {
            "estacionalidad_detectada": bool(fuerza > 0.05),
            "periodo_sugerido": int(mejor["periodo"]) if mejor["periodo"] else None,
            "modelo": mejor["modelo"],
            "fuerza_estacionalidad": fuerza,
            "evaluaciones": evals,
            "frecuencia_inferida": freq,
            "grafico_path": path_png
        }
        path_json = os.path.join(out_dir, "estacionalidad.json")
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(resumen, f, ensure_ascii=False, indent=2)

        return json.dumps(resumen, ensure_ascii=False)
        
def normalize_series(self, json_df: str, robusto: bool = False) -> str:
    import os, json
    from io import StringIO
    import pandas as pd

    df = pd.read_json(StringIO(json_df))
    df.index = pd.to_datetime(df.index)

    if 'valor_raw' not in df.columns:
        df['valor_raw'] = df['valor']

    if robusto:
        mediana = df['valor'].median()
        q1 = df['valor'].quantile(0.25)
        q3 = df['valor'].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            iqr = df['valor'].std(ddof=1)
        if iqr == 0 or pd.isna(iqr):
            iqr = 1.0
        df['valor'] = (df['valor'] - mediana) / iqr
    else:
        media = df['valor'].mean()
        desv = df['valor'].std(ddof=1)
        if desv == 0 or pd.isna(desv):
            desv = 1.0
        df['valor'] = (df['valor'] - media) / desv

    return df.to_json(date_format='iso')    
def analisis_pre_arima(self, json_df: str, modelo: str, out_dir: str = None) -> str:
    import os, io, json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import acf as _acf
    from componentes.settings import RUTA_ANOMALIAS_METODO

    df = pd.read_json(io.StringIO(json_df))
    df.index = pd.to_datetime(df.index)
    s = df['valor'].astype(float).dropna().copy()
    s = s.asfreq('D')

    if out_dir is None:
        out_dir = os.path.join(RUTA_ANOMALIAS_METODO, modelo)
    os.makedirs(out_dir, exist_ok=True)

    def _safe_adf(x):
        x = pd.Series(x).dropna()
        if len(x) < 20:
            return {"stat": np.nan, "pvalue": np.nan}
        try:
            res = adfuller(x, autolag='AIC')
            return {"stat": float(res[0]), "pvalue": float(res[1])}
        except Exception:
            return {"stat": np.nan, "pvalue": np.nan}

    def _safe_kpss_test(x):
        x = pd.Series(x).dropna()
        if len(x) < 20:
            return {"stat": np.nan, "pvalue": np.nan}
        try:
            stat, p, *_ = kpss(x, regression='c', nlags='auto')
            return {"stat": float(stat), "pvalue": float(p)}
        except Exception:
            return {"stat": np.nan, "pvalue": np.nan}

    # ✅ NUEVO: salida temprana — solo calcula d1/d2 si es necesario
    adf_0  = _safe_adf(s.values)
    kpss_0 = _safe_kpss_test(s.values)

    s_d1   = s.diff().dropna()
    adf_1  = _safe_adf(s_d1.values)
    kpss_1 = _safe_kpss_test(s_d1.values)

    # Solo calcula d2 si d1 tampoco es estacionaria
    if not (adf_1.get("pvalue", 1) < 0.05):
        s_d2   = s_d1.diff().dropna()
        adf_2  = _safe_adf(s_d2.values)
        kpss_2 = _safe_kpss_test(s_d2.values)
    else:
        s_d2   = s_d1.diff().dropna()
        adf_2  = {"stat": np.nan, "pvalue": np.nan}
        kpss_2 = {"stat": np.nan, "pvalue": np.nan}

    # Sugerencia de d
    d_sugerido = 0
    if not (adf_0["pvalue"] < 0.05):
        d_sugerido = 1
        if not (adf_1["pvalue"] < 0.05):
            d_sugerido = 2
    serie_diff = {0: s, 1: s_d1, 2: s_d2}[d_sugerido]

    # ACF
    max_lag = min(365, max(10, len(serie_diff) // 3))
    acf_vals = _acf(serie_diff.dropna(), nlags=max_lag, fft=True)
    candidatos = [7, 30, 365]
    estacionalidad = sorted(
        [{"lag": lag, "acf": float(acf_vals[lag])} for lag in candidatos if lag <= max_lag],
        key=lambda x: -abs(x["acf"])
    )

    # ✅ NUEVO: dpi reducido de 150 a 100
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(2, 1, 1)
    plot_acf(serie_diff.dropna(), ax=ax1, lags=min(60, max_lag))
    ax1.set_title(f"ACF (serie diferenciada d={d_sugerido})")
    ax2 = plt.subplot(2, 1, 2)
    plot_pacf(serie_diff.dropna(), ax=ax2, lags=min(60, max_lag), method='ywm')
    ax2.set_title(f"PACF (serie diferenciada d={d_sugerido})")
    plt.tight_layout()
    ruta_graf = os.path.join(out_dir, f"arima_acf_pacf_d{d_sugerido}.png")
    plt.savefig(ruta_graf, dpi=100)
    plt.close(fig)

    resumen = {
        "adf":  {"original": adf_0,  "d1": adf_1,  "d2": adf_2},
        "kpss": {"original": kpss_0, "d1": kpss_1, "d2": kpss_2},
        "d_sugerido": int(d_sugerido),
        "pistas_estacionalidad": estacionalidad[:3],
        "ruta_grafico_acf_pacf": ruta_graf,
    }

    ruta_rep = os.path.join(out_dir, "reporte_analisis_pre_arima.txt")
    with open(ruta_rep, "w", encoding="utf-8") as f:
        f.write("ANÁLISIS PREVIO PARA ARIMA\n===========================\n\n")
        f.write(f"Tamaño serie: {len(s)}\n")
        f.write("\nPruebas ADF (p<0.05 sugiere estacionariedad):\n")
        f.write(f" - Original: p={adf_0['pvalue']:.4f}\n")
        f.write(f" - d=1:      p={adf_1['pvalue']:.4f}\n")
        f.write(f" - d=2:      p={adf_2['pvalue']:.4f}\n")
        f.write("\nPruebas KPSS (p>0.05 sugiere estacionariedad):\n")
        f.write(f" - Original: p={kpss_0['pvalue']:.4f}\n")
        f.write(f" - d=1:      p={kpss_1['pvalue']:.4f}\n")
        f.write(f" - d=2:      p={kpss_2['pvalue']:.4f}\n")
        f.write(f"\nSugerencia de d (ADF): d={d_sugerido}\n")
        if estacionalidad:
            f.write("\nPistas de estacionalidad (ACF por lag):\n")
            for e in estacionalidad[:3]:
                f.write(f" - lag {e['lag']}: ACF={e['acf']:.3f}\n")
        f.write(f"\nGráfico ACF/PACF: {ruta_graf}\n")

    with open(os.path.join(out_dir, "resumen_pre_arima.json"), "w", encoding="utf-8") as fjs:
        json.dump(resumen, fjs, ensure_ascii=False, indent=2)

    return json_df    
def analyze_seasonality(self, json_df: str) -> str:
    df = pd.read_json(json_df)
    df.index = pd.to_datetime(df.index)
    y = df["valor"]
    if y.isna().any():
        raise ValueError("❌ No se puede analizar estacionalidad con valores nulos.")

    # --- Inferir frecuencia ---
    freq = pd.infer_freq(y.index) or y.index.inferred_freq
    if not freq:
        step = (y.index.to_series().diff().median()).total_seconds()
        if step <= 3600:
            freq = "H"
        elif step <= 86400:
            freq = "D"
        else:
            freq = "M"

    if freq.startswith("H"):
        candidatos = [24, 24*7]
    elif freq.startswith("D"):
        candidatos = [7, 30, 365]
    elif freq.startswith(("W", "2W")):
        candidatos = [52]
    elif freq.startswith(("M", "MS")):
        candidatos = [12]
    elif freq.startswith(("Q", "QS")):
        candidatos = [4]
    elif freq.startswith(("A", "Y")):
        candidatos = [1]
    else:
        y = y.asfreq("D").interpolate("time")
        candidatos = [7, 30, 365]
        freq = "D"

    # ✅ NUEVO: series con ceros/negativos no admiten multiplicativo
    tiene_no_positivos = bool((y <= 0).any())

    # --- Métrica de fuerza (sin cambios) ---
    def fuerza_estacionalidad(resid, seasonal):
        r = pd.Series(resid).dropna().to_numpy()
        s = pd.Series(seasonal).dropna().to_numpy()
        if r.size < 2 or s.size < 2:
            return 0.0
        var_r = float(np.nanvar(r, ddof=1))
        var_sum = float(np.nanvar(r + s, ddof=1))
        if var_sum <= 0:
            return 0.0
        return max(0.0, 1.0 - var_r / var_sum)

    # --- Evaluar candidatos ---
    evals = []
    mejor = {"periodo": None, "fuerza": -1.0, "modelo": None, "result": None}

    for p in candidatos:
        # ✅ NUEVO: salir temprano si ya encontramos estacionalidad fuerte
        if mejor["fuerza"] > 0.8:
            break

        modelos = ["additive"]
        if not tiene_no_positivos:
            modelos.append("multiplicative")

        for modelo in modelos:
            try:
                try:
                    # ✅ NUEVO: inner_iter y outer_iter reducidos para velocidad
                    res = STL(
                        y, period=p, robust=False,
                        inner_iter=2, outer_iter=1
                    ).fit()
                    seasonal, resid, trend = res.seasonal, res.resid, res.trend
                except Exception:
                    decomp = seasonal_decompose(
                        y, model=modelo, period=p, extrapolate_trend="freq"
                    )
                    seasonal, resid, trend = decomp.seasonal, decomp.resid, decomp.trend

                f = fuerza_estacionalidad(resid, seasonal)
                evals.append({"periodo": p, "modelo": modelo, "fuerza": float(round(f, 4))})

                if f > mejor["fuerza"]:
                    mejor = {"periodo": p, "fuerza": f, "modelo": modelo,
                             "result": (seasonal, resid, trend)}

            except Exception:
                evals.append({"periodo": p, "modelo": modelo, "fuerza": 0.0})

    # --- Sin resultado ---
    if mejor["result"] is None:
        out_dir = os.path.dirname(GRAFICO_SERIE)
        os.makedirs(out_dir, exist_ok=True)
        resumen = {
            "estacionalidad_detectada": False,
            "periodo_sugerido": None,
            "modelo": None,
            "fuerza_estacionalidad": 0.0,
            "evaluaciones": evals,
            "frecuencia_inferida": freq,
            "grafico_path": None
        }
        path_json = os.path.join(out_dir, "estacionalidad.json")
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(resumen, f, ensure_ascii=False, indent=2)
        return json.dumps(resumen, ensure_ascii=False)

    # --- Graficar ---
    seasonal, resid, trend = mejor["result"]
    out_dir = os.path.dirname(GRAFICO_SERIE)
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(12.5, 9.0), sharex=True)
    axes[0].plot(y.index, y.values);        axes[0].set_title("Valor")
    axes[1].plot(trend.index, trend.values); axes[1].set_title("Tendencia")
    axes[2].plot(seasonal.index, seasonal.values); axes[2].set_title("Estacional")
    axes[3].plot(resid.index, resid.values); axes[3].set_title("Residuo")
    plt.suptitle(
        f"Descomposición (freq={freq}, periodo={mejor['periodo']}, modelo={mejor['modelo']})",
        fontsize=14
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path_png = os.path.join(out_dir, "descomposicion.png")
    # ✅ NUEVO: dpi reducido de 130 a 100 (menos tiempo de escritura)
    plt.savefig(path_png, dpi=100)
    plt.close(fig)

    # --- Resumen JSON ---
    fuerza = float(round(mejor["fuerza"], 4))
    resumen = {
        "estacionalidad_detectada": bool(fuerza > 0.05),
        "periodo_sugerido": int(mejor["periodo"]) if mejor["periodo"] else None,
        "modelo": mejor["modelo"],
        "fuerza_estacionalidad": fuerza,
        "evaluaciones": evals,
        "frecuencia_inferida": freq,
        "grafico_path": path_png
    }
    path_json = os.path.join(out_dir, "estacionalidad.json")
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)
    return json.dumps(resumen, ensure_ascii=False)