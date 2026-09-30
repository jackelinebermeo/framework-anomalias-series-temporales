# componentes/prediction.py
from __future__ import annotations

# --- Stdlib
import os
import io
import json
import warnings
from typing import Dict, Optional, Tuple
from pathlib import Path

# --- Third-party (ligeros y transversales)|
import numpy as np
import pandas as pd

# --- Plotting (headless/servidor)
import matplotlib
matplotlib.use("Agg")                 # asegura backend sin display antes de importar pyplot

# --- Time series (statsmodels)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# --- Config local
from componentes import settings as CFG

# --- Paths globales derivados de settings
pred_root = Path(getattr(CFG, "RUTA_PREDICCION",
                         getattr(CFG, "RUTA_SALIDA", "/home/jacky")))

# --- Higiene de warnings (evita ruido en logs de Airflow)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- Lazy import helpers para dependencias pesadas (se usarán en métodos)
def _tf():
    """
    Carga perezosa de TensorFlow. Úsala dentro de runners que lo requieran.
    Ejemplo: tf = _tf(); model = tf.keras.Sequential(...)
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silencia INFO/DEBUG de TF
    import tensorflow as tf  # noqa: F401
    try:
        tf.get_logger().setLevel("ERROR")
    except Exception:
        pass
    return tf

def _sklearn():
    """
    Paquete perezoso de símbolos habituales de scikit-learn.
    Usa solo lo que necesites: s = _sklearn(); StandardScaler = s["StandardScaler"]
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    return {
        "StandardScaler": StandardScaler,
        "MultiOutputRegressor": MultiOutputRegressor,
        "Pipeline": Pipeline,
        "SVR": SVR,
        "RandomForestRegressor": RandomForestRegressor,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Helpers 
# ─────────────────────────────────────────────────────────────────────────────
#arima
def _arima_best_order(y_train: pd.Series, trend: str, maxiter: int) -> tuple:
    """Busca el mejor orden ARIMA por AIC entre candidatos comunes."""
    import warnings
    from statsmodels.tsa.arima.model import ARIMA

    candidates = [
        (1, 1, 1), (1, 1, 2), (2, 1, 1), (2, 1, 2),
        (1, 2, 1), (2, 2, 1), (0, 1, 1), (0, 2, 1),
    ]

    best_aic   = np.inf
    best_order = (1, 1, 1)
    best_res   = None

    for candidate in candidates:
        try:
            p, d, q = candidate
            t = "n" if d >= 1 else trend
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = ARIMA(y_train, order=candidate, trend=t,
                          enforce_stationarity=False, enforce_invertibility=False)
                r = m.fit(low_memory=True, method_kwargs={"maxiter": maxiter})
            print(f"[ARIMA] orden={candidate} AIC={r.aic:.2f}")
            if r.aic < best_aic:
                best_aic   = r.aic
                best_order = candidate
                best_res   = r
        except Exception as e:
            print(f"[ARIMA] orden={candidate} falló: {e}")
            continue

    print(f"[ARIMA] Mejor orden={best_order} AIC={best_aic:.2f}")
    return best_order, best_res

def _arima_best_order_feedback(y_train: pd.Series, trend: str, maxiter: int) -> tuple:
    """Grid search específico para FEEDBACK - incluye órdenes con d=0 que pueden oscilar."""
    candidates = [
        (2, 0, 2), (3, 0, 2), (2, 0, 1), (3, 0, 1),
        (4, 0, 2), (2, 0, 3), (3, 0, 3),
        (1, 1, 2), (2, 1, 2),  # también evalúa d=1 para comparar
    ]

    best_aic = np.inf
    best_order = (2, 0, 2)
    best_res   = None

    for candidate in candidates:
        try:
            p, d, q = candidate
            t = "n" if d >= 1 else trend
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = ARIMA(y_train, order=candidate, trend=t,
                          enforce_stationarity=False, enforce_invertibility=False)
                r = m.fit(low_memory=True, method_kwargs={"maxiter": maxiter})
            print(f"[ARIMA_FB] orden={candidate} AIC={r.aic:.2f}")
            if r.aic < best_aic:
                best_aic   = r.aic
                best_order = candidate
                best_res   = r
        except Exception as e:
            print(f"[ARIMA_FB] orden={candidate} falló: {e}")
            continue

    print(f"[ARIMA_FB] Mejor orden={best_order} AIC={best_aic:.2f}")
    return best_order, best_res

#svr
def _build_lag_matrix(df_in: pd.DataFrame, lags: int = 64):
    """
    Construye una matriz de retardos (ventanas autoregresivas) a partir de una serie.
    Devuelve:
        X : matriz (n - lags, lags)
        y : vector (n - lags,)
        last : último vector de tamaño 'lags' (para predicción iterativa)
    """
    import numpy as np
    import pandas as pd

    if df_in is None or df_in.empty:
        raise ValueError("_build_lag_matrix: DataFrame vacío o None.")

    # Asegurar orden y limpieza
    df = df_in.copy().sort_values("fecha").dropna(subset=["valor"])
    y = df["valor"].astype(float).to_numpy()
    n = len(y)

    if n <= lags:
        raise ValueError(f"Serie insuficiente para construir lag matrix (n={n}, lags={lags}).")

    # Matriz X: cada fila contiene los últimos 'lags' valores para predecir el siguiente
    # Cada columna se desfasa una posición respecto a la anterior.
    X = np.column_stack([y[i : n - lags + i] for i in range(lags)])  # shape: (n - lags, lags)
    y_out = y[lags:]                                                 # shape: (n - lags,)
    last = y[-lags:].copy()                                          # último bloque (para feedback)

    # Validaciones finales
    if X.shape[0] != y_out.shape[0]:
        raise RuntimeError(
            f"_build_lag_matrix inconsistente: X.shape={X.shape}, y_out.shape={y_out.shape}"
        )

    return X, y_out, last

def _build_direct_matrices(df_in: pd.DataFrame, lags: int, horizon: int):
    """
    Directo multi-salida: X[t] = ventana de lags, Y[t] = [y(t+1)..y(t+h)].
    """
    df = df_in.copy().sort_values("fecha").dropna(subset=["valor"])
    y = df["valor"].astype(float).to_numpy()
    n = len(y)
    min_len = lags + horizon
    if n <= min_len:
        raise ValueError(f"Serie insuficiente para lags={lags} y horizon={horizon} (n={n}).")
    X_list, Y_list = [], []
    for t in range(n - min_len + 1):
        X_list.append(y[t: t + lags])
        Y_list.append(y[t + lags: t + lags + horizon])
    X = np.asarray(X_list, dtype=float)
    Y = np.asarray(Y_list, dtype=float)
    last = y[-lags:].copy()
    return X, Y, last

def _detect_anomalies_mad(fecha_idx: np.ndarray,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          k: float):
    """
    ÚNICA implementación: detecta anomalías por residuo con k·MAD (fallback a std).
    Retorna: (idxs, fechas, valores)
    """
    resid = (y_true - y_pred).astype(float)
    med = np.median(resid) if resid.size else 0.0
    sig = _mad(resid)  # robusto
    if not np.isfinite(sig) or sig <= 0:
        # Fallback a std si MAD ~ 0
        sig = np.std(resid)
        if not np.isfinite(sig) or sig <= 0:
            return np.array([], dtype=int), np.array([], dtype="datetime64[ns]"), np.array([])
    thr = float(k) * sig
    mask = np.abs(resid - med) > thr
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return idxs, np.array([], dtype="datetime64[ns]"), np.array([])
    fechas = fecha_idx[idxs]
    valores = y_true[idxs]
    return idxs, fechas, valores

def _backtest_kmad_and_save(model_tag: str,
                            fecha_idx: np.ndarray,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            cfg=CFG,
                            filename: str | None = None) -> Optional[str]:
    """
    Estandariza backtest k·MAD y guardado de anomalías (fecha,valor) en RUTA_ANOMALIAS_METODO.
    - model_tag: ej. "PROPHET_DIRECT", "NBEATS_FEEDBACK", etc.
    - filename:  opcional; por defecto usa f"{model_tag}_puntos.csv".
    Devuelve ruta del CSV escrito (o None si no se guardó).
    """
    try:
        k = float(getattr(cfg, "ANOM_K_SIGMA", 3.5))
    except Exception:
        k = 3.5

    idxs, f_anom, v_anom = _detect_anomalies_mad(fecha_idx, y_true, y_pred, k)
    ruta_anom = getattr(cfg, "RUTA_ANOMALIAS_METODO", None)
    if ruta_anom and f_anom.size:
        try:
            os.makedirs(ruta_anom, exist_ok=True)
            anom_df = pd.DataFrame({"fecha": f_anom, "valor": v_anom})
            # salida homogénea de fecha como string (ISO) para facilitar ingestión
            if hasattr(anom_df["fecha"], "dt"):
                anom_df["fecha"] = pd.to_datetime(anom_df["fecha"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            fname = filename or f"{model_tag}_puntos.csv"
            out_path = os.path.join(ruta_anom, fname)
            anom_df.to_csv(out_path, index=False)
            return out_path
        except Exception as e:
            print(f"[{model_tag}] Advertencia: no se pudo guardar anomalías en {ruta_anom}: {e}")
    return None

def _assemble_outputs(df_hist: pd.DataFrame,
                      idx_future: np.ndarray,
                      preds_future: np.ndarray,
                      model_tag: str,
                      out_dir: str,
                      cfg=CFG) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Construye df_future y df_full estandarizados y delega en _save_outputs para gráficos/CSVs.
    Retorna: (df_full, df_future, paths_dict)
    """
    if df_hist is None or df_hist.empty:
        raise ValueError("_assemble_outputs: df_hist vacío")

    df_future = pd.DataFrame({
        "fecha": idx_future,
        "valor": preds_future,
        "pred":  preds_future,
        "fuente": getattr(cfg, "PRED_FUENTE", "sintetica")
    })

    df_hist2 = df_hist.copy()
    if "pred" not in df_hist2.columns:
        df_hist2["pred"] = np.nan
    if "fuente" not in df_hist2.columns:
        df_hist2["fuente"] = getattr(cfg, "ORIG_FUENTE", "original")

    try:
        df_hist2["fecha"] = pd.to_datetime(df_hist2["fecha"])
    except Exception:
        pass

    df_full = pd.concat(
        [df_hist2[["fecha","valor","pred","fuente"]],
         df_future[["fecha","valor","pred","fuente"]]],
        ignore_index=True
    )

    paths = _save_outputs(
        model_tag,
        out_dir,
        df_full=df_full,
        df_future=df_future
    )
    return df_full, df_future, paths

def _future_jump_anomalies(preds: np.ndarray,
                           fecha_idx: np.ndarray,
                           y_hist: Optional[np.ndarray] = None,
                           cfg=CFG) -> Tuple[np.ndarray, np.ndarray]:
    """
    Marca anomalías FUTURAS por saltos de la predicción (Δpred).
    - Escala robusta con MAD de deltas históricas si y_hist se provee; si no, usa std de Δpred.
    Devuelve (fechas_anom, deltas_anom) para CSV opcional.
    """
    if preds is None or len(preds) < 2:
        return np.array([]), np.array([])
    diffs_pred = np.diff(preds.astype(float))
    if y_hist is not None and len(y_hist) > 2:
        diffs_hist = np.diff(np.asarray(y_hist, dtype=float))
        med = np.median(diffs_hist) if diffs_hist.size else 0.0
        mad = np.median(np.abs(diffs_hist - med)) if diffs_hist.size else 0.0
        scale = mad if mad > 0 else (np.std(diffs_hist) + 1e-9)
        center = med
    else:
        scale = np.std(diffs_pred) + 1e-9
        center = np.median(diffs_pred)
    k_delta = float(getattr(cfg, "ANOM_K_SIGMA_DELTA", 3.5))
    flags = np.where(np.abs(diffs_pred - center) > (k_delta * scale))[0]  # salto entre t y t+1
    if flags.size == 0:
        return np.array([]), np.array([])
    fechas = fecha_idx[flags + 1]  # el salto afecta a la posición siguiente
    return fechas, diffs_pred[flags]

def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

def _infer_freq(fechas) -> str:
    try:
        f = pd.infer_freq(pd.to_datetime(fechas))
        return f or "D"
    except Exception:
        return "D"

def _future_index(last_date, freq: str, horizon: int):
    last = pd.to_datetime(last_date)
    return pd.date_range(start=last, periods=int(horizon)+1, freq=freq)[1:]

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _load_series_from_json(json_or_csv: str) -> pd.DataFrame:
    """
    Carga serie desde JSON (orient=records) o ruta CSV.
    Devuelve columnas: fecha (datetime y ordenada), valor (float), y conserva extras (p.ej. fuente).
    """
    # CSV (ruta)
    if isinstance(json_or_csv, str) and json_or_csv.lower().endswith(".csv") and os.path.exists(json_or_csv):
        df = pd.read_csv(json_or_csv)
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        else:
            first = df.columns[0]
            df.rename(columns={first: "fecha"}, inplace=True)
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = df.dropna(subset=["fecha"]).sort_values("fecha")
        if "valor" not in df.columns:
            candidates = [c for c in df.columns if c != "fecha"]
            if candidates:
                df.rename(columns={candidates[0]: "valor"}, inplace=True)
        if "valor" in df.columns:
            df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        if "valor_raw" in df.columns:
            df["valor_raw"] = pd.to_numeric(df["valor_raw"], errors="coerce")
        df = df.dropna(subset=["valor"]).reset_index(drop=True)
        return df

    # JSON literal
    df = pd.read_json(io.StringIO(json_or_csv))
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        df["fecha"] = df.index
    if "valor" not in df.columns:
        first = [c for c in df.columns if c != "fecha"][0]
        df.rename(columns={first: "valor"}, inplace=True)
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    if "valor_raw" in df.columns:
        df["valor_raw"] = pd.to_numeric(df["valor_raw"], errors="coerce")
    df = df.dropna(subset=["fecha", "valor"]).sort_values("fecha").reset_index(drop=True)
    return df


def _save_outputs(model_tag: str,
                  out_dir: str,
                  df_full: pd.DataFrame,
                  df_future: pd.DataFrame) -> dict:
    """
    Guardado estandarizado (definitivo):
      - CSV 1:  1_<MODELO>_sintetica.csv   (solo futuro: fecha, valor, fuente, metodo_prediccion)
      - CSV 3:  3_<MODELO>_series_full.csv (serie completa: fecha, valor, pred, fuente)

      - Gráficos de diagnóstico en: <out_dir>/Graficos/
        * <MODELO>_series_full.png
        * 1_<MODELO>_sintetica.png

    NOTA: El CSV 2 (<MODELO>_anomalies.csv) lo genera la ETAPA DE DETECCIÓN.
    """
    import os
    import numpy as np
    import pandas as pd

    # Backend no interactivo para Airflow/servidor
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    #_ensure_dir(out_dir)
    #graf_dir = os.path.join(out_dir, "Graficos")
    #_ensure_dir(graf_dir)
    _ensure_dir(out_dir)
    graf_dir = out_dir 
    
    # -------- Normalización df_full (serie completa)
    df_full = df_full.copy()
    # Asegurar columnas
    if "pred" not in df_full.columns:
        df_full["pred"] = np.nan
    if "fuente" not in df_full.columns:
        # Usa CFG si existe, si no "original"
        fuente_orig = getattr(CFG, "ORIG_FUENTE", "original")
        df_full["fuente"] = fuente_orig

    # Tipos/orden
    df_full["fecha"] = pd.to_datetime(df_full["fecha"], errors="coerce")
    df_full = df_full.dropna(subset=["fecha"])
    df_full = df_full[["fecha", "valor", "pred", "fuente"]].sort_values("fecha").reset_index(drop=True)

    # -------- Normalización df_future (SOLO futuro → archivo 1)
    df_future = df_future.copy()
    if "valor" not in df_future.columns and "pred" in df_future.columns:
        df_future["valor"] = df_future["pred"]
    if "pred" not in df_future.columns and "valor" in df_future.columns:
        df_future["pred"] = df_future["valor"]

    df_future["fecha"] = pd.to_datetime(df_future["fecha"], errors="coerce")
    df_future = df_future.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)

    df_future["fuente"] = getattr(CFG, "PRED_FUENTE", "sintetica")
    df_future["metodo_prediccion"] = model_tag

    df1 = df_future[["fecha", "valor", "fuente", "metodo_prediccion"]]

    # ── Redondear valores al guardar (aplica a TODOS los métodos) ────
    # Evita que ruido decimal sea detectado como anomalía
    n_dec = getattr(CFG, "PRED_ROUND_DECIMALS", 2)
    df1 = df1.copy()
    df1["valor"] = df1["valor"].round(n_dec)
    print(f"[SAVE] {model_tag} → valores redondeados a {n_dec} decimales | "
          f"rango: [{df1['valor'].min():.{n_dec}f}, {df1['valor'].max():.{n_dec}f}]")
    # ─────────────────────────────────────────────────────────────────

   # csv_1 = os.path.join(out_dir, f"1_{model_tag}_sintetica.csv")
   # df1.to_csv(csv_1, index=False)


    # -------- Guardado CSVs
    csv_1 = os.path.join(out_dir, f"1_{model_tag}_sintetica.csv")
    csv_3 = os.path.join(out_dir, f"3_{model_tag}_series_full.csv")
    df1.to_csv(csv_1, index=False)
    df_full.to_csv(csv_3, index=False)

    # -------- Gráfico completo (diagnóstico)
    graph_full = None
    try:
        if not df_full.empty:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(df_full["fecha"], df_full["valor"], label="valor", linewidth=1.5)
            # Solo traza pred si tiene algún dato no nulo
            if df_full["pred"].notna().any():
                ax.plot(df_full["fecha"], df_full["pred"], label="pred", linewidth=1.0)
            ax.set_title(str(model_tag))
            ax.legend()
            fig.tight_layout()
            graph_full = os.path.join(graf_dir,f"3_{model_tag}_series_full.png")
            fig.savefig(graph_full, dpi=150, bbox_inches="tight")
            plt.close(fig)
    except Exception:
        graph_full = None

    # -------- Gráfico de futuro (diagnóstico)
    graph_future = None
    try:
        if not df1.empty:
            fig2, ax2 = plt.subplots(figsize=(6, 3.2))
            ax2.plot(df1["fecha"], df1["valor"], linewidth=1.5)
            ax2.set_title(f"{model_tag} (futuro)")
            plt.setp(ax2.get_xticklabels(), rotation=90, ha="center", fontsize=7)
            fig2.tight_layout()
            graph_future = os.path.join(graf_dir, f"1_{model_tag}_sintetica.png")
            fig2.savefig(graph_future, dpi=150, bbox_inches="tight")
            plt.close(fig2)
    except Exception:
        graph_future = None

    return {
        "csv_1_sintetica": csv_1,
        "csv_3_series_full": csv_3,
        "graph_series_full": graph_full,
        "graph_sintetica": graph_future,
    }

def _import_prophet():
    try:
        from prophet import Prophet
        return Prophet
    except Exception:
        try:
            from fbprophet import Prophet
            return Prophet
        except Exception as e:
            raise ImportError(
                "Falta Prophet. Instala con: pip install prophet  (o fbprophet en entornos antiguos)"
            ) from e

def _prophet_from_cfg(cfg):
    Prophet = _import_prophet()
    return Prophet(
        growth=str(getattr(cfg, "PROPHET_GROWTH", "linear")),
        yearly_seasonality=bool(getattr(cfg, "PROPHET_YEARLY_SEASONALITY", True)),
        weekly_seasonality=bool(getattr(cfg, "PROPHET_WEEKLY_SEASONALITY", False)),
        daily_seasonality=bool(getattr(cfg, "PROPHET_DAILY_SEASONALITY", False)),
        seasonality_mode=str(getattr(cfg, "PROPHET_SEASONALITY_MODE", "additive")),
        n_changepoints=int(getattr(cfg, "PROPHET_N_CHANGEPOINTS", 25)),
        changepoint_prior_scale=float(getattr(cfg, "PROPHET_CHANGEPOINT_PRIOR", 0.05)),
        seasonality_prior_scale=float(getattr(cfg, "PROPHET_SEASONALITY_PRIOR", 10.0)),
        holidays_prior_scale=float(getattr(cfg, "PROPHET_HOLIDAYS_PRIOR", 10.0)),
        interval_width=float(getattr(cfg, "PRED_LEVEL", 0.9)),
    )
def _apply_logistic_bounds(df_ds_y: pd.DataFrame, cfg):
    """Si growth='logistic', garantiza columnas cap/floor tanto en train como en future."""
    growth = str(getattr(cfg, "PROPHET_GROWTH", "linear"))
    if growth != "logistic":
        return df_ds_y, None, None
    cap = getattr(cfg, "PROPHET_CAP", None)
    floor = getattr(cfg, "PROPHET_FLOOR", None)
    y = df_ds_y["y"].astype(float).values
    if cap is None:
        cap = float(np.nanmax(y) * 1.2) if np.isfinite(np.nanmax(y)) else 1.0
    if floor is None:
        floor = float(np.nanmin(y)) if np.isfinite(np.nanmin(y)) else 0.0
    df_ds_y = df_ds_y.copy()
    df_ds_y["cap"] = cap
    df_ds_y["floor"] = floor
    return df_ds_y, cap, floor
def _prophet_predict(model, future_ds: pd.DatetimeIndex, cap=None, floor=None):
    """Construye el DF futuro para Prophet y devuelve yhat como np.array."""
    future_df = pd.DataFrame({"ds": pd.to_datetime(future_ds)})
    if cap is not None:
        future_df["cap"] = float(cap)
    if floor is not None:
        future_df["floor"] = float(floor)
    fcst = model.predict(future_df)
    # yhat es la predicción puntual; podrías exponer también yhat_lower/upper si lo deseas
    return fcst["yhat"].astype(float).values
def _deepar_recursive_forecast(model, last_window, scaler_X, scaler_y, L, horizon):
    """
    FEEDBACK forecast: predice 1 paso, inyecta, repite hasta completar horizon.

    last_window: array de longitud L en escala original.
    Retorna: np.array de longitud horizon en escala original.
    """
    import numpy as np

    w = np.asarray(last_window, dtype=np.float32).reshape(-1)
    if w.shape[0] != int(L):
        raise ValueError(f"last_window debe tener longitud L={L}, recibido {w.shape[0]}")

    preds = []
    for _ in range(int(horizon)):
        x_raw = w[-int(L):].reshape(1, int(L))                 # (1, L) escala original
        x_scaled_2d = scaler_X.transform(x_raw)                # (1, L)
        x_scaled = x_scaled_2d.reshape(1, int(L), 1)           # (1, L, 1)
        mu_scaled = model.predict(x_scaled, verbose=0).reshape(-1)[0]
        mu = scaler_y.inverse_transform([[mu_scaled]])[0][0]   # escala original
        preds.append(float(mu))
        w = np.concatenate([w, np.asarray([mu], dtype=np.float32)])
    return np.asarray(preds, dtype=np.float32)

def _deepar_fit_and_backtest(df, L, cfg, epochs, batch_sz, val_split, patience, rs=None):
    """
    Entrena un modelo tipo DeepAR (implementación práctica con RNN regresor univariado)
    y hace backtest 1-step-ahead sobre todas las ventanas:
      X[i] = y[i:i+L]  -> predice y[i+L]

    Retorna (7):
      model, scaler_X, scaler_y, X, y, last_window, mu_back
    donde:
      - X: ventanas (N-L, L, 1) escaladas (para reutilizar si necesitas)
      - y: targets (N-L,) en escala original
      - last_window: últimos L valores (escala original) para forecast futuro
      - mu_back: predicción 1-step para cada ventana (escala original), len = N-L
    """
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler

    # --- Seed determinista si se solicita ---
    if rs is not None:
        try:
            np.random.seed(int(rs))
            tf.random.set_seed(int(rs))
        except Exception:
            pass

    # --- Preparación serie ---
    df_ = df.copy()
    # Intenta detectar columnas típicas (respeta tu estilo: en tus métodos ya armas y_series)
    if "fecha" in df_.columns:
        df_["fecha"] = pd.to_datetime(df_["fecha"], errors="coerce")
        df_ = df_.dropna(subset=["fecha"]).sort_values("fecha")
        y_series = df_["valor"].astype(float).values if "valor" in df_.columns else df_.iloc[:, -1].astype(float).values
    else:
        # Si df ya viene ordenado y solo trae valor
        y_series = df_.iloc[:, -1].astype(float).values

    N = len(y_series)
    if N <= L + 2:
        raise ValueError(f"Serie demasiado corta para L={L}. N={N}")

    # --- Construcción ventanas 1-step ---
    X_list = []
    y_list = []
    for i in range(0, N - L):
        X_list.append(y_series[i:i+L])
        y_list.append(y_series[i+L])
    X_raw = np.asarray(X_list, dtype=np.float32)         # (N-L, L)
    y_raw = np.asarray(y_list, dtype=np.float32)         # (N-L,)

    # Guardar y en escala original (tu pipeline lo usa para residual/anomalías)
    y = y_raw.copy()

    # Última ventana real para arrancar forecast
    last_window = y_series[-L:].astype(np.float32)

    # --- Escalado (consistente con LSTM: StandardScaler) ---
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled_2d = scaler_X.fit_transform(X_raw)                          # (N-L, L)
    y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).reshape(-1)  # (N-L,)

    # Reshape para RNN: (samples, timesteps, features=1)
    X_scaled = X_scaled_2d.reshape((X_scaled_2d.shape[0], X_scaled_2d.shape[1], 1))

    # --- Split temporal (NO aleatorio) ---
    n_samples = X_scaled.shape[0]
    n_val = int(max(1, round(n_samples * float(val_split)))) if val_split else 0
    n_train = n_samples - n_val
    if n_train < 2:
        # si te queda muy poco train, desactiva validación
        n_train = n_samples
        n_val = 0

    X_tr = X_scaled[:n_train]
    y_tr = y_scaled[:n_train]
    if n_val > 0:
        X_va = X_scaled[n_train:]
        y_va = y_scaled[n_train:]
        validation_data = (X_va, y_va)
    else:
        validation_data = None

    # --- Hiperparámetros desde cfg (con defaults robustos) ---
    units = getattr(cfg, "DEEPAR_UNITS", 30)
    layers = getattr(cfg, "DEEPAR_LAYERS", 2)
    dropout = getattr(cfg, "DEEPAR_DROPOUT", 0.1)
    lr = getattr(cfg, "DEEPAR_LR", 1e-3)  # si no existe, default

    # --- Modelo: RNN regresor (GRU por estabilidad; puedes cambiar a LSTM si prefieres) ---
    inp = tf.keras.Input(shape=(L, 1))
    x = inp
    for li in range(int(layers)):
        return_sequences = (li < int(layers) - 1)
        x = tf.keras.layers.GRU(int(units), dropout=float(dropout), return_sequences=return_sequences)(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inp, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)),
        loss="mse",
    )

    callbacks = []
    if patience and patience > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if validation_data is not None else "loss",
            patience=int(patience),
            restore_best_weights=True
        ))

    # shuffle=False por series
    model.fit(
        X_tr, y_tr,
        epochs=int(epochs),
        batch_size=int(batch_sz),
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=0,
        shuffle=False
    )

    # --- Backtest 1-step sobre TODAS las ventanas ---
    mu_scaled = model.predict(X_scaled, verbose=0).reshape(-1)  # escala y_scaled
    mu_back = scaler_y.inverse_transform(mu_scaled.reshape(-1, 1)).reshape(-1).astype(np.float32)

    return model, scaler_X, scaler_y, X_scaled, y, last_window, mu_back


def _nbeats_build_model(L: int, H: int, cfg=CFG):
    import tensorflow as tf

    stacks = int(getattr(cfg, "NBEATS_STACKS", 2))
    blocks_per_stack = int(getattr(cfg, "NBEATS_BLOCKS_PER_STACK", 3))
    width = int(getattr(cfg, "NBEATS_WIDTH", 256))
    fc_layers = int(getattr(cfg, "NBEATS_FC_LAYERS", 4))
    activation = str(getattr(cfg, "NBEATS_ACTIVATION", "relu"))
    dropout = float(getattr(cfg, "NBEATS_DROPOUT", 0.0))
    loss = str(getattr(cfg, "NBEATS_LOSS", "huber"))
    optimizer = str(getattr(cfg, "NBEATS_OPTIMIZER", "adam"))

    inp = tf.keras.Input(shape=(L,), name="input_window")
    x = inp

    # Head simple multi-salida (si tu NBEATS es más complejo, igual aplica la idea de nombres únicos)
    for s in range(stacks):
        for b in range(blocks_per_stack):
            h = x
            for k in range(fc_layers):
                h = tf.keras.layers.Dense(
                    width,
                    activation=activation,
                    name=f"fc_s{s}_b{b}_k{k}",   # ← único
                )(h)
                if dropout and dropout > 0:
                    h = tf.keras.layers.Dropout(dropout, name=f"drop_s{s}_b{b}_k{k}")(h)
            # “block output” (puede ser residual / backcast/forecast según tu implementación)
            x = h

    out = tf.keras.layers.Dense(H, name="forecast")(x)
    model = tf.keras.Model(inp, out, name=f"nbeats_L{L}_H{H}")
    model.compile(optimizer=optimizer, loss=loss)
    return model

# --- Helpers consolidación fechas (NO tocan tu lógica de predicción/detección) ---
import os, re, glob
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _norm_ddmmyyyy(ts: Optional[pd.Timestamp]) -> Optional[str]:
    if ts is pd.NaT or ts is None:
        return None
    # fuerza dd/mm/yyyy
    return ts.strftime("%d/%m/%Y")

_ddmmyyyy_re = re.compile(r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\s*$")

def _safe_parse_fecha(txt: str) -> Tuple[pd.Timestamp, Optional[str]]:
    """
    Devuelve (ts_orden, fecha_norm_ddmmyyyy)
    - ts_orden: Timestamp para ordenar/agrupación; NaT si no se pudo.
    - fecha_norm_ddmmyyyy: string dd/mm/yyyy si se pudo normalizar; None si no.
    """
    if txt is None:
        return (pd.NaT, None)
    s = str(txt).strip()

    # Intento 1: formato fijo dd/mm/yyyy
    try:
        ts = pd.to_datetime(s, format="%d/%m/%Y", errors="raise")
        return (ts, _norm_ddmmyyyy(ts))
    except Exception:
        pass

    # Intento 2: tolerante con dayfirst
    try:
        ts = pd.to_datetime(s, dayfirst=True, errors="raise")
        return (ts, _norm_ddmmyyyy(ts))
    except Exception:
        pass

    # Intento 3: regex dd-mm-yyyy o dd/m/yy, etc.
    m = _ddmmyyyy_re.match(s)
    if m:
        d, mth, y = m.groups()
        d = int(d); mth = int(mth); y = int(y)
        if y < 100:  # asume 2000-2099 si llegan 2 dígitos
            y += 2000
        try:
            ts = pd.Timestamp(year=y, month=mth, day=d)
            return (ts, _norm_ddmmyyyy(ts))
        except Exception:
            return (pd.NaT, None)

    return (pd.NaT, None)

def _read_detectado_puntos(model_tag: str, pred_dir: str) -> pd.DataFrame:
    """
    Lee TODOS los *_puntos*.csv bajo Prediccion/<MODEL_TAG>/DetectadoXModelo/** (dtype=str).
    Devuelve DF con columnas estandarizadas.
    """
    base = Path(pred_dir) / "DetectadoXModelo"
    patrones = [
        str(base / "**" / "*puntos*.csv"),
        str(base / "**" / "puntos.csv"),
    ]
    files = []
    for pat in patrones:
        files.extend(glob.glob(pat, recursive=True))

    if not files:
        return pd.DataFrame(columns=[
            "fecha", "valor", "metodo_deteccion", "metodo_prediccion", "fuente", "score"
        ])

    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp, dtype=str)
        except Exception:
            continue
        # normaliza columnas mínimas esperadas
        cols = {c.lower().strip(): c for c in df.columns}
        def pick(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None

        c_fecha  = pick("fecha")
        c_valor  = pick("valor", "y", "pred", "value")
        c_md     = pick("metodo_deteccion", "metodo", "detector")
        c_mp     = pick("metodo_prediccion", "modelo_prediccion", "modelo")
        c_fuente = pick("fuente")
        c_score  = pick("score", "anomaly_score")

        out = pd.DataFrame({
            "fecha":  df[c_fecha]  if c_fecha  else np.nan,
            "valor":  df[c_valor]  if c_valor  else np.nan,
            "metodo_deteccion": df[c_md] if c_md else "desconocido",
            "metodo_prediccion": df[c_mp] if c_mp else str(model_tag),
            "fuente": df[c_fuente] if c_fuente else "sintetica",
            "score":  df[c_score]  if c_score else np.nan,
        })
        dfs.append(out)

    if not dfs:
        return pd.DataFrame(columns=[
            "fecha", "valor", "metodo_deteccion", "metodo_prediccion", "fuente", "score"
        ])

    df_all = pd.concat(dfs, ignore_index=True)
    # columnas auxiliares para orden/agrupado
    ts, norm = zip(*df_all["fecha"].map(_safe_parse_fecha))
    df_all["__fecha_ord"] = list(ts)
    df_all["__fecha_norm"] = list(norm)
    return df_all

def _consolidar_puntos_por_modelo(df_puntos: pd.DataFrame, model_tag: str) -> pd.DataFrame:
    """
    - Ordena por fecha, deduplica
    - id contiguo con padding 0001
    - fecha de salida SIEMPRE dd/mm/yyyy (si no se pudo, deja el texto original)
    """
    if df_puntos.empty:
        return pd.DataFrame(columns=[
            "id","fecha","valor","metodo_deteccion","metodo_prediccion","fuente","score"
        ])

    df = df_puntos.copy()

    # fecha final a escribir: intenta __fecha_norm; si falta, deja 'fecha' original
    fecha_final = np.where(df["__fecha_norm"].notna(), df["__fecha_norm"], df["fecha"])
    df["__fecha_final"] = fecha_final

    # orden para consolidar
    df = df.sort_values(["metodo_prediccion","metodo_deteccion","__fecha_ord","valor"], kind="mergesort")

    # deduplicación conservadora (mismo método, misma fecha final, mismo valor)
    keep_cols = ["__fecha_final","valor","metodo_deteccion","metodo_prediccion","fuente","score"]
    df = df.drop_duplicates(subset=keep_cols).reset_index(drop=True)

    # id contiguo
    df.insert(0, "id", (df.index + 1).astype(int).astype(str).str.zfill(4))

    # salida final
    out = df.rename(columns={"__fecha_final": "fecha"})[
        ["id","fecha","valor","metodo_deteccion","metodo_prediccion","fuente","score"]
    ]

    return out

def _agrupar_fechas_consecutivas(df_cons: pd.DataFrame) -> pd.DataFrame:
    """
    Solo agrupa tramos con diferencia EXACTA de 1 día dentro de:
    (metodo_prediccion, metodo_deteccion)
    """
    if df_cons.empty:
        return pd.DataFrame(columns=[
            "id","metodo_prediccion","metodo_deteccion",
            "fecha_inicio","fecha_fin","fechas_concat","n_puntos","valores_concat","fuente"
        ])

    df = df_cons.copy()

    # reconstruye timestamp para agrupado desde 'fecha' (ya dd/mm/yyyy)
    ts = pd.to_datetime(df["fecha"], format="%d/%m/%Y", errors="coerce")
    df["__ts"] = ts

    # orden base
    df = df.sort_values(["metodo_prediccion","metodo_deteccion","__ts"]).reset_index(drop=True)

    # define cortes de grupo (nuevo grupo si cambia método o si la diferencia != 1 día)
    key = ["metodo_prediccion","metodo_deteccion"]
    df["__grp"] = (
        (df.groupby(key)["__ts"]
           .apply(lambda s: (s.diff().dt.days != 1).fillna(True))
           .reset_index(level=key, drop=True))
        | (df[key] != df[key].shift(1)).any(axis=1).fillna(True)
    ).cumsum()

    # agrega
    agg = df.groupby(key + ["__grp"]).agg(
        fecha_inicio=("__ts", lambda s: _norm_ddmmyyyy(s.min())),
        fecha_fin   =("__ts", lambda s: _norm_ddmmyyyy(s.max())),
        fechas_concat=("fecha", lambda s: ",".join(map(str, s.tolist()))),
        n_puntos=("fecha", "count"),
        valores_concat=("valor", lambda s: ",".join(map(str, s.tolist()))),
        fuente=("fuente", lambda s: s.iloc[0] if len(s) else "sintetica"),
    ).reset_index(drop=False)

    # id contiguo
    agg.insert(0, "id", (agg.index + 1).astype(int).astype(str).str.zfill(4))

    out = agg[[
        "id","metodo_prediccion","metodo_deteccion",
        "fecha_inicio","fecha_fin","fechas_concat","n_puntos","valores_concat","fuente"
    ]]
    return out

def _write_consolidados(df_cons: pd.DataFrame, df_grp: pd.DataFrame, out_dir: str, model_tag: str) -> Dict[str,str]:
    _ensure_dir(out_dir)
    f1 = os.path.join(out_dir, f"{model_tag}_anomalies.csv")
    f2 = os.path.join(out_dir, f"{model_tag}_anomalies_agrupado.csv")

    # aseguramos columnas (aunque estén vacías)
    cols1 = ["id","fecha","valor","metodo_deteccion","metodo_prediccion","fuente","score"]
    cols2 = ["id","metodo_prediccion","metodo_deteccion","fecha_inicio","fecha_fin",
             "fechas_concat","n_puntos","valores_concat","fuente"]

    pd.DataFrame(columns=cols1).to_csv(f1, index=False) if df_cons.empty else df_cons.to_csv(f1, index=False)
    pd.DataFrame(columns=cols2).to_csv(f2, index=False) if df_grp.empty else df_grp.to_csv(f2, index=False)

    return {"csv_anomalies": f1, "csv_anomalies_agrupado": f2}


#------------------------------------------
#ARIMA 1
#------------------------------------------
def run_arimav1(df_in: pd.DataFrame, horizon: int, out_dir: str, cfg=CFG) -> Dict:
    if df_in is None or df_in.empty:
        raise ValueError("Serie vacía en run_arima.")

    # Orden y limpieza mínimos (igual que antes)
    df = df_in.copy().sort_values("fecha").dropna(subset=["fecha", "valor"])
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])
    df["valor"] = df["valor"].astype(float)

    # Serie indexada por fecha
    y = pd.Series(df["valor"].values, index=df["fecha"])
    if len(y) < 20:
        raise ValueError("Se requieren al menos 20 puntos para ARIMA.")

    # Parámetros
    order = tuple(getattr(cfg, "ARIMA_ORDER_predict", (3, 0, 2)))
    trend = str(getattr(cfg, "ARIMA_TREND_predict", "n") or "n")
    maxiter = int(getattr(cfg, "MAXITER_ARIMA", 60))

    # Frecuencia explícita (igual enfoque)
    try:
        if y.index.inferred_freq is None:
            freq = _infer_freq(y.index.values.astype("datetime64[ns]")) or "D"
            y = y.asfreq(freq)
    except Exception:
        pass

    # Guardas de trend vs d (idéntico)
    p, d, q = order
    if d > 0 and trend in ("c", "ct"):
        trend = "t"

    # Ajuste ARIMA y forecast multi-paso (igual)
    model = ARIMA(y, order=order, trend=trend,
                  enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(low_memory=True, method_kwargs={"maxiter": maxiter})

    fc = res.get_forecast(steps=int(horizon)).predicted_mean
    preds = np.asarray(fc, dtype=float)

    # Índice futuro consistente (misma política de frecuencia)
    last = y.index[-1]
    freq = y.index.freqstr or _infer_freq(y.index.values.astype("datetime64[ns]")) or "D"
    idx_fut = pd.date_range(start=last, periods=int(horizon) + 1, freq=freq)[1:]

    # Salidas estandarizadas
    df_full, df_future, paths = _assemble_outputs(
        df_hist=df[["fecha", "valor"]],
        idx_future=idx_fut,
        preds_future=preds,
        model_tag="ARIMA_DIRECT",
        out_dir=out_dir,
        cfg=cfg
    )

    # --- Shim de compatibilidad de claves (no cambia lógica) ---
    mapped = dict(paths)
    if "csv_1_sintetica" not in mapped and "csv_completo_path" in mapped:
        mapped["csv_1_sintetica"] = mapped.pop("csv_completo_path")
    if "csv_3_series_full" not in mapped and "csv_futuros_path" in mapped:
        # Nota: antes “csv_futuros_path” era 2_; ahora el 3_ es “series_full”.
        # Si _assemble_outputs ya usa _save_outputs nuevo, esta línea no se usa.
        mapped["csv_3_series_full"] = mapped.pop("csv_futuros_path")
    if "grafico_series_full_path" not in mapped and "grafico_path" in mapped:
        mapped["grafico_series_full_path"] = mapped.pop("grafico_path")
    if "grafico_sintetica_path" not in mapped and "grafico_pred_only_path" in mapped:
        mapped["grafico_sintetica_path"] = mapped.pop("grafico_pred_only_path")

    return {
        "modelo": "ARIMA",
        "params": {"order": order, "trend": trend, "maxiter": maxiter},
        "n_futuros": int(horizon),
        **mapped
    }
def run_arima(df_in: pd.DataFrame, horizon: int, out_dir: str, cfg=CFG) -> Dict:
    if df_in is None or df_in.empty:
        raise ValueError("Serie vacía en run_arima.")

    df = df_in.copy().sort_values("fecha").dropna(subset=["fecha", "valor"])
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])
    df["valor"] = df["valor"].astype(float)

    y = pd.Series(df["valor"].values, index=df["fecha"])
    if len(y) < 20:
        raise ValueError("Se requieren al menos 20 puntos para ARIMA.")

    order  = tuple(getattr(cfg, "ARIMA_ORDER_predict", (2, 1, 2)))
    trend  = str(getattr(cfg, "ARIMA_TREND_predict", "n") or "n")
    maxiter = int(getattr(cfg, "MAXITER_ARIMA", 200))

    try:
        if y.index.inferred_freq is None:
            freq = _infer_freq(y.index.values.astype("datetime64[ns]")) or "D"
            y = y.asfreq(freq)
    except Exception:
        pass

    # ── Ventana de entrenamiento ───────────────────────────────────
    train_years = int(getattr(cfg, "ARIMA_TRAIN_YEARS", 2))
    cutoff  = y.index[-1] - pd.DateOffset(years=train_years)
    y_train = y[y.index >= cutoff]
    print(f"[ARIMA] Entrenando con {len(y_train)} puntos desde {y_train.index[0].date()} (ventana={train_years} años)")

    # ── Selección de orden: grid search o fijo ─────────────────────
    use_auto = bool(getattr(cfg, "ARIMA_AUTO_ORDER", True))

    if use_auto:
        order, res = _arima_best_order(y_train, trend, maxiter)
    else:
        p, d, q = order
        t = "n" if d >= 1 else trend
        model = ARIMA(y_train, order=order, trend=t,
                      enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(low_memory=True, method_kwargs={"maxiter": maxiter})

    print(f"[ARIMA] AIC={res.aic:.2f} | orden={order} | trend={trend}")

    # ── Forecast ───────────────────────────────────────────────────
    fc    = res.get_forecast(steps=int(horizon)).predicted_mean
    preds = np.asarray(fc, dtype=float)

    last = y_train.index[-1]
    freq = y_train.index.freqstr or _infer_freq(y_train.index.values.astype("datetime64[ns]")) or "D"
    idx_fut = pd.date_range(start=last, periods=int(horizon) + 1, freq=freq)[1:]

    # ── Salidas ────────────────────────────────────────────────────
    df_full, df_future, paths = _assemble_outputs(
        df_hist=df[["fecha", "valor"]],
        idx_future=idx_fut,
        preds_future=preds,
        model_tag="ARIMA_DIRECT",
        out_dir=out_dir,
        cfg=cfg
    )

    mapped = dict(paths)
    return {
        "modelo": "ARIMA",
        "params": {"order": order, "trend": trend, "maxiter": maxiter},
        "n_futuros": int(horizon),
        **mapped
    }
def run_arima_feedback(df_in: pd.DataFrame, horizon: int, out_dir: str, cfg=CFG) -> Dict:
    if df_in is None or df_in.empty:
        raise ValueError("Serie vacía en run_arima_feedback.")

    df = df_in.copy().sort_values("fecha").dropna(subset=["fecha", "valor"])
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])
    df["valor"] = df["valor"].astype(float)

    y = pd.Series(df["valor"].values, index=df["fecha"])
    if len(y) < 20:
        raise ValueError("Se requieren al menos 20 puntos para ARIMA_FEEDBACK.")

    trend    = str(getattr(cfg, "ARIMA_TREND_predict", "n") or "n")
    seasonal = tuple(getattr(cfg, "ARIMA_SEASONAL_ORDER_predict", (0, 0, 0, 0)))
    maxiter  = int(getattr(cfg, "MAXITER_ARIMA", 200))

    try:
        if y.index.inferred_freq is None:
            freq = _infer_freq(y.index.values.astype("datetime64[ns]")) or "D"
            y = y.asfreq(freq)
    except Exception:
        pass

    # ── Ventana de entrenamiento ───────────────────────────────────
    train_years = int(getattr(cfg, "ARIMA_TRAIN_YEARS", 2))
    cutoff  = y.index[-1] - pd.DateOffset(years=train_years)
    y_train = y[y.index >= cutoff]
    print(f"[ARIMA_FB] Entrenando con {len(y_train)} puntos desde {y_train.index[0].date()} (ventana={train_years} años)")

    # ── Frecuencia para construir índice futuro ────────────────────
    freq = (y_train.index.freqstr
            or _infer_freq(y_train.index.values.astype("datetime64[ns]"))
            or "D")

    # ── Selección de orden: grid search o fijo ─────────────────────
    use_auto = bool(getattr(cfg, "ARIMA_AUTO_ORDER", True))
    # En vez de _arima_best_order usa el específico para feedback
    if use_auto:
        order, _ = _arima_best_order_feedback(y_train, trend, maxiter)
    else:
        order = tuple(getattr(cfg, "ARIMA_ORDER_predict", (2, 0, 2)))


    p, d, q = order
    t = "n" if d >= 1 else trend
    print(f"[ARIMA_FB] orden={order} | trend={t}")

    # ── Forecast iterativo REAL: reajusta en cada paso ─────────────
    # Cada predicción se agrega a la serie y el modelo se reajusta,
    # lo que produce comportamiento genuinamente diferente a DIRECT.
    preds  = []
    y_ext  = y_train.copy()

    for i in range(int(horizon)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = SARIMAX(
                y_ext,
                order=order,
                seasonal_order=seasonal,
                trend=t,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            r = m.fit(disp=False, maxiter=maxiter)

        yhat = float(r.forecast(steps=1).iloc[0])
        preds.append(yhat)

        if i < 5:
            print(f"[ARIMA_FB] paso={i+1} yhat={yhat:.6f}")

        # Agregar predicción como nuevo punto real para el siguiente paso
        new_date = y_ext.index[-1] + pd.tseries.frequencies.to_offset(freq)
        y_ext = pd.concat([y_ext, pd.Series([yhat], index=[new_date])])

    print(f"[ARIMA_FB] Forecast completo: min={min(preds):.2f} max={max(preds):.2f}")

    preds   = np.asarray(preds, dtype=float)
    last_idx = y_train.index[-1]
    idx_fut  = pd.date_range(start=last_idx, periods=int(horizon) + 1, freq=freq)[1:]

    # ── Salidas ────────────────────────────────────────────────────
    df_full, df_future, paths = _assemble_outputs(
        df_hist=df[["fecha", "valor"]],
        idx_future=idx_fut,
        preds_future=preds,
        model_tag="ARIMA_FEEDBACK",
        out_dir=out_dir,
        cfg=cfg
    )

    mapped = dict(paths)
    return {
        "modelo": "ARIMA_FEEDBACK",
        "params": {"order": order, "trend": t, "seasonal": seasonal, "maxiter": maxiter},
        "n_futuros": int(horizon),
        **mapped
    }
#------------------------------------------
#SVR 2
#------------------------------------------
def run_svr_direct(df_in: pd.DataFrame, horizon: int, out_dir: str, cfg=CFG) -> Dict:
    try:
        s = _sklearn()
        StandardScaler = s["StandardScaler"]
        MultiOutputRegressor = __import__("sklearn.multioutput").multioutput.MultiOutputRegressor
        SVR = s["SVR"]
    except Exception:
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        from sklearn.multioutput import MultiOutputRegressor

    if df_in is None or df_in.empty:
        raise ValueError("Serie vacía en run_svr_direct.")

    # ── Parámetros ────────────────────────────────────────────────
    L        = int(getattr(cfg, "SVR_WINDOW_LEN", 60))
    C        = float(getattr(cfg, "SVR_C", 10.0))
    gamma    = getattr(cfg, "SVR_GAMMA", "scale")
    eps      = float(getattr(cfg, "SVR_EPSILON", 0.05))
    n_jobs   = int(getattr(cfg, "N_JOBS", -1))
    kernel   = str(getattr(cfg, "SVR_KERNEL", "rbf"))
    tol      = float(getattr(cfg, "SVR_TOL", 1e-3))
    max_iter = int(getattr(cfg, "SVR_MAX_ITER", 3000))
    cache_mb = float(getattr(cfg, "SVR_CACHE_MB", 1000.0))
    stride   = int(getattr(cfg, "SVR_STRIDE", 1))

    # ── Ventana de entrenamiento ──────────────────────────────────
    train_years = int(getattr(cfg, "SVR_TRAIN_YEARS", 3))
    cutoff = pd.to_datetime(df_in["fecha"]).max() - pd.DateOffset(years=train_years)
    df_train = df_in[pd.to_datetime(df_in["fecha"]) >= cutoff].copy()
    print(f"[SVR_D] Entrenando con {len(df_train)} puntos desde {cutoff.date()} (ventana={train_years} años)")

    # ── Matrices ──────────────────────────────────────────────────
    X, Y, last = _build_direct_matrices(df_train, lags=L, horizon=int(horizon))

    if stride > 1 and X.shape[0] >= 2 * stride:
        X = X[::stride]
        Y = Y[::stride]

    X    = np.ascontiguousarray(X,    dtype=np.float32)
    Y    = np.ascontiguousarray(Y,    dtype=np.float32)
    last = np.ascontiguousarray(last, dtype=np.float32)

    # ── Escalado ──────────────────────────────────────────────────
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    last_s = scaler.transform(last.reshape(1, -1))

    # ── Modelo ───────────────────────────────────────────────────
    base  = SVR(kernel=kernel, C=C, gamma=gamma if kernel != "linear" else "scale",
                epsilon=eps, tol=tol, max_iter=max_iter, cache_size=cache_mb)
    model = MultiOutputRegressor(base, n_jobs=n_jobs)
    model.fit(Xs, Y)

    preds = model.predict(last_s).ravel().astype(float)
    print(f"[SVR_D] pred: min={preds.min():.2f} max={preds.max():.2f} mean={preds.mean():.2f}")

    # ── Índice futuro ─────────────────────────────────────────────
    fechas  = pd.to_datetime(df_in["fecha"])
    freq    = _infer_freq(fechas)
    idx_fut = _future_index(fechas.iloc[-1], freq, int(horizon))

    steps   = np.arange(1, int(horizon) + 1, dtype=int)
    uses_fb = np.zeros(int(horizon), dtype=bool)

    df_future = pd.DataFrame({
        "fecha":         idx_fut,
        "pred":          preds,
        "step":          steps,
        "uses_feedback": uses_fb,
        "fuente":        getattr(cfg, "PRED_FUENTE", "sintetica"),
    })

    df_full = df_in.copy().sort_values("fecha")
    df_full["pred"] = np.nan
    fut = df_future.rename(columns={"pred": "valor"})
    for col in df_full.columns:
        if col not in fut.columns:
            fut[col] = np.nan
    df_full = pd.concat([df_full, fut.assign(pred=lambda d: d["valor"])], ignore_index=True)

    paths = _save_outputs("SVR_DIRECT", out_dir, df_full, df_future)

    return {
        "modelo": "SVR_DIRECT",
        "params": {"lags": L, "C": C, "gamma": gamma, "epsilon": eps,
                   "kernel": kernel, "tol": tol, "max_iter": max_iter,
                   "train_years": train_years},
        "n_futuros": int(horizon),
        **paths
    }
def run_svr_feedback(df_in: pd.DataFrame, horizon: int, out_dir: str, cfg=CFG) -> Dict:
    try:
        s = _sklearn()
        StandardScaler = s["StandardScaler"]
        Pipeline = s["Pipeline"]
        SVR = s["SVR"]
    except Exception:
        from sklearn.svm import SVR
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

    if df_in is None or df_in.empty:
        raise ValueError("Serie vacía en run_svr_feedback.")

    # ── Parámetros — mismos que DIRECT para comparación justa ────
    L        = int(getattr(cfg, "SVR_WINDOW_LEN", 60))
    C        = float(getattr(cfg, "SVR_C", 10.0))
    gamma    = getattr(cfg, "SVR_GAMMA", "scale")
    eps      = float(getattr(cfg, "SVR_EPSILON", 0.05))

    # ── Ventana de entrenamiento ──────────────────────────────────
    train_years = int(getattr(cfg, "SVR_TRAIN_YEARS", 3))
    cutoff = pd.to_datetime(df_in["fecha"]).max() - pd.DateOffset(years=train_years)
    df_train = df_in[pd.to_datetime(df_in["fecha"]) >= cutoff].copy()
    print(f"[SVR_FB] Entrenando con {len(df_train)} puntos desde {cutoff.date()} (ventana={train_years} años)")

    # ── Matrices ──────────────────────────────────────────────────
    X, y, last = _build_lag_matrix(df_train, lags=L)

    # ── Modelo ───────────────────────────────────────────────────
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="rbf", C=C, gamma=gamma, epsilon=eps)),
    ])
    model.fit(X, y)

    # ── Predicción iterativa con feedback ────────────────────────
    preds = []
    cur   = last.copy()
    for i in range(int(horizon)):
        pred = float(model.predict(cur.reshape(1, -1))[0])
        preds.append(pred)
        cur       = np.roll(cur, -1)
        cur[-1]   = pred
        if i < 5:
            print(f"[SVR_FB] paso={i+1} pred={pred:.4f}")

    print(f"[SVR_FB] pred: min={min(preds):.2f} max={max(preds):.2f} mean={np.mean(preds):.2f}")

    # ── Índice futuro ─────────────────────────────────────────────
    fechas  = pd.to_datetime(df_in["fecha"])
    freq    = _infer_freq(fechas)
    idx_fut = _future_index(fechas.iloc[-1], freq, int(horizon))

    steps   = np.arange(1, int(horizon) + 1, dtype=int)
    uses_fb = np.array([False] + [True] * max(int(horizon) - 1, 0), dtype=bool)

    df_future = pd.DataFrame({
        "fecha":         idx_fut,
        "pred":          np.array(preds, dtype=float),
        "step":          steps,
        "uses_feedback": uses_fb,
        "fuente":        getattr(cfg, "PRED_FUENTE", "sintetica"),
    })

    df_full = df_in.copy().sort_values("fecha")
    df_full["pred"] = np.nan
    fut = df_future.rename(columns={"pred": "valor"})
    for col in df_full.columns:
        if col not in fut.columns:
            fut[col] = np.nan
    df_full = pd.concat([df_full, fut.assign(pred=lambda d: d["valor"])], ignore_index=True)

    paths = _save_outputs("SVR_FEEDBACK", out_dir, df_full, df_future)

    return {
        "modelo": "SVR_FEEDBACK",
        "params": {"lags": L, "C": C, "gamma": gamma, "epsilon": eps,
                   "train_years": train_years},
        "n_futuros": int(horizon),
        **paths
    }
# ─────────────────────────────────────────────────────────────────────────────
# Runners: RandomForest (IN y DIRECT) 3
# ─────────────────────────────────────────────────────────────────────────────
def run_rf_feedback(df_in: pd.DataFrame, horizon: int, out_dir: str, cfg=CFG) -> Dict:
    """
    RF_FEEDBACK: RandomForest iterativo (con retroalimentación). Entrena una vez con originales
    y a partir del 2º paso se realimenta con sus propias predicciones.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise ImportError("Falta scikit-learn para RF: pip install scikit-learn") from e

    if df_in is None or df_in.empty:
        raise ValueError("Serie vacía en run_rf_feedback.")

    L = int(getattr(cfg, "WINDOW_LEN", 64))
    X, y, last = _build_lag_matrix(df_in, lags=L)

    n_estimators = int(getattr(cfg, "RF_N_ESTIMATORS", 300))
    max_depth = getattr(cfg, "RF_MAX_DEPTH", None)
    rs = int(getattr(cfg, "RANDOM_SEED_PREDICT", 42)) if getattr(cfg, "RANDOM_SEED_PREDICT", None) is not None else None

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None if max_depth in (None, "None") else int(max_depth),
        random_state=rs,
        n_jobs=-1,
    )
    rf.fit(Xz, y)

    preds = []
    cur = last.copy()
    for _ in range(int(horizon)):
        pred = float(rf.predict(scaler.transform(cur.reshape(1, -1)))[0])
        preds.append(pred)
        cur = np.roll(cur, -1)
        cur[-1] = pred  # feedback

    freq = _infer_freq(pd.to_datetime(df_in["fecha"]))
    idx_fut = _future_index(pd.to_datetime(df_in["fecha"].iloc[-1]), freq, int(horizon))

    steps = np.arange(1, int(horizon)+1, dtype=int)
    uses_fb = np.array([False] + [True]*(max(int(horizon)-1, 0)), dtype=bool)

    df_future = pd.DataFrame({
        "fecha": idx_fut,
        "pred": np.array(preds, dtype=float),
        "step": steps,
        "uses_feedback": uses_fb,
        "fuente": getattr(cfg, "PRED_FUENTE", "sintetica"),
    })

    df_full = df_in.copy().sort_values("fecha")
    df_full["pred"] = np.nan
    fut = df_future.rename(columns={"pred": "valor"})
    for col in df_full.columns:
        if col not in fut.columns:
            fut[col] = np.nan
    df_full = pd.concat([df_full, fut.assign(pred=lambda d: d["valor"])], ignore_index=True)

    paths = _save_outputs("RF_FEEDBACK", out_dir, df_full, df_future)
    return {
        "modelo": "RF_FEEDBACK",
        "params": {"lags": L, "n_estimators": n_estimators, "max_depth": max_depth},
        "n_futuros": int(horizon),
        **paths
    }
def run_rf_direct(df_in: pd.DataFrame, horizon: int, out_dir: str, cfg=CFG) -> Dict:
    """
    RandomForest DIRECT (sin feedback): entrena multi-salida y predice todo el vector futuro.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise ImportError("Falta scikit-learn para RF DIRECT: pip install scikit-learn") from e

    if df_in is None or df_in.empty:
        raise ValueError("Serie vacía en run_rf_direct.")

    L = int(getattr(cfg, "WINDOW_LEN", 30))
    X, Y, last = _build_direct_matrices(df_in, lags=L, horizon=int(horizon))

    n_estimators = int(getattr(cfg, "RF_N_ESTIMATORS", 300))
    max_depth = getattr(cfg, "RF_MAX_DEPTH", None)
    rs = int(getattr(cfg, "RANDOM_SEED_PREDICT", 42)) if getattr(cfg, "RANDOM_SEED_PREDICT", None) is not None else None

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None if max_depth in (None, "None") else int(max_depth),
        random_state=rs,
        n_jobs=-1,
    )
    rf.fit(Xz, Y)  # RF soporta multi-salida

    preds = rf.predict(scaler.transform(last.reshape(1, -1))).ravel().astype(float)

    freq = _infer_freq(pd.to_datetime(df_in["fecha"]))
    idx_fut = _future_index(pd.to_datetime(df_in["fecha"].iloc[-1]), freq, int(horizon))

    steps = np.arange(1, int(horizon)+1, dtype=int)
    uses_fb = np.full(int(horizon), False, dtype=bool)

    df_future = pd.DataFrame({
        "fecha": idx_fut,
        "pred": preds,
        "step": steps,
        "uses_feedback": uses_fb,
        "fuente": getattr(cfg, "PRED_FUENTE", "sintetica"),
    })

    df_full = df_in.copy().sort_values("fecha")
    df_full["pred"] = np.nan
    fut = df_future.rename(columns={"pred": "valor"})
    for col in df_full.columns:
        if col not in fut.columns:
            fut[col] = np.nan
    df_full = pd.concat([df_full, fut.assign(pred=lambda d: d["valor"])], ignore_index=True)

    paths = _save_outputs("RF_DIRECT", out_dir, df_full, df_future)
    return {
        "modelo": "RF_DIRECT",
        "params": {"lags": L, "n_estimators": n_estimators, "max_depth": max_depth},
        "n_futuros": int(horizon),
        **paths
    }
#---------------------------------------------------------------------------
#LSTM 4
# --------------------------------------------------------------------------
def run_lstm_direct(df_in: pd.DataFrame, horizon: int, out_dir: str, cfg=CFG) -> Dict:
    """
    LSTM_DIRECT: predicción multi-paso directa con LSTM (sin feedback).
    Cambios clave:
      - Entrenamiento temporalmente ordenado: shuffle=False
      - Validación ordenada: validation_split (por cfg o 0.15)
      - Escalado de Y UNIFICADO (vector) para evitar "medianas por paso"
    """
    # ── Dependencias ──
    try:
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise ImportError("Falta scikit-learn para LSTM_DIRECT: pip install scikit-learn") from e
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        tf.get_logger().setLevel("ERROR")
    except Exception as e:
        raise ImportError("Falta TensorFlow para LSTM_DIRECT: pip install tensorflow") from e

    if df_in is None or df_in.empty:
        raise ValueError("Serie vacía en run_lstm_direct.")

    # ── Validación de settings ──
    required_keys = [
        "WINDOW_LEN", "PRED_FUENTE_ORIG", "PRED_FUENTE", "RUTA_PREDICCION",
        "LSTM_HIDDEN", "LSTM_DROPOUT", "LSTM_EPOCHS", "LSTM_BATCH_SIZE",
        "LSTM_EARLY_STOP_PATIENCE", "LSTM_LOSS", "LSTM_OPTIMIZER",
    ]
    missing = [k for k in required_keys if not hasattr(cfg, k)]
    if missing:
        raise ValueError(f"[LSTM_DIRECT] Faltan claves en settings.py: {', '.join(missing)}")

    # ── Datos supervisados ──
    L = int(getattr(cfg, "WINDOW_LEN"))
    X, Y, last = _build_direct_matrices(df_in, lags=L, horizon=int(horizon))

    # ⚠️ Escaladores separados (X) y UNIFICADO (Y)
    scaler_X = StandardScaler()
    Xz = scaler_X.fit_transform(X)

    # Escala Y como un vector único (no por columna/horizonte)
    scaler_y = StandardScaler()
    Y_flat = Y.reshape(-1, 1)                 # (n_samples*horizon, 1)
    Yz_flat = scaler_y.fit_transform(Y_flat)  # escala conjunta
    Yz = Yz_flat.reshape(Y.shape)             # vuelve a (n_samples, horizon)

    Xz_seq = Xz.reshape(Xz.shape[0], L, 1)

    # ── Hiperparámetros ──
    hidden   = int(getattr(cfg, "LSTM_HIDDEN"))
    dropout  = float(getattr(cfg, "LSTM_DROPOUT"))
    epochs   = int(getattr(cfg, "LSTM_EPOCHS"))
    batch_sz = int(getattr(cfg, "LSTM_BATCH_SIZE"))
    patience = int(getattr(cfg, "LSTM_EARLY_STOP_PATIENCE"))
    loss_fn  = str(getattr(cfg, "LSTM_LOSS"))
    optim    = str(getattr(cfg, "LSTM_OPTIMIZER"))
    rs       = getattr(cfg, "RANDOM_SEED_PREDICT", None)
    val_split= float(getattr(cfg, "LSTM_VAL_SPLIT", 0.15))
    monitor  = str(getattr(cfg, "LSTM_EARLY_STOP_MONITOR", "val_loss"))

    try:
        if rs is not None:
            np.random.seed(int(rs))
            tf.random.set_seed(int(rs))
    except Exception:
        pass

    # ── Modelo ──
    model = Sequential()
    model.add(LSTM(hidden, input_shape=(L, 1), return_sequences=False))
    if dropout and dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(int(horizon)))
    model.compile(optimizer=optim, loss=loss_fn)

    es = EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True, verbose=0)
    model.fit(
        Xz_seq, Yz,
        epochs=epochs,
        batch_size=batch_sz,
        verbose=0,
        validation_split=max(0.0, min(val_split, 0.4)),
        shuffle=False,                        # ← clave para series temporales
        callbacks=[es],
    )

    # ── Predicción directa ──
    last_z = scaler_X.transform(last.reshape(1, -1)).reshape(1, L, 1)
    preds_z = model.predict(last_z, verbose=0).ravel()
    preds = scaler_y.inverse_transform(preds_z.reshape(-1, 1)).ravel()

    # ── DataFrames de salida ──
    freq = _infer_freq(pd.to_datetime(df_in["fecha"]))
    idx_fut = _future_index(pd.to_datetime(df_in["fecha"].iloc[-1]), freq, int(horizon))

    fuente_orig = getattr(cfg, "PRED_FUENTE_ORIG")
    fuente_synth = getattr(cfg, "PRED_FUENTE")

    df_future = pd.DataFrame({
        "fecha": idx_fut,
        "valor": preds.astype(float),
        "pred":  preds.astype(float),
        "fuente": fuente_synth,
    })

    df_hist = df_in.copy().sort_values("fecha")
    df_hist["pred"] = np.nan
    df_hist["fuente"] = fuente_orig

    df_full = pd.concat([df_hist[["fecha", "valor", "pred", "fuente"]], df_future], ignore_index=True)
    df_full["valor"] = df_full["valor"].astype(float)

    paths = _save_outputs("LSTM_DIRECT", out_dir, df_full, df_future[["fecha", "valor", "pred", "fuente"]])
    return {
        "modelo": "LSTM_DIRECT",
        "params": {
            "lags": L,
            "hidden": hidden,
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_sz,
            "loss": loss_fn,
            "optimizer": optim,
        },
        "n_futuros": int(horizon),
        **paths
    }
def run_lstm_feedback(df_in: pd.DataFrame, horizon: int, out_dir: str, cfg=CFG) -> Dict:
    """
    LSTM_FEEDBACK con predicción iterativa en ESPACIO DE PASOS (delta).
    - Evita la reversión a un nivel constante.
    - El "clamp" limita el paso; el "shrink" encoge el paso; alpha amortigua el paso.
    """
    # Dependencias
    try:
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise ImportError("Falta scikit-learn para LSTM_FEEDBACK") from e
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        tf.get_logger().setLevel("ERROR")
    except Exception as e:
        raise ImportError("Falta TensorFlow para LSTM_FEEDBACK") from e

    if df_in is None or df_in.empty:
        raise ValueError("Serie vacía en run_lstm_feedback.")

    # Validación de claves
    required = [
        "WINDOW_LEN","PRED_FUENTE_ORIG","PRED_FUENTE","RUTA_PREDICCION",
        "LSTM_HIDDEN","LSTM_DROPOUT","LSTM_EPOCHS","LSTM_BATCH_SIZE",
        "LSTM_EARLY_STOP_PATIENCE","LSTM_LOSS","LSTM_OPTIMIZER",
        "LSTM_DETREND_MEDIAN_WIN","LSTM_CLIP_TARGETS_MAD_K","LSTM_CLIP_TARGETS_QTL",
        "LSTM_USE_DIFF1","LSTM_VAL_SPLIT","LSTM_EARLY_STOP_MONITOR",
        "LSTM_FEEDBACK_DAMPING","LSTM_RESID_SHRINK","LSTM_PERSISTENCE_MIX",
        "LSTM_MAX_STEP_SIGMAS","LSTM_CLAMP_USE_MAD",
    ]
    missing = [k for k in required if not hasattr(cfg, k)]
    if missing:
        raise ValueError(f"[LSTM_FEEDBACK] Faltan claves en settings.py: {', '.join(missing)}")

    # Datos ordenados
    df = df_in.copy().sort_values("fecha").dropna(subset=["fecha","valor"])
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])
    s = df["valor"].astype(float).reset_index(drop=True)
    fechas = df["fecha"].reset_index(drop=True)

    # Detrend (mediana móvil centrada)
    win = int(getattr(cfg,"LSTM_DETREND_MEDIAN_WIN"))
    use_detrend = bool(win) and win > 0
    if use_detrend:
        trend = s.rolling(win, center=True, min_periods=1).median()
        trend = trend.fillna(method="bfill").fillna(method="ffill")
        resid = s - trend
    else:
        trend = pd.Series(0.0, index=s.index)
        resid = s.copy()

    # Dif. de 1er orden (opcional)
    use_diff1 = bool(getattr(cfg,"LSTM_USE_DIFF1"))
    if use_diff1:
        series_train = resid.diff().dropna().reset_index(drop=True)      # diferencias (Δy)
        fechas_train = fechas.iloc[1:].reset_index(drop=True)
        last_level_trans = float(resid.iloc[-1])                         # último nivel (resid)
    else:
        series_train = resid.copy()                                      # niveles (resid)
        fechas_train = fechas.copy()
        last_level_trans = float(resid.iloc[-1])

    # Clipping robusto (opcional)
    def _mad(x):
        med = np.median(x)
        return np.median(np.abs(x - med)) * 1.4826

    y_train = series_train.values.astype(float)
    mad_k = float(getattr(cfg,"LSTM_CLIP_TARGETS_MAD_K"))
    qtls  = getattr(cfg,"LSTM_CLIP_TARGETS_QTL")
    if mad_k and mad_k > 0:
        med = np.median(y_train); mad = _mad(y_train)
        if mad > 0:
            lo, hi = med - mad_k*mad, med + mad_k*mad
            y_train = np.clip(y_train, lo, hi)
    if isinstance(qtls,(list,tuple)) and len(qtls)==2:
        qlo, qhi = float(qtls[0]), float(qtls[1])
        lo, hi = np.quantile(y_train, qlo), np.quantile(y_train, qhi)
        y_train = np.clip(y_train, lo, hi)

    # Lags
    L = int(getattr(cfg,"WINDOW_LEN"))
    tmp = pd.DataFrame({"fecha":fechas_train, "valor":y_train})
    X, y, last = _build_lag_matrix(tmp, lags=L)

    # Escalado
    scaler_X = StandardScaler(); Xz = scaler_X.fit_transform(X)
    scaler_y = StandardScaler(); yz = scaler_y.fit_transform(y.reshape(-1,1)).ravel()
    Xz_seq = Xz.reshape(Xz.shape[0], L, 1)

    # Hiperparámetros / frenos
    hidden   = int(getattr(cfg,"LSTM_HIDDEN"))
    dropout  = float(getattr(cfg,"LSTM_DROPOUT"))
    epochs   = int(getattr(cfg,"LSTM_EPOCHS"))
    batch_sz = int(getattr(cfg,"LSTM_BATCH_SIZE"))
    patience = int(getattr(cfg,"LSTM_EARLY_STOP_PATIENCE"))
    loss_fn  = str(getattr(cfg,"LSTM_LOSS"))
    optim    = str(getattr(cfg,"LSTM_OPTIMIZER"))
    val_split= float(getattr(cfg,"LSTM_VAL_SPLIT"))
    monitor  = str(getattr(cfg,"LSTM_EARLY_STOP_MONITOR"))
    alpha    = float(getattr(cfg,"LSTM_FEEDBACK_DAMPING"))   # amortiguación del PASO
    shrink   = float(getattr(cfg,"LSTM_RESID_SHRINK"))       # encoge el PASO (0..1)
    mix_pers = float(getattr(cfg,"LSTM_PERSISTENCE_MIX"))    # mezcla con persistencia
    ksig     = float(getattr(cfg,"LSTM_MAX_STEP_SIGMAS"))    # límite en sigmas del objetivo
    use_mad  = bool(getattr(cfg,"LSTM_CLAMP_USE_MAD"))
    rs       = getattr(cfg,"RANDOM_SEED_PREDICT", None)
    try:
        if rs is not None:
            np.random.seed(int(rs)); tf.random.set_seed(int(rs))
    except Exception:
        pass

    # Dispersión para clamp (en espacio objetivo del modelo)
    disp = (_mad(y_train) if use_mad else np.std(y_train))
    max_step = float(ksig * disp) if disp > 0 else np.inf

    # Modelo
    model = Sequential()
    model.add(tf.keras.Input(shape=(L,1)))
    model.add(LSTM(hidden, return_sequences=False))
    if dropout and dropout>0: model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer=optim, loss=loss_fn)

    es = EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True, verbose=0)
    model.fit(
        Xz_seq, yz,
        epochs=epochs,
        batch_size=batch_sz,
        verbose=0,
        validation_split=max(0.0, min(val_split,0.4)),
        shuffle=False,   # ← clave para series temporales
        callbacks=[es],
    )

    # ─────────────────────────────────────────────────────────────────
    # Pred iterativa en ESPACIO DE PASOS (delta)
    #   - Si use_diff1=True: la red ya predice Δy. Clamp/shrink/alpha se aplican al Δ.
    #   - Si use_diff1=False: la red predice nivel resid; trabajamos con Δnivel = yhat - y_prev.
    # ─────────────────────────────────────────────────────────────────
    preds = []
    cur = last.astype(float).copy()                  # últimos L valores del objetivo (Δ o nivel)
    cur_last_level_trans = float(last_level_trans)   # último nivel (resid) para reconstrucción
    last_orig_value = float(s.iloc[-1])              # para mezcla con persistencia

    for _ in range(int(horizon)):
        # 1) Predicción en espacio transformado
        cur_z = scaler_X.transform(cur.reshape(1,-1)).ravel().reshape(1,L,1)
        yhat_z = float(model.predict(cur_z, verbose=0).ravel()[0])
        yhat_trans = float(scaler_y.inverse_transform([[yhat_z]])[0,0])

        # 2) Paso (delta) en espacio transformado
        if use_diff1:
            # La red predice directamente el Δy (siguiente diferencia)
            step_trans = yhat_trans
        else:
            # La red predice nivel resid; usamos Δnivel respecto al último valor de la ventana
            y_prev = float(cur[-1])
            step_trans = (yhat_trans - y_prev)

        # Encogimiento y amortiguación del PASO
        step_trans = (1.0 - shrink) * step_trans
        step_trans = alpha * step_trans

        # Clamp del PASO
        if np.isfinite(max_step) and abs(step_trans) > max_step:
            step_trans = np.sign(step_trans) * max_step

        # 3) Actualizar nivel transformado
        yhat_level_trans = cur_last_level_trans + step_trans

        # 4) Reconstrucción a escala original (añadiendo tendencia si aplica)
        yhat_orig = (yhat_level_trans + float(trend.iloc[-1])) if use_detrend else yhat_level_trans

        # 5) Mezcla con persistencia (si está activada)
        yhat_orig = (1.0 - mix_pers) * yhat_orig + mix_pers * last_orig_value

        preds.append(float(yhat_orig))

        # 6) Feedback: deslizar ventana en el espacio objetivo del modelo
        cur = np.roll(cur, -1)
        if use_diff1:
            # La ventana contiene diferencias → añadimos la diferencia predicha (Δy)
            cur[-1] = step_trans
        else:
            # La ventana contiene niveles (resid) → añadimos el nuevo nivel
            cur[-1] = cur[-2] + step_trans  # equivalente a y_prev + Δnivel

        cur_last_level_trans = yhat_level_trans
        last_orig_value = preds[-1]

    # Índices y salidas
    freq = _infer_freq(pd.to_datetime(df["fecha"]))
    idx_fut = _future_index(pd.to_datetime(df["fecha"].iloc[-1]), freq, int(horizon))

    fuente_orig = getattr(cfg,"PRED_FUENTE_ORIG")
    fuente_synth = getattr(cfg,"PRED_FUENTE")

    df_future = pd.DataFrame({
        "fecha": idx_fut,
        "valor": np.array(preds, dtype=float),
        "pred":  np.array(preds, dtype=float),
        "fuente": fuente_synth,
    })
    df_hist = df[["fecha","valor"]].copy()
    df_hist["pred"] = np.nan
    df_hist["fuente"] = fuente_orig

    df_full = pd.concat([df_hist[["fecha","valor","pred","fuente"]], df_future], ignore_index=True)
    df_full["valor"] = df_full["valor"].astype(float)

    paths = _save_outputs("LSTM_FEEDBACK", out_dir, df_full, df_future[["fecha","valor","pred","fuente"]])
    return {
        "modelo": "LSTM_FEEDBACK",
        "params": {
            "lags": L, "hidden": hidden, "dropout": dropout, "epochs": epochs, "batch_size": batch_sz,
            "loss": loss_fn, "optimizer": optim,
            "detrend_win": win, "clip_mad_k": mad_k, "clip_qtl": qtls,
            "use_diff1": use_diff1, "val_split": val_split, "monitor": monitor,
            "damping": alpha, "shrink": shrink, "persistence_mix": mix_pers,
            "max_step_sigmas": ksig, "clamp_use_mad": use_mad,
        },
        "n_futuros": int(horizon),
        **paths
    }
#-----------------------------------------------------------------------
# propheat 5
#-----------------------------------------------------------------------
def run_prophet_direct(df_in: pd.DataFrame, horizon: int, out_dir: str, cfg=CFG) -> Dict:
    """
    PROPHET_DIRECT: un solo ajuste y predicción multi-paso (h) de una vez.
    Salida estándar:
      - df_full: fecha|valor|pred|fuente
      - df_future: solo futuro con predicciones
    """
    Prophet = _import_prophet()
    if df_in is None or df_in.empty:
        raise ValueError("Serie vacía en run_prophet_direct.")

    # Datos base
    df = df_in.copy().sort_values("fecha").dropna(subset=["fecha", "valor"])
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])
    df_ds_y = pd.DataFrame({"ds": df["fecha"], "y": df["valor"].astype(float)})

    # Modelo
    m = _prophet_from_cfg(cfg)
    df_ds_y_log, cap, floor = _apply_logistic_bounds(df_ds_y, cfg)
    m.fit(df_ds_y_log)

    # Fechas futuras desde helpers del pipeline
    freq = _infer_freq(df["fecha"])
    idx_fut = _future_index(df["fecha"].iloc[-1], freq, int(horizon))

    # Predicción
    yhat = _prophet_predict(m, idx_fut, cap=cap, floor=floor)

    # Salidas estándar
    fuente_orig = getattr(cfg, "PRED_FUENTE_ORIG")
    fuente_synth = getattr(cfg, "PRED_FUENTE")

    df_future = pd.DataFrame({
        "fecha": idx_fut,
        "valor": yhat.astype(float),
        "pred":  yhat.astype(float),
        "fuente": fuente_synth,
    })
    df_hist = df[["fecha", "valor"]].copy()
    df_hist["pred"] = np.nan
    df_hist["fuente"] = fuente_orig
    df_full = pd.concat([df_hist[["fecha","valor","pred","fuente"]], df_future], ignore_index=True)
    df_full["valor"] = df_full["valor"].astype(float)

    paths = _save_outputs("PROPHET_DIRECT", out_dir, df_full, df_future[["fecha","valor","pred","fuente"]])
    return {
        "modelo": "PROPHET_DIRECT",
        "params": {
            "growth": getattr(cfg,"PROPHET_GROWTH","linear"),
            "seasonality_mode": getattr(cfg,"PROPHET_SEASONALITY_MODE","additive"),
            "n_changepoints": getattr(cfg,"PROPHET_N_CHANGEPOINTS",25),
            "changepoint_prior": getattr(cfg,"PROPHET_CHANGEPOINT_PRIOR",0.05),
        },
        "n_futuros": int(horizon),
        **paths
    }
def run_prophet_feedback(df_in: pd.DataFrame, horizon: int, out_dir: str, cfg=CFG) -> Dict:
    """
    PROPHET_FEEDBACK: predicción iterativa h-pasos con *re-ajuste por paso*.
    (Similar a "feedback": cada nuevo punto predicho se agrega al historial y se re-ajusta el modelo.)
    Nota: es más costoso (O(h) ajustes).
    """
    Prophet = _import_prophet()
    if df_in is None or df_in.empty:
        raise ValueError("Serie vacía en run_prophet_feedback.")

    df = df_in.copy().sort_values("fecha").dropna(subset=["fecha", "valor"])
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])
    base_ds_y = pd.DataFrame({"ds": df["fecha"], "y": df["valor"].astype(float)})

    # Fechas futuras (todas) para ir pronosticando paso a paso
    freq = _infer_freq(df["fecha"])
    idx_fut = _future_index(df["fecha"].iloc[-1], freq, int(horizon))

    preds = []
    train_ds_y = base_ds_y.copy()
    cap = floor = None

    for dt in idx_fut:
        m = _prophet_from_cfg(cfg)
        train_ds_y_log, cap, floor = _apply_logistic_bounds(train_ds_y, cfg)
        m.fit(train_ds_y_log)

        # predecir SOLO el siguiente paso
        yhat_next = float(_prophet_predict(m, pd.DatetimeIndex([dt]), cap=cap, floor=floor)[0])
        preds.append(yhat_next)

        # retroalimentación: añadir el punto predicho a la serie de entrenamiento
        train_ds_y = pd.concat([train_ds_y, pd.DataFrame({"ds":[dt], "y":[yhat_next]})], ignore_index=True)

    # Salidas estándar
    fuente_orig = getattr(cfg, "PRED_FUENTE_ORIG")
    fuente_synth = getattr(cfg, "PRED_FUENTE")

    df_future = pd.DataFrame({
        "fecha": idx_fut,
        "valor": np.array(preds, dtype=float),
        "pred":  np.array(preds, dtype=float),
        "fuente": fuente_synth,
    })
    df_hist = df[["fecha","valor"]].copy()
    df_hist["pred"] = np.nan
    df_hist["fuente"] = fuente_orig
    df_full = pd.concat([df_hist[["fecha","valor","pred","fuente"]], df_future], ignore_index=True)
    df_full["valor"] = df_full["valor"].astype(float)

    paths = _save_outputs("PROPHET_FEEDBACK", out_dir, df_full, df_future[["fecha","valor","pred","fuente"]])
    return {
        "modelo": "PROPHET_FEEDBACK",
        "params": {
            "growth": getattr(cfg,"PROPHET_GROWTH","linear"),
            "seasonality_mode": getattr(cfg,"PROPHET_SEASONALITY_MODE","additive"),
            "n_changepoints": getattr(cfg,"PROPHET_N_CHANGEPOINTS",25),
            "changepoint_prior": getattr(cfg,"PROPHET_CHANGEPOINT_PRIOR",0.05),
        },
        "n_futuros": int(horizon),
        **paths
    }
#-----------
#nbeats 6
#----------------------------------
def run_nbeats_direct(df_in: pd.DataFrame, horizon: int, out_dir: str, cfg=CFG) -> Dict:

    # Dependencias
    try:
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise ImportError("Falta scikit-learn para N-BEATS: pip install scikit-learn") from e
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
    except Exception as e:
        raise ImportError("Falta TensorFlow para N-BEATS: pip install tensorflow") from e

    if df_in is None or df_in.empty:
        raise ValueError("Serie vacía en run_nbeats_direct.")

    # Config
    L = int(getattr(cfg, "WINDOW_LEN", 30))
    val_split = float(getattr(cfg, "NBEATS_VAL_SPLIT", 0.15))
    epochs = int(getattr(cfg, "NBEATS_EPOCHS", 200))
    batch_sz = int(getattr(cfg, "NBEATS_BATCH_SIZE", 64))
    patience = int(getattr(cfg, "NBEATS_EARLY_STOP_PATIENCE", 10))
    rs = getattr(cfg, "RANDOM_SEED_PREDICT", None)
    k_sigma = float(getattr(cfg, "ANOM_K_SIGMA", 3.0))
    back_steps = int(getattr(cfg, "PRED_BACKTEST_STEPS", 90))

    # Datos ordenados
    df = df_in.copy().sort_values("fecha").dropna(subset=["fecha", "valor"])
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])
    y_series = df["valor"].astype(float).values
    fechas = df["fecha"].values

    if len(y_series) < (L + max(2, horizon)):
        raise ValueError(f"NBEATS_DIRECT requiere al menos L+H puntos. L={L}, H={horizon}")

    # Ventanas supervisadas multi-horizon y última ventana
    X, Y, last = _build_direct_matrices(df, lags=L, horizon=int(horizon))  # usa tu helper

    # Escalado: X estándar, Y estándar unificado (flatten)
    scaler_X = StandardScaler()
    Xz = scaler_X.fit_transform(X)
    scaler_y = StandardScaler()
    Yz = scaler_y.fit_transform(Y.reshape(-1, 1)).reshape(Y.shape)

    # Semilla
    try:
        if rs is not None:
            np.random.seed(int(rs))
            tf.random.set_seed(int(rs))
    except Exception:
        pass

    # Modelo N-BEATS
    model = _nbeats_build_model(L, int(horizon), cfg=cfg)

    # Early stopping
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0
    )

    # Entrenamiento (muestras temporales; no barajamos)
    model.fit(
        Xz, Yz,
        epochs=epochs, batch_size=batch_sz, verbose=0,
        validation_split=max(0.0, min(val_split, 0.4)),
        shuffle=False, callbacks=[es]
    )

    # Predicción del horizonte (desde la última ventana real)
    last_z = scaler_X.transform(last.reshape(1, -1))
    preds_z = model.predict(last_z, verbose=0).ravel()
    preds = scaler_y.inverse_transform(preds_z.reshape(-1, 1)).ravel()

    # Salidas estándar
    freq = _infer_freq(df["fecha"])
    idx_fut = _future_index(df["fecha"].iloc[-1], freq, int(horizon))
    fuente_orig = getattr(cfg, "PRED_FUENTE_ORIG", "original")
    fuente_synth = getattr(cfg, "PRED_FUENTE", "sintetica")

    df_future = pd.DataFrame({
        "fecha": idx_fut,
        "valor": preds.astype(float),
        "pred":  preds.astype(float),
        "fuente": fuente_synth,
    })
    df_hist = df[["fecha", "valor"]].copy()
    df_hist["pred"] = np.nan
    df_hist["fuente"] = fuente_orig
    df_full = pd.concat([df_hist[["fecha","valor","pred","fuente"]], df_future], ignore_index=True)
    df_full["valor"] = df_full["valor"].astype(float)

    paths = _save_outputs("NBEATS_DIRECT", out_dir, df_full, df_future[["fecha","valor","pred","fuente"]])

    # ── ANOMALÍAS (backtest 1-paso) en los últimos PRED_BACKTEST_STEPS ──
    # Usamos la primera columna (t+1) de la salida multi-h
    # Mapeo de ventanas -> fecha real: ventana i predice y en t = L+i
    yhat_train_z = model.predict(Xz, verbose=0)                      # (n_samples, H)
    yhat_train = scaler_y.inverse_transform(yhat_train_z.reshape(-1,1)).reshape(yhat_train_z.shape)
    y_true_next = []
    y_pred_next = []
    fechas_next = []
    n_samples = X.shape[0]
    for i in range(n_samples):
        t_next = L + i  # índice en la serie original del punto y_{t+1}
        if t_next < len(y_series):
            y_true_next.append(y_series[t_next])
            y_pred_next.append(yhat_train[i, 0])  # 1er paso de la pred multi-h
            fechas_next.append(fechas[t_next])

    y_true_next = np.array(y_true_next, dtype=float)
    y_pred_next = np.array(y_pred_next, dtype=float)
    fechas_next = np.array(fechas_next, dtype="datetime64[ns]")

    # Tomamos solo los últimos 'back_steps'
    if back_steps > 0 and len(y_true_next) > back_steps:
        y_true_bt = y_true_next[-back_steps:]
        y_pred_bt = y_pred_next[-back_steps:]
        fechas_bt = fechas_next[-back_steps:]
    else:
        y_true_bt, y_pred_bt, fechas_bt = y_true_next, y_pred_next, fechas_next

    idxs_anom, fechas_anom, vals_anom = _detect_anomalies_mad(fechas_bt, y_true_bt, y_pred_bt, k_sigma)

    # Guardar anomalías crudas (solo fecha, valor)
    anom_path = None
    ruta_anom = getattr(cfg, "RUTA_ANOMALIAS_METODO", None)
    if ruta_anom:
        try:
            os.makedirs(ruta_anom, exist_ok=True)
            anom_df = pd.DataFrame({"fecha": fechas_anom, "valor": vals_anom})
            # aseguremos formato estándar de fecha si tu _save_outputs usa str
            if hasattr(anom_df["fecha"], "dt"):
                anom_df["fecha"] = pd.to_datetime(anom_df["fecha"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            #anom_path = os.path.join(ruta_anom, "NBEATS_puntos.csv")
            anom_path = os.path.join(ruta_anom, "NBEATS_DIRECT_puntos.csv")

            anom_df.to_csv(anom_path, index=False)
        except Exception as e:
            # no interrumpir el runner por un fallo de guardado de anomalías
            print(f"[NBEATS] Advertencia: no se pudo guardar anomalías en {ruta_anom}: {e}")

    return {
        "modelo": "NBEATS_DIRECT",
        "params": {
            "lags": L,
            "stacks": int(getattr(cfg, "NBEATS_STACKS", 2)),
            "blocks_per_stack": int(getattr(cfg, "NBEATS_BLOCKS_PER_STACK", 3)),
            "width": int(getattr(cfg, "NBEATS_WIDTH", 256)),
            "fc_layers": int(getattr(cfg, "NBEATS_FC_LAYERS", 4)),
            "activation": str(getattr(cfg, "NBEATS_ACTIVATION", "relu")),
            "dropout": float(getattr(cfg, "NBEATS_DROPOUT", 0.0)),
            "loss": str(getattr(cfg, "NBEATS_LOSS", "huber")),
            "optimizer": str(getattr(cfg, "NBEATS_OPTIMIZER", "adam")),
            "val_split": float(getattr(cfg, "NBEATS_VAL_SPLIT", 0.15)),
        },
        "n_futuros": int(horizon),
        "anomalias_csv": anom_path,
        **paths
    }

def run_nbeats_feedbackv1(df_in: pd.DataFrame, horizon: int, out_dir: str, cfg=CFG) -> Dict:
    """
    NBEATS_FEEDBACK:
      - Entrena N-BEATS para 1 paso (y_{t+1}) con ventanas de longitud WINDOW_LEN.
      - Pronostica h pasos de forma RECURSIVA (feedback) desde la última ventana real.
      - Genera ANOMALÍAS (backtest 1-paso) con umbral robusto k·MAD en los últimos PRED_BACKTEST_STEPS.
    Salidas estándar:
      - df_full (histórico+futuro): fecha|valor|pred|fuente
      - df_future (futuro):         fecha|valor|pred|fuente
      - Guarda anomalías crudas en: RUTA_ANOMALIAS_METODO/NBEATS_FEEDBACK_puntos.csv (fecha, valor)
    """
    # Dependencias
    try:
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise ImportError("Falta scikit-learn para N-BEATS: pip install scikit-learn") from e
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
    except Exception as e:
        raise ImportError("Falta TensorFlow para N-BEATS: pip install tensorflow") from e

    if df_in is None or df_in.empty:
        raise ValueError("Serie vacía en run_nbeats_feedback.")

    # Config
    L         = int(getattr(cfg, "WINDOW_LEN", 30))
    val_split = float(getattr(cfg, "NBEATS_VAL_SPLIT", 0.15))
    epochs    = int(getattr(cfg, "NBEATS_EPOCHS", 200))
    batch_sz  = int(getattr(cfg, "NBEATS_BATCH_SIZE", 64))
    patience  = int(getattr(cfg, "NBEATS_EARLY_STOP_PATIENCE", 10))
    rs        = getattr(cfg, "RANDOM_SEED_PREDICT", None)
    k_sigma   = float(getattr(cfg, "ANOM_K_SIGMA", 3.0))
    back_steps= int(getattr(cfg, "PRED_BACKTEST_STEPS", 90))

    # Datos ordenados
    df = df_in.copy().sort_values("fecha").dropna(subset=["fecha","valor"])
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])
    y_series = df["valor"].astype(float).values
    fechas   = df["fecha"].values

    if len(y_series) < (L + max(2, horizon)):
        raise ValueError(f"NBEATS_FEEDBACK requiere al menos L+H puntos. L={L}, H={horizon}")

    # Ventanas 1-paso (X: Lags, y: y_{t+1}) + última ventana real
    X, y, last = _build_lag_matrix(df[["fecha","valor"]], lags=L)  # usa tu helper

    # Escalado: X estándar, y estándar
    scaler_X = StandardScaler(); Xz = scaler_X.fit_transform(X)
    scaler_y = StandardScaler(); yz = scaler_y.fit_transform(y.reshape(-1,1)).ravel()

    # Semilla
    try:
        if rs is not None:
            np.random.seed(int(rs))
            tf.random.set_seed(int(rs))
    except Exception:
        pass

    # Modelo N-BEATS (salida 1)
    model = _nbeats_build_model(L, 1, cfg=cfg)

    # Early stopping
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0
    )

    # Entrenamiento (orden temporal; sin barajar)
    model.fit(
        Xz, yz,
        epochs=epochs, batch_size=batch_sz, verbose=0,
        validation_split=max(0.0, min(val_split, 0.4)),
        shuffle=False, callbacks=[es]
    )

    # ── Predicción recursiva h-pasos desde la última ventana real ──
    preds = []
    cur_window = last.astype(float).copy()  # en espacio ORIGINAL (no escalado)
    for _ in range(int(horizon)):
        # Escalar la ventana actual a espacio de features
        cur_z = scaler_X.transform(cur_window.reshape(1, -1))
        yhat_z = float(model.predict(cur_z, verbose=0).ravel()[0])  # escala de y
        yhat   = float(scaler_y.inverse_transform([[yhat_z]])[0,0]) # vuelve a original
        preds.append(yhat)
        # feedback: deslizar ventana con la predicción
        cur_window = np.roll(cur_window, -1)
        cur_window[-1] = yhat

    # Salidas estándar
    freq = _infer_freq(df["fecha"])
    idx_fut = _future_index(df["fecha"].iloc[-1], freq, int(horizon))
    fuente_orig  = getattr(cfg, "PRED_FUENTE_ORIG", "original")
    fuente_synth = getattr(cfg, "PRED_FUENTE", "sintetica")

    df_future = pd.DataFrame({
        "fecha": idx_fut,
        "valor": np.array(preds, dtype=float),
        "pred":  np.array(preds, dtype=float),
        "fuente": fuente_synth,
    })
    df_hist = df[["fecha","valor"]].copy()
    df_hist["pred"] = np.nan
    df_hist["fuente"] = fuente_orig
    df_full = pd.concat([df_hist[["fecha","valor","pred","fuente"]], df_future], ignore_index=True)
    df_full["valor"] = df_full["valor"].astype(float)

    paths = _save_outputs("NBEATS_FEEDBACK", out_dir, df_full, df_future[["fecha","valor","pred","fuente"]])

    # ── ANOMALÍAS (backtest 1-paso) ──
    # Predicción 1-paso sobre todas las ventanas del dataset
    yhat_train_z = model.predict(Xz, verbose=0).ravel()
    yhat_train   = scaler_y.inverse_transform(yhat_train_z.reshape(-1,1)).ravel()

    # Mapear cada ventana i a la fecha real t=L+i (el punto y_{t+1})
    y_true_next = []
    y_pred_next = []
    fechas_next = []
    n_samples = X.shape[0]
    for i in range(n_samples):
        t_next = L + i
        if t_next < len(y_series):
            y_true_next.append(y_series[t_next])
            y_pred_next.append(yhat_train[i])
            fechas_next.append(fechas[t_next])

    y_true_next = np.array(y_true_next, dtype=float)
    y_pred_next = np.array(y_pred_next, dtype=float)
    fechas_next = np.array(fechas_next, dtype="datetime64[ns]")

    # Últimos 'back_steps'
    if back_steps > 0 and len(y_true_next) > back_steps:
        y_true_bt = y_true_next[-back_steps:]
        y_pred_bt = y_pred_next[-back_steps:]
        fechas_bt = fechas_next[-back_steps:]
    else:
        y_true_bt, y_pred_bt, fechas_bt = y_true_next, y_pred_next, fechas_next

    # Detección robusta (k·MAD)
    idxs_anom, fechas_anom, vals_anom = _detect_anomalies_mad(fechas_bt, y_true_bt, y_pred_bt, k_sigma)

    # Guardar anomalías crudas (solo fecha, valor)
    anom_path = None
    ruta_anom = getattr(cfg, "RUTA_ANOMALIAS_METODO", None)
    if ruta_anom:
        try:
            os.makedirs(ruta_anom, exist_ok=True)
            anom_df = pd.DataFrame({"fecha": fechas_anom, "valor": vals_anom})
            if hasattr(anom_df["fecha"], "dt"):
                anom_df["fecha"] = pd.to_datetime(anom_df["fecha"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            anom_path = os.path.join(ruta_anom, "NBEATS_FEEDBACK_puntos.csv")
            anom_df.to_csv(anom_path, index=False)
        except Exception as e:
            print(f"[NBEATS_FEEDBACK] Advertencia: no se pudo guardar anomalías en {ruta_anom}: {e}")

    return {
        "modelo": "NBEATS_FEEDBACK",
        "params": {
            "lags": L,
            "stacks": int(getattr(cfg, "NBEATS_STACKS", 2)),
            "blocks_per_stack": int(getattr(cfg, "NBEATS_BLOCKS_PER_STACK", 3)),
            "width": int(getattr(cfg, "NBEATS_WIDTH", 256)),
            "fc_layers": int(getattr(cfg, "NBEATS_FC_LAYERS", 4)),
            "activation": str(getattr(cfg, "NBEATS_ACTIVATION", "relu")),
            "dropout": float(getattr(cfg, "NBEATS_DROPOUT", 0.0)),
            "loss": str(getattr(cfg, "NBEATS_LOSS", "huber")),
            "optimizer": str(getattr(cfg, "NBEATS_OPTIMIZER", "adam")),
            "val_split": float(getattr(cfg, "NBEATS_VAL_SPLIT", 0.15)),
        },
        "n_futuros": int(horizon),
        "anomalias_csv": anom_path,
        **paths
    }

def run_nbeats_feedback(df_in: pd.DataFrame, horizon: int, out_dir: str, cfg=CFG) -> Dict:
    """
    NBEATS_FEEDBACK corregido:
    - Entrena sobre DIFERENCIAS (Δy) en vez de niveles
    - Esto garantiza que la ventana final tenga variación incluso si la serie está plana
    - La predicción se reconstruye sumando las diferencias al último valor real
    """
    try:
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise ImportError("Falta scikit-learn para N-BEATS: pip install scikit-learn") from e
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
    except Exception as e:
        raise ImportError("Falta TensorFlow para N-BEATS: pip install tensorflow") from e

    if df_in is None or df_in.empty:
        raise ValueError("Serie vacía en run_nbeats_feedback.")

    # Config
    L          = int(getattr(cfg, "WINDOW_LEN", 30))
    val_split  = float(getattr(cfg, "NBEATS_VAL_SPLIT", 0.15))
    epochs     = int(getattr(cfg, "NBEATS_EPOCHS", 200))
    batch_sz   = int(getattr(cfg, "NBEATS_BATCH_SIZE", 64))
    patience   = int(getattr(cfg, "NBEATS_EARLY_STOP_PATIENCE", 10))
    rs         = getattr(cfg, "RANDOM_SEED_PREDICT", None)
    k_sigma    = float(getattr(cfg, "ANOM_K_SIGMA", 3.0))
    back_steps = int(getattr(cfg, "PRED_BACKTEST_STEPS", 90))

    # Datos ordenados
    df = df_in.copy().sort_values("fecha").dropna(subset=["fecha", "valor"])
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])
    y_series = df["valor"].astype(float).values
    fechas   = df["fecha"].values

    if len(y_series) < (L + max(2, horizon)):
        raise ValueError(f"NBEATS_FEEDBACK requiere al menos L+H puntos. L={L}, H={horizon}")

    # ── CORRECCIÓN: trabajar en espacio de DIFERENCIAS ──────────────────
    # Δy[t] = y[t] - y[t-1]
    # Esto garantiza variación incluso si la serie termina plana
    y_diff = np.diff(y_series)           # (N-1,) diferencias
    last_real_value = float(y_series[-1])  # para reconstruir niveles al final

    # Construir lag matrix sobre diferencias
    df_diff = pd.DataFrame({
        "fecha": fechas[1:],   # una menos por el diff
        "valor": y_diff
    })
    X, y, last_diff = _build_lag_matrix(df_diff, lags=L)

    # Escalado estándar sobre diferencias
    scaler_X = StandardScaler(); Xz = scaler_X.fit_transform(X)
    scaler_y = StandardScaler(); yz = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # Semilla
    try:
        if rs is not None:
            np.random.seed(int(rs))
            tf.random.set_seed(int(rs))
    except Exception:
        pass

    # Modelo N-BEATS (salida 1)
    model = _nbeats_build_model(L, 1, cfg=cfg)

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0
    )

    model.fit(
        Xz, yz,
        epochs=epochs, batch_size=batch_sz, verbose=0,
        validation_split=max(0.0, min(val_split, 0.4)),
        shuffle=False, callbacks=[es]
    )

    # ── Predicción recursiva en espacio de DIFERENCIAS ──────────────────
    preds_diff = []
    cur_window = last_diff.astype(float).copy()  # últimas L diferencias reales

    for _ in range(int(horizon)):
        cur_z  = scaler_X.transform(cur_window.reshape(1, -1))
        dhat_z = float(model.predict(cur_z, verbose=0).ravel()[0])
        dhat   = float(scaler_y.inverse_transform([[dhat_z]])[0, 0])  # Δy predicho
        preds_diff.append(dhat)

        # feedback en espacio de diferencias
        cur_window = np.roll(cur_window, -1)
        cur_window[-1] = dhat

    # ── Reconstruir niveles: y[t] = last_real + cumsum(Δy) ──────────────
    preds = last_real_value + np.cumsum(preds_diff)
    print(f"[NBEATS_FB] pred: min={preds.min():.2f} max={preds.max():.2f} "
          f"mean={preds.mean():.2f} | último real={last_real_value:.2f}")

    # Salidas estándar
    freq = _infer_freq(df["fecha"])
    idx_fut = _future_index(df["fecha"].iloc[-1], freq, int(horizon))
    fuente_orig  = getattr(cfg, "PRED_FUENTE_ORIG", "original")
    fuente_synth = getattr(cfg, "PRED_FUENTE", "sintetica")

    df_future = pd.DataFrame({
        "fecha": idx_fut,
        "valor": np.array(preds, dtype=float),
        "pred":  np.array(preds, dtype=float),
        "fuente": fuente_synth,
    })
    df_hist = df[["fecha", "valor"]].copy()
    df_hist["pred"] = np.nan
    df_hist["fuente"] = fuente_orig
    df_full = pd.concat([df_hist[["fecha", "valor", "pred", "fuente"]], df_future], ignore_index=True)
    df_full["valor"] = df_full["valor"].astype(float)

    paths = _save_outputs("NBEATS_FEEDBACK", out_dir, df_full, df_future[["fecha", "valor", "pred", "fuente"]])

    # ── ANOMALÍAS (backtest 1-paso en espacio de diferencias) ──────────
    yhat_train_z = model.predict(Xz, verbose=0).ravel()
    yhat_train_diff = scaler_y.inverse_transform(yhat_train_z.reshape(-1, 1)).ravel()

    # Reconstruir niveles del backtest
    y_true_next = []
    y_pred_next = []
    fechas_next = []
    n_samples = X.shape[0]
    for i in range(n_samples):
        t_next = L + 1 + i  # +1 por el diff inicial
        if t_next < len(y_series):
            y_true_next.append(y_series[t_next])
            # reconstruir nivel: nivel anterior + delta predicho
            y_pred_next.append(y_series[t_next - 1] + yhat_train_diff[i])
            fechas_next.append(fechas[t_next])

    y_true_next = np.array(y_true_next, dtype=float)
    y_pred_next = np.array(y_pred_next, dtype=float)
    fechas_next = np.array(fechas_next, dtype="datetime64[ns]")

    if back_steps > 0 and len(y_true_next) > back_steps:
        y_true_bt = y_true_next[-back_steps:]
        y_pred_bt = y_pred_next[-back_steps:]
        fechas_bt = fechas_next[-back_steps:]
    else:
        y_true_bt, y_pred_bt, fechas_bt = y_true_next, y_pred_next, fechas_next

    idxs_anom, fechas_anom, vals_anom = _detect_anomalies_mad(fechas_bt, y_true_bt, y_pred_bt, k_sigma)

    anom_path = None
    ruta_anom = getattr(cfg, "RUTA_ANOMALIAS_METODO", None)
    if ruta_anom:
        try:
            os.makedirs(ruta_anom, exist_ok=True)
            anom_df = pd.DataFrame({"fecha": fechas_anom, "valor": vals_anom})
            if hasattr(anom_df["fecha"], "dt"):
                anom_df["fecha"] = pd.to_datetime(anom_df["fecha"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            anom_path = os.path.join(ruta_anom, "NBEATS_FEEDBACK_puntos.csv")
            anom_df.to_csv(anom_path, index=False)
        except Exception as e:
            print(f"[NBEATS_FEEDBACK] Advertencia: no se pudo guardar anomalías en {ruta_anom}: {e}")

    return {
        "modelo": "NBEATS_FEEDBACK",
        "params": {
            "lags": L,
            "stacks": int(getattr(cfg, "NBEATS_STACKS", 2)),
            "blocks_per_stack": int(getattr(cfg, "NBEATS_BLOCKS_PER_STACK", 3)),
            "width": int(getattr(cfg, "NBEATS_WIDTH", 256)),
            "fc_layers": int(getattr(cfg, "NBEATS_FC_LAYERS", 4)),
            "activation": str(getattr(cfg, "NBEATS_ACTIVATION", "relu")),
            "dropout": float(getattr(cfg, "NBEATS_DROPOUT", 0.0)),
            "loss": str(getattr(cfg, "NBEATS_LOSS", "huber")),
            "optimizer": str(getattr(cfg, "NBEATS_OPTIMIZER", "adam")),
            "val_split": float(getattr(cfg, "NBEATS_VAL_SPLIT", 0.15)),
            "use_diff": True,
        },
        "n_futuros": int(horizon),
        "anomalias_csv": anom_path,
        **paths
    }


#-----------------------------------
#deepar 7
#------------------------------------

def run_deepar_direct(df_in: pd.DataFrame, horizon: int, out_dir: str, cfg=CFG) -> Dict:
    """
    DEEPAR_DIRECT (CORREGIDO para tu definición):
      - 1 solo fit
      - 1 sola predicción que devuelve H valores (multi-salida real, NO recursivo)
      - Backtest de anomalías usando el 1er paso (t+1) de la salida multi-H sobre ventanas de train
      - Guarda anomalías en: RUTA_ANOMALIAS_METODO/DEEPAR_DIRECT_puntos.csv
    Salidas estándar:
      - df_full (histórico+futuro): fecha|valor|pred|fuente
      - df_future (futuro):         fecha|valor|pred|fuente
    """
    # ── Dependencias ──
    try:
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise ImportError("DeepAR_DIRECT requiere scikit-learn. Instala: pip install scikit-learn") from e

    try:
        import tensorflow as tf
        from tensorflow.keras import Model
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        tf.get_logger().setLevel("ERROR")
    except Exception as e:
        raise ImportError("DeepAR_DIRECT requiere tensorflow. Instala: pip install tensorflow") from e

    if df_in is None or df_in.empty:
        raise ValueError("Serie vacía en run_deepar_direct.")

    # ── Config ──
    H          = int(horizon)
    L          = int(getattr(cfg, "WINDOW_LEN", 30))
    epochs     = int(getattr(cfg, "DEEPAR_EPOCHS", 120))
    batch_sz   = int(getattr(cfg, "DEEPAR_BATCH_SIZE", 64))
    val_split  = float(getattr(cfg, "DEEPAR_VAL_SPLIT", 0.15))
    patience   = int(getattr(cfg, "DEEPAR_EARLY_STOP_PATIENCE", 8))
    rs         = getattr(cfg, "RANDOM_SEED_PREDICT", None)
    k_sigma    = float(getattr(cfg, "ANOM_K_SIGMA", 3.0))
    back_steps = int(getattr(cfg, "PRED_BACKTEST_STEPS", 90))

    units      = int(getattr(cfg, "DEEPAR_UNITS", 64))
    layers_n   = int(getattr(cfg, "DEEPAR_LAYERS", 1))   # n capas LSTM apiladas
    dropout    = float(getattr(cfg, "DEEPAR_DROPOUT", 0.1))
    loss_fn    = str(getattr(cfg, "DEEPAR_LOSS", "huber"))
    optim      = str(getattr(cfg, "DEEPAR_OPTIMIZER", "adam"))

    # ── Datos ordenados (CORRECTO: convertir a datetime ANTES de ordenar) ──
    df = df_in.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha", "valor"]).sort_values("fecha")
    df["valor"] = df["valor"].astype(float)

    y_series = df["valor"].values.astype(float)
    fechas   = df["fecha"].values

    if len(y_series) < (L + max(2, H)):
        raise ValueError(f"DEEPAR_DIRECT requiere al menos L+H puntos. L={L}, H={H}, n={len(y_series)}")

    # ── Ventanas multi-horizon (X, Y) + última ventana ──
    X, Y, last = _build_direct_matrices(df, lags=L, horizon=H)

    # ── Escalado: X estándar; Y estándar unificado (flatten) ──
    scaler_X = StandardScaler()
    Xz = scaler_X.fit_transform(X)

    scaler_y = StandardScaler()
    Yz = scaler_y.fit_transform(Y.reshape(-1, 1)).reshape(Y.shape)

    Xz_seq = Xz.reshape(Xz.shape[0], L, 1)

    # ── Semilla ──
    try:
        if rs is not None:
            np.random.seed(int(rs))
            tf.random.set_seed(int(rs))
    except Exception:
        pass

    # ── Modelo "DeepAR-like" multi-salida (H) ──
    inp = Input(shape=(L, 1))
    x = inp
    for li in range(max(1, layers_n)):
        return_seq = (li < max(1, layers_n) - 1)
        x = LSTM(units, return_sequences=return_seq)(x)
        if dropout and dropout > 0:
            x = Dropout(dropout)(x)
    out = Dense(H)(x)

    model = Model(inp, out)
    model.compile(optimizer=optim, loss=loss_fn)

    es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0)
    model.fit(
        Xz_seq, Yz,
        epochs=epochs,
        batch_size=batch_sz,
        verbose=0,
        validation_split=max(0.0, min(val_split, 0.4)),
        shuffle=False,
        callbacks=[es],
    )

    # ── Predicción DIRECT: 1 sola llamada -> H valores ──
    last_z = scaler_X.transform(last.reshape(1, -1)).reshape(1, L, 1)
    preds_z = model.predict(last_z, verbose=0).ravel()
    preds = scaler_y.inverse_transform(preds_z.reshape(-1, 1)).ravel()

    if preds.shape[0] != H:
        raise ValueError(f"[DEEPAR_DIRECT] preds tiene {preds.shape[0]} pero horizon={H}")
    if np.any(~np.isfinite(preds)):
        raise ValueError("[DEEPAR_DIRECT] preds contiene NaN/inf")

    # ── Índice futuro ──
    freq = _infer_freq(df["fecha"])
    idx_fut = _future_index(df["fecha"].iloc[-1], freq, H)
    if (len(idx_fut) != H) or (not idx_fut.is_monotonic_increasing) or (not idx_fut.is_unique):
        raise ValueError("[DEEPAR_DIRECT] idx_fut inválido (len/orden/unicidad). Revisa _infer_freq/_future_index.")

    # ── Salida estándar ──
    fuente_orig  = getattr(cfg, "PRED_FUENTE_ORIG", "original")
    fuente_synth = getattr(cfg, "PRED_FUENTE", "sintetica")

    df_future = pd.DataFrame({
        "fecha": idx_fut,
        "valor": preds.astype(float),
        "pred":  preds.astype(float),
        "fuente": fuente_synth,
    })

    df_hist = df[["fecha", "valor"]].copy()
    df_hist["pred"] = np.nan
    df_hist["fuente"] = fuente_orig

    df_full = pd.concat([df_hist[["fecha","valor","pred","fuente"]], df_future], ignore_index=True)
    df_full["valor"] = df_full["valor"].astype(float)

    paths = _save_outputs("DEEPAR_DIRECT", out_dir, df_full, df_future[["fecha","valor","pred","fuente"]])

    # ── ANOMALÍAS (backtest 1-paso) usando la 1ra columna (t+1) de la salida multi-H ──
    yhat_train_z = model.predict(Xz_seq, verbose=0)  # (n_samples, H)
    yhat_train = scaler_y.inverse_transform(yhat_train_z.reshape(-1, 1)).reshape(yhat_train_z.shape)

    y_true_next, y_pred_next, fechas_next = [], [], []
    n_samples = X.shape[0]
    for i in range(n_samples):
        t_next = L + i
        if t_next < len(y_series):
            y_true_next.append(y_series[t_next])
            y_pred_next.append(yhat_train[i, 0])
            fechas_next.append(fechas[t_next])

    y_true_next = np.array(y_true_next, dtype=float)
    y_pred_next = np.array(y_pred_next, dtype=float)
    fechas_next = np.array(fechas_next, dtype="datetime64[ns]")

    if back_steps > 0 and len(y_true_next) > back_steps:
        y_true_bt = y_true_next[-back_steps:]
        y_pred_bt = y_pred_next[-back_steps:]
        fechas_bt = fechas_next[-back_steps:]
    else:
        y_true_bt, y_pred_bt, fechas_bt = y_true_next, y_pred_next, fechas_next

    idxs_anom, fechas_anom, vals_anom = _detect_anomalies_mad(fechas_bt, y_true_bt, y_pred_bt, k_sigma)

    anom_path = None
    ruta_anom = getattr(cfg, "RUTA_ANOMALIAS_METODO", None)
    if ruta_anom:
        try:
            os.makedirs(ruta_anom, exist_ok=True)
            anom_df = pd.DataFrame({"fecha": fechas_anom, "valor": vals_anom})
            if hasattr(anom_df["fecha"], "dt"):
                anom_df["fecha"] = pd.to_datetime(anom_df["fecha"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            anom_path = os.path.join(ruta_anom, "DEEPAR_DIRECT_puntos.csv")
            anom_df.to_csv(anom_path, index=False)
        except Exception as e:
            print(f"[DEEPAR_DIRECT] Advertencia: no se pudo guardar anomalías: {e}")

    return {
        "modelo": "DEEPAR_DIRECT",
        "params": {
            "lags": L,
            "units": units,
            "layers": layers_n,
            "dropout": dropout,
            "loss": loss_fn,
            "optimizer": optim,
            "epochs": epochs,
            "batch_size": batch_sz,
            "val_split": val_split,
            "patience": patience,
        },
        "n_futuros": H,
        "anomalias_csv": anom_path,
        **paths
    }

def run_deepar_feedback(df_in: pd.DataFrame, horizon: int, out_dir: str, cfg=CFG) -> Dict:
    """
    DEEPAR_FEEDBACK:
      - Entrena DeepAR a 1 paso (y_{t+1})
      - Pronostica H pasos de forma recursiva (feedback) desde la última ventana real
    Salida estándar:
      - df_full (histórico+futuro): fecha|valor|pred|fuente
      - df_future (futuro):         fecha|valor|pred|fuente
      - Guarda outputs con prefijo DEEPAR_FEEDBACK
    """
    # ── Dependencias (mismo patrón que otros runners) ──
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from sklearn.preprocessing import StandardScaler  # verificación de dependencia
    except Exception as e:
        raise ImportError("DeepAR_FEEDBACK requiere tensorflow y scikit-learn instalados.") from e

    if df_in is None or df_in.empty:
        raise ValueError("Serie vacía en run_deepar_feedback.")

    H = int(horizon)
    if H <= 0:
        raise ValueError(f"Horizon inválido en run_deepar_feedback: {horizon}")

    # ── Datos ordenados (CORRECTO: convertir a datetime ANTES de ordenar) ──
    df = df_in.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha", "valor"]).sort_values("fecha")
    df["valor"] = df["valor"].astype(float)

    # ── Hiperparámetros DeepAR (desde settings.py) ──
    L         = int(getattr(cfg, "WINDOW_LEN", 30))
    epochs    = int(getattr(cfg, "DEEPAR_EPOCHS", 120))             # default igual a DIRECT (comparación limpia)
    batch_sz  = int(getattr(cfg, "DEEPAR_BATCH_SIZE", 64))          # default igual a DIRECT
    val_split = float(getattr(cfg, "DEEPAR_VAL_SPLIT", 0.15))
    patience  = int(getattr(cfg, "DEEPAR_EARLY_STOP_PATIENCE", 8))  # default igual a DIRECT
    rs        = getattr(cfg, "RANDOM_SEED_PREDICT", None)

    # ── Entrenamiento 1-paso + backtest (helper existente) ──
    model, scaler_X, scaler_y, X, y, last_window, mu_back = _deepar_fit_and_backtest(
        df, L=L, cfg=cfg, epochs=epochs, batch_sz=batch_sz, val_split=val_split, patience=patience, rs=rs
    )

    # ── Pronóstico recursivo con realimentación ──
    preds = _deepar_recursive_forecast(model, last_window, scaler_X, scaler_y, L=L, horizon=H)

    preds = np.array(preds, dtype=float).ravel()
    if preds.shape[0] != H:
        raise ValueError(f"[DEEPAR_FEEDBACK] preds tiene {preds.shape[0]} pero horizon={H}")
    if np.any(~np.isfinite(preds)):
        raise ValueError("[DEEPAR_FEEDBACK] preds contiene NaN/inf")

    # ── Índice futuro y ensamblado de dataframes estándar ──
    freq   = _infer_freq(df["fecha"])
    idx_ft = _future_index(df["fecha"].iloc[-1], freq, H)
    if (len(idx_ft) != H) or (not idx_ft.is_monotonic_increasing) or (not idx_ft.is_unique):
        raise ValueError("[DEEPAR_FEEDBACK] idx_ft inválido (len/orden/unicidad). Revisa _infer_freq/_future_index.")

    fuente_synth = getattr(cfg, "PRED_FUENTE", "sintetica")
    fuente_orig  = getattr(cfg, "PRED_FUENTE_ORIG", "original")

    df_future = pd.DataFrame({
        "fecha": idx_ft,
        "valor": preds.astype(float),
        "pred":  preds.astype(float),
        "fuente": fuente_synth,
    })

    df_hist = df[["fecha", "valor"]].copy()
    df_hist["pred"] = np.nan
    df_hist["fuente"] = fuente_orig

    df_full = pd.concat([df_hist[["fecha", "valor", "pred", "fuente"]], df_future], ignore_index=True)
    df_full["valor"] = df_full["valor"].astype(float)

    # ── Guardado con prefijo correcto ──
    paths = _save_outputs("DEEPAR_FEEDBACK", out_dir, df_full, df_future[["fecha", "valor", "pred", "fuente"]])

    return {
        "modelo": "DEEPAR_FEEDBACK",
        "params": {
            "lags": L,
            "units": int(getattr(cfg, "DEEPAR_UNITS", 64)),
            "layers": int(getattr(cfg, "DEEPAR_LAYERS", 2)),
            "dropout": float(getattr(cfg, "DEEPAR_DROPOUT", 0.1)),
            "epochs": epochs,
            "batch_size": batch_sz,
            "val_split": val_split,
            "patience": patience,
        },
        "n_futuros": H,
        **paths
    }


#------------------------------------------

def exportar_ventanas_sinteticas(json_map: dict, cfg=None) -> str:
    """
    SOLO detección sobre series SINTÉTICAS.
    - No crea carpetas de ventanas ni PNG.
    - No modifica flags globales.
    - No genera archivos 'anomalias_sinteticas_*' ni '__anom_ventanas*.csv'.
    """
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")

    # -- Resolver CFG
    if cfg is None:
        from componentes import settings as CFG
    else:
        CFG = cfg

    # Validaciones
    if json_map is None:
        json_map = {}
    if not isinstance(json_map, dict):
        raise TypeError(f"[exportar_ventanas_sinteticas] json_map debe ser dict, recibido: {type(json_map)}")

    # SOLO ejecutar detectores por cada serie sintética
    for series_tag, json_df in sorted(json_map.items()):
        _process_one_sintetica(
            series_tag=series_tag,
            json_df=json_df,
            cfg=CFG
        )

    return "OK"


def _process_one_sintetica(series_tag: str, json_df, cfg) -> None:
    """
    Para UNA serie SINTÉTICA:
      - Aplica TODOS los detectores en settings.MODELOS_ACTIVOS
      - SOLO genera los archivos de detección (<modelo>_puntos*.csv) en
        <RUTA_PREDICCION>/<SERIE>/DetectadoXModelo/
      - No genera ventanas, PNGs ni índices.
    """
    import pandas as pd
    from pathlib import Path

    # --- Imports de componentes
    from componentes.settings import MODELOS_ACTIVOS as DETECTORES
    from componentes.detection import AnomaliesDetector
    from componentes.prediction import _load_series_from_json

    # --- Parámetros y rutas
    pred_root = Path(getattr(cfg, "RUTA_PREDICCION", getattr(cfg, "RUTA_SALIDA", "/home/jacky"))).resolve()
    pred_dir = pred_root / series_tag
    pred_dir.mkdir(parents=True, exist_ok=True)

    # --- Cargar serie
    try:
        serie_df = _load_series_from_json(json_df)
    except Exception as e:
        print(f"[SINT][ERROR] No se pudo cargar la serie ({series_tag}): {e}")
        return

    if serie_df is None or serie_df.empty:
        print(f"[SINT][WARN] Serie vacía en {series_tag}; omito detección.")
        return

    col_fecha = getattr(cfg, "COL_FECHA", "fecha")
    col_valor = getattr(cfg, "COL_VALOR", "valor")
    if col_fecha in serie_df.columns:
        serie_df[col_fecha] = pd.to_datetime(serie_df[col_fecha], errors="coerce")
    serie_df = (
        serie_df.dropna(subset=[col_fecha, col_valor])
                .sort_values(col_fecha)
                .reset_index(drop=True)
    )
    N_DECIMALES = getattr(cfg, "PRED_ROUND_DECIMALS", 2)
    serie_df[col_valor] = serie_df[col_valor].round(N_DECIMALES)
    # --- Flags (se restauran al final)
    old_dir = getattr(cfg, "RUTA_ANOMALIAS_METODO", str(pred_dir))
    old_ts  = getattr(cfg, "EXPORTAR_CON_TIMESTAMP", False)
    cfg.EXPORTAR_CON_TIMESTAMP = False

    try:
        # Detección por cada detector configurado
        det_root = pred_dir / "DetectadoXModelo"
        det_root.mkdir(parents=True, exist_ok=True)

        # Redirige la salida de los detectores a la subcarpeta por método
        cfg.RUTA_ANOMALIAS_METODO = str(det_root)

        for modelo in list(DETECTORES):
            print(f"[SINT] {series_tag} -> detect {modelo}")
            try:
                AnomaliesDetector().execute_model(
                    modelo,
                    serie_df.to_json(orient="records", date_unit="ns", date_format="iso"),
                    "sintetica"
                )
            except Exception as e:
                print(f"[SINT][DETECT][WARN] {series_tag}/{modelo}: {e}")

    finally:
        # Restaurar flags locales
        try:
            cfg.RUTA_ANOMALIAS_METODO = old_dir
            cfg.EXPORTAR_CON_TIMESTAMP = old_ts
        except Exception:
            pass



PRED_DISPATCHER = {
    "ARIMA_DIRECT": run_arima,    "ARIMA_FEEDBACK": run_arima_feedback,
    "SVR_DIRECT": run_svr_direct,
    "SVR_FEEDBACK": run_svr_feedback,
    "RF_DIRECT": run_rf_direct, "RF_FEEDBACK": run_rf_feedback, 
    "PROPHET_DIRECT": run_prophet_direct, "PROPHET_FEEDBACK": run_prophet_feedback,
    "LSTM_DIRECT": run_lstm_direct, "LSTM_FEEDBACK": run_lstm_feedback,
    "NBEATS_DIRECT":run_nbeats_direct, "NBEATS_FEEDBACK":run_nbeats_feedback,
    "DEEPAR_DIRECT":run_deepar_direct, "DEEPAR_FEEDBACK":run_deepar_feedback
}

# -*- coding: utf-8 -*-
from typing import Dict

def _pack_small_json(res: Dict) -> None:
    """Arma el JSON pequeño para XCom en res['json'] (mismo contenido que antes)."""
    import json as _json
    small = {
        "modelo": res["modelo"],
        "n_futuros": res["n_futuros"],
        "csv_1_sintetica":       res["csv_1_sintetica"],
        "csv_3_series_full":     res["csv_3_series_full"],
        "serie_full_path":       res.get("serie_full_path"),
        "serie_sintetica_path":  res.get("serie_sintetica_path"),
        "salidas": {
            "1":  {"nombre": "csv sintética",            "path": res["csv_1_sintetica"]},
            "3":  {"nombre": "csv serie completa",       "path": res["csv_3_series_full"]},
            "G1": {"nombre": "gráfico serie completa",   "path": res.get("serie_full_path")},
            "G2": {"nombre": "gráfico parte sintética",  "path": res.get("serie_sintetica_path")},
        }
    }
    res["json"] = _json.dumps(small, ensure_ascii=False)


def _auto_detect_from_synthetic(res: Dict, out_dir: str, model_tag: str, CFG) -> None:
    """
    1) Carga CSV sintético, normaliza fechas (DMY, sin TZ, a día).
    2) Construye payload (DMY) y ejecuta detectores con ese payload.
    """
    import os
    import json as _json
    import pandas as pd
    from componentes.detection import AnomaliesDetector
    from componentes.settings import MODELOS_ACTIVOS as _DETECTORES

    csv_sint = res.get("csv_1_sintetica")
    if not (csv_sint and os.path.isfile(csv_sint)):
        print("[AUTO-DETECT][WARN] No se encontró 'csv_1_sintetica' para lanzar detección.")
        return

    # --- Cargar exactamente el CSV sintético generado por la predicción ---
    df_s = pd.read_csv(csv_sint)

    # Columnas esperadas por tu pipeline
    COL_FECHA = getattr(CFG, "COL_FECHA", "fecha")
    COL_VALOR = getattr(CFG, "COL_VALOR", "valor")

    if COL_FECHA not in df_s.columns:
        df_s.rename(columns={df_s.columns[0]: COL_FECHA}, inplace=True)
    if COL_VALOR not in df_s.columns and "pred" in df_s.columns:
        df_s.rename(columns={"pred": COL_VALOR}, inplace=True)

    # ---- Parseo de fecha respetando settings ----
    DATE_FMT = getattr(CFG, "DATE_FMT", None)       # ej: "%d/%m/%Y"
    DAYFIRST = bool(getattr(CFG, "DATE_DAYFIRST", True))

    if DATE_FMT:
        dt = pd.to_datetime(df_s[COL_FECHA], format=DATE_FMT, errors="coerce")
    else:
        dt = pd.to_datetime(df_s[COL_FECHA], errors="coerce", dayfirst=DAYFIRST)

    # Normalizar: sin TZ, a medianoche
    try:
        dt = dt.dt.tz_localize(None)
    except Exception:
        try:
            dt = dt.dt.tz_convert(None)
        except Exception:
            pass
    dt = dt.dt.normalize()

    df_s[COL_FECHA] = dt
    df_s = (df_s.dropna(subset=[COL_FECHA, COL_VALOR])
                .sort_values(COL_FECHA)
                .reset_index(drop=True))

    if "fuente" not in df_s.columns:
        df_s["fuente"] = "sintetica"

    # ---- Payload a detectores: DMY SIEMPRE ----
    OUT_DATE_FMT = getattr(CFG, "OUT_DATE_FMT", "%d/%m/%Y")
    df_emit = df_s.copy()
    df_emit[COL_FECHA] = df_emit[COL_FECHA].dt.strftime(OUT_DATE_FMT)
    payload_records = df_emit[[COL_FECHA, COL_VALOR, "fuente"]].to_dict(orient="records")
    json_det = _json.dumps(payload_records, ensure_ascii=False)

    # ---- LOG simple de control ----
    n = len(payload_records)
    if n:
        first_date = payload_records[0][COL_FECHA]
        last_date = payload_records[-1][COL_FECHA]
        print(f"[AUTO-DETECT] {model_tag}: {n} filas  Rango={first_date}..{last_date}")
    else:
        print(f"[AUTO-DETECT][WARN] {model_tag}: payload vacío, no se lanzarán detectores.")
        return

    # ---- Ejecución de detectores ----
    _old_dir = getattr(CFG, "RUTA_ANOMALIAS_METODO", out_dir)
    _old_ts  = getattr(CFG, "EXPORTAR_CON_TIMESTAMP", False)
    CFG.RUTA_ANOMALIAS_METODO = out_dir
    CFG.EXPORTAR_CON_TIMESTAMP = False

    try:
        for det_model in list(_DETECTORES):
            print(f"[AUTO-DETECT] {model_tag} -> {det_model} (sintética)")
            AnomaliesDetector().execute_model(det_model, json_det, "sintetica")
    finally:
        try:
            CFG.RUTA_ANOMALIAS_METODO = _old_dir
            CFG.EXPORTAR_CON_TIMESTAMP = _old_ts
        except Exception:
            pass

def run_prediction(modelo: str, json_df: str, out_root: str | None = None) -> Dict:
    """
    Orquestador de predicción:
    - Carga la serie desde JSON
    - Selecciona y ejecuta el modelo (PRED_DISPATCHER)
    - Empaqueta artefactos mínimos para XCom
    - Ejecuta AUTO-DETECCIÓN de anomalías sobre la salida sintética
    - NO ejecuta consolidación ni pasos aguas abajo
    """
    import os
    import pandas as pd
    from componentes import settings as CFG

    # ------------------ carga de serie de entrada ------------------
    df = _load_series_from_json(json_df)
    if df is None or df.empty:
        raise ValueError("No se pudo cargar la serie (json_df).")

    # Raíz de predicción y carpeta del modelo
    if out_root is None:
            out_root = getattr(CFG, "RUTA_PREDICCION", getattr(CFG, "RUTA_SALIDA", "/home/jacky"))
    
    _ensure_dir(out_root)
    model_tag = (modelo or "").strip()
    out_dir = os.path.join(out_root, model_tag)
    _ensure_dir(out_dir)

    horizon = int(getattr(CFG, "PRED_HORIZON", 60))

    # ------------------ lookup del modelo ------------------
    name_upper = model_tag.upper()
    fn = PRED_DISPATCHER.get(name_upper)
    if fn is None:
        raise ValueError(f"Modelo de predicción no soportado: {modelo}")

    # ------------------ ejecución del modelo ------------------
    res = fn(df, horizon, out_dir, CFG)

    # ------------------ empaquetado mínimo (XCom) ------------------
    _pack_small_json(res)

    # ------------------ AUTO-DETECCIÓN (sin consolidación) ------------------
    try:
        _auto_detect_from_synthetic(res, out_dir, model_tag, CFG)
    except Exception as _e_autodet:
        print(f"[AUTO-DETECT][ERROR] {model_tag}: {_e_autodet}")

    # ------------------ FIN DEL PASO ------------------
    # Importante: NO llamar consolidación ni otras fases downstream.
    return res


# componentes/procesos/anomalies_phase_a.py

#-------------------------
#agrupa las predxicciones sobre sintetica
#------------------------

from pathlib import Path
from typing import Dict, Iterable, List
import os
import re
import pandas as pd
from componentes import settings as CFG


def anomalies_by_prediction_method_all_models(
    include_provenance: bool = False,
    chunksize: int = 100_000,
    debug_sample_rows: int = 3,
) -> Dict[str, str]:
    """
    FASE A con LOGS detallados (CORREGIDA para evitar duplicados):
      - Modelos = subcarpetas de RUTA_PREDICCION (ej. SVR_DIRECT, SVR_FEEDBACK).
      - Busca SOLO en <MODELO>/DetectadoXModelo/ archivos *_puntos*.csv (rglob).
      - Concatena sin transformar y DEDUPLICA por: fecha, valor, fuente, metodo_prediccion, metodo_deteccion.
      - Salida por modelo: <RUTA_PREDICCION>/<MODELO>/anomalies_by_prediction_method_raw.csv
      - Columnas (en este orden): fecha, valor, fuente, metodo_prediccion, metodo_deteccion
      - Prioridad metodo_prediccion: (existente) -> nombre carpeta modelo
    """
    import os
    from pathlib import Path
    import pandas as pd

    root_pred = Path(getattr(CFG, "RUTA_PREDICCION", getattr(CFG, "RUTA_SALIDA", "/home/jacky"))).resolve()
    print(f"[DEBUG] RUTA_PREDICCION: {root_pred}")

    if not root_pred.exists():
        raise FileNotFoundError(f"RUTA_PREDICCION no existe: {root_pred}")

    # Detecta modelos como subcarpetas (omite utilitarias)
    models = [d.name for d in sorted(root_pred.iterdir()) if d.is_dir() and not d.name.startswith("_")]
    print(f"[DEBUG] Modelos detectados (subcarpetas): {models}")
    if not models:
        print("[WARN] No se detectaron subcarpetas de modelos en RUTA_PREDICCION.")
        return {}

    results: Dict[str, str] = {}

    for model in models:
        model_dir = (root_pred / model).resolve()
        detect_root = model_dir / "DetectadoXModelo"
        out_path = model_dir / "anomalies_by_prediction_method_raw.csv"

        # Limpiar salida previa
        if out_path.exists():
            try:
                out_path.unlink()
                print(f"[DEBUG] Limpiando salida previa: {out_path}")
            except Exception as e:
                print(f"[WARN] No se pudo limpiar salida previa ({out_path}): {e}")

        print(f"\n[INFO] ===== MODELO: {model} =====")
        print(f"[DEBUG] Carpeta del modelo:      {model_dir}")
        print(f"[DEBUG] Carpeta de detecciones:  {detect_root}")
        print(f"[DEBUG] CSV de salida:           {out_path}")

        if not detect_root.exists():
            print(f"[WARN] {model}: no existe {detect_root}; omito.")
            results[model] = str(out_path)
            # Crear CSV vacío con cabecera esperada
            header = ["fecha", "valor", "fuente", "metodo_prediccion", "metodo_deteccion"]
            if include_provenance:
                header += ["source_file", "load_ts"]
            pd.DataFrame(columns=header).to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
            continue

        # Recolectar *_puntos*.csv únicos (rutas absolutas resueltas)
        csvs = sorted({p.resolve() for p in detect_root.rglob("*_puntos*.csv") if p.is_file()})
        print(f"[DEBUG] Archivos *_puntos*.csv encontrados ({len(csvs)}):")
        for p in csvs:
            try:
                size = p.stat().st_size
            except Exception:
                size = -1
            rel_show = str(p.relative_to(model_dir)).replace(os.sep, "/")
            print(f"        - {rel_show}  (size={size} bytes)")

        frames_model = []   # acumulador por modelo

        if not csvs:
            header = ["fecha", "valor", "fuente", "metodo_prediccion", "metodo_deteccion"]
            if include_provenance:
                header += ["source_file", "load_ts"]
            pd.DataFrame(columns=header).to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
            print(f"[INFO] {model}: sin archivos *_puntos*; creado CSV vacío: {out_path}")
            results[model] = str(out_path)
            continue

        for fpath in csvs:
            rel = fpath.relative_to(model_dir)
            print(f"\n[READ] {rel}")
            try:
                for chunk_idx, chunk in enumerate(pd.read_csv(
                    fpath, chunksize=chunksize, dtype=str, encoding="utf-8", on_bad_lines="skip"
                ), start=1):
                    cols_orig = list(chunk.columns)
                    n_in = len(chunk)
                    print(f"  [CHUNK {chunk_idx}] filas={n_in} | columnas originales={cols_orig}")

                    # Normaliza headers y selecciona 5 columnas canónicas
                    chunk_norm = _normalize_headers(chunk)
                    df_out = _select_five_columns(chunk_norm, model=model, rel=rel)

                    # Asegurar columnas mínimas
                    for col in ("fecha", "valor", "fuente", "metodo_prediccion", "metodo_deteccion"):
                        if col not in df_out.columns:
                            df_out[col] = ""

                    # Prioridad metodo_prediccion: mantener si existe; si está vacío, usar carpeta modelo
                    if "metodo_prediccion" not in df_out.columns:
                        df_out["metodo_prediccion"] = ""
                    mask_empty = df_out["metodo_prediccion"].astype(str).str.strip().eq("")
                    if mask_empty.any():
                        df_out.loc[mask_empty, "metodo_prediccion"] = str(model).upper()

                    # Procedencia opcional
                    if include_provenance:
                        df_out["source_file"] = str(rel).replace(os.sep, "/")
                        df_out["load_ts"] = pd.Timestamp.utcnow().isoformat(timespec="seconds")

                    # Debug: muestra previa
                    if debug_sample_rows > 0:
                        cols_out = ["fecha", "valor", "fuente", "metodo_prediccion", "metodo_deteccion"]
                        if include_provenance:
                            cols_out += ["source_file", "load_ts"]
                        print("    -> muestra a acumular:")
                        print(df_out[cols_out].head(debug_sample_rows).to_string(index=False))

                    frames_model.append(df_out)

            except Exception as e:
                print(f"[WARN] Error leyendo {rel} (modelo {model}): {e}")
                continue

        # Concat + DEDUPE por claves canónicas
        if frames_model:
            df_all = pd.concat(frames_model, ignore_index=True)

            cols_out = ["fecha", "valor", "fuente", "metodo_prediccion", "metodo_deteccion"]
            if include_provenance:
                cols_out += ["source_file", "load_ts"]

            # Desduplicar registros idénticos (evita dobles lecturas/rutas)
            df_all = df_all[cols_out].drop_duplicates(
                subset=["fecha", "valor", "fuente", "metodo_prediccion", "metodo_deteccion"],
                keep="first"
            )

            # Escribir una sola vez por modelo
            df_all.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
            print(f"[DONE] {model}: filas escritas={len(df_all)} → {out_path}")
        else:
            header = ["fecha", "valor", "fuente", "metodo_prediccion", "metodo_deteccion"]
            if include_provenance:
                header += ["source_file", "load_ts"]
            pd.DataFrame(columns=header).to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
            print(f"[INFO] {model}: sin datos tras lectura; creado CSV vacío: {out_path}")

        results[model] = str(out_path)

    return results


# ---------- Helpers ----------

def _collect_csvs(root: Path, patterns: Iterable[str], excludes: Iterable[str]) -> List[Path]:
    found: List[Path] = []
    for pat in patterns:
        for p in root.glob(pat):
            if not p.is_file():
                continue
            name = p.name.lower()
            if any(ex.lower() in name for ex in excludes):
                continue
            found.append(p.resolve())
    return sorted(set(found), key=lambda p: str(p))


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .map(lambda c: str(c).strip().lower())
        .map(lambda c: re.sub(r"\s+", "_", c))
    )
    return df


def _select_five_columns(df: pd.DataFrame, model: str = None, rel=None) -> pd.DataFrame:
    """
    Selecciona y normaliza las cinco columnas clave del consolidado:
    fecha, valor, fuente, metodo_prediccion, metodo_deteccion.

    Correcciones:
      - 'modelo' ahora se mapea a 'metodo_deteccion' (ya no a metodo_prediccion).
      - metodo_prediccion se conserva del CSV; si está vacío, se usa el nombre de la carpeta (model).
      - metodo_deteccion se completa si está vacío (desde 'modelo' o nombre del archivo).
    """

    # --- MAPEOS DE COLUMNAS ---
    alias_map = {
        "fecha": "fecha",
        "valor": "valor",
        "fuente": "fuente",
        "modelo": "metodo_deteccion",    # <- correcto: el detector (ARIMA/DIF)
        "metodo_prediccion": "metodo_prediccion",
        "metodo_deteccion": "metodo_deteccion",
    }

    # --- NORMALIZACIÓN ---
    keep = {}
    for col in df.columns:
        tgt = alias_map.get(col, None)
        if tgt and tgt not in keep:
            keep[tgt] = col

    out = pd.DataFrame(index=df.index)
    for tgt in ["fecha", "valor", "fuente", "metodo_prediccion", "metodo_deteccion"]:
        if tgt in keep:
            out[tgt] = df[keep[tgt]]
        else:
            out[tgt] = ""

    # --- PRIORIDAD metodo_prediccion ---
    # Conservar el valor del CSV si existe; si está vacío, usar el nombre del modelo de predicción
    out["metodo_prediccion"] = out.get("metodo_prediccion")
    out["metodo_prediccion"] = out["metodo_prediccion"].replace("", pd.NA)
    out["metodo_prediccion"] = out["metodo_prediccion"].fillna(model)

    # --- COMPLETAR metodo_deteccion SI QUEDÓ VACÍO ---
    if "metodo_deteccion" in out.columns:
        # Si hay NaN y existe la columna original 'modelo', usarla
        if out["metodo_deteccion"].isna().any() and "modelo" in df.columns:
            mask = out["metodo_deteccion"].isna()
            out.loc[mask, "metodo_deteccion"] = df.loc[mask, "modelo"]
        # Si sigue vacío, infiere del nombre del archivo
        if out["metodo_deteccion"].isna().any() and rel is not None:
            det = rel.name.split("_", 1)[0].upper()
            out["metodo_deteccion"] = out["metodo_deteccion"].fillna(det)

    return out





# ----------------------- Helpers internos -----------------------

def _get_active_models_from_cfg(root_pred: Path) -> List[str]:
    """Obtiene modelos activos desde settings; si no hay, lista subcarpetas de RUTA_PREDICCION (no '_' )."""
    candidates = [
        getattr(CFG, "MODELOS_PREDICCION_ACTIVOS", None),
        getattr(CFG, "PRED_MODELS_ACTIVE", None),
        getattr(CFG, "MODELOS_ACTIVOS", None),
        getattr(CFG, "ACTIVE_MODELS", None),
    ]
    for c in candidates:
        if c and isinstance(c, (list, tuple)):
            cleaned = [str(x).strip() for x in c if str(x).strip()]
            if cleaned:
                return cleaned
    # Fallback: subcarpetas
    models = []
    for d in root_pred.iterdir():
        if d.is_dir() and not d.name.startswith("_"):
            models.append(d.name)
    return sorted(models)



#------------------------------------------------------------
# AGRUPA POR FECHAS Y POR MODELO
#------------------------------------------------------------
def anomalies_by_date_all_models(CFG) -> None:
    """
    Lee cada '*/anomalies_by_prediction_method_raw.csv' en RUTA_PREDICCION y
    escribe un archivo agrupado POR MODELO en su propia carpeta:
      RUTA_PREDICCION/<MODELO>/anomalies_grouped.csv

    No mezcla métodos: la clave de agrupación es
      ('modelo', 'metodo_prediccion', 'metodo_deteccion', 'fuente')

    Si no hay fechas consecutivas, cada fila del RAW produce un grupo (inicio==fin).
    Exporta fechas en ISO (YYYY-MM-DD) y 'valores' como JSON (lista).
    """
    from pathlib import Path
    import json
    import pandas as pd
    import numpy as np

    pred_dir = Path(CFG.RUTA_PREDICCION)
    raw_paths = sorted(pred_dir.glob("*/anomalies_by_prediction_method_raw.csv"))
    if not raw_paths:
        print("[WARN] No hay RAWs para agrupar en:", pred_dir)
        return

    for raw_path in raw_paths:
        try:
            modelo_tag = raw_path.parent.name  # carpeta del modelo de predicción
            df = pd.read_csv(raw_path)

            if df.empty:
                print(f"[WARN] {raw_path} vacío. Se omite.")
                continue

            # --- Normalizar columnas clave ---
            for c in ["metodo_prediccion", "metodo_deteccion", "fuente", "valor", "fecha"]:
                if c not in df.columns:
                    df[c] = ""

            # Trazabilidad de modelo/carpeta
            df["modelo"] = modelo_tag

            # metodo_prediccion: usar el del CSV; si no hay, usar nombre de carpeta (modelo_tag)
            df["metodo_prediccion"] = df["metodo_prediccion"].astype(str).str.strip()
            df.loc[df["metodo_prediccion"].eq("") | df["metodo_prediccion"].isna(), "metodo_prediccion"] = modelo_tag

            # valor: intentar a numérico (sin romper strings)
            df["valor"] = pd.to_numeric(df["valor"], errors="coerce")

            # --- Parseo ROBUSTO de fecha (una sola vez, sin reparseos) ---
            # Regla: si contiene '-', asumimos ISO (YYYY-MM-DD) -> sin dayfirst
            #        si contiene '/', asumimos DD/MM/YYYY -> dayfirst=True
            fecha_raw = df["fecha"].astype(str)
            mask_iso = fecha_raw.str.contains("-", regex=False)
            mask_slash = fecha_raw.str.contains("/", regex=False)

            fechas = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
            if mask_iso.any():
                fechas.loc[mask_iso] = pd.to_datetime(fecha_raw.loc[mask_iso], errors="coerce", infer_datetime_format=False)
            if mask_slash.any():
                fechas.loc[mask_slash] = pd.to_datetime(fecha_raw.loc[mask_slash], errors="coerce", dayfirst=True, infer_datetime_format=False)

            df["fecha"] = fechas
            df = df.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)

            # --- Agrupación por combinación completa (no mezclar métodos) ---
            key_cols = ["modelo", "metodo_prediccion", "metodo_deteccion", "fuente"]
            grouped_rows = []

            for key, g in df.groupby(key_cols, dropna=False, as_index=False):
                g = g.sort_values("fecha").reset_index(drop=True)
                if g.empty:
                    continue

                # Recorrido para tramos de días consecutivos
                start = g.loc[0, "fecha"]
                last = g.loc[0, "fecha"]
                vals = [g.loc[0, "valor"] if not pd.isna(g.loc[0, "valor"]) else None]

                for i in range(1, len(g)):
                    f = g.loc[i, "fecha"]
                    v = g.loc[i, "valor"] if not pd.isna(g.loc[i, "valor"]) else None
                    # Consecutivo exacto si la diferencia es 1 día
                    if (f - last).days == 1:
                        vals.append(v)
                        last = f
                    else:
                        # Cierra tramo anterior
                        grouped_rows.append({
                            "modelo": key[0],
                            "metodo_prediccion": key[1],
                            "metodo_deteccion": key[2],
                            "fuente": key[3],
                            "fecha_inicio": start.strftime("%Y-%m-%d"),
                            "fecha_fin": last.strftime("%Y-%m-%d"),
                            "valores": json.dumps(vals, ensure_ascii=False),
                        })
                        # Inicia nuevo tramo
                        start = f
                        last = f
                        vals = [v]

                # Cierra último tramo
                grouped_rows.append({
                    "modelo": key[0],
                    "metodo_prediccion": key[1],
                    "metodo_deteccion": key[2],
                    "fuente": key[3],
                    "fecha_inicio": start.strftime("%Y-%m-%d"),
                    "fecha_fin": last.strftime("%Y-%m-%d"),
                    "valores": json.dumps(vals, ensure_ascii=False),
                })

            # --- Escritura por modelo ---
            out_file = raw_path.parent / "anomalies_grouped.csv"
            if grouped_rows:
                pd.DataFrame(grouped_rows).to_csv(out_file, index=False)
                print(f"[DONE] {modelo_tag}: {len(grouped_rows)} tramos agrupados -> {out_file}")
            else:
                # Si no hay consecutivas y df no estaba vacío, cada fila debería ser un tramo.
                # Si grouped_rows quedó vacío, probablemente se filtraron todas las fechas (revisa formato).
                print(f"[WARN] {modelo_tag}: no se generaron tramos (verifica formato de 'fecha').")

        except Exception as e:
            print(f"[ERROR] Procesando {raw_path}: {e}")

#----------------------------------------------------------
#---- GEnera las ventanas desde archivos consolidados
#-------------------------------------------------------------
def exportar_ventanas_desde_grouped(CFG) -> str:
    """
    Lee, por cada modelo de predicción (carpeta en CFG.RUTA_PREDICCION),
    el archivo 'anomalies_grouped.csv' y genera para CADA anomalía:

      1) Ventana COMPLETA (contexto):
         - Archivo: anomalias_sinteticas_csv/NNN__<metodo_prediccion>__<metodo_deteccion>.csv
         - Columnas: id,valor,fuente,fecha,metodo_prediccion,metodo_deteccion,anomalo
         - Rango: [fecha_inicio - VENTANA_CTX_DIAS, fecha_fin + VENTANA_CTX_DIAS]
         - Si faltan días al inicio/fin (por límite de la serie), se recorta sin rellenar.

      2) Ventana LIMPIA (solo valores):
         - Archivo: anomalias_sinteticas_limpias_csv/NNN__<metodo_prediccion>__<metodo_deteccion>.csv
         - Columnas: id,valor

      3) Índice maestro global (SOLO catálogo, sin datos de ventanas):
         - anomalias_sinteticas_csv/anom_ventanas_index_global.csv
         - Columnas: n,modelo,metodo_prediccion,metodo_deteccion,fuente,fecha_inicio,fecha_fin,valores,csv_path,csv_clean_path

    Notas:
    - Serie base: SIEMPRE se toma '1_*_sintetica.csv' en la carpeta del modelo (case-insensitive).
    - Numeración global única y continua (001, 002, ...), usada también como id.
    - Orden de modelos según CFG.PRED_MODELOS_ACTIVOS; dentro: metodo_deteccion (A-Z), fecha_inicio asc.
    """
    from pathlib import Path
    import pandas as pd
    import json

    pred_root = Path(CFG.RUTA_PREDICCION)
    out_full = Path(CFG.ANOMALIAS_SINTETICAS_CSV)
    out_clean = Path(CFG.ANOMALIAS_SINTETICAS_CSV_LIMPIAS)
    out_full.mkdir(parents=True, exist_ok=True)
    out_clean.mkdir(parents=True, exist_ok=True)

    modelos_orden = list(getattr(CFG, "PRED_MODELOS_ACTIVOS", []))
    if not modelos_orden:
        modelos_orden = sorted([p.name for p in pred_root.iterdir() if p.is_dir()])

    def _find_serie_sintetica(model_dir: Path, modelo_tag: str) -> Path | None:
        candidates = [p for p in model_dir.glob("1_*_sintetica.csv") if p.is_file()]
        if not candidates:
            looser = [p for p in model_dir.glob("*_sintetica.csv") if p.is_file()]
            if not looser:
                return None
            modelo_lower = modelo_tag.lower()
            prefer = [p for p in looser if modelo_lower in p.name.lower()]
            return prefer[0] if prefer else looser[0]
        if len(candidates) == 1:
            return candidates[0]
        modelo_lower = modelo_tag.lower()
        prefer = [p for p in candidates if modelo_lower in p.name.lower()]
        return prefer[0] if prefer else candidates[0]

    dias_antes = int(getattr(CFG, "VENTANA_CTX_DIAS", 10))
    dias_despues = int(getattr(CFG, "VENTANA_CTX_DIAS", 10))

    idx_global = []
    n = 1  # numeración global única

    for modelo_tag in modelos_orden:
        model_dir = pred_root / modelo_tag
        gpath = model_dir / "anomalies_grouped.csv"
        if not gpath.exists():
            continue

        df = pd.read_csv(gpath)
        if df.empty:
            continue

        for c in ["metodo_prediccion", "metodo_deteccion", "fuente", "fecha_inicio", "fecha_fin", "valores"]:
            if c not in df.columns:
                df[c] = ""
        df["metodo_prediccion"] = df["metodo_prediccion"].astype(str).str.strip()
        df.loc[df["metodo_prediccion"].eq("") | df["metodo_prediccion"].isna(), "metodo_prediccion"] = modelo_tag

        df["fecha_inicio"] = pd.to_datetime(df["fecha_inicio"], errors="coerce", infer_datetime_format=False)
        df["fecha_fin"] = pd.to_datetime(df["fecha_fin"], errors="coerce", infer_datetime_format=False)
        df = df.dropna(subset=["fecha_inicio", "fecha_fin"]).reset_index(drop=True)

        serie_path = _find_serie_sintetica(model_dir, modelo_tag)
        if serie_path is None:
            print(f"[WARN] No se encontró '1_*_sintetica.csv' en {model_dir}. Se omite {modelo_tag}.")
            continue

        s = pd.read_csv(serie_path)
        if "fecha" not in s.columns or "valor" not in s.columns:
            print(f"[WARN] Serie {serie_path.name} sin columnas requeridas ('fecha','valor'). Se omite {modelo_tag}.")
            continue

        s["fecha"] = pd.to_datetime(s["fecha"], errors="coerce", infer_datetime_format=False)
        s = s.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)

        df = df.sort_values(["metodo_deteccion", "fecha_inicio"]).reset_index(drop=True)

        for _, row in df.iterrows():
            m_pred = (str(row["metodo_prediccion"]).strip() or modelo_tag)
            m_det = (str(row["metodo_deteccion"]).strip() or "NA")
            fuente = str(row["fuente"]).strip() or "sintetica"
            f_ini = row["fecha_inicio"]
            f_fin = row["fecha_fin"]
            valores_json = row.get("valores", "")

            if pd.isna(f_ini) or pd.isna(f_fin):
                continue

            win_ini = f_ini - pd.Timedelta(days=dias_antes)
            win_fin = f_fin + pd.Timedelta(days=dias_despues)
            mask = (s["fecha"] >= win_ini) & (s["fecha"] <= win_fin)
            win = s.loc[mask, ["valor", "fuente", "fecha"]].copy().sort_values("fecha")

            fname = f"{n:03d}_{m_pred}_{m_det}.csv"
            f_full = out_full / fname
            f_clean = out_clean / fname

            # A) Ventana COMPLETA (agrega columna 'anomalo')
            if not win.empty:
                wout = win.copy()
                wout.insert(0, "id", n)
                wout["fecha"] = pd.to_datetime(wout["fecha"], errors="coerce")
                wout["anomalo"] = wout["fecha"].between(f_ini, f_fin, inclusive="both")
                wout["fecha"] = wout["fecha"].dt.strftime("%Y-%m-%d")
                if "fuente" not in wout.columns:
                    wout["fuente"] = fuente
                wout["metodo_prediccion"] = m_pred
                wout["metodo_deteccion"] = m_det
                # Reordenar columnas
                wout = wout[["id","valor","fuente","fecha","metodo_prediccion","metodo_deteccion","anomalo"]]
                wout.to_csv(f_full, index=False)
            else:
                pd.DataFrame(columns=[
                    "id","valor","fuente","fecha","metodo_prediccion","metodo_deteccion","anomalo"
                ]).to_csv(f_full, index=False)

            # B) Ventana LIMPIA (solo valores)
            if not win.empty:
                wclean = win.loc[:, ["fecha", "valor"]].copy()
                wclean["fecha"] = pd.to_datetime(wclean["fecha"], errors="coerce").dt.strftime("%Y-%m-%d")
                wclean.insert(0, "id", n)
                wclean = wclean[["id", "fecha", "valor"]]
                wclean.to_csv(f_clean, index=False)
            else:
                pd.DataFrame(columns=["id","fecha","valor"]).to_csv(f_clean, index=False)

            # C) Índice maestro (solo catálogo)
            idx_global.append({
                "n": n,
                "modelo": modelo_tag,
                "metodo_prediccion": m_pred,
                "metodo_deteccion": m_det,
                "fuente": fuente,
                "fecha_inicio": f_ini.strftime("%Y-%m-%d"),
                "fecha_fin": f_fin.strftime("%Y-%m-%d"),
                "valores": valores_json if isinstance(valores_json, str) else json.dumps(valores_json, ensure_ascii=False),
                "csv_path": str(f_full),
                "csv_clean_path": str(f_clean),
            })

            n += 1

    idx_path = out_full / "anom_ventanas_index_global.csv"
    if idx_global:
        pd.DataFrame(idx_global).to_csv(idx_path, index=False)

    print(f"[DONE] Ventanas generadas:\n - Full:  {out_full}\n - Clean: {out_clean}\nÍndice: {idx_path}")
    return "OK"


def crear_csv_comparacion_numerica_sinteticas(
    idx_path: str | None = None,
    csv_out: str | None = None,
) -> str:
    """
    Genera el catálogo numérico de anomalías sintéticas.
    AHORA incluye la columna 'serie_base' (nombre de la serie origen),
    para que las IRIs se construyan correctamente aguas abajo.
    """
    import os
    import pandas as pd
    import numpy as np

    base_dir = CFG.ANOMALIAS_SINTETICAS_CSV
    idx_path = idx_path or os.path.join(base_dir, "anom_ventanas_index_global.csv")
    csv_out  = csv_out  or os.path.join(base_dir, "anomalias_comparacion_numerica_sinteticas.csv")

    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    # ===== SERIE BASE (NOMBRE, NO IRI) =====
    if not hasattr(CFG, "IRI") or not CFG.IRI:
        raise ValueError("CFG.IRI no definido (ej: anom:series_riesgo_pais_1)")

    serie_base = str(CFG.IRI).replace("anom:", "").strip()
    if not serie_base.startswith("series_"):
        raise ValueError(f"CFG.IRI no parece una serie válida: {CFG.IRI}")

    if not os.path.isfile(idx_path):
        pd.DataFrame().to_csv(csv_out, index=False)
        return csv_out

    df = pd.read_csv(idx_path)

    # Blindaje: eliminar columna duplicada si existiera
    if "method" in df.columns:
        df = df.drop(columns=["method"])

    # Asegurar columnas mínimas
    for c in [
        "n","modelo","metodo_prediccion","metodo_deteccion","fuente",
        "fecha_inicio","fecha_fin","valores","csv_path","csv_clean_path"
    ]:
        if c not in df.columns:
            df[c] = pd.NA

    # Fechas
    df["start_date"] = pd.to_datetime(df["fecha_inicio"], errors="coerce")
    df["end_date"]   = pd.to_datetime(df["fecha_fin"], errors="coerce")
    df = df.dropna(subset=["start_date","end_date"]).reset_index(drop=True)

    # Duración y centro
    df["duracion_dias"] = (df["end_date"] - df["start_date"]).dt.days + 1
    center_dt = df["start_date"] + (df["end_date"] - df["start_date"]) / 2
    df["center_date"] = center_dt.dt.date.astype(str)

    # Parseo de valores
    def parse_lista(x):
        if isinstance(x, str):
            txt = x.strip().replace("[", "").replace("]", "")
            vals = []
            for t in txt.split(","):
                try:
                    vals.append(float(t.strip()))
                except Exception:
                    pass
            return vals
        return []

    vals = df["valores"].apply(parse_lista)

    df["min_value"]  = [min(v) if v else np.nan for v in vals]
    df["max_value"]  = [max(v) if v else np.nan for v in vals]
    df["amplitude"]  = df["max_value"] - df["min_value"]
    df["mean_value"] = [np.mean(v) if v else np.nan for v in vals]
    df["std_value"]  = [np.std(v)  if v else np.nan for v in vals]
    df["num_values"] = [len(set(v)) for v in vals]
    df["count"]      = [len(v) for v in vals]
    df["peak_value"] = df["max_value"]
    df["peak_time"]  = df["start_date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    out = pd.DataFrame({
        "id_anomalia": df["n"],
        "modelo": df["modelo"],
        "metodo_prediccion": df["metodo_prediccion"],
        "serie_base": serie_base,                 # <<< CLAVE
        "metodo_deteccion": df["metodo_deteccion"],
        "start_date": df["start_date"].dt.date.astype(str),
        "end_date": df["end_date"].dt.date.astype(str),
        "center_date": df["center_date"],
        "duracion_dias": df["duracion_dias"],
        "min_value": df["min_value"],
        "max_value": df["max_value"],
        "amplitude": df["amplitude"],
        "peak_value": df["peak_value"],
        "peak_time": df["peak_time"],
        "mean_value": df["mean_value"],
        "std_value": df["std_value"],
        "num_values": df["num_values"],
        "count": df["count"],
        "valores": df["valores"],
        "csv_path": df["csv_path"],
        "csv_clean_path": df["csv_clean_path"],
        "tipo": "predicha",
        "fuente": "sintetica",
    })

    # Reordenar: serie_base justo después de metodo_prediccion
    cols = list(out.columns)
    cols.remove("serie_base")
    idx = cols.index("metodo_prediccion")
    cols.insert(idx + 1, "serie_base")
    out = out[cols]

    out.to_csv(csv_out, index=False)
    return csv_out



def plot_ventanas_desde_csvv2(CFG) -> str:
    """
    Recorre anomalias_sinteticas_csv/ y para cada archivo NNN__<pred>__<det>.csv
    genera dos gráficos con el MISMO nombre base:
      1) Contextual  -> <GRAFICOS_ANOMALIAS>/NNN__<pred>__<det>.png
         - Título: nombre del archivo
         - Marca fecha de inicio/fin (banda o líneas verticales)
         - Puntos rojos en los días anómalos (anomalo=True)
         - Anota valor mínimo y valor máximo dentro de la ventana
      2) Limpio      -> <GRAFICOS_ANOMALIAS_LIMPIOS>/NNN__<pred>__<det>.png
         - Solo curva fecha–valor (minimal), sin marcadores ni texto extra

    Requisitos del CSV de entrada (por fila de ventana):
      id, valor, fuente, fecha, metodo_prediccion, metodo_deteccion, anomalo
    """
    import os
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt

    # Rutas base
    in_dir  = Path(CFG.ANOMALIAS_SINTETICAS_CSV).resolve()
    out_ctx = Path(getattr(CFG, "GRAFICOS_ANOMALIAS",
                           str(Path(CFG.RUTA_PREDICCION).resolve() / "anomalias_sinteticos"))).resolve()
    out_cln = Path(getattr(CFG, "GRAFICOS_ANOMALIAS_LIMPIOS",
                           str(Path(CFG.RUTA_PREDICCION).resolve() / "anomalias_sinteticos_limpios"))).resolve()
    out_ctx.mkdir(parents=True, exist_ok=True)
    out_cln.mkdir(parents=True, exist_ok=True)

    # Parámetros opcionales
    figsize_ctx   = tuple(getattr(CFG, "PNG_CTX_FIGSIZE", (8, 3.6)))
    figsize_clean = tuple(getattr(CFG, "PNG_LIMPIO_FIGSIZE", (6, 3.2)))
    dpi_ctx       = int(getattr(CFG, "PNG_CTX_DPI", 150))
    dpi_clean     = int(getattr(CFG, "PNG_LIMPIO_DPI", 150))
    lw_ctx        = float(getattr(CFG, "PNG_CTX_LINEWIDTH", 1.6))
    lw_clean      = float(getattr(CFG, "PNG_LIMPIO_LINEWIDTH", 1.5))

    # Listar archivos de ventanas
    csv_files = sorted(p for p in in_dir.glob("*.csv") if p.name != "anom_ventanas_index_global.csv")
    if not csv_files:
        print(f"[plot] No hay CSVs en {in_dir}")
        return "NO_CSV"

    total = 0
    for f in csv_files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[plot][WARN] No se pudo leer {f.name}: {e}")
            continue

        # Validar columnas mínimas
        needed = {"fecha", "valor"}
        if not needed.issubset(df.columns):
            print(f"[plot][WARN] {f.name} sin columnas requeridas {needed}")
            continue

        # Parseo y orden
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = df.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)

        # anomalo -> boolean
        if "anomalo" in df.columns:
            df["anomalo"] = df["anomalo"].astype(str).str.lower().isin(["true", "1", "t", "yes", "y"])
        else:
            df["anomalo"] = False

        # Rango anómalo (si existe)
        if df["anomalo"].any():
            f_ini = df.loc[df["anomalo"], "fecha"].min()
            f_fin = df.loc[df["anomalo"], "fecha"].max()
        else:
            f_ini = f_fin = None

        # ---------- Gráfico contextual ----------
        try:
            fig = plt.figure(figsize=figsize_ctx, dpi=dpi_ctx)
            ax = fig.add_subplot(111)

            ax.plot(df["fecha"].values, df["valor"].values, linewidth=lw_ctx)

            # Puntos rojos en días anómalos
            if df["anomalo"].any():
                dfa = df[df["anomalo"]]
                ax.scatter(dfa["fecha"].values, dfa["valor"].values, s=18)  # sin color explícito

                # Línea(s) vertical(es) inicio/fin o banda
                ax.axvline(f_ini, linestyle="--", alpha=0.35)
                if f_fin is not None and f_fin != f_ini:
                    ax.axvline(f_fin, linestyle="--", alpha=0.35)

            # Anotar min y max en la ventana
            # (usar todo el df; si quieres limitar al rango extendido, ya viene listo)
            vmin_idx = df["valor"].idxmin()
            vmax_idx = df["valor"].idxmax()
            for idx, label in [(vmin_idx, "min"), (vmax_idx, "max")]:
                x = df.loc[idx, "fecha"]
                y = df.loc[idx, "valor"]
                ax.scatter([x], [y], s=28)  # marcador
                ax.annotate(f"{label}: {y:.4f}", (x, y),
                            xytext=(8, 8), textcoords="offset points")

            ax.set_xlabel("fecha")
            ax.set_ylabel("valor")
            ax.set_title(f.name.replace(".csv", ""))

            fig.tight_layout()
            out_png = out_ctx / f.with_suffix(".png").name
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"[plot][WARN] Falló gráfico contextual {f.name}: {e}")

        # ---------- Gráfico limpio ----------
        try:
            fig2 = plt.figure(figsize=figsize_clean, dpi=dpi_clean)
            ax2 = fig2.add_subplot(111)
            ax2.plot(df["fecha"].values, df["valor"].values, linewidth=lw_clean)

            # Opcional: línea vertical en el centro del rango anómalo
           # if f_ini is not None:
           #     center = f_ini if (f_fin is None or f_fin == f_ini) else f_ini + (f_fin - f_ini) / 2
           #     ax2.axvline(center, alpha=0.15)

            # Estilo minimal
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            ax2.set_xlabel("")
            ax2.set_ylabel("")
            for spine in ax2.spines.values():
                spine.set_visible(False)

            fig2.tight_layout(pad=0.2)
            out_png_clean = out_cln / f.with_suffix(".png").name
            fig2.savefig(out_png_clean, bbox_inches="tight", pad_inches=0.0)
            plt.close(fig2)
        except Exception as e:
            print(f"[plot][WARN] Falló gráfico limpio {f.name}: {e}")

        total += 1

    print(f"[plot][DONE] PNGs generados: {total}  →  {out_ctx}  |  {out_cln}")
    return "OK"

def plot_ventanas_desde_csv3(CFG) -> str:
    """
    Recorre anomalias_sinteticas_csv/ y para cada archivo NNN__<pred>__<det>.csv
    genera dos gráficos con el MISMO nombre base:
      1) Contextual  -> <GRAFICOS_ANOMALIAS>/NNN__<pred>__<det>.png
         - Título: nombre del archivo
         - Marca fecha de inicio/fin (líneas verticales)
         - Puntos en los días anómalos (anomalo=True)
         - Anota valor mínimo y valor máximo dentro de la ventana
      2) Limpio      -> <GRAFICOS_ANOMALIAS_LIMPIOS>/NNN__<pred>__<det>.png
         - Solo curva fecha–valor (minimal), sin marcadores ni texto extra
 
    Requisitos del CSV de entrada (por fila de ventana):
      id, valor, fuente, fecha, metodo_prediccion, metodo_deteccion, anomalo
    """
    import os
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
 
    # Rutas base
    in_dir  = Path(CFG.ANOMALIAS_SINTETICAS_CSV).resolve()
    out_ctx = Path(getattr(CFG, "GRAFICOS_ANOMALIAS",
                           str(Path(CFG.RUTA_PREDICCION).resolve() / "anomalias_sinteticos"))).resolve()
    out_cln = Path(getattr(CFG, "GRAFICOS_ANOMALIAS_LIMPIOS",
                           str(Path(CFG.RUTA_PREDICCION).resolve() / "anomalias_sinteticos_limpios"))).resolve()
    out_ctx.mkdir(parents=True, exist_ok=True)
    out_cln.mkdir(parents=True, exist_ok=True)
 
    # Parámetros opcionales
    figsize_ctx   = tuple(getattr(CFG, "PNG_CTX_FIGSIZE", (10, 4.0)))
    figsize_clean = tuple(getattr(CFG, "PNG_LIMPIO_FIGSIZE", (6, 3.2)))
    dpi_ctx       = int(getattr(CFG, "PNG_CTX_DPI", 150))
    dpi_clean     = int(getattr(CFG, "PNG_LIMPIO_DPI", 150))
    lw_ctx        = float(getattr(CFG, "PNG_CTX_LINEWIDTH", 1.6))
    lw_clean      = float(getattr(CFG, "PNG_LIMPIO_LINEWIDTH", 1.5))
 
    # Listar archivos de ventanas
    csv_files = sorted(p for p in in_dir.glob("*.csv") if p.name != "anom_ventanas_index_global.csv")
    if not csv_files:
        print(f"[plot] No hay CSVs en {in_dir}")
        return "NO_CSV"
 
    total = 0
    for f in csv_files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[plot][WARN] No se pudo leer {f.name}: {e}")
            continue
 
        # Validar columnas mínimas
        needed = {"fecha", "valor"}
        if not needed.issubset(df.columns):
            print(f"[plot][WARN] {f.name} sin columnas requeridas {needed}")
            continue
 
        # Parseo y orden
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = df.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)
 
        # anomalo -> boolean
        if "anomalo" in df.columns:
            df["anomalo"] = df["anomalo"].astype(str).str.lower().isin(["true", "1", "t", "yes", "y"])
        else:
            df["anomalo"] = False
 
        # Rango anómalo (si existe)
        if df["anomalo"].any():
            f_ini = df.loc[df["anomalo"], "fecha"].min()
            f_fin = df.loc[df["anomalo"], "fecha"].max()
        else:
            f_ini = f_fin = None
 
        # ---------- Gráfico contextual ----------
        try:
            fig = plt.figure(figsize=figsize_ctx, dpi=dpi_ctx)
            ax = fig.add_subplot(111)
 
            ax.plot(df["fecha"].values, df["valor"].values, linewidth=lw_ctx)
 
            # Puntos en días anómalos
            if df["anomalo"].any():
                dfa = df[df["anomalo"]]
                ax.scatter(dfa["fecha"].values, dfa["valor"].values, s=18, zorder=5)
 
                # Líneas verticales inicio/fin
                ax.axvline(f_ini, linestyle="--", alpha=0.35)
                if f_fin is not None and f_fin != f_ini:
                    ax.axvline(f_fin, linestyle="--", alpha=0.35)
 
            # ── Anotar min y max con valor REAL (2 decimales) ──
            vmin_idx = df["valor"].idxmin()
            vmax_idx = df["valor"].idxmax()
            for idx, label in [(vmin_idx, "min"), (vmax_idx, "max")]:
                x = df.loc[idx, "fecha"]
                y = df.loc[idx, "valor"]
                ax.scatter([x], [y], s=28, zorder=6)
                ax.annotate(
                    f"{label}: {y:.2f}",   # ← 2 decimales, valor real
                    (x, y),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=8,
                )
 
            ax.set_xlabel("fecha", fontsize=9)
            ax.set_ylabel("valor", fontsize=9)
            ax.set_title(f.name.replace(".csv", ""), fontsize=9, pad=6)
 
            # ── Fechas en eje X: sin solapamiento ──
            n_points = len(df)
            if n_points <= 14:
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            elif n_points <= 30:
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            else:
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
 
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
 
            ax.yaxis.set_tick_params(labelsize=8)
            ax.grid(axis="y", alpha=0.15)
 
            fig.tight_layout()
            out_png = out_ctx / f.with_suffix(".png").name
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"[plot][WARN] Falló gráfico contextual {f.name}: {e}")
 
        # ---------- Gráfico limpio ----------
        try:
            fig2 = plt.figure(figsize=figsize_clean, dpi=dpi_clean)
            ax2 = fig2.add_subplot(111)
            ax2.plot(df["fecha"].values, df["valor"].values, linewidth=lw_clean)
 
            # Estilo minimal
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            ax2.set_xlabel("")
            ax2.set_ylabel("")
            for spine in ax2.spines.values():
                spine.set_visible(False)
 
            fig2.tight_layout(pad=0.2)
            out_png_clean = out_cln / f.with_suffix(".png").name
            fig2.savefig(out_png_clean, bbox_inches="tight", pad_inches=0.0)
            plt.close(fig2)
        except Exception as e:
            print(f"[plot][WARN] Falló gráfico limpio {f.name}: {e}")
 
        total += 1
 
    print(f"[plot][DONE] PNGs generados: {total}  →  {out_ctx}  |  {out_cln}")
    return "OK"
 
def plot_ventanas_desde_csvV_aplana(CFG) -> str:
    """
    Recorre anomalias_sinteticas_csv/ y para cada archivo NNN__<pred>__<det>.csv
    genera dos gráficos con el MISMO nombre base:
      1) Contextual  -> <GRAFICOS_ANOMALIAS>/NNN__<pred>__<det>.png
         - Título: nombre del archivo
         - Marca fecha de inicio/fin (banda o líneas verticales)
         - Puntos rojos en los días anómalos (anomalo=True)
         - Anota valor mínimo y valor máximo dentro de la ventana
      2) Limpio      -> <GRAFICOS_ANOMALIAS_LIMPIOS>/NNN__<pred>__<det>.png
         - Solo curva fecha–valor (minimal), sin marcadores ni texto extra
 
    Requisitos del CSV de entrada (por fila de ventana):
      id, valor, fuente, fecha, metodo_prediccion, metodo_deteccion, anomalo
    """
    import os
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
 
    # Rutas base
    in_dir  = Path(CFG.ANOMALIAS_SINTETICAS_CSV).resolve()
    out_ctx = Path(getattr(CFG, "GRAFICOS_ANOMALIAS",
                           str(Path(CFG.RUTA_PREDICCION).resolve() / "anomalias_sinteticos"))).resolve()
    out_cln = Path(getattr(CFG, "GRAFICOS_ANOMALIAS_LIMPIOS",
                           str(Path(CFG.RUTA_PREDICCION).resolve() / "anomalias_sinteticos_limpios"))).resolve()
    out_ctx.mkdir(parents=True, exist_ok=True)
    out_cln.mkdir(parents=True, exist_ok=True)
 
    # Parámetros opcionales
    figsize_ctx   = tuple(getattr(CFG, "PNG_CTX_FIGSIZE", (8, 3.6)))
    figsize_clean = tuple(getattr(CFG, "PNG_LIMPIO_FIGSIZE", (6, 3.2)))
    dpi_ctx       = int(getattr(CFG, "PNG_CTX_DPI", 150))
    dpi_clean     = int(getattr(CFG, "PNG_LIMPIO_DPI", 150))
    lw_ctx        = float(getattr(CFG, "PNG_CTX_LINEWIDTH", 1.6))
    lw_clean      = float(getattr(CFG, "PNG_LIMPIO_LINEWIDTH", 1.5))
 
    # Listar archivos de ventanas
    csv_files = sorted(p for p in in_dir.glob("*.csv") if p.name != "anom_ventanas_index_global.csv")
    if not csv_files:
        print(f"[plot] No hay CSVs en {in_dir}")
        return "NO_CSV"
 
    # ── Cargar escala global desde serie original (UNA VEZ, antes del loop) ──
    # Esto garantiza que todos los PNGs limpios usen la misma escala Y,
    # evitando que series sintéticas casi planas parezcan tener grandes variaciones
    # por el autoescalado de matplotlib, lo que confundiría a CLIP en la comparación.
    _y_min = None
    _y_max = None
    try:
        _df_orig = pd.read_csv(CFG.SERIE_LIMPIA)
        _df_orig.columns = [c.lower().strip() for c in _df_orig.columns]
        _col_valor = next((c for c in _df_orig.columns if c in ["valor", "value"]), None)
        if _col_valor:
            _df_orig[_col_valor] = pd.to_numeric(_df_orig[_col_valor], errors="coerce")
            _y_min = _df_orig[_col_valor].min() * 0.95
            _y_max = _df_orig[_col_valor].max() * 1.05
            print(f"[plot] Escala global fijada desde serie original: [{_y_min:.2f}, {_y_max:.2f}]")
        else:
            print("[plot][WARN] No se encontró columna 'valor' en serie original — se usará escala automática")
    except Exception as e:
        print(f"[plot][WARN] No se pudo cargar serie original para escala fija: {e} — se usará escala automática")
 
    total = 0
    for f in csv_files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[plot][WARN] No se pudo leer {f.name}: {e}")
            continue
 
        # Validar columnas mínimas
        needed = {"fecha", "valor"}
        if not needed.issubset(df.columns):
            print(f"[plot][WARN] {f.name} sin columnas requeridas {needed}")
            continue
 
        # Parseo y orden
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = df.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)
 
        # anomalo -> boolean
        if "anomalo" in df.columns:
            df["anomalo"] = df["anomalo"].astype(str).str.lower().isin(["true", "1", "t", "yes", "y"])
        else:
            df["anomalo"] = False
 
        # Rango anómalo (si existe)
        if df["anomalo"].any():
            f_ini = df.loc[df["anomalo"], "fecha"].min()
            f_fin = df.loc[df["anomalo"], "fecha"].max()
        else:
            f_ini = f_fin = None
 
        # ---------- Gráfico contextual ----------
        try:
            fig = plt.figure(figsize=figsize_ctx, dpi=dpi_ctx)
            ax = fig.add_subplot(111)
 
            ax.plot(df["fecha"].values, df["valor"].values, linewidth=lw_ctx)
 
            # Puntos rojos en días anómalos
            if df["anomalo"].any():
                dfa = df[df["anomalo"]]
                ax.scatter(dfa["fecha"].values, dfa["valor"].values, s=18)  # sin color explícito
 
                # Línea(s) vertical(es) inicio/fin o banda
                ax.axvline(f_ini, linestyle="--", alpha=0.35)
                if f_fin is not None and f_fin != f_ini:
                    ax.axvline(f_fin, linestyle="--", alpha=0.35)
 
            # Anotar min y max en la ventana
            # (usar todo el df; si quieres limitar al rango extendido, ya viene listo)
            vmin_idx = df["valor"].idxmin()
            vmax_idx = df["valor"].idxmax()
            for idx, label in [(vmin_idx, "min"), (vmax_idx, "max")]:
                x = df.loc[idx, "fecha"]
                y = df.loc[idx, "valor"]
                ax.scatter([x], [y], s=28)  # marcador
                ax.annotate(f"{label}: {y:.4f}", (x, y),
                            xytext=(8, 8), textcoords="offset points")
 
            ax.set_xlabel("fecha")
            ax.set_ylabel("valor")
            ax.set_title(f.name.replace(".csv", ""))
 
            fig.tight_layout()
            out_png = out_ctx / f.with_suffix(".png").name
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"[plot][WARN] Falló gráfico contextual {f.name}: {e}")
 
        # ---------- Gráfico limpio ----------
        try:
            fig2 = plt.figure(figsize=figsize_clean, dpi=dpi_clean)
            ax2 = fig2.add_subplot(111)
            ax2.plot(df["fecha"].values, df["valor"].values, linewidth=lw_clean)
 
            # ── Escala Y fija al rango de la serie original ──────────────────
            # Evita que series casi planas parezcan tener grandes variaciones
            # por el autoescalado, lo que confundiría a CLIP en la comparación.
            if _y_min is not None and _y_max is not None:
                ax2.set_ylim(_y_min, _y_max)
 
            # Estilo minimal
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            ax2.set_xlabel("")
            ax2.set_ylabel("")
            for spine in ax2.spines.values():
                spine.set_visible(False)
 
            fig2.tight_layout(pad=0.2)
            out_png_clean = out_cln / f.with_suffix(".png").name
            fig2.savefig(out_png_clean, bbox_inches="tight", pad_inches=0.0)
            plt.close(fig2)
        except Exception as e:
            print(f"[plot][WARN] Falló gráfico limpio {f.name}: {e}")
 
        total += 1
 
    print(f"[plot][DONE] PNGs generados: {total}  →  {out_ctx}  |  {out_cln}")
    return "OK"

def plot_ventanas_desde_csvv4(CFG) -> str:
    """
    Recorre anomalias_sinteticas_csv/ y para cada archivo NNN__<pred>__<det>.csv
    genera dos gráficos con el MISMO nombre base:
      1) Contextual  -> <GRAFICOS_ANOMALIAS>/NNN__<pred>__<det>.png
         - Título: nombre del archivo
         - Marca fecha de inicio/fin (líneas verticales)
         - Puntos en los días anómalos (anomalo=True)
         - Solo muestra fechas de inicio y fin en eje X (sin solapamiento)
      2) Limpio      -> <GRAFICOS_ANOMALIAS_LIMPIOS>/NNN__<pred>__<det>.png
         - Solo curva fecha–valor (minimal), sin marcadores ni texto extra

    Requisitos del CSV de entrada (por fila de ventana):
      id, valor, fuente, fecha, metodo_prediccion, metodo_deteccion, anomalo
    """
    import os
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # Rutas base
    in_dir  = Path(CFG.ANOMALIAS_SINTETICAS_CSV).resolve()
    out_ctx = Path(getattr(CFG, "GRAFICOS_ANOMALIAS",
                           str(Path(CFG.RUTA_PREDICCION).resolve() / "anomalias_sinteticos"))).resolve()
    out_cln = Path(getattr(CFG, "GRAFICOS_ANOMALIAS_LIMPIOS",
                           str(Path(CFG.RUTA_PREDICCION).resolve() / "anomalias_sinteticos_limpios"))).resolve()
    out_ctx.mkdir(parents=True, exist_ok=True)
    out_cln.mkdir(parents=True, exist_ok=True)

    # Parámetros opcionales — proporciones originales preservadas
    figsize_ctx   = tuple(getattr(CFG, "PNG_CTX_FIGSIZE",   (8, 3.6)))
    figsize_clean = tuple(getattr(CFG, "PNG_LIMPIO_FIGSIZE", (6, 3.2)))
    dpi_ctx       = int(getattr(CFG, "PNG_CTX_DPI",   150))
    dpi_clean     = int(getattr(CFG, "PNG_LIMPIO_DPI", 150))
    lw_ctx        = float(getattr(CFG, "PNG_CTX_LINEWIDTH",   1.6))
    lw_clean      = float(getattr(CFG, "PNG_LIMPIO_LINEWIDTH", 1.5))

    # Listar archivos de ventanas
    csv_files = sorted(
        p for p in in_dir.glob("*.csv")
        if p.name != "anom_ventanas_index_global.csv"
    )
    if not csv_files:
        print(f"[plot] No hay CSVs en {in_dir}")
        return "NO_CSV"

    total = 0
    for f in csv_files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[plot][WARN] No se pudo leer {f.name}: {e}")
            continue

        # Validar columnas mínimas
        needed = {"fecha", "valor"}
        if not needed.issubset(df.columns):
            print(f"[plot][WARN] {f.name} sin columnas requeridas {needed}")
            continue

        # Parseo y orden
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = df.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)

        # anomalo -> boolean
        if "anomalo" in df.columns:
            df["anomalo"] = df["anomalo"].astype(str).str.lower().isin(["true", "1", "t", "yes", "y"])
        else:
            df["anomalo"] = False

        # Rango anómalo (si existe)
        if df["anomalo"].any():
            f_ini = df.loc[df["anomalo"], "fecha"].min()
            f_fin = df.loc[df["anomalo"], "fecha"].max()
        else:
            f_ini = f_fin = None

        # ---------- Gráfico contextual ----------
        try:
            fig = plt.figure(figsize=figsize_ctx, dpi=dpi_ctx)
            ax = fig.add_subplot(111)

            ax.plot(df["fecha"].values, df["valor"].values, linewidth=lw_ctx)

            # Puntos en días anómalos + líneas verticales
            if df["anomalo"].any():
                dfa = df[df["anomalo"]]
                ax.scatter(dfa["fecha"].values, dfa["valor"].values, s=18)
                ax.axvline(f_ini, linestyle="--", alpha=0.35)
                if f_fin is not None and f_fin != f_ini:
                    ax.axvline(f_fin, linestyle="--", alpha=0.35)

            # Marcadores de min y max SIN texto (evita sobreescritura)
            vmin_idx = df["valor"].idxmin()
            vmax_idx = df["valor"].idxmax()
            for idx in [vmin_idx, vmax_idx]:
                ax.scatter([df.loc[idx, "fecha"]], [df.loc[idx, "valor"]], s=28)

            # Eje Y sin offset automático (muestra valores reales)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x:.2f}")
            )

            # Eje X: solo fecha inicio y fin de anomalía para evitar solapamiento
            if f_ini is not None and f_fin is not None and f_ini != f_fin:
                ax.set_xticks([f_ini, f_fin])
                ax.set_xticklabels(
                    [f_ini.strftime("%Y-%m-%d"), f_fin.strftime("%Y-%m-%d")],
                    fontsize=7, rotation=90, ha="center"
                )
            elif f_ini is not None:
                ax.set_xticks([f_ini])
                ax.set_xticklabels([f_ini.strftime("%Y-%m-%d")], fontsize=8)
            else:
                ax.set_xticks([df["fecha"].iloc[0], df["fecha"].iloc[-1]])
                ax.set_xticklabels(
                    [df["fecha"].iloc[0].strftime("%Y-%m-%d"),
                     df["fecha"].iloc[-1].strftime("%Y-%m-%d")],
                    fontsize=8
                )

            ax.set_xlabel("fecha")
            ax.set_ylabel("valor")
            ax.set_title(f.name.replace(".csv", ""))

            fig.tight_layout()
            out_png = out_ctx / f.with_suffix(".png").name
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"[plot][WARN] Falló gráfico contextual {f.name}: {e}")

        # ---------- Gráfico limpio ----------
        try:
            fig2 = plt.figure(figsize=figsize_clean, dpi=dpi_clean)
            ax2 = fig2.add_subplot(111)
            ax2.plot(df["fecha"].values, df["valor"].values, linewidth=lw_clean)

            # Estilo minimal — sin nada que corrompa el gráfico para CLIP
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            ax2.set_xlabel("")
            ax2.set_ylabel("")
            ax2.tick_params(left=False, bottom=False)
            for spine in ax2.spines.values():
                spine.set_visible(False)

            fig2.tight_layout(pad=0.2)
            out_png_clean = out_cln / f.with_suffix(".png").name
            fig2.savefig(out_png_clean, bbox_inches="tight", pad_inches=0.0)
            plt.close(fig2)
        except Exception as e:
            print(f"[plot][WARN] Falló gráfico limpio {f.name}: {e}")

        total += 1

    print(f"[plot][DONE] PNGs generados: {total}  →  {out_ctx}  |  {out_cln}")
    return "OK"

def plot_ventanas_desde_csv(CFG) -> str:
    """
    Recorre anomalias_sinteticas_csv/ y para cada archivo NNN__<pred>__<det>.csv
    genera dos gráficos con el MISMO nombre base:
      1) Contextual  -> <GRAFICOS_ANOMALIAS>/NNN__<pred>__<det>.png
         - Título: nombre del archivo
         - Puntos en los días anómalos (anomalo=True)
         - Fecha arriba si es 1 punto, abajo en vertical si son varios
      2) Limpio      -> <GRAFICOS_ANOMALIAS_LIMPIOS>/NNN__<pred>__<det>.png
         - Solo curva fecha–valor (minimal), sin marcadores ni texto extra

    Requisitos del CSV de entrada (por fila de ventana):
      id, valor, fuente, fecha, metodo_prediccion, metodo_deteccion, anomalo
    """
    import os
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # Rutas base
    in_dir  = Path(CFG.ANOMALIAS_SINTETICAS_CSV).resolve()
    out_ctx = Path(getattr(CFG, "GRAFICOS_ANOMALIAS",
                           str(Path(CFG.RUTA_PREDICCION).resolve() / "anomalias_sinteticos"))).resolve()
    out_cln = Path(getattr(CFG, "GRAFICOS_ANOMALIAS_LIMPIOS",
                           str(Path(CFG.RUTA_PREDICCION).resolve() / "anomalias_sinteticos_limpios"))).resolve()
    out_ctx.mkdir(parents=True, exist_ok=True)
    out_cln.mkdir(parents=True, exist_ok=True)

    # Parámetros opcionales — proporciones originales preservadas
    figsize_ctx   = tuple(getattr(CFG, "PNG_CTX_FIGSIZE",   (8, 3.6)))
    figsize_clean = tuple(getattr(CFG, "PNG_LIMPIO_FIGSIZE", (6, 3.2)))
    dpi_ctx       = int(getattr(CFG, "PNG_CTX_DPI",   150))
    dpi_clean     = int(getattr(CFG, "PNG_LIMPIO_DPI", 150))
    lw_ctx        = float(getattr(CFG, "PNG_CTX_LINEWIDTH",   1.6))
    lw_clean      = float(getattr(CFG, "PNG_LIMPIO_LINEWIDTH", 1.5))

    # Listar archivos de ventanas
    csv_files = sorted(
        p for p in in_dir.glob("*.csv")
        if p.name != "anom_ventanas_index_global.csv"
    )
    if not csv_files:
        print(f"[plot] No hay CSVs en {in_dir}")
        return "NO_CSV"

    total = 0
    for f in csv_files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[plot][WARN] No se pudo leer {f.name}: {e}")
            continue

        # Validar columnas mínimas
        needed = {"fecha", "valor"}
        if not needed.issubset(df.columns):
            print(f"[plot][WARN] {f.name} sin columnas requeridas {needed}")
            continue

        # Parseo y orden
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = df.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)

        # anomalo -> boolean
        if "anomalo" in df.columns:
            df["anomalo"] = df["anomalo"].astype(str).str.lower().isin(["true", "1", "t", "yes", "y"])
        else:
            df["anomalo"] = False

        # ---------- Gráfico contextual ----------
        try:
            fig = plt.figure(figsize=figsize_ctx, dpi=dpi_ctx)
            ax = fig.add_subplot(111)

            ax.plot(df["fecha"].values, df["valor"].values, linewidth=lw_ctx)

            # Eje Y sin offset automático (muestra valores reales)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x:.2f}")
            )

            # Puntos anómalos + fechas
            if df["anomalo"].any():
                dfa = df[df["anomalo"]].copy()
                ax.scatter(dfa["fecha"].values, dfa["valor"].values,
                           s=20, color="tab:orange", zorder=5)

                n_anom = len(dfa)
                if n_anom == 1:
                    # Un solo punto: fecha arriba del punto
                    x = dfa["fecha"].iloc[0]
                    y = dfa["valor"].iloc[0]
                    ax.annotate(
                        x.strftime("%Y-%m-%d"),
                        (x, y),
                        xytext=(0, 8),
                        textcoords="offset points",
                        fontsize=7,
                        ha="center",
                        color="tab:orange"
                    )
                else:
                    # Varios puntos: fechas abajo en vertical
                    for _, row in dfa.iterrows():
                        ax.annotate(
                            row["fecha"].strftime("%Y-%m-%d"),
                            (row["fecha"], df["valor"].min()),
                            xytext=(0, -4),
                            textcoords="offset points",
                            fontsize=7,
                            ha="center",
                            va="top",
                            rotation=90,
                            color="tab:orange"
                        )

            # Sin ticks en eje X
            ax.set_xticks([])
            ax.set_xlabel("fecha")
            ax.set_ylabel("valor")
            ax.set_title(f.name.replace(".csv", ""))

            fig.tight_layout()
            out_png = out_ctx / f.with_suffix(".png").name
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"[plot][WARN] Falló gráfico contextual {f.name}: {e}")

        # ---------- Gráfico limpio ----------
        try:
            fig2 = plt.figure(figsize=figsize_clean, dpi=dpi_clean)
            ax2 = fig2.add_subplot(111)
            ax2.plot(df["fecha"].values, df["valor"].values, linewidth=lw_clean)

            # Estilo minimal — sin nada que corrompa el gráfico para CLIP
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            ax2.set_xlabel("")
            ax2.set_ylabel("")
            ax2.tick_params(left=False, bottom=False)
            for spine in ax2.spines.values():
                spine.set_visible(False)

            fig2.tight_layout(pad=0.2)
            out_png_clean = out_cln / f.with_suffix(".png").name
            fig2.savefig(out_png_clean, bbox_inches="tight", pad_inches=0.0)
            plt.close(fig2)
        except Exception as e:
            print(f"[plot][WARN] Falló gráfico limpio {f.name}: {e}")

        total += 1

    print(f"[plot][DONE] PNGs generados: {total}  →  {out_ctx}  |  {out_cln}")
    return "OK"

def consolidar_puntos_sinteticos_all_models():
    """
    Recorre RUTA_PREDICCION/*/1_*_sintetica.csv (incluye carpetas tipo 'RF:DIRECT')
    y consolida TODOS los puntos sintéticos en un solo CSV.

    Salida: <RUTA_RESULTADOS>/puntos_sinteticos_consolidados.csv (o en RUTA_PREDICCION si no hay RUTA_RESULTADOS)

    Columnas de salida (estándar):
      - fecha (YYYY-MM-DD)
      - valor (float)
      - fuente (sintetica)
      - metodo_prediccion (p.ej. RF_DIRECT, ARIMA_FEEDBACK, etc.)
      - serie_id (si CFG.SERIE_ID existe)
      - series_iri (si CFG.IRI existe)
      - source_file (ruta del csv origen)
    """
    import os
    import glob
    import re
    import pandas as pd

    pred_dir = getattr(CFG, "RUTA_PREDICCION", "/home/jacky/DatosSerie/Resultados/Prediccion")
    ruta_resultados = getattr(CFG, "RUTA_RESULTADOS", None)

    pred_dir = str(pred_dir).strip().strip('"')
    if ruta_resultados:
        ruta_resultados = str(ruta_resultados).strip().strip('"')

    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(f"[SINT][ERROR] No existe RUTA_PREDICCION: {pred_dir}")

    out_dir = ruta_resultados if ruta_resultados else pred_dir
    os.makedirs(out_dir, exist_ok=True)
    output_csv = os.path.join(out_dir, "puntos_sinteticos_consolidados.csv")

    series_id = getattr(CFG, "SERIE_ID", "")
    series_iri = getattr(CFG, "IRI", "")

    # Busca en subcarpetas por modelo: .../<CARPETA_MODELO>/1_*_sintetica.csv
    pattern = os.path.join(pred_dir, "*", "1_*_sintetica.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise ValueError(f"[SINT][ERROR] No encontré archivos con patrón: {pattern}")

    def _infer_method_from_filename(fp: str) -> str:
        # 1_RF_DIRECT_sintetica.csv -> RF_DIRECT
        base = os.path.basename(fp)
        base = re.sub(r"\.csv$", "", base, flags=re.IGNORECASE)
        base = base.replace("1_", "", 1)
        base = re.sub(r"_sintetica$", "", base, flags=re.IGNORECASE)
        base = base.strip("_").strip()
        return base.upper() if base else "UNKNOWN"

    dfs = []
    for fp in files:
        df = pd.read_csv(fp)

        # Normalización mínima de columnas
        cols_lower = {c: str(c).strip().lower() for c in df.columns}

        # fecha
        if "fecha" not in [v for v in cols_lower.values()]:
            # si no existe "fecha", usar la primera columna
            first = df.columns[0]
            df = df.rename(columns={first: "fecha"})
        else:
            # renombrar columna que sea 'fecha' (case-insensitive) a 'fecha'
            for c, cl in cols_lower.items():
                if cl == "fecha":
                    df = df.rename(columns={c: "fecha"})
                    break

        # valor
        if "valor" not in [v for v in cols_lower.values()]:
            # fallback típico
            if "pred" in df.columns:
                df = df.rename(columns={"pred": "valor"})
            elif "yhat" in df.columns:
                df = df.rename(columns={"yhat": "valor"})
            else:
                # si no hay columna clara, intenta usar segunda columna
                if len(df.columns) >= 2 and "valor" not in df.columns:
                    df = df.rename(columns={df.columns[1]: "valor"})

        if "fecha" not in df.columns or "valor" not in df.columns:
            raise ValueError(f"[SINT][ERROR] {fp} no tiene columnas mínimas 'fecha' y 'valor'. Columnas: {list(df.columns)}")

        # metodo_prediccion
        if "metodo_prediccion" not in df.columns:
            df["metodo_prediccion"] = _infer_method_from_filename(fp)
        else:
            # Si viene vacío o nulo, completar desde filename
            df["metodo_prediccion"] = df["metodo_prediccion"].fillna("").astype(str).str.strip()
            df.loc[df["metodo_prediccion"] == "", "metodo_prediccion"] = _infer_method_from_filename(fp)

        # fuente
        if "fuente" not in df.columns:
            df["fuente"] = "sintetica"
        else:
            df["fuente"] = df["fuente"].fillna("").astype(str).str.strip()
            df.loc[df["fuente"] == "", "fuente"] = "sintetica"

        # Limpieza de fecha y valor
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date
        df = df.dropna(subset=["fecha"]).copy()

        # Valor numérico
        df["valor"] = df["valor"].astype(str).str.replace(",", ".", regex=False).str.strip()
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        df = df.dropna(subset=["valor"]).copy()

        # Metadata útil
        df["series_id"] = str(series_id) if series_id is not None else ""
        df["series_iri"] = str(series_iri) if series_iri is not None else ""
        df["source_file"] = fp

        df = df[["fecha", "valor", "fuente", "metodo_prediccion", "series_id", "series_iri", "source_file"]]
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values(["metodo_prediccion", "fecha"]).reset_index(drop=True)

    # Guardar en formato estable (fecha como YYYY-MM-DD)
    out["fecha"] = out["fecha"].astype(str)
    out.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"[SINT] OK - archivos: {len(files)} | filas: {len(out)} | salida: {output_csv}")
    return output_csv

def consolidar_anomalias_sinteticas(
    out_name: str = "anomalias_sinteticas_consolidadas.csv",
    chunksize: int = 200_000,
    debug_sample_rows: int = 3,
    include_provenance: bool = True,
) -> str:
    """
    Consolida en UN SOLO CSV todas las anomalías sintéticas por modelo, generadas por:
      <RUTA_PREDICCION>/<MODELO>/anomalies_by_prediction_method_raw.csv

    Salida:
      <RUTA_RESULTADOS>/<out_name>

    Columnas canónicas:
      fecha, valor, fuente, metodo_prediccion, metodo_deteccion
    Opcional (si include_provenance=True):
      model_folder, source_file, load_ts

    Dedup global por:
      fecha, valor, fuente, metodo_prediccion, metodo_deteccion
    """
    import os
    from pathlib import Path
    from datetime import datetime
    import pandas as pd

    root_pred = Path(getattr(CFG, "RUTA_PREDICCION", getattr(CFG, "RUTA_SALIDA", "/home/jacky"))).resolve()
    root_res  = Path(getattr(CFG, "RUTA_RESULTADOS", getattr(CFG, "RUTA_SALIDA", "/home/jacky"))).resolve()

    print(f"[SYN-ANOM][DEBUG] RUTA_PREDICCION: {root_pred}")
    print(f"[SYN-ANOM][DEBUG] RUTA_RESULTADOS: {root_res}")

    if not root_pred.exists():
        raise FileNotFoundError(f"RUTA_PREDICCION no existe: {root_pred}")

    root_res.mkdir(parents=True, exist_ok=True)
    out_path = (root_res / out_name).resolve()

    # Detecta modelos como subcarpetas (omite utilitarias)
    models = [d for d in sorted(root_pred.iterdir()) if d.is_dir() and not d.name.startswith("_")]
    model_names = [d.name for d in models]
    print(f"[SYN-ANOM][INFO] Modelos detectados: {model_names}")

    # Recolectar rutas de CSV por modelo
    inputs = []
    for d in models:
        p = (d / "anomalies_by_prediction_method_raw.csv").resolve()
        if p.exists() and p.is_file():
            inputs.append((d.name, p))
        else:
            print(f"[SYN-ANOM][WARN] No existe CSV para modelo {d.name}: {p}")

    if not inputs:
        # crear vacío con cabecera esperada
        header = ["fecha", "valor", "fuente", "metodo_prediccion", "metodo_deteccion"]
        if include_provenance:
            header += ["model_folder", "source_file", "load_ts"]
        pd.DataFrame(columns=header).to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
        print(f"[SYN-ANOM][DONE] Sin insumos; creado CSV vacío: {out_path}")
        return str(out_path)

    frames = []
    total_rows_in = 0

    for model_folder, csv_path in inputs:
        rel_show = str(csv_path.relative_to(root_pred)).replace(os.sep, "/")
        print(f"\n[SYN-ANOM][READ] {model_folder} -> {rel_show}")

        try:
            for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize, dtype=str, encoding="utf-8", on_bad_lines="skip"), start=1):
                n_in = len(chunk)
                total_rows_in += n_in
                cols_orig = list(chunk.columns)
                print(f"  [CHUNK {chunk_idx}] filas={n_in} | columnas={cols_orig}")

                # Normalizar encabezados si tienes helpers; si no, mínimo aseguramos canónicas
                # Si ya existen _normalize_headers/_select_five_columns en tu módulo, úsalo:
                try:
                    chunk_norm = _normalize_headers(chunk)
                except Exception:
                    chunk_norm = chunk.copy()
                    chunk_norm.columns = [str(c).strip().lower() for c in chunk_norm.columns]

                # Selección canónica tolerante (por si viene con nombres raros)
                colmap = {
                    "fecha": ["fecha", "date", "ds", "timestamp"],
                    "valor": ["valor", "value", "y", "yhat", "pred", "prediccion"],
                    "fuente": ["fuente", "source", "origen"],
                    "metodo_prediccion": ["metodo_prediccion", "prediction_method", "model", "metodo"],
                    "metodo_deteccion": ["metodo_deteccion", "detection_method", "detector"],
                }

                def pick_col(df, candidates):
                    for c in candidates:
                        if c in df.columns:
                            return c
                    return None

                out = pd.DataFrame()
                for canon, candidates in colmap.items():
                    c = pick_col(chunk_norm, candidates)
                    out[canon] = chunk_norm[c].astype(str) if c else ""

                # Si metodo_prediccion está vacío, usar carpeta del modelo
                mask_empty = out["metodo_prediccion"].astype(str).str.strip().eq("")
                if mask_empty.any():
                    out.loc[mask_empty, "metodo_prediccion"] = str(model_folder).upper()

                if include_provenance:
                    out["model_folder"] = str(model_folder)
                    out["source_file"] = rel_show
                    out["load_ts"] = datetime.utcnow().isoformat(timespec="seconds")

                # Debug
                if debug_sample_rows > 0:
                    cols_dbg = ["fecha", "valor", "fuente", "metodo_prediccion", "metodo_deteccion"]
                    if include_provenance:
                        cols_dbg += ["model_folder", "source_file", "load_ts"]
                    print("    -> muestra a acumular:")
                    print(out[cols_dbg].head(debug_sample_rows).to_string(index=False))

                frames.append(out)

        except Exception as e:
            print(f"[SYN-ANOM][WARN] Error leyendo {csv_path}: {e}")
            continue

    if not frames:
        header = ["fecha", "valor", "fuente", "metodo_prediccion", "metodo_deteccion"]
        if include_provenance:
            header += ["model_folder", "source_file", "load_ts"]
        pd.DataFrame(columns=header).to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
        print(f"[SYN-ANOM][DONE] Sin datos tras lectura; creado CSV vacío: {out_path}")
        return str(out_path)

    df_all = pd.concat(frames, ignore_index=True)

    # Dedupe global
    df_all = df_all.drop_duplicates(
        subset=["fecha", "valor", "fuente", "metodo_prediccion", "metodo_deteccion"],
        keep="first"
    )

    df_all.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")

    print(f"\n[SYN-ANOM][DONE] Filas leídas total={total_rows_in} | Filas finales (dedupe)={len(df_all)}")
    print(f"[SYN-ANOM][DONE] Consolidado escrito en: {out_path}")

    return str(out_path)