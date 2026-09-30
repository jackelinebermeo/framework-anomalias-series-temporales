from __future__ import annotations

# --- Stdlib
import os
from io import StringIO
from typing import Optional

# --- Third-party
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# --- Local
import componentes.settings as CFG  # Usamos el módulo para validar presencia de claves


# ===========================
# Validaciones de configuración
# ===========================
def _require(attr: str):
    if not hasattr(CFG, attr):
        raise ValueError(f"Falta '{attr}' en componentes.settings")
    return getattr(CFG, attr)

def _empty_result_json():
    return pd.DataFrame(columns=["fecha", "valor"]).to_json()

# Requeridos
_OUT_METODO           = _require("RUTA_ANOMALIAS_METODO")
_EXPORTAR_CON_TIMESTAMP = _require("EXPORTAR_CON_TIMESTAMP")
_ARIMA_ORDER          = _require("ARIMA_ORDER")  # tupla/lista (p,d,q)


_ARIMA_THRESHOLD_METHOD = _require("ARIMA_THRESHOLD_METHOD")
if _ARIMA_THRESHOLD_METHOD not in ("percentile", "std"):
    raise ValueError("ARIMA_THRESHOLD_METHOD debe ser 'percentile' o 'std'")

if _ARIMA_THRESHOLD_METHOD == "percentile":
    _ARIMA_PERCENTIL = _require("ARIMA_PERCENTIL")  # 0..100 (float/int)
else:
    _ARIMA_SIGMAS = _require("ARIMA_SIGMAS")        # float



# ----------------------Utilidades comunes

def _load_series_json(json_df: str) -> Optional[pd.DataFrame]:
    """Carga serie desde JSON literal, asegura índice datetime, 'valor_raw' y 'fuente'."""
    from io import StringIO
    import pandas as pd

    try:
        df = pd.read_json(StringIO(json_df))
    except Exception:
        return None

    if df.empty or "valor" not in df.columns:
        return None

    # --- Manejo robusto de fechas ---
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=["fecha"]).sort_values("fecha").set_index("fecha")
    else:
        df.index = pd.to_datetime(df.index, errors="coerce").tz_localize(None)
        df = df[~df.index.isna()]
        df = df.sort_index()

    if df.empty:
        return None

    # Preservar valor original si no viene
    if "valor_raw" not in df.columns:
        df["valor_raw"] = df["valor"]

    # Marcar fuente
    if "fuente" not in df.columns:
        df["fuente"] = "original"
    else:
        df["fuente"] = df["fuente"].astype(str).str.lower()

    # ── Redondear solo series sintéticas ────────────────────────
    # Evita que variaciones numéricas insignificantes (ej: 0.003 bps)
    # sean detectadas como anomalías por los algoritmos
    if df["fuente"].iloc[0] == "sintetica":
        n_dec = getattr(CFG, "PRED_ROUND_DECIMALS", 2)
        df["valor"]     = df["valor"].round(n_dec)
        df["valor_raw"] = df["valor_raw"].round(n_dec)
        print(f"[DETECT] Serie sintética redondeada a {n_dec} decimales | "
              f"rango: [{df['valor'].min():.{n_dec}f}, {df['valor'].max():.{n_dec}f}]")
    # ────────────────────────────────────────────────────────────

    return df

def _export_points(modelo: str, fechas, valores, fuente: str, pred_model: str | None = None) -> str:
    """
    (… docstring igual …) + Si fuente == 'sintetica', agrega columna metodo_prediccion.
    """
    import re
    import pandas as pd
    import os
    from componentes import settings as CFG

    #base_out = getattr(CFG, "RUTA_ANOMALIAS_METODO", "/home/jacky/DatosSerie/Resultados/DetectadoXMetodo")
    base_out = getattr(CFG, "RUTA_ANOMALIAS_METODO")

    fuente_norm = (fuente or "original").strip().lower()
    if fuente_norm not in ("original", "sintetica"):
        fuente_norm = "original"

    out_dir = os.path.join(base_out, "Original") if fuente_norm == "original" else base_out
    os.makedirs(out_dir, exist_ok=True)

    dfp = pd.DataFrame({
        "fecha": pd.to_datetime(fechas, errors="coerce", dayfirst=True),
        "valor": pd.to_numeric(valores, errors="coerce"),
    })
    try:
        dfp["fecha"] = dfp["fecha"].dt.tz_localize(None)
    except Exception:
        pass

    modelo_up = str(modelo).upper()
    dfp["modelo"] = modelo_up
    dfp["fuente"] = fuente_norm

    # NUEVO: si es sintética, añade metodo_prediccion (si no viene, deja 'DESCONOCIDO')
    if fuente_norm == "sintetica":
        from pathlib import Path
        # 1) Si no vino por argumento, intenta leer del contexto global
        tag = (pred_model or getattr(CFG, "CONTEXTO_PREDICCION_TAG", None))

        # 2) Si aún no hay tag, intenta inferirlo desde la RUTA base
        #    (…/Resultados/Prediccion/<TAG>/DetectadoXModelo/…)
        if not tag:
            try:
                parts = Path(base_out).resolve().parts
                i = parts.index("Prediccion")
                tag = parts[i + 1]  # carpeta siguiente a 'Prediccion'
            except Exception:
                tag = None

        dfp["metodo_prediccion"] = (str(tag).upper() if tag else "DESCONOCIDO")


    dfp = dfp.dropna(subset=["fecha", "valor"]).sort_values("fecha").reset_index(drop=True)

    def _fmt_max2(x):
        s = f"{float(x):.2f}"
        return s.rstrip("0").rstrip(".")
    if not dfp.empty:
        dfp["valor"] = dfp["valor"].map(_fmt_max2)

    export_ts = bool(getattr(CFG, "EXPORTAR_CON_TIMESTAMP", False))
    modelo_safe = re.sub(r"[^A-Z0-9._-]+", "_", modelo_up)
    if export_ts:
        stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        fname = f"{modelo_safe}_puntos_{stamp}.csv"
    else:
        fname = f"{modelo_safe}_puntos.csv"
    full_path = os.path.join(out_dir, fname)

    # Orden de columnas: en original se mantiene igual; en sintética se inserta metodo_prediccion
    cols_orig = ["fecha", "valor", "modelo", "fuente"]
    cols_sint = ["fecha", "valor", "modelo", "fuente", "metodo_prediccion"]

    if dfp.empty:
        pd.DataFrame(columns=(cols_sint if fuente_norm == "sintetica" else cols_orig)).to_csv(
            full_path, index=False, encoding="utf-8"
        )
        return full_path

    dfp = dfp.copy()
    dfp["fecha"] = pd.to_datetime(dfp["fecha"], errors="coerce", dayfirst=True).dt.strftime("%Y-%m-%d")
    cols_final = cols_sint if fuente_norm == "sintetica" else cols_orig
    dfp = dfp[cols_final].drop_duplicates(subset=cols_final).reset_index(drop=True)
    dfp.to_csv(full_path, index=False, encoding="utf-8")
    return full_path

# ===========================
# Núcleo ARIMA y Umbralización
# ===========================
def _run_arima_in_sample(data: np.ndarray, order: tuple[int, int, int]) -> np.ndarray:
    """
    Entrena ARIMA(p,d,q) y devuelve residuales absolutos in-sample como 'score'.
    """
    model = ARIMA(data, order=order).fit()
    fitted = model.fittedvalues  # misma longitud que 'data'
    resid = np.abs(np.asarray(data, dtype=float) - np.asarray(fitted, dtype=float))
    return resid

def _extract_anomalies_by_policy(df: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """
    Convierte un score continuo en anomalías de punto según la política declarada en settings.
    Devuelve DataFrame con columnas ['fecha','valor'] (sin agrupar).
    """
    s = np.asarray(score, dtype=float)

    if _ARIMA_THRESHOLD_METHOD == "percentile":
        thr = np.percentile(s, float(_ARIMA_PERCENTIL))
    else:  # "std"
        mu = float(np.nanmean(s))
        sd = float(np.nanstd(s, ddof=1))
        if not np.isfinite(sd) or sd == 0:
            return pd.DataFrame(columns=["fecha", "valor"])
        thr = mu + float(_ARIMA_SIGMAS) * sd

    mask = s > thr
    if not np.any(mask):
        return pd.DataFrame(columns=["fecha", "valor"])

    fechas = df.index[mask]
    valores = df.loc[mask, "valor_raw"].values
    return pd.DataFrame({"fecha": fechas, "valor": valores})

# ===========================
# API pública
# ===========================
class AnomaliesDetector:
    """
    Detector modular. Por ahora implementa solo ARIMA.
    - execute_model('ARIMA', json_df) -> JSON agrupado estándar.
    """

    def _detect_arima(self, df: pd.DataFrame,fuente: str) -> str:
        # Salida vacía uniforme
        EMPTY = _empty_result_json()

        # Longitud mínima (simple y directa)
        p, d, q = (int(x) for x in _ARIMA_ORDER)
        if len(df) < max(p + d + q + 3, 8):
            return EMPTY

        # ARIMA in-sample y extracción de puntos (sin agrupar)
        try:
            score = _run_arima_in_sample(df["valor"].values, (p, d, q))
        except Exception:
            return EMPTY

        anom_df = _extract_anomalies_by_policy(df, score)
        if anom_df.empty:
            return EMPTY

        # Solo puntos crudos
        puntos = anom_df[["fecha", "valor"]]

        # Exportar SOLO puntos (ARIMA_puntos.csv)
        #_export_points("ARIMA", puntos["fecha"], puntos["valor"],fuente)
        _export_points("ARIMA", puntos["fecha"], puntos["valor"], fuente, getattr(CFG, "CONTEXTO_PREDICCION_TAG", None))


        # Retornar SOLO puntos
        return puntos.to_json()
        
    def _detect_dif(self, df: pd.DataFrame,fuente: str) -> str:
        """
        Detecta anomalías por diferencia absoluta (|Δ valor|) usando un umbral por percentil.
        - Usa settings.DIF_PERCENTIL (float, ej. 90.0)
        - Devuelve SOLO puntos: columnas ['fecha','valor'] en JSON
        - Exporta <RUTA_ANOMALIAS_METODO>/DIF_puntos.csv
        """
        EMPTY = _empty_result_json()

        # --- Config desde settings (sin hardcode) ---
        try:
            cfg = getattr(self, "cfg", CFG)
            DIF_PERCENTIL = float(getattr(cfg, "DIF_PERCENTIL"))  # debe existir en settings
        except Exception:
            # Si falta en settings, devuelve vacío para no “quemar” un valor
            return EMPTY

        # --- Validaciones ligeras ---
        if df is None or df.empty or "valor" not in df.columns:
            return EMPTY

        # Índice datetime ordenado
        if not isinstance(df.index, pd.DatetimeIndex):
            if "fecha" not in df.columns:
                return EMPTY
            df = df.copy()
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
            df = df.dropna(subset=["fecha"]).set_index("fecha")
        if len(df) < 3:
            return EMPTY
        df = df.sort_index()

        # --- |Δ valor| y umbral por percentil ---
        # Usar 'valor' como float
        v = pd.to_numeric(df["valor"], errors="coerce")
        dif = v.diff().abs()  # primer valor NaN
        if dif.dropna().empty:
            return EMPTY

        thr = np.nanpercentile(dif.values, DIF_PERCENTIL)
        if not np.isfinite(thr):
            return EMPTY

        mask = dif >= thr
        if not mask.any():
            return EMPTY

        # --- Construir puntos crudos (fecha, valor) ---
        fechas = dif.index[mask]
        valores = (df["valor_raw"] if "valor_raw" in df.columns else df["valor"]).reindex(fechas)
        anom_df = pd.DataFrame({"fecha": fechas, "valor": valores.values})

        # --- Exportar SOLO puntos ---
        #_export_points("DIF", anom_df["fecha"], anom_df["valor"],fuente)
        _export_points("DIF", anom_df["fecha"], anom_df["valor"], fuente, getattr(CFG, "CONTEXTO_PREDICCION_TAG", None))


        # --- Retornar SOLO puntos (sin agrupar) ---
        return anom_df[["fecha", "valor"]].to_json()

    def _detect_iforest(self, df: pd.DataFrame,fuente: str) -> str:
        """
        Isolation Forest:
        - Toma TODOS los parámetros desde settings: IF_N_ESTIMATORS, IF_CONTAMINATION,
        IF_MAX_SAMPLES, IF_RANDOM_STATE, IF_KEEP_FRACTION.
        - Devuelve SOLO puntos (['fecha','valor']) en JSON.
        - Exporta <RUTA_ANOMALIAS_METODO>/IFOREST_puntos.csv
        - SIN agrupar y SIN aplicar distancia mínima aquí (eso va en consolidación).
        """
        import numpy as np
        from sklearn.ensemble import IsolationForest

        EMPTY = _empty_result_json()

        # --- Config desde settings (sin hardcode) ---
        try:
            cfg = getattr(self, "cfg", CFG)
            IF_N_ESTIMATORS   = int(getattr(cfg, "IF_N_ESTIMATORS"))
            IF_CONTAMINATION  = float(getattr(cfg, "IF_CONTAMINATION"))
            IF_MAX_SAMPLES    = getattr(cfg, "IF_MAX_SAMPLES")  # puede ser int, float o 'auto'
            IF_RANDOM_STATE   = int(getattr(cfg, "IF_RANDOM_STATE"))
            IF_KEEP_FRACTION  = float(getattr(cfg, "IF_KEEP_FRACTION"))
        except Exception:
            return EMPTY  # falta algún parámetro en settings

        # --- Validaciones mínimas ---
        if df is None or df.empty or "valor" not in df.columns:
            return EMPTY

        # Índice datetime ordenado
        if not isinstance(df.index, pd.DatetimeIndex):
            if "fecha" not in df.columns:
                return EMPTY
            df = df.copy()
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
            df = df.dropna(subset=["fecha"]).set_index("fecha")
        df = df.sort_index()

        # Señal numérica
        v = pd.to_numeric(df["valor"], errors="coerce")
        X = v.to_numpy().reshape(-1, 1)
        if X.size == 0 or np.all(~np.isfinite(X)):
            return EMPTY

        # --- Modelo ---
        try:
            clf = IsolationForest(
                n_estimators=IF_N_ESTIMATORS,
                contamination=IF_CONTAMINATION,
                max_samples=IF_MAX_SAMPLES,
                random_state=IF_RANDOM_STATE,
                n_jobs=-1,
            ).fit(X)
        except Exception:
            return EMPTY

        y = clf.predict(X)  # -1 = outlier
        if not np.any(y == -1):
            return EMPTY

        # Severidad (decision_function: inliers ↑, outliers ↓) → ordenar por más severos
        scores = clf.decision_function(X).ravel()
        out_idx = np.where(y == -1)[0]
        sev = -scores[out_idx]                   # mayor = más severo
        out_idx = out_idx[np.argsort(-sev)]     # ordenar desc

        # Mantener fracción indicada (al menos 1)
        keep = max(1, int(np.ceil(len(out_idx) * IF_KEEP_FRACTION)))
        out_idx = out_idx[:keep]
        if out_idx.size == 0:
            return EMPTY

        # --- Construir puntos crudos ---
        fechas = df.index.values[out_idx]
        valores = (df["valor_raw"].values[out_idx]
                if "valor_raw" in df.columns else v.values[out_idx])
        anom_df = pd.DataFrame({"fecha": fechas, "valor": valores}).sort_values("fecha")

        # --- Exportar SOLO puntos ---
        try:
            #_export_points("IFOREST", anom_df["fecha"], anom_df["valor"],fuente)
            _export_points("IFOREST", anom_df["fecha"], anom_df["valor"], fuente, getattr(CFG, "CONTEXTO_PREDICCION_TAG", None))


        except Exception:
            pass

        # --- Retornar SOLO puntos (sin agrupar) ---
        return anom_df[["fecha", "valor"]].to_json()    

    def _detect_dbscan(self, df: pd.DataFrame,fuente: str) -> str:
        """
        DBSCAN univariado:
        - Usa settings: DBSCAN_TREND_WIN, DBSCAN_FEATURE ('resid'|'raw'|'zresid'),
        DBSCAN_EPS, DBSCAN_MIN_SAMPLES, DBSCAN_KEEP_FRACTION.
        - Devuelve SOLO puntos ['fecha','valor'] en JSON.
        - Exporta <RUTA_ANOMALIAS_METODO>/DBSCAN_puntos.csv
        - Sin agrupación ni distancia mínima aquí (eso va en consolidación).
        """
        EMPTY = _empty_result_json()

        # --- Config desde settings (sin hardcode) ---
        try:
            cfg = getattr(self, "cfg", CFG)
            WIN              = int(getattr(cfg, "DBSCAN_TREND_WIN"))
            FEAT             = str(getattr(cfg, "DBSCAN_FEATURE")).lower()  # 'resid'|'raw'|'zresid'
            EPS              = float(getattr(cfg, "DBSCAN_EPS"))
            MIN_SAMPLES      = int(getattr(cfg, "DBSCAN_MIN_SAMPLES"))
            KEEP_FRACTION    = float(getattr(cfg, "DBSCAN_KEEP_FRACTION"))
        except Exception:
            return EMPTY

        # --- Validaciones mínimas y normalización de índice ---
        if df is None or df.empty or "valor" not in df.columns:
            return EMPTY
        if not isinstance(df.index, pd.DatetimeIndex):
            if "fecha" not in df.columns:
                return EMPTY
            df = df.copy()
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
            df = df.dropna(subset=["fecha"]).set_index("fecha")
        df = df.sort_index()
        v = pd.to_numeric(df["valor"], errors="coerce")
        if v.dropna().empty:
            return EMPTY

        x = v.to_numpy()

        # --- Tendencia robusta y residuo ---
        try:
            trend = pd.Series(x, index=df.index, dtype=float).rolling(WIN, center=True, min_periods=1).median().to_numpy()
        except Exception:
            return EMPTY
        resid = x - trend

        # --- z-score robusto por IQR (fallback a std si IQR=0) ---
        med = float(np.nanmedian(resid))
        q1, q3 = np.nanpercentile(resid, [25, 75])
        iqr = float(q3 - q1)
        if np.isfinite(iqr) and iqr > 0:
            zres = (resid - med) / iqr
        else:
            sd = float(np.nanstd(resid, ddof=1))
            zres = (resid - med) / (sd if sd > 0 else 1.0)

        # --- Selección de feature para DBSCAN ---
        if FEAT == "raw":
            X = x.reshape(-1, 1)
            sev_base = np.abs(zres)
        elif FEAT == "zresid":
            X = zres.reshape(-1, 1)
            sev_base = np.abs(zres)
        else:  # 'resid' por defecto
            X = resid.reshape(-1, 1)
            sev_base = np.abs(zres)

        # --- DBSCAN ---
        try:
            from sklearn.cluster import DBSCAN
            labels = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric="euclidean", n_jobs=-1).fit_predict(X)
            # ── diagnóstico ──────────────────────────────────────
            n_noise = np.sum(labels == -1)
            n_total = len(labels)
            print(f"[DBSCAN] noise={n_noise}/{n_total} ({100*n_noise/n_total:.1f}%) EPS={EPS} feat={FEAT}")
            # ─────────────────────────────────────────────────────
        except Exception:
            return EMPTY

        idx = np.where(labels == -1)[0]  # ruido = anomalías
        if idx.size == 0:
            return EMPTY

        # --- Mantener solo la fracción más severa (ordenado por |zres|) ---
        sev = sev_base[idx]
        idx = idx[np.argsort(-sev)]
        k = max(1, int(np.ceil(len(idx) * KEEP_FRACTION)))
        idx = idx[:k]
        if idx.size == 0:
            return EMPTY

        # --- Salida cruda (sin agrupar) + export ---
        fechas  = df.index.values[idx]
        valores = (df["valor_raw"].values[idx] if "valor_raw" in df.columns else v.values[idx])
        anom_df = pd.DataFrame({"fecha": fechas, "valor": valores}).sort_values("fecha")

        try:
            _export_points("DBSCAN",  anom_df["fecha"], anom_df["valor"], fuente, getattr(CFG, "CONTEXTO_PREDICCION_TAG", None))
        except Exception:
            pass

        return anom_df[["fecha", "valor"]].to_json()
      
    def _detect_tranad(self, df: pd.DataFrame,fuente: str) -> str:
        """
        TRANAD-lite (LSTM AE CPU) con fallback estadístico.
        - Usa SOLO parámetros de settings: TRANAD_WINDOW, TRANAD_EPOCHS, TRANAD_HIDDEN,
        TRANAD_LR, TRANAD_BATCH, TRANAD_PERCENTIL, TRANAD_STANDARDIZE, TRANAD_KEEP_FRACTION,
        TRANAD_SMOOTH.
        - Devuelve SOLO puntos ['fecha','valor'] en JSON (sin agrupación ni distancias mínimas).
        - Exporta <RUTA_ANOMALIAS_METODO>/TRANAD_puntos.csv
        """
        import numpy as np
        EMPTY = _empty_result_json()

        # --- Config desde settings (sin hardcode) ---
        try:
            cfg = getattr(self, "cfg", CFG)
            W          = int(getattr(cfg, "TRANAD_WINDOW"))
            EPOCHS     = int(getattr(cfg, "TRANAD_EPOCHS"))
            H          = int(getattr(cfg, "TRANAD_HIDDEN"))
            LR         = float(getattr(cfg, "TRANAD_LR"))
            BATCH      = int(getattr(cfg, "TRANAD_BATCH"))
            PCTL       = float(getattr(cfg, "TRANAD_PERCENTIL"))
            STDZ       = bool(getattr(cfg, "TRANAD_STANDARDIZE"))
            KEEP_F     = float(getattr(cfg, "TRANAD_KEEP_FRACTION"))
            SMOOTH     = int(getattr(cfg, "TRANAD_SMOOTH"))
        except Exception:
            return EMPTY

        # --- Validaciones mínimas e índice datetime ---
        if df is None or df.empty or "valor" not in df.columns:
            return EMPTY
        if not isinstance(df.index, pd.DatetimeIndex):
            if "fecha" not in df.columns:
                return EMPTY
            df = df.copy()
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
            df = df.dropna(subset=["fecha"]).set_index("fecha")
        df = df.sort_index()
        v = pd.to_numeric(df["valor"], errors="coerce")
        if v.dropna().empty or len(v) <= max(3, W):
            return EMPTY

        x = v.to_numpy(dtype=float)

        # --- Escalado robusto opcional ---
        def _robust_scale_1d(a: np.ndarray) -> np.ndarray:
            a = np.asarray(a, dtype=float).ravel()
            med = np.nanmedian(a)
            q1, q3 = np.nanpercentile(a, [25, 75])
            iqr = q3 - q1
            if np.isfinite(iqr) and iqr > 0:
                z = (a - med) / iqr
            else:
                sd = np.nanstd(a, ddof=1)
                z = (a - med) / (sd if sd > 0 else 1.0)
            return z

        xz = _robust_scale_1d(x) if STDZ else x.copy()

        # --- Ventanas [m, W, 1] ---
        try:
            m = len(xz) - W + 1
            if m <= 0:
                return EMPTY
            Xseq = np.lib.stride_tricks.sliding_window_view(xz, W).copy().reshape(m, W, 1).astype("float32")
        except Exception:
            return EMPTY

        # --- Intento DL (PyTorch); si falla, fallback estadístico ---
        err_seq = None
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            torch.manual_seed(42)
            device = torch.device("cpu")

            class LSTMAE(nn.Module):
                def __init__(self, hidden: int):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
                    self.lin  = nn.Linear(hidden, 1)
                def forward(self, x):
                    out, _ = self.lstm(x)     # [B,W,H]
                    return self.lin(out)      # [B,W,1]

            ds = TensorDataset(torch.from_numpy(Xseq), torch.from_numpy(Xseq))
            dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)

            model = LSTMAE(H).to(device)
            opt   = torch.optim.Adam(model.parameters(), lr=LR)
            lossf = nn.MSELoss()

            model.train()
            for _ in range(EPOCHS):
                for xb, yb in dl:
                    xb = xb.to(device); yb = yb.to(device)
                    opt.zero_grad()
                    yhat = model(xb)
                    loss = lossf(yhat, yb)
                    loss.backward()
                    opt.step()

            model.eval()
            with torch.no_grad():
                Yhat = model(torch.from_numpy(Xseq).to(device)).cpu().numpy()  # [m,W,1]
            err_seq = ((Yhat.squeeze(-1) - Xseq.squeeze(-1)) ** 2).mean(axis=1).astype("float32")  # [m]

        except Exception:
            # Fallback: energía del residuo robusto en ventana W
            try:
                s = pd.Series(x, index=df.index, dtype=float)
                w_med = W | 1  # asegurar impar
                trend = s.rolling(w_med, center=True, min_periods=1).median().to_numpy()
                resid = x - trend
                rz = _robust_scale_1d(resid)
                rw = np.lib.stride_tricks.sliding_window_view(rz, W)
                err_seq = (rw ** 2).mean(axis=1).astype("float32")
            except Exception:
                return EMPTY

        if err_seq is None or err_seq.size == 0:
            return EMPTY

        # --- Alinear error al tiempo (usar fin de cada ventana) ---
        idx_err = df.index[W-1:]
        if len(idx_err) != len(err_seq):
            return EMPTY
        err = pd.Series(err_seq, index=idx_err)
        if SMOOTH and SMOOTH > 1:
            err = err.rolling(SMOOTH, center=True, min_periods=1).median()
        if err.empty:
            return EMPTY

        # --- Selección por percentil alto + KEEP_F (sin MIN_DIST aquí) ---
        thr = np.nanpercentile(err.values, PCTL)
        cand = err[err >= thr]
        if cand.empty:
            return EMPTY

        order = np.argsort(-cand.values)                      # desc por severidad
        idx_sel = cand.index[order]
        k = max(1, int(np.ceil(len(idx_sel) * KEEP_F)))
        idx_sel = idx_sel[:k]
        if len(idx_sel) == 0:
            return EMPTY

        # --- Construir puntos crudos + export ---
        valores = (df.loc[idx_sel, "valor_raw"].values
                if "valor_raw" in df.columns else df.loc[idx_sel, "valor"].values)
        anom_df = pd.DataFrame({"fecha": idx_sel, "valor": valores}).sort_values("fecha")
        if anom_df.empty:
            return EMPTY

        try:
            #_export_points("TRANAD", anom_df["fecha"], anom_df["valor"],fuente)
            _export_points("TRANAD", anom_df["fecha"], anom_df["valor"], fuente, getattr(CFG, "CONTEXTO_PREDICCION_TAG", None))

        except Exception:
            pass

        # --- Retornar SOLO puntos (sin agrupar) ---
        return anom_df[["fecha", "valor"]].to_json()

    def _detect_times_net(self, df: pd.DataFrame,fuente: str) -> str:
            """
            TimesNet-lite (Conv1D AE en CPU) con fallback estadístico.
            - Usa SOLO parámetros de settings:
            TIMESNET_WINDOW, TIMESNET_EPOCHS, TIMESNET_HIDDEN, TIMESNET_N_BLOCKS,
            TIMESNET_KERNEL_SIZE, TIMESNET_LR, TIMESNET_BATCH, TIMESNET_PERCENTIL,
            TIMESNET_STANDARDIZE, TIMESNET_KEEP_FRACTION, TIMESNET_SMOOTH.
            - Devuelve SOLO puntos ['fecha','valor'] en JSON (sin agrupar ni espaciar).
            - Exporta <RUTA_ANOMALIAS_METODO>/TIMESNET_puntos.csv
            """
            import numpy as np
            EMPTY = _empty_result_json()

            # --- Config desde settings (sin hardcode) ---
            try:
                cfg = getattr(self, "cfg", CFG)
                W        = int(getattr(cfg, "TIMESNET_WINDOW"))
                EPOCHS   = int(getattr(cfg, "TIMESNET_EPOCHS"))
                HIDDEN   = int(getattr(cfg, "TIMESNET_HIDDEN"))
                N_BLOCKS = int(getattr(cfg, "TIMESNET_N_BLOCKS"))
                KSIZE    = int(getattr(cfg, "TIMESNET_KERNEL_SIZE"))
                LR       = float(getattr(cfg, "TIMESNET_LR"))
                BATCH    = int(getattr(cfg, "TIMESNET_BATCH"))
                PCTL     = float(getattr(cfg, "TIMESNET_PERCENTIL"))
                STDZ     = bool(getattr(cfg, "TIMESNET_STANDARDIZE"))
                KEEP_F   = float(getattr(cfg, "TIMESNET_KEEP_FRACTION"))
                SMOOTH   = int(getattr(cfg, "TIMESNET_SMOOTH"))
            except Exception:
                return EMPTY

            # --- Validaciones mínimas e índice datetime ---
            if df is None or df.empty or "valor" not in df.columns:
                return EMPTY
            if not isinstance(df.index, pd.DatetimeIndex):
                if "fecha" not in df.columns:
                    return EMPTY
                df = df.copy()
                df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
                df = df.dropna(subset=["fecha"]).set_index("fecha")
            df = df.sort_index()
            v = pd.to_numeric(df["valor"], errors="coerce")
            if v.dropna().empty or len(v) <= W:
                return EMPTY

            x = v.to_numpy(dtype=float)

            # --- Escalado robusto opcional ---
            def _robust_scale_1d(a: np.ndarray) -> np.ndarray:
                a = np.asarray(a, dtype=float).ravel()
                med = np.nanmedian(a)
                q1, q3 = np.nanpercentile(a, [25, 75])
                iqr = q3 - q1
                if np.isfinite(iqr) and iqr > 0:
                    return (a - med) / iqr
                sd = np.nanstd(a, ddof=1)
                return (a - med) / (sd if sd > 0 else 1.0)

            xz = _robust_scale_1d(x) if STDZ else x.copy()

            # --- Ventanas para AE: [m, 1, W] ---
            try:
                m = len(xz) - W + 1
                if m <= 0:
                    return EMPTY
                X = np.lib.stride_tricks.sliding_window_view(xz, W).copy().reshape(m, 1, W).astype("float32")
            except Exception:
                return EMPTY

            # --- Intento DL (PyTorch); fallback estadístico si falla ---
            err_seq = None
            try:
                import torch
                import torch.nn as nn
                from torch.utils.data import DataLoader, TensorDataset

                torch.manual_seed(42)
                device = torch.device("cpu")

                class TimesBlock(nn.Module):
                    def __init__(self, c_in, c_hidden, k):
                        super().__init__()
                        p = k // 2
                        self.net = nn.Sequential(
                            nn.Conv1d(c_in, c_hidden, k, padding=p),
                            nn.GELU(),
                            nn.Conv1d(c_hidden, c_in, k, padding=p),
                        )
                    def forward(self, x):
                        return x + self.net(x)

                class TimesAE(nn.Module):
                    def __init__(self, hidden, k, n_blocks):
                        super().__init__()
                        self.blocks = nn.Sequential(*[TimesBlock(1, hidden, k) for _ in range(n_blocks)])
                    def forward(self, x):
                        return self.blocks(x)

                ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(X))
                dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)

                model = TimesAE(HIDDEN, KSIZE, N_BLOCKS).to(device)
                opt   = torch.optim.Adam(model.parameters(), lr=LR)
                lossf = nn.MSELoss()

                model.train()
                for _ in range(EPOCHS):
                    for xb, yb in dl:
                        xb = xb.to(device); yb = yb.to(device)
                        opt.zero_grad();  yhat = model(xb)
                        loss = lossf(yhat, yb); loss.backward(); opt.step()

                model.eval()
                with torch.no_grad():
                    Yhat = model(torch.from_numpy(X).to(device)).cpu().numpy()  # [m,1,W]
                err_seq = ((Yhat - X) ** 2).mean(axis=(1, 2)).astype("float32")

            except Exception:
                # Fallback: energía del residuo robusto en ventana W
                try:
                    s = pd.Series(x, index=df.index, dtype=float)
                    w_med = W | 1  # asegurar impar
                    trend = s.rolling(w_med, center=True, min_periods=1).median().to_numpy()
                    resid = x - trend
                    rz = _robust_scale_1d(resid)
                    rw = np.lib.stride_tricks.sliding_window_view(rz, W)
                    err_seq = (rw ** 2).mean(axis=1).astype("float32")
                except Exception:
                    return EMPTY

            if err_seq is None or err_seq.size == 0:
                return EMPTY

            # --- Alinear error al tiempo (fin de cada ventana) ---
            idx_err = df.index[W-1:]
            if len(idx_err) != len(err_seq):
                return EMPTY

            err = pd.Series(err_seq, index=idx_err)
            if SMOOTH and SMOOTH > 1:
                err = err.rolling(SMOOTH, center=True, min_periods=1).median()
            if err.empty:
                return EMPTY

            # --- Selección por percentil alto + KEEP_F (sin MIN_DIST aquí) ---
            thr = np.nanpercentile(err.values, PCTL)
            cand = err[err >= thr]
            if cand.empty:
                return EMPTY
            order = np.argsort(-cand.values)
            idx_sel = cand.index[order][:max(1, int(np.ceil(len(cand) * KEEP_F)))]
            if len(idx_sel) == 0:
                return EMPTY

            # --- Construir puntos crudos + export ---
            valores = (df.loc[idx_sel, "valor_raw"].values
                    if "valor_raw" in df.columns else df.loc[idx_sel, "valor"].values)
            anom_df = pd.DataFrame({"fecha": idx_sel, "valor": valores}).sort_values("fecha")
            if anom_df.empty:
                return EMPTY

            try:
                #_export_points("TIMESNET", anom_df["fecha"], anom_df["valor"],fuente)
                _export_points("TIMESNET", anom_df["fecha"], anom_df["valor"], fuente, getattr(CFG, "CONTEXTO_PREDICCION_TAG", None))

            except Exception:
                pass

            # --- Retornar SOLO puntos (sin agrupar) ---
            return anom_df[["fecha", "valor"]].to_json()

    def _detect_couta(self, df: pd.DataFrame, fuente: str) -> str:
        """
        COUTA (Page-Hinkley bidireccional)
        ----------------------------------
        Detecta cambios abruptos en la serie usando el algoritmo Page-Hinkley.
        - Usa settings.COUTA_DELTA (tolerancia al drift)
        - Usa settings.COUTA_LAMBDA (umbral de disparo)
        - Marca fechas donde el estadístico acumulado supera el umbral
        CON REINICIO tras cada detección para evitar marcar toda la serie.
        - Devuelve SOLO puntos ['fecha','valor'] en JSON.
        - Exporta <RUTA_ANOMALIAS_METODO>/COUTA_puntos.csv
        """
        EMPTY = _empty_result_json()

        # --- Cargar parámetros desde settings ---
        try:
            cfg = getattr(self, "cfg", CFG)
            DELTA   = float(getattr(cfg, "COUTA_DELTA"))
            LAMBDA  = float(getattr(cfg, "COUTA_LAMBDA"))
        except Exception:
            return EMPTY

        # --- Validaciones de DataFrame ---
        if df is None or df.empty or "valor" not in df.columns:
            return EMPTY

        # Asegurar índice datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if "fecha" not in df.columns:
                return EMPTY
            df = df.copy()
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
            df = df.dropna(subset=["fecha"]).set_index("fecha")
        df = df.sort_index()
        v = pd.to_numeric(df["valor"], errors="coerce")
        if v.dropna().empty or len(v) < 5:
            return EMPTY

        # --- Inicialización del Page-Hinkley ---
        x = v.to_numpy(dtype=float)
        n = len(x)

        mean  = x[0]
        m_pos = 0.0
        m_neg = 0.0

        alarms = []

        # --- Iteración principal CON REINICIO tras cada detección ---
        for i in range(1, n):
            xi   = x[i]
            mean = mean + (xi - mean) / (i + 1)

            dev   = xi - mean - DELTA
            m_pos = max(0.0, m_pos + dev)
            m_neg = min(0.0, m_neg + dev)

            if m_pos > LAMBDA or abs(m_neg) > LAMBDA:
                alarms.append(i)
                m_pos = 0.0   # ← reiniciar tras detección
                m_neg = 0.0   # ← reiniciar tras detección

        if not alarms:
            return EMPTY

        # --- Construcción de resultados ---
        fechas  = df.index[alarms]
        valores = (df["valor_raw"] if "valor_raw" in df.columns else df["valor"]).reindex(fechas)
        anom_df = pd.DataFrame({"fecha": fechas, "valor": valores.values}).sort_values("fecha")

        if anom_df.empty:
            return EMPTY

        # --- Exportar resultados ---
        try:
            _export_points(
                "COUTA",
                anom_df["fecha"],
                anom_df["valor"],
                fuente,
                getattr(CFG, "CONTEXTO_PREDICCION_TAG", None),
            )
        except Exception:
            pass

        return anom_df[["fecha", "valor"]].to_json()
    def _detect_coutav1(self, df: pd.DataFrame, fuente: str) -> str:
        """
        COUTA (Page-Hinkley bidireccional)
        ----------------------------------
        Detecta cambios abruptos en la serie usando el algoritmo Page-Hinkley.
        - Usa settings.COUTA_DELTA (tolerancia al drift)
        - Usa settings.COUTA_LAMBDA (umbral de disparo)
        - Marca TODAS las fechas donde el estadístico acumulado supera el umbral
          (modo puntual, sin reinicio) → ideal para luego agrupar.
        - Devuelve SOLO puntos ['fecha','valor'] en JSON.
        - Exporta <RUTA_ANOMALIAS_METODO>/COUTA_puntos.csv
        """
        EMPTY = _empty_result_json()

        # --- Cargar parámetros desde settings ---
        try:
            cfg = getattr(self, "cfg", CFG)
            DELTA   = float(getattr(cfg, "COUTA_DELTA"))
            LAMBDA  = float(getattr(cfg, "COUTA_LAMBDA"))
        except Exception:
            return EMPTY  # si faltan parámetros, retorna vacío

        # --- Validaciones de DataFrame ---
        if df is None or df.empty or "valor" not in df.columns:
            return EMPTY

        # Asegurar índice datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if "fecha" not in df.columns:
                return EMPTY
            df = df.copy()
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
            df = df.dropna(subset=["fecha"]).set_index("fecha")
        df = df.sort_index()
        v = pd.to_numeric(df["valor"], errors="coerce")
        if v.dropna().empty or len(v) < 5:
            return EMPTY

        # --- Inicialización del Page-Hinkley ---
        x = v.to_numpy(dtype=float)
        n = len(x)

        # Media incremental y acumuladores (positivo / negativo)
        mean = x[0]
        m_pos = 0.0   # desviación acumulada positiva
        m_neg = 0.0   # desviación acumulada negativa

        alarms = []   # índices detectados

        # --- Iteración principal ---
        # Modo puntual: no se reinicia tras detección.
        for i in range(1, n):
            xi = x[i]
            mean = mean + (xi - mean) / (i + 1)

            # Incremento de desviación con tolerancia al drift
            dev = xi - mean - DELTA
            m_pos = max(0.0, m_pos + dev)
            m_neg = min(0.0, m_neg + dev)

            # Detección bidireccional
            if m_pos > LAMBDA or abs(m_neg) > LAMBDA:
                alarms.append(i)

        if not alarms:
            return EMPTY

        # --- Construcción de resultados crudos ---
        fechas = df.index[alarms]
        valores = (df["valor_raw"] if "valor_raw" in df.columns else df["valor"]).reindex(fechas)
        anom_df = pd.DataFrame({"fecha": fechas, "valor": valores.values}).sort_values("fecha")

        if anom_df.empty:
            return EMPTY

        # --- Exportar resultados ---
        try:
            _export_points(
                "COUTA",
                anom_df["fecha"],
                anom_df["valor"],
                fuente,
                getattr(CFG, "CONTEXTO_PREDICCION_TAG", None),
            )
        except Exception:
            pass

        # --- Retornar SOLO puntos ---
        return anom_df[["fecha", "valor"]].to_json()

    def _detect_timegpt(self, df: pd.DataFrame,fuente: str) -> str:
        """
        TimeGPT-lite (forecast robusto + selección por percentil del error).
        - Entrada: df con índice datetime y columna 'valor' (opcional 'valor_raw').
        - Parámetros: TODOS desde componentes.settings (sin defaults aquí).
        Requeridos:
            TIMEGPT_WINDOW, TIMEGPT_TREND_WIN, TIMEGPT_SEASONAL_PERIOD,
            TIMEGPT_STANDARDIZE, TIMEGPT_PERCENTIL, TIMEGPT_KEEP_FRACTION,
            ANOMALIAS_MIN_DIST, TIMEGPT_SMOOTH
        - Exporta CSV con alias 'TIMEGPT' (compatibilidad con tu graficador).
        - Devuelve SOLO puntos en JSON (sin agrupar).
        """
        import numpy as np
        import pandas as pd
        from componentes import settings as CFG

        # -------- Validaciones básicas --------
        if df is None or df.empty or "valor" not in df.columns:
            return _empty_result_json()

        if not isinstance(df.index, pd.DatetimeIndex):
            if "fecha" in df.columns:
                df = df.copy()
                df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
                df = df.set_index("fecha").sort_index()
            else:
                return _empty_result_json()

        df = df.dropna(subset=["valor"]).sort_index()
        if df.empty:
            return _empty_result_json()

        # -------- Parámetros SOLO desde settings --------
        required = [
            "TIMEGPT_WINDOW", "TIMEGPT_TREND_WIN", "TIMEGPT_SEASONAL_PERIOD",
            "TIMEGPT_STANDARDIZE", "TIMEGPT_PERCENTIL", "TIMEGPT_KEEP_FRACTION",
            "ANOMALIAS_MIN_DIST", "TIMEGPT_SMOOTH"
        ]
        if any(not hasattr(CFG, k) for k in required):
            return _empty_result_json()

        W        = int(CFG.TIMEGPT_WINDOW)
        TWIN     = int(CFG.TIMEGPT_TREND_WIN)
        PERIOD   = int(CFG.TIMEGPT_SEASONAL_PERIOD)
        STDZ     = bool(CFG.TIMEGPT_STANDARDIZE)
        PCTL     = float(CFG.TIMEGPT_PERCENTIL)         # percentil ALTO del score
        KEEP_F   = float(CFG.TIMEGPT_KEEP_FRACTION)
        MIN_D    = int(CFG.ANOMALIAS_MIN_DIST)
        SMOOTH   = int(CFG.TIMEGPT_SMOOTH)

        x = df["valor"].astype(float).to_numpy()
        n = len(x)
        if n < max(W, TWIN, 3):
            return _empty_result_json()

        # -------- Utilidades robustas --------
        def _robust_scale_1d(a: np.ndarray) -> np.ndarray:
            a = np.asarray(a, dtype=float).ravel()
            med = np.median(a)
            q1, q3 = np.percentile(a, [25, 75])
            iqr = q3 - q1
            if np.isfinite(iqr) and iqr > 0:
                z = (a - med) / iqr
            else:
                sd = np.std(a, ddof=1)
                z = (a - med) / (sd if sd > 0 else 1.0)
            return z

        # -------- Tendencia robusta (mediana móvil) --------
        win_tr = TWIN if TWIN % 2 == 1 else TWIN + 1  # asegurar impar
        s = pd.Series(x, index=df.index, dtype=float)
        trend = s.rolling(max(3, win_tr), center=True, min_periods=1).median().to_numpy()

        # -------- Estacionalidad opcional --------
        resid = x - trend
        if PERIOD and PERIOD > 1 and PERIOD < n // 2:
            idx = np.arange(n)
            phase = idx % PERIOD
            phase_med = np.zeros(PERIOD, dtype=float)
            for p in range(PERIOD):
                vals = resid[phase == p]
                phase_med[p] = np.median(vals) if vals.size > 0 else 0.0
            seasonal = phase_med[phase]
            resid = resid - seasonal

        # -------- Estandarización robusta del residuo (opcional) --------
        rz = _robust_scale_1d(resid) if STDZ else resid.copy()

        # -------- Score: energía media en ventana W --------
        if W < 3:
            W = 3
        try:
            rw = np.lib.stride_tricks.sliding_window_view(rz, W)  # [n-W+1, W]
            score_vals = (rw**2).mean(axis=1)                     # energía media
            score_idx  = df.index[W-1:]                           # alinea al final
        except Exception:
            return _empty_result_json()

        score = pd.Series(score_vals.astype("float32"), index=score_idx)

        # Suavizado opcional del score (mediana móvil)
        if SMOOTH and SMOOTH > 1:
            score = score.rolling(SMOOTH, center=True, min_periods=1).median()

        if score.empty:
            return _empty_result_json()

        # -------- Selección por percentil ALTO + post-filtros --------
        try:
            thr = np.percentile(score.values, PCTL)
            cand = score[score >= thr]
        except Exception:
            return _empty_result_json()

        if cand.empty:
            return _empty_result_json()

        idx = cand.index
        sev = cand.values
        order = np.argsort(-sev)             # severidad descendente
        idx = idx[order]

        keep = max(1, int(np.ceil(len(idx) * KEEP_F)))
        idx = idx[:keep]

        if MIN_D and MIN_D > 1 and len(idx) > 1:
            pos = df.index.get_indexer(idx)
            pos = pos[pos >= 0]
            pos.sort()
            selected, last = [], -10**9
            for p in pos:
                if p - last >= MIN_D:
                    selected.append(p); last = p
            idx = df.index[selected]

        if len(idx) == 0:
            return _empty_result_json()

        # -------- Salida: exportar y devolver SOLO puntos --------
        valores = (df.loc[idx, "valor_raw"].values
                if "valor_raw" in df.columns else df.loc[idx, "valor"].values)
        anom_df = pd.DataFrame({"fecha": idx, "valor": valores}).sort_values("fecha")
        if anom_df.empty:
            return _empty_result_json()

        try:
            #_export_points("TIMEGPT", anom_df["fecha"], anom_df["valor"],fuente)
            _export_points("TIMEGPT", anom_df["fecha"], anom_df["valor"], fuente, getattr(CFG, "CONTEXTO_PREDICCION_TAG", None))

        except Exception:
            pass

        return anom_df[["fecha", "valor"]].to_json()   

    def execute_model(self, nombre_modelo: str, json_df: str, fuente: str) -> str:
        """
        Orquestador por modelo.
        - Carga la serie desde JSON.
        - Normaliza 'fuente' a {'original','sintetica'}.
        - Despacha al handler correspondiente pasando (df, fuente).
        - Devuelve lo que retorne el handler (contrato original).
        """
        # Clave estable para el modelo
        key = (nombre_modelo or "").strip().lower().replace(" ", "").replace("-", "").replace("_", "")

        # Normalizar/validar 'fuente'
        fuente = (fuente or "original").strip().lower()
        if fuente not in ("original", "sintetica"):
            fuente = "original"

        # Cargar serie desde el JSON
        df = _load_series_json(json_df)
        if df is None:
            # Mantener contrato previo: retorno vacío si no hay datos
            return _empty_result_json()

        # Mapa de handlers: TODOS reciben (df, fuente)
        name_map = {
            "arima":     lambda: self._detect_arima(df, fuente),
            "dif":       lambda: self._detect_dif(df, fuente),
            "iforest":   lambda: self._detect_iforest(df, fuente),
            "dbscan":    lambda: self._detect_dbscan(df, fuente),
            "couta":     lambda: self._detect_couta(df, fuente),
            "tranad":    lambda: self._detect_tranad(df, fuente),
            "timesnet":  lambda: self._detect_times_net(df, fuente),
            "timegpt":   lambda: self._detect_timegpt(df, fuente),
        
        }

        handler = name_map.get(key)
        if handler is None:
            raise ValueError(f"Modelo '{nombre_modelo}' no está implementado.")

        # Ejecutar y devolver tal cual (los handlers exportan usando 'fuente')
        return handler()
    
    # ============================================================
    # DETECCIÓN SOBRE SERIES DE PREDICCIÓN (SINTÉTICAS / FUTURO)
    # ============================================================


def _call_detector_prediction(detector_name: str, df_future: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Llama a un detector existente sobre la SERIE DE PREDICCIÓN (sintética).
    Devuelve SIEMPRE DataFrame con columnas: ['fecha','valor'] (puntos crudos).
    Si falla o no hay puntos, devuelve DataFrame vacío con esas columnas.
    """
    try:
        det_lower = (detector_name or "").lower()
        det_upper = (detector_name or "").upper()

        fn = globals().get(f"run_{det_lower}")
        if fn is None:
            # intentos de alias mínimos (mismos que tu versión)
            alias = {
                "IFOREST": "run_isolation_forest",
                "TRANAD":  "run_tranAD",
                "TIMESNET":"run_times_net",
                "TIMEGPT": "run_timegpt",
                "DBSCAN":  "run_dbscan",
                "COUTA":   "run_couta",
                "ARIMA":   "run_arima",
                "DIF":     "run_dif",
            }
            fn = globals().get(alias.get(det_upper, ""))

        if fn is None:
            return pd.DataFrame(columns=["fecha", "valor"])

        # Ejecuta el detector. Se asume firma existente compatible (df, cfg=CFG).
        res = fn(df_future.copy(), cfg=cfg)

        # Normaliza salida a DataFrame (sin cambiar columnas)
        if isinstance(res, pd.DataFrame):
            out = res.copy()
        elif isinstance(res, dict) and "puntos" in res:
            out = pd.DataFrame(res["puntos"])
        elif res is None:
            return pd.DataFrame(columns=["fecha", "valor"])
        else:
            out = pd.DataFrame(res)

        if out is None or out.empty:
            return pd.DataFrame(columns=["fecha", "valor"])

        # Renombres mínimos (mismos que tu versión)
        if "fecha" not in out.columns and "timestamp" in out.columns:
            out = out.rename(columns={"timestamp": "fecha"})
        if "valor" not in out.columns and "pred" in out.columns:
            out = out.rename(columns={"pred": "valor"})

        # Asegurar columnas mínimas (mantiene 'score' si existe)
        if "fecha" not in out.columns or "valor" not in out.columns:
            return pd.DataFrame(columns=["fecha", "valor"])

        keep = ["fecha", "valor"] + (["score"] if "score" in out.columns else [])
        out = out[keep].dropna()

        # Parse y limpieza de fechas
        out["fecha"] = pd.to_datetime(out["fecha"], errors="coerce")
        out = out.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)

        # Si se quedó vacío, devolver vacío estándar
        if out.empty:
            return pd.DataFrame(columns=["fecha", "valor"])

        return out

    except Exception:
        return pd.DataFrame(columns=["fecha", "valor"])


def detect_all_on_prediction_series(df_future: pd.DataFrame, model_tag: str, cfg=CFG) -> dict:
    import os

    # --- Normalización mínima del df_future (SERIE DE PREDICCIÓN)
    # Copia solo una vez (igual que antes), y evita trabajo duplicado dentro del loop.
    df = df_future.copy()

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    if "valor" not in df.columns:
        if "pred" in df.columns:
            df["valor"] = df["pred"]
        else:
            df["valor"] = pd.NA

    df["fuente"] = "sintetica"
    df = df.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)

    # --- Rutas de salida (estructura pedida)
    #base_root = getattr(cfg, "RUTA_ANOMALIAS_METODO", "/home/jacky/DatosSerie/Resultados/DetectadoXMetodo")
    base_out = getattr(CFG, "RUTA_ANOMALIAS_METODO")
    base_dir = os.path.join(base_root, "sintetica", model_tag)
    xmet_dir = os.path.join(base_dir, "XMetodo")
    os.makedirs(xmet_dir, exist_ok=True)

    activos = list(getattr(cfg, "MODELOS_ACTIVOS", [])) or ["ARIMA", "DIF", "IFOREST"]

    # Constantes para evitar recrear listas en cada iteración
    _cols_base = ("fecha", "valor", "score")
    _order_base = ("fecha", "valor", "metodo_deteccion", "metodo_prediccion", "fuente", "score")

    filas = []

    # --- Ejecuta detectores EN SERIE para evitar OOM
    for det in activos:
        det_up = det.upper()

        puntos = _call_detector_prediction(det, df, cfg)
        if puntos is None or puntos.empty:
            continue

        # guarda crudos por detector (rastro fino)
        det_dir = os.path.join(xmet_dir, det_up)
        os.makedirs(det_dir, exist_ok=True)

        cols = [c for c in _cols_base if c in puntos.columns]
        p2 = puntos[cols].copy()
        p2["metodo_deteccion"] = det_up
        p2["metodo_prediccion"] = model_tag
        p2["fuente"] = "sintetica"

        order = [c for c in _order_base if c in p2.columns]
        p2[order].to_csv(os.path.join(det_dir, "puntos.csv"), index=False, encoding="utf-8")

        # Para el apilado final solo necesitamos lo que ya está en p2 (incluye score si existía)
        filas.append(p2)

    # --- Apila detecciones
    if filas:
        apilado = pd.concat(filas, ignore_index=True)
    else:
        apilado = pd.DataFrame(columns=["fecha", "valor", "metodo_deteccion", "metodo_prediccion", "fuente"])

    # Orden y columnas finales (preserva 'score' si existe)
    base_cols = ["fecha", "valor", "metodo_deteccion", "metodo_prediccion", "fuente"]
    cols = base_cols + (["score"] if "score" in apilado.columns else [])
    apilado = apilado[cols].sort_values(["metodo_deteccion", "fecha"]).reset_index(drop=True)
    apilado.insert(0, "id", apilado.index + 1)

    csv_anoms = os.path.join(base_dir, f"{model_tag}_anomalies.csv")
    apilado.to_csv(csv_anoms, index=False, encoding="utf-8")

    # --- (OPCIONAL) Agrupar consecutivas
    do_group = bool(getattr(cfg, "CONS_SINT_CONSECUTIVAS", True))
    csv_group = None
    if do_group:
        csv_group = os.path.join(base_dir, f"{model_tag}_anomalies_agrupado.csv")
        consolidate_prediction_series_consecutive(apilado, csv_group)

    return {
        "csv_anomalies": csv_anoms,
        "csv_anomalies_agrupado": csv_group,
        "base_dir": base_dir,
        "modelo_prediccion": model_tag,
    }

def consolidate_prediction_series_consecutive(df_in: pd.DataFrame | str, csv_out: str) -> None:
        """
        CONSECUTIVAS para SERIES DE PREDICCIÓN (sintéticas).
        Agrupa por (metodo_deteccion, metodo_prediccion) uniendo días consecutivos.
        Salida: id, start_date, end_date, count, metodos_deteccion, metodo_prediccion,
                min_value, max_value, peak_value, peak_time, mean_value, std_value
        """
        import numpy as np

        if isinstance(df_in, str):
            df = pd.read_csv(df_in, parse_dates=["fecha"])
        else:
            df = df_in.copy()
            if not np.issubdtype(df["fecha"].dtype, np.datetime64):
                df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

        grupos = []
        keys = ["metodo_deteccion", "metodo_prediccion"]
        for key_vals, g in df.groupby(keys, dropna=False):
            g = g.sort_values("fecha").reset_index(drop=True)
            if g.empty:
                continue

            det, mp = key_vals
            run_start = g.loc[0, "fecha"]
            prev_t    = g.loc[0, "fecha"]
            vals      = [g.loc[0, "valor"]]
            times     = [g.loc[0, "fecha"]]

            def flush():
                if not vals:
                    return
                seg = pd.DataFrame({"valor": vals, "fecha": times})
                row = {
                    "start_date": run_start,
                    "end_date":   prev_t,
                    "count":      len(seg),
                    "metodos_deteccion": det,
                    "metodo_prediccion": mp,
                    "min_value":  seg["valor"].min(),
                    "max_value":  seg["valor"].max(),
                    "peak_value": seg.loc[seg["valor"].idxmax(), "valor"],
                    "peak_time":  seg.loc[seg["valor"].idxmax(), "fecha"],
                    "mean_value": seg["valor"].mean(),
                    "std_value":  seg["valor"].std(ddof=0),
                }
                grupos.append(row)

            for i in range(1, len(g)):
                t = g.loc[i, "fecha"]; v = g.loc[i, "valor"]
                if (t - prev_t).days == 1:
                    vals.append(v); times.append(t); prev_t = t
                else:
                    flush()
                    run_start = t
                    prev_t = t
                    vals  = [v]
                    times = [t]
            flush()

        if grupos:
            out = pd.DataFrame(grupos).sort_values(["metodo_prediccion", "start_date"]).reset_index(drop=True)
            out.insert(0, "id", out.index + 1)
        else:
            out = pd.DataFrame(columns=[
                "id","start_date","end_date","count","metodos_deteccion","metodo_prediccion",
                "min_value","max_value","peak_value","peak_time","mean_value","std_value"
            ])

        out.to_csv(csv_out, index=False, encoding="utf-8")
