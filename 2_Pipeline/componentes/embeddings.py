import os
import math
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
from sentence_transformers import SentenceTransformer

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

_MODEL_NAME = "clip-ViT-B-32"


def _load_img_model(model_name: str = _MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def _embed_pngs_from_dir(dir_in: str, model: SentenceTransformer) -> pd.DataFrame:
    """
    Lee TODOS los .png de dir_in (recursivo), calcula embeddings L2-normalizados.
    Devuelve DataFrame con columnas: file_path, file_name, folder, vec(list[float]).
    """
    rows = []
    base_dir = Path(dir_in)

    for root, _, files in os.walk(base_dir):
        for f in sorted(files):
            if not f.lower().endswith(".png"):
                continue

            p = Path(root) / f
            img = Image.open(p).convert("RGB")
            emb = model.encode(img, convert_to_numpy=True, normalize_embeddings=True)

            rows.append({
                "file_path": str(p.resolve()),
                "file_name": f,
                "folder": str(base_dir.resolve()),
                "vec": emb.astype("float32").tolist(),
            })

    return pd.DataFrame(rows)


def _cosine_topk(q: np.ndarray, M: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna (indices_topk, scores_topk) para q frente a M (coseno).
    q y M deben estar L2-normalizados. Usa producto punto.
    """
    sims = (M @ q)  # (N,)
    order = np.argsort(-sims)
    k = min(int(k), len(order))
    idx = order[:k]
    return idx, sims[idx]


def build_corpus_and_queries_indices(
    synth_dir: str,
    orig_dir: str,
    out_dir: str,
    model_name: str,
) -> Tuple[str, str]:
    """
    Firma intacta.

    ACORDADO (CORREGIDO):
      - queries (df_q)  = SINTÉTICAS (synth_dir)
      - corpus  (df_c)  = ORIGINALES (orig_dir)

    Mantiene nombres de salida (por compatibilidad con el pipeline):
      - queries_synthetic.parquet   -> ahora contiene SINTÉTICAS
      - corpus_original.parquet     -> ahora contiene ORIGINALES
    """
    os.makedirs(out_dir, exist_ok=True)
    model = _load_img_model(model_name)

    # Queries = SINTÉTICAS
    df_q = _embed_pngs_from_dir(synth_dir, model)

    # Corpus = ORIGINALES
    df_c = _embed_pngs_from_dir(orig_dir, model)

    q_path = os.path.join(out_dir, "queries_synthetic.parquet")
    c_path = os.path.join(out_dir, "corpus_original.parquet")

    df_q.to_parquet(q_path, index=False)
    df_c.to_parquet(c_path, index=False)

    return q_path, c_path

  
def compare_synthetic_against_originalsv1(
    queries_parquet: str,
    corpus_parquet: str,
    out_csv: str,
    cmp_min_sim: float,
    topk_per_synthetic: int,
    model_name: str,
    model_weights: str,
    consolidado_synth_csv: str,
    consolidado_orig_csv: str,
) -> str:
    """
    Compara anomalías SINTÉTICAS (queries) contra ORIGINALES (corpus).
    - Queries: PNGs de anomalias_sinteticos_limpios/
    - Corpus:  PNGs de GraficosOriginalLimpios/
    - Fechas sintéticas: desde anomalias_comparacion_numerica_sinteticas.csv
    - Fechas originales: desde anomalias_consolidado.csv
    - Salida: ComparacionesEmbeddings/similaridades_embeddings.csv
    """
    import os, time
    import numpy as np
    import pandas as pd

    def _cosine_topk(q_vec, C_mat, k):
        q = np.asarray(q_vec, dtype=np.float32)
        C = np.asarray(C_mat, dtype=np.float32)
        qn = float(np.linalg.norm(q)) + 1e-12
        Cn = np.linalg.norm(C, axis=1) + 1e-12
        sims = (C @ q) / (Cn * qn)
        k = max(1, int(k))
        if k >= sims.shape[0]:
            idx = np.argsort(-sims)
            return idx.tolist(), sims[idx].astype(np.float32).tolist()
        part = np.argpartition(-sims, k - 1)[:k]
        part = part[np.argsort(-sims[part])]
        return part.tolist(), sims[part].astype(np.float32).tolist()

    def _safe_str(x):
        if x is None: return ""
        if isinstance(x, float) and np.isnan(x): return ""
        return str(x)

    # ── Cargar parquets ────────────────────────────────────────────────────
    df_q = pd.read_parquet(queries_parquet)  # SINTÉTICAS
    df_c = pd.read_parquet(corpus_parquet)   # ORIGINALES

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # ── Lookup fechas SINTÉTICAS ───────────────────────────────────────────
    _lookup_synth = {}
    if os.path.isfile(consolidado_synth_csv):
        _df = pd.read_csv(consolidado_synth_csv)[["id_anomalia", "start_date", "end_date"]]
        _lookup_synth = {
            int(row["id_anomalia"]): {
                "start_date": str(row["start_date"])[:10],
                "end_date":   str(row["end_date"])[:10],
            }
            for _, row in _df.iterrows()
        }
        print(f"[EMB_SYN] lookup sintéticas: {len(_lookup_synth)} registros desde {consolidado_synth_csv}")
    else:
        print(f"[EMB_SYN][WARN] No encontrado: {consolidado_synth_csv}")

    # ── Lookup fechas ORIGINALES ───────────────────────────────────────────
    _lookup_orig = {}
    if os.path.isfile(consolidado_orig_csv):
        _df = pd.read_csv(consolidado_orig_csv)[["id", "start_date", "end_date"]]
        _lookup_orig = {
            int(row["id"]): {
                "start_date": str(row["start_date"])[:10],
                "end_date":   str(row["end_date"])[:10],
            }
            for _, row in _df.iterrows()
        }
        print(f"[EMB_SYN] lookup originales: {len(_lookup_orig)} registros desde {consolidado_orig_csv}")
    else:
        print(f"[EMB_SYN][WARN] No encontrado: {consolidado_orig_csv}")

    if df_q.empty or df_c.empty:
        print("[EMB_SYN] Parquet vacío — CSV vacío generado.")
        pd.DataFrame().to_csv(out_csv, index=False)
        return out_csv

    Q = np.vstack(df_q["vec"].to_list()).astype("float32")
    C = np.vstack(df_c["vec"].to_list()).astype("float32")

    RATIO_DELTA   = 0.001
    GAMMA         = 4.0
    topk_internal = max(int(topk_per_synthetic), 2)
    run_ts        = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    nc, nq        = C.shape[0], Q.shape[0]

    print(f"[EMB_SYN] Q shape={Q.shape} C shape={C.shape}")
    print(f"[EMB_SYN] RATIO_DELTA={RATIO_DELTA} GAMMA={GAMMA} topk_internal={topk_internal}")

    rows = []
    skipped_ratio = 0
    skipped_threshold = 0

    for i in range(nq):
        q_row      = df_q.iloc[i]
        syn_name   = q_row["file_name"]
        syn_id     = os.path.splitext(syn_name)[0]  # ej: 047_RF_DIRECT_DBSCAN
        syn_id_int = int(syn_id.split("_")[0]) if syn_id.split("_")[0].isdigit() else None

        idx, scores = _cosine_topk(Q[i], C, topk_internal)

        # ── Ratio test ────────────────────────────────────────────────────
        if len(scores) >= 2:
            s1, s2 = float(scores[0]), float(scores[1])
            if (s1 - s2) < RATIO_DELTA:
                skipped_ratio += 1
                continue

        kept = 0
        passed = False

        for rnk, (j, s) in enumerate(zip(idx, scores), start=1):
            if kept >= int(topk_per_synthetic):
                break

            s = float(s)
            if s < float(cmp_min_sim):
                continue

            passed = True
            c_row       = df_c.iloc[int(j)]
            orig_name   = os.path.splitext(c_row["file_name"])[0]
            orig_id_int = int(orig_name.split("_")[0]) if orig_name.split("_")[0].isdigit() else None

            # ── Fechas desde lookups separados ────────────────────────────
            syn_fechas  = _lookup_synth.get(syn_id_int,  {})
            orig_fechas = _lookup_orig.get(orig_id_int,  {})

            s_strict = max(0.0, min(1.0, s)) ** GAMMA

            rows.append({
                "synthetic_id":         syn_id,
                "original_id":          orig_name,
                "synthetic_name":       syn_name,
                "original_name":        c_row["file_name"],
                "synthetic_path":       _safe_str(q_row.get("file_path", "")),
                "original_path":        _safe_str(c_row.get("file_path", "")),
                "synthetic_folder":     _safe_str(q_row.get("folder", "")),
                "original_folder":      _safe_str(c_row.get("folder", "")),
                "synthetic_start_date": syn_fechas.get("start_date", ""),
                "synthetic_end_date":   syn_fechas.get("end_date",   ""),
                "original_start_date":  orig_fechas.get("start_date", ""),
                "original_end_date":    orig_fechas.get("end_date",   ""),
                "similarity_cosine":    s,
                "similarity_pct":       f"{s_strict * 100:.1f}%",
                "rank_in_corpus":       int(rnk),
                "topk_used":            int(topk_per_synthetic),
                "threshold_used":       float(cmp_min_sim),
                "passed_threshold":     True,
                "corpus_size":          int(nc),
                "query_set_size":       int(nq),
                "model_name":           model_name,
                "model_weights":        model_weights,
                "run_timestamp":        run_ts,
                "notes":                f"gamma={GAMMA}|ratio_delta={RATIO_DELTA}",
            })
            kept += 1

        if not passed:
            skipped_threshold += 1

    print(f"[EMB_SYN] nq={nq} nc={nc} | skipped_ratio={skipped_ratio} "
          f"skipped_threshold={skipped_threshold} | rows={len(rows)}")

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values(
            ["synthetic_name", "similarity_cosine"],
            ascending=[True, False]
        )
    df_out.to_csv(out_csv, index=False)
    print(f"[EMB_SYN] CSV escrito: {out_csv} | filas: {len(df_out)}")
    return out_csv


def compare_synthetic_against_originals(
    queries_parquet: str,
    corpus_parquet: str,
    out_csv: str,
    cmp_min_sim: float,
    topk_per_synthetic: int,
    model_name: str,
    model_weights: str,
    consolidado_synth_csv: str,
    consolidado_orig_csv: str,
) -> str:
    """
    Compara anomalías SINTÉTICAS (queries) contra ORIGINALES (corpus).
    - Queries: PNGs de anomalias_sinteticos_limpios/
    - Corpus:  PNGs de GraficosOriginalLimpios/
    - Fechas sintéticas: desde anomalias_comparacion_numerica_sinteticas.csv
    - Fechas originales: desde anomalias_consolidado.csv
    - Salida: ComparacionesEmbeddings/similaridades_embeddings.csv
    - Tendencia sintéticas: media pre/post desde serie sintética predicha
    - Tendencia originales: media pre/post desde SerieOriginal.csv
    """
    import os, time
    import numpy as np
    import pandas as pd
    from pathlib import Path

    def _cosine_topk(q_vec, C_mat, k):
        q = np.asarray(q_vec, dtype=np.float32)
        C = np.asarray(C_mat, dtype=np.float32)
        qn = float(np.linalg.norm(q)) + 1e-12
        Cn = np.linalg.norm(C, axis=1) + 1e-12
        sims = (C @ q) / (Cn * qn)
        k = max(1, int(k))
        if k >= sims.shape[0]:
            idx = np.argsort(-sims)
            return idx.tolist(), sims[idx].astype(np.float32).tolist()
        part = np.argpartition(-sims, k - 1)[:k]
        part = part[np.argsort(-sims[part])]
        return part.tolist(), sims[part].astype(np.float32).tolist()

    def _safe_str(x):
        if x is None: return ""
        if isinstance(x, float) and np.isnan(x): return ""
        return str(x)

    # ── Tendencia desde serie (aplica a sintéticas y originales) ──────────
    VENTANA_TENDENCIA = 5

    def _tendencia_desde_serie(serie_df, start_date, end_date) -> str:
        try:
            start = pd.to_datetime(start_date)
            end   = pd.to_datetime(end_date)
            pre  = serie_df[
                (serie_df["fecha"] >= start - pd.Timedelta(days=VENTANA_TENDENCIA)) &
                (serie_df["fecha"] <  start)
            ]["valor"]
            post = serie_df[
                (serie_df["fecha"] >  end) &
                (serie_df["fecha"] <= end + pd.Timedelta(days=VENTANA_TENDENCIA))
            ]["valor"]
            if pre.empty or post.empty:
                return "indefinida"
            delta  = post.mean() - pre.mean()
            umbral = pre.mean() * 0.01
            if delta > umbral:
                return "subida"
            elif delta < -umbral:
                return "bajada"
            else:
                return "indefinida"
        except Exception:
            return "indefinida"

    # ── Cargar SerieOriginal una vez ───────────────────────────────────────
    RUTA_RESULTADOS = "/home/jacky/DatosSerie/Resultados/"
    serie_orig_path = os.path.join(
        os.path.dirname(RUTA_RESULTADOS.rstrip("/")), "SerieOriginal.csv"
    )
    serie_orig_df = None
    if os.path.isfile(serie_orig_path):
        serie_orig_df = pd.read_csv(serie_orig_path)
        serie_orig_df["fecha"] = pd.to_datetime(serie_orig_df["fecha"])
        serie_orig_df = serie_orig_df.sort_values("fecha").reset_index(drop=True)
        print(f"[EMB_SYN] SerieOriginal cargada: {len(serie_orig_df)} filas")
    else:
        print(f"[EMB_SYN][WARN] SerieOriginal no encontrada: {serie_orig_path}")

    # ── Cargar 14 series sintéticas en diccionario {modelo: df} ───────────
    pred_root = os.path.join(RUTA_RESULTADOS, "Prediccion")
    _series_sinteticas = {}
    if os.path.isdir(pred_root):
        for modelo_dir in sorted(Path(pred_root).iterdir()):
            if not modelo_dir.is_dir():
                continue
            candidates = list(modelo_dir.glob("1_*_sintetica.csv"))
            if candidates:
                try:
                    df_s = pd.read_csv(candidates[0])
                    df_s["fecha"] = pd.to_datetime(df_s["fecha"], errors="coerce")
                    df_s = df_s.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)
                    _series_sinteticas[modelo_dir.name] = df_s
                    print(f"[EMB_SYN] Serie sintética cargada: {modelo_dir.name} "
                          f"({len(df_s)} filas)")
                except Exception as e:
                    print(f"[EMB_SYN][WARN] No se pudo cargar {candidates[0]}: {e}")
    print(f"[EMB_SYN] Total series sintéticas cargadas: {len(_series_sinteticas)}")

    # ── Cargar parquets ────────────────────────────────────────────────────
    df_q = pd.read_parquet(queries_parquet)  # SINTÉTICAS
    df_c = pd.read_parquet(corpus_parquet)   # ORIGINALES

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # ── Lookup SINTÉTICAS (con metodo_prediccion) ──────────────────────────
    _lookup_synth = {}
    if os.path.isfile(consolidado_synth_csv):
        _df = pd.read_csv(consolidado_synth_csv)
        for _, row in _df.iterrows():
            try:
                anom_id = int(row["id_anomalia"])
            except Exception:
                continue
            _lookup_synth[anom_id] = {
                "start_date":        str(row["start_date"])[:10],
                "end_date":          str(row["end_date"])[:10],
                "metodo_prediccion": str(row.get("metodo_prediccion", "")),
            }
        print(f"[EMB_SYN] lookup sintéticas: {len(_lookup_synth)} registros "
              f"desde {consolidado_synth_csv}")
    else:
        print(f"[EMB_SYN][WARN] No encontrado: {consolidado_synth_csv}")

    # ── Lookup ORIGINALES ──────────────────────────────────────────────────
    _lookup_orig = {}
    if os.path.isfile(consolidado_orig_csv):
        _df = pd.read_csv(consolidado_orig_csv)[["id", "start_date", "end_date"]]
        _lookup_orig = {
            int(row["id"]): {
                "start_date": str(row["start_date"])[:10],
                "end_date":   str(row["end_date"])[:10],
            }
            for _, row in _df.iterrows()
        }
        print(f"[EMB_SYN] lookup originales: {len(_lookup_orig)} registros "
              f"desde {consolidado_orig_csv}")
    else:
        print(f"[EMB_SYN][WARN] No encontrado: {consolidado_orig_csv}")

    # ── Caché de tendencia para ORIGINALES ────────────────────────────────
    _tendencia_cache_orig = {}

    def _tendencia_original(anom_id_int: int) -> str:
        if serie_orig_df is None:
            return "indefinida"
        if anom_id_int in _tendencia_cache_orig:
            return _tendencia_cache_orig[anom_id_int]
        fechas = _lookup_orig.get(anom_id_int, {})
        if not fechas:
            _tendencia_cache_orig[anom_id_int] = "indefinida"
            return "indefinida"
        tend = _tendencia_desde_serie(
            serie_orig_df,
            fechas["start_date"],
            fechas["end_date"],
        )
        _tendencia_cache_orig[anom_id_int] = tend
        return tend

    if df_q.empty or df_c.empty:
        print("[EMB_SYN] Parquet vacío — CSV vacío generado.")
        pd.DataFrame().to_csv(out_csv, index=False)
        return out_csv

    Q = np.vstack(df_q["vec"].to_list()).astype("float32")
    C = np.vstack(df_c["vec"].to_list()).astype("float32")

    RATIO_DELTA   = 0.001
    GAMMA         = 4.0
    # Ampliado para compensar descartes por tendencia
    topk_internal = max(int(topk_per_synthetic) * 5, 10)
    run_ts        = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    nc, nq        = C.shape[0], Q.shape[0]

    print(f"[EMB_SYN] Q shape={Q.shape} C shape={C.shape}")
    print(f"[EMB_SYN] RATIO_DELTA={RATIO_DELTA} GAMMA={GAMMA} "
          f"topk_internal={topk_internal}")
    print(f"[EMB_SYN] VENTANA_TENDENCIA={VENTANA_TENDENCIA} días")

    rows              = []
    skipped_ratio     = 0
    skipped_threshold = 0
    skipped_tendencia = 0

    for i in range(nq):
        q_row      = df_q.iloc[i]
        syn_name   = q_row["file_name"]
        syn_id     = os.path.splitext(syn_name)[0]
        syn_id_int = int(syn_id.split("_")[0]) if syn_id.split("_")[0].isdigit() else None

        # Tendencia sintética desde su serie predicha
        syn_fechas = _lookup_synth.get(syn_id_int, {})
        modelo     = syn_fechas.get("metodo_prediccion", "")
        serie_syn  = _series_sinteticas.get(modelo)
        tend_syn   = _tendencia_desde_serie(
            serie_syn,
            syn_fechas.get("start_date", ""),
            syn_fechas.get("end_date",   ""),
        ) if serie_syn is not None else "indefinida"

        idx, scores = _cosine_topk(Q[i], C, topk_internal)

        # ── Ratio test ────────────────────────────────────────────────────
        if len(scores) >= 2:
            s1, s2 = float(scores[0]), float(scores[1])
            if (s1 - s2) < RATIO_DELTA:
                skipped_ratio += 1
                continue

        kept   = 0
        passed = False

        for rnk, (j, s) in enumerate(zip(idx, scores), start=1):
            if kept >= int(topk_per_synthetic):
                break

            s = float(s)
            if s < float(cmp_min_sim):
                continue

            c_row       = df_c.iloc[int(j)]
            orig_name   = os.path.splitext(c_row["file_name"])[0]
            orig_id_int = int(orig_name.split("_")[0]) \
                          if orig_name.split("_")[0].isdigit() else None

            # Tendencia original desde SerieOriginal.csv
            tend_orig = _tendencia_original(orig_id_int) if orig_id_int else "indefinida"

            # ── Filtro tendencia: solo descarta opuestos definidos ─────────
            if (tend_syn  == "subida" and tend_orig == "bajada") or \
               (tend_syn  == "bajada" and tend_orig == "subida"):
                skipped_tendencia += 1
                print(f"[EMB_SYN][TEND] Descartado: {syn_id}({tend_syn}) "
                      f"↔ {orig_name}({tend_orig})")
                continue

            passed      = True
            orig_fechas = _lookup_orig.get(orig_id_int, {})
            s_strict    = max(0.0, min(1.0, s)) ** GAMMA

            rows.append({
                "synthetic_id":         syn_id,
                "original_id":          orig_name,
                "synthetic_name":       syn_name,
                "original_name":        c_row["file_name"],
                "synthetic_path":       _safe_str(q_row.get("file_path", "")),
                "original_path":        _safe_str(c_row.get("file_path", "")),
                "synthetic_folder":     _safe_str(q_row.get("folder", "")),
                "original_folder":      _safe_str(c_row.get("folder", "")),
                "synthetic_start_date": syn_fechas.get("start_date", ""),
                "synthetic_end_date":   syn_fechas.get("end_date",   ""),
                "original_start_date":  orig_fechas.get("start_date", ""),
                "original_end_date":    orig_fechas.get("end_date",   ""),
                "similarity_cosine":    s,
                "similarity_pct":       f"{s_strict * 100:.1f}%",
                "rank_in_corpus":       int(rnk),
                "topk_used":            int(topk_per_synthetic),
                "threshold_used":       float(cmp_min_sim),
                "passed_threshold":     True,
                "corpus_size":          int(nc),
                "query_set_size":       int(nq),
                "model_name":           model_name,
                "model_weights":        model_weights,
                "run_timestamp":        run_ts,
                "tend_synthetic":       tend_syn,
                "tend_original":        tend_orig,
                "notes":                f"gamma={GAMMA}|ratio_delta={RATIO_DELTA}"
                                        f"|ventana_tend={VENTANA_TENDENCIA}",
            })
            kept += 1

        if not passed:
            skipped_threshold += 1

    print(f"[EMB_SYN] nq={nq} nc={nc} | skipped_ratio={skipped_ratio} | "
          f"skipped_tendencia={skipped_tendencia} | "
          f"skipped_threshold={skipped_threshold} | rows={len(rows)}")

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values(
            ["synthetic_name", "similarity_cosine"],
            ascending=[True, False]
        )
    df_out.to_csv(out_csv, index=False)
    print(f"[EMB_SYN] CSV escrito: {out_csv} | filas: {len(df_out)}")
    return out_csv


def compare_historical_against_historicalv1(
    queries_parquet: str,
    corpus_parquet: str,
    out_csv: str,
    cmp_min_sim: float,
    topk_per_query: int,
    model_name: str,
    model_weights: str,
    consolidado_orig_csv: str,
) -> str:
    """
    Compara anomalías RECIENTES (queries) contra HISTÓRICAS (corpus).
    Ambos grupos vienen de la serie original — no hay predicción.
    - Queries: PNGs de anomalías recientes en GraficosOriginalLimpios/
    - Corpus:  PNGs de anomalías históricas en GraficosOriginalLimpios/
    - Fechas ambos grupos: desde anomalias_consolidado.csv
    - Salida: ComparacionesHist/similaridades_embeddings_hist.csv
    """
    import os, time
    import numpy as np
    import pandas as pd

    def _cosine_topk(q_vec, C_mat, k):
        q = np.asarray(q_vec, dtype=np.float32)
        C = np.asarray(C_mat, dtype=np.float32)
        qn = float(np.linalg.norm(q)) + 1e-12
        Cn = np.linalg.norm(C, axis=1) + 1e-12
        sims = (C @ q) / (Cn * qn)
        k = max(1, int(k))
        if k >= sims.shape[0]:
            idx = np.argsort(-sims)
            return idx.tolist(), sims[idx].astype(np.float32).tolist()
        part = np.argpartition(-sims, k - 1)[:k]
        part = part[np.argsort(-sims[part])]
        return part.tolist(), sims[part].astype(np.float32).tolist()

    def _safe_str(x):
        if x is None: return ""
        if isinstance(x, float) and np.isnan(x): return ""
        return str(x)

    # ── Cargar parquets ────────────────────────────────────────────────────
    df_q = pd.read_parquet(queries_parquet)  # RECIENTES
    df_c = pd.read_parquet(corpus_parquet)   # HISTÓRICAS

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # ── Lookup fechas — ambos grupos usan anomalias_consolidado.csv ────────
    _lookup = {}
    if os.path.isfile(consolidado_orig_csv):
        _df = pd.read_csv(consolidado_orig_csv)[["id", "start_date", "end_date"]]
        _lookup = {
            int(row["id"]): {
                "start_date": str(row["start_date"])[:10],
                "end_date":   str(row["end_date"])[:10],
            }
            for _, row in _df.iterrows()
        }
        print(f"[EMB_HIST] lookup: {len(_lookup)} registros desde {consolidado_orig_csv}")
    else:
        print(f"[EMB_HIST][WARN] No encontrado: {consolidado_orig_csv}")

    if df_q.empty or df_c.empty:
        print("[EMB_HIST] Parquet vacío — CSV vacío generado.")
        pd.DataFrame().to_csv(out_csv, index=False)
        return out_csv

    Q = np.vstack(df_q["vec"].to_list()).astype("float32")
    C = np.vstack(df_c["vec"].to_list()).astype("float32")

    RATIO_DELTA   = 0.001
    GAMMA         = 4.0
    topk_internal = max(int(topk_per_query), 2)
    run_ts        = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    nc, nq        = C.shape[0], Q.shape[0]

    print(f"[EMB_HIST] Q shape={Q.shape} C shape={C.shape}")
    print(f"[EMB_HIST] RATIO_DELTA={RATIO_DELTA} GAMMA={GAMMA} topk_internal={topk_internal}")

    rows = []
    skipped_ratio     = 0
    skipped_threshold = 0

    for i in range(nq):
        q_row         = df_q.iloc[i]
        reciente_name = q_row["file_name"]
        reciente_id   = os.path.splitext(reciente_name)[0]
        reciente_id_int = int(reciente_id.split("_")[0]) if reciente_id.split("_")[0].isdigit() else None

        idx, scores = _cosine_topk(Q[i], C, topk_internal)

        # ── Ratio test ────────────────────────────────────────────────────
        if len(scores) >= 2:
            s1, s2 = float(scores[0]), float(scores[1])
            if (s1 - s2) < RATIO_DELTA:
                skipped_ratio += 1
                continue

        kept   = 0
        passed = False

        for rnk, (j, s) in enumerate(zip(idx, scores), start=1):
            if kept >= int(topk_per_query):
                break

            s = float(s)
            if s < float(cmp_min_sim):
                continue

            passed        = True
            c_row         = df_c.iloc[int(j)]
            historico_name = c_row["file_name"]
            historico_id   = os.path.splitext(historico_name)[0]
            historico_id_int = int(historico_id.split("_")[0]) if historico_id.split("_")[0].isdigit() else None

            reciente_fechas  = _lookup.get(reciente_id_int,  {})
            historico_fechas = _lookup.get(historico_id_int, {})

            s_strict = max(0.0, min(1.0, s)) ** GAMMA

            rows.append({
                # ── Anomalía reciente (queries) ──────────────────────────
                "synthetic_id":         reciente_id,
                "synthetic_name":       reciente_name,
                "synthetic_path":       _safe_str(q_row.get("file_path", "")),
                "synthetic_folder":     _safe_str(q_row.get("folder", "")),
                "synthetic_start_date": reciente_fechas.get("start_date", ""),
                "synthetic_end_date":   reciente_fechas.get("end_date",   ""),
                # ── Anomalía histórica (corpus) ──────────────────────────
                "original_id":          historico_id,
                "original_name":        historico_name,
                "original_path":        _safe_str(c_row.get("file_path", "")),
                "original_folder":      _safe_str(c_row.get("folder", "")),
                "original_start_date":  historico_fechas.get("start_date", ""),
                "original_end_date":    historico_fechas.get("end_date",   ""),
                # ── Métricas ─────────────────────────────────────────────
                "similarity_cosine":    s,
                "similarity_pct":       f"{s_strict * 100:.1f}%",
                "rank_in_corpus":       int(rnk),
                "topk_used":            int(topk_per_query),
                "threshold_used":       float(cmp_min_sim),
                "passed_threshold":     True,
                "corpus_size":          int(nc),
                "query_set_size":       int(nq),
                "model_name":           model_name,
                "model_weights":        model_weights,
                "run_timestamp":        run_ts,
                "notes":                f"gamma={GAMMA}|ratio_delta={RATIO_DELTA}",
            })
            kept += 1

        if not passed:
            skipped_threshold += 1

    print(f"[EMB_HIST] nq={nq} nc={nc} | skipped_ratio={skipped_ratio} "
          f"skipped_threshold={skipped_threshold} | rows={len(rows)}")

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values(
            ["synthetic_name", "similarity_cosine"],
            ascending=[True, False]
        )
    df_out.to_csv(out_csv, index=False)
    print(f"[EMB_HIST] CSV escrito: {out_csv} | filas: {len(df_out)}")
    return out_csv


def compare_historical_against_historical(
    queries_parquet: str,
    corpus_parquet: str,
    out_csv: str,
    cmp_min_sim: float,
    topk_per_query: int,
    model_name: str,
    model_weights: str,
    consolidado_orig_csv: str,
) -> str:
    """
    Compara anomalías RECIENTES (queries) contra HISTÓRICAS (corpus).
    Ambos grupos vienen de la serie original — no hay predicción.
    Incluye filtro de tendencia: solo compara pares con misma dirección
    (subida vs subida, bajada vs bajada) usando ventana de contexto
    de la SerieOriginal.csv.
    """
    import os, time
    import numpy as np
    import pandas as pd

    def _cosine_topk(q_vec, C_mat, k):
        q = np.asarray(q_vec, dtype=np.float32)
        C = np.asarray(C_mat, dtype=np.float32)
        qn = float(np.linalg.norm(q)) + 1e-12
        Cn = np.linalg.norm(C, axis=1) + 1e-12
        sims = (C @ q) / (Cn * qn)
        k = max(1, int(k))
        if k >= sims.shape[0]:
            idx = np.argsort(-sims)
            return idx.tolist(), sims[idx].astype(np.float32).tolist()
        part = np.argpartition(-sims, k - 1)[:k]
        part = part[np.argsort(-sims[part])]
        return part.tolist(), sims[part].astype(np.float32).tolist()

    def _safe_str(x):
        if x is None: return ""
        if isinstance(x, float) and np.isnan(x): return ""
        return str(x)

    # ── Cargar parquets ────────────────────────────────────────────────────
    df_q = pd.read_parquet(queries_parquet)  # RECIENTES
    df_c = pd.read_parquet(corpus_parquet)   # HISTÓRICAS

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # ── Cargar serie original para calcular tendencias ────────────────────
    RUTA_RESULTADOS = "/home/jacky/DatosSerie/Resultados/"
    serie_path = os.path.join(os.path.dirname(RUTA_RESULTADOS.rstrip("/")), "SerieOriginal.csv")
    serie_df   = None
    VENTANA_TENDENCIA = 5  # días antes y después de la anomalía

    if os.path.isfile(serie_path):
        serie_df = pd.read_csv(serie_path)
        serie_df["fecha"] = pd.to_datetime(serie_df["fecha"])
        serie_df = serie_df.sort_values("fecha").reset_index(drop=True)
        print(f"[EMB_HIST] SerieOriginal cargada: {len(serie_df)} filas | {serie_path}")
    else:
        print(f"[EMB_HIST][WARN] SerieOriginal no encontrada: {serie_path} — filtro de tendencia desactivado")

    # ── Lookup fechas — ambos grupos usan anomalias_consolidado.csv ────────
    _lookup = {}
    if os.path.isfile(consolidado_orig_csv):
        _df = pd.read_csv(consolidado_orig_csv)[["id", "start_date", "end_date"]]
        _lookup = {
            int(row["id"]): {
                "start_date": str(row["start_date"])[:10],
                "end_date":   str(row["end_date"])[:10],
            }
            for _, row in _df.iterrows()
        }
        print(f"[EMB_HIST] lookup: {len(_lookup)} registros desde {consolidado_orig_csv}")
    else:
        print(f"[EMB_HIST][WARN] No encontrado: {consolidado_orig_csv}")

    # ── Pre-calcular tendencia de todas las anomalías ─────────────────────
    _tendencia_cache = {}

    def _calcular_tendencia(anom_id_int: int) -> str:
        """
        Calcula tendencia de una anomalía usando ventana de contexto
        en SerieOriginal.csv.
        subida  → media post > media pre
        bajada  → media post < media pre
        plana   → sin diferencia significativa
        indefinida → sin datos suficientes
        """
        if serie_df is None:
            return "indefinida"
        if anom_id_int in _tendencia_cache:
            return _tendencia_cache[anom_id_int]

        fechas = _lookup.get(anom_id_int, {})
        if not fechas:
            _tendencia_cache[anom_id_int] = "indefinida"
            return "indefinida"

        start = pd.to_datetime(fechas["start_date"])
        end   = pd.to_datetime(fechas["end_date"])

        pre  = serie_df[
            (serie_df["fecha"] >= start - pd.Timedelta(days=VENTANA_TENDENCIA)) &
            (serie_df["fecha"] <  start)
        ]["valor"]
        post = serie_df[
            (serie_df["fecha"] >  end) &
            (serie_df["fecha"] <= end + pd.Timedelta(days=VENTANA_TENDENCIA))
        ]["valor"]

        if pre.empty or post.empty:
            _tendencia_cache[anom_id_int] = "indefinida"
            return "indefinida"

        delta = post.mean() - pre.mean()

        # Umbral mínimo del 1% del valor pre para evitar clasificar como
        # subida/bajada diferencias insignificantes
        umbral = pre.mean() * 0.01

        if delta > umbral:
            tendencia = "subida"
        elif delta < -umbral:
            tendencia = "bajada"
        else:
            tendencia = "plana"

        _tendencia_cache[anom_id_int] = tendencia
        return tendencia

    if df_q.empty or df_c.empty:
        print("[EMB_HIST] Parquet vacío — CSV vacío generado.")
        pd.DataFrame().to_csv(out_csv, index=False)
        return out_csv

    Q = np.vstack(df_q["vec"].to_list()).astype("float32")
    C = np.vstack(df_c["vec"].to_list()).astype("float32")

    RATIO_DELTA   = 0.001
    GAMMA         = 4.0
    #topk_internal = max(int(topk_per_query), 2)
    topk_internal = max(int(topk_per_query) * 5, 10)
    run_ts        = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    nc, nq        = C.shape[0], Q.shape[0]

    print(f"[EMB_HIST] Q shape={Q.shape} C shape={C.shape}")
    print(f"[EMB_HIST] RATIO_DELTA={RATIO_DELTA} GAMMA={GAMMA} topk_internal={topk_internal}")
    print(f"[EMB_HIST] VENTANA_TENDENCIA={VENTANA_TENDENCIA} días")

    rows = []
    skipped_ratio      = 0
    skipped_threshold  = 0
    skipped_tendencia  = 0

    for i in range(nq):
        q_row           = df_q.iloc[i]
        reciente_name   = q_row["file_name"]
        reciente_id     = os.path.splitext(reciente_name)[0]
        reciente_id_int = int(reciente_id.split("_")[0]) if reciente_id.split("_")[0].isdigit() else None

        # Tendencia de la anomalía reciente
        tend_reciente = _calcular_tendencia(reciente_id_int) if reciente_id_int else "indefinida"

        idx, scores = _cosine_topk(Q[i], C, topk_internal)

        # ── Ratio test ────────────────────────────────────────────────────
        if len(scores) >= 2:
            s1, s2 = float(scores[0]), float(scores[1])
            if (s1 - s2) < RATIO_DELTA:
                skipped_ratio += 1
                continue

        kept   = 0
        passed = False

        for rnk, (j, s) in enumerate(zip(idx, scores), start=1):
            if kept >= int(topk_per_query):
                break

            s = float(s)
            if s < float(cmp_min_sim):
                continue

            c_row            = df_c.iloc[int(j)]
            historico_name   = c_row["file_name"]
            historico_id     = os.path.splitext(historico_name)[0]
            historico_id_int = int(historico_id.split("_")[0]) if historico_id.split("_")[0].isdigit() else None

            # ── Filtro de tendencia ───────────────────────────────────────
            tend_historico = _calcular_tendencia(historico_id_int) if historico_id_int else "indefinida"

            if (tend_reciente != "indefinida" and
                tend_historico != "indefinida" and
                tend_reciente  != tend_historico):
                skipped_tendencia += 1
                print(f"[EMB_HIST][TEND] Descartado: {reciente_id}({tend_reciente}) ↔ "
                      f"{historico_id}({tend_historico}) — dirección opuesta")
                continue

            passed        = True
            reciente_fechas  = _lookup.get(reciente_id_int,  {})
            historico_fechas = _lookup.get(historico_id_int, {})

            s_strict = max(0.0, min(1.0, s)) ** GAMMA

            rows.append({
                # ── Anomalía reciente (queries) ───────────────────────────
                "synthetic_id":           reciente_id,
                "synthetic_name":         reciente_name,
                "synthetic_path":         _safe_str(q_row.get("file_path", "")),
                "synthetic_folder":       _safe_str(q_row.get("folder", "")),
                "synthetic_start_date":   reciente_fechas.get("start_date", ""),
                "synthetic_end_date":     reciente_fechas.get("end_date",   ""),
                "synthetic_tendencia":    tend_reciente,
                # ── Anomalía histórica (corpus) ───────────────────────────
                "original_id":            historico_id,
                "original_name":          historico_name,
                "original_path":          _safe_str(c_row.get("file_path", "")),
                "original_folder":        _safe_str(c_row.get("folder", "")),
                "original_start_date":    historico_fechas.get("start_date", ""),
                "original_end_date":      historico_fechas.get("end_date",   ""),
                "original_tendencia":     tend_historico,
                # ── Métricas ──────────────────────────────────────────────
                "similarity_cosine":      s,
                "similarity_pct":         f"{s_strict * 100:.1f}%",
                "rank_in_corpus":         int(rnk),
                "topk_used":              int(topk_per_query),
                "threshold_used":         float(cmp_min_sim),
                "passed_threshold":       True,
                "corpus_size":            int(nc),
                "query_set_size":         int(nq),
                "model_name":             model_name,
                "model_weights":          model_weights,
                "run_timestamp":          run_ts,
                "notes":                  f"gamma={GAMMA}|ratio_delta={RATIO_DELTA}|ventana_tend={VENTANA_TENDENCIA}",
            })
            kept += 1

        if not passed:
            skipped_threshold += 1

    print(f"[EMB_HIST] nq={nq} nc={nc} | skipped_ratio={skipped_ratio} "
          f"skipped_threshold={skipped_threshold} | skipped_tendencia={skipped_tendencia} "
          f"| rows={len(rows)}")

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values(
            ["synthetic_name", "similarity_cosine"],
            ascending=[True, False]
        )
    df_out.to_csv(out_csv, index=False)
    print(f"[EMB_HIST] CSV escrito: {out_csv} | filas: {len(df_out)}")
    return out_csv
    
def generate_contact_sheets_per_synthetic(
    results_csv: str,
    output_dir: str,
    topk_per_synthetic: int,
    gap_px: int,
    thumb_width_px: int,
    show_labels: bool,
    show_percent: bool,
):
    """
    Firma intacta.

    CORREGIDO para el nuevo flujo:
      - Genera 1 lámina POR SINTÉTICA:
        SINTÉTICA grande + Top-K ORIGINALES (más parecidas)
    """
    df = pd.read_csv(results_csv)
    if df.empty:
        return

    out_dir = Path(output_dir) / "laminas_per_sintetica"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_w = int(thumb_width_px * 2.5)
    img_h_no_label = int(round(img_w * 0.75))
    label_h = 60 if show_labels else 0
    tile_h = img_h_no_label + label_h

    font_size = 30
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = None

    # Helper robusto para leer columnas opcionales
    def _pick(row, keys, default=""):
        for k in keys:
            if k in row and pd.notna(row[k]) and str(row[k]).strip():
                return str(row[k]).strip()
        return default

    # Agrupar por SINTÉTICA (porque ahora cada sintética tiene sus top-K originales)
    for syn_name, g in df.groupby("synthetic_name", sort=False):
        g = g.sort_values("similarity_cosine", ascending=False).head(int(topk_per_synthetic))
        if g.empty:
            continue

        # -------------------------
        # SINTÉTICA (tile grande)
        # -------------------------
        syn_path = Path(g.iloc[0]["synthetic_path"])
        syn_img = Image.open(syn_path).convert("RGB")
        syn_id  = _pick(g.iloc[0], ["synthetic_id"], "?")
        syn_ini = _pick(g.iloc[0], ["synthetic_start_date"], "")
        syn_fin = _pick(g.iloc[0], ["synthetic_end_date"],   "")
        syn_resized = ImageOps.contain(syn_img, (img_w, img_h_no_label))
        syn_tile = Image.new("RGB", (img_w, img_h_no_label), "white")
        syn_tile.paste(
            syn_resized,
            ((img_w - syn_resized.width) // 2, (img_h_no_label - syn_resized.height) // 2),
        )

        thumbs = [syn_tile]

        # Label SINTÉTICA (id + fechas si existieran; si no, al menos el nombre)
       
        if syn_ini and syn_fin:
            labels = [f"SINTÉTICA | id={syn_id} | {syn_ini}–{syn_fin}"]
        else:
            labels = [f"SINTÉTICA | id={syn_id} | {syn_name}"]

        # -------------------------
        # ORIGINALES (Top-K)
        # -------------------------
        for _, row in g.iterrows():
            im = Image.open(row["original_path"]).convert("RGB")
            im_resized = ImageOps.contain(im, (img_w, img_h_no_label))
            tile = Image.new("RGB", (img_w, img_h_no_label), "white")
            tile.paste(
                im_resized,
                ((img_w - im_resized.width) // 2, (img_h_no_label - im_resized.height) // 2),
            )
            thumbs.append(tile)

            lbl = f"{row['original_name']}"
            if show_percent:
                pct = row.get("similarity_pct", "")
                if pd.notna(pct) and str(pct).strip():
                    pct_fmt = str(pct)
                    lbl = f"{lbl} | {pct_fmt}"
            labels.append(lbl)

        # Layout
        k = len(thumbs)
        cols = min(3, k)
        nrows = math.ceil(k / cols)

        W = cols * img_w + (cols - 1) * int(gap_px)
        H = nrows * tile_h + (nrows - 1) * int(gap_px)

        canvas = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(canvas)

        for j in range(k):
            r = j // cols
            c = j % cols
            x = c * (img_w + int(gap_px))
            y = r * (tile_h + int(gap_px))

            canvas.paste(thumbs[j], (x, y))
            if show_labels:
                draw.rectangle([x, y + img_h_no_label, x + img_w, y + tile_h], fill=(240, 240, 240))
                draw.text((x + 10, y + img_h_no_label + 12), labels[j], fill=(0, 0, 0), font=font)

        out_path = out_dir / f"{Path(syn_name).stem}__top{len(g)}.png"
        canvas.save(out_path)

def run_embeddings_comparison_and_sheets(CFG) -> dict:
    """
    Orquesta la comparación visual para anomalías SINTÉTICAS (predichas) vs ORIGINALES.

    Flujo:
      1) Lee anomalías sintéticas desde el consolidado de predicciones
         y anomalías originales desde CFG.ANOMALIAS_CONSOLIDADO
      2) Calcula embeddings CLIP para ambos grupos leyendo los PNGs
         con sistema de checkpoint por lotes (igual que comparar_anomalias_embeddings_historicas_dtw)
      3) Para cada sintética busca las top-K originales más similares
      4) Guarda parquets, CSV de similitudes y láminas visuales
         en CFG.RUTA_RESULTADOS/ComparacionesEmbeddings/

    Archivo de salida:
        CFG.RUTA_RESULTADOS/ComparacionesEmbeddings/similaridades_embeddings.csv

    Retorna dict con rutas clave.
    """
    import os
    import gc
    from pathlib import Path
    from PIL import Image
    import pandas as pd

    # ── Rutas ─────────────────────────────────────────────────────────────────
    synth_dir   = os.path.join(CFG.RUTA_RESULTADOS, "Prediccion", "anomalias_sinteticos_limpios")
    orig_dir    = os.path.join(CFG.RUTA_RESULTADOS, "GraficosConsolidado", "GraficosOriginalLimpios")
    out_root    = os.path.join(CFG.RUTA_RESULTADOS, "ComparacionesEmbeddings")

    # CSV de anomalías para lookup de fechas
    consolidado_orig  = CFG.ANOMALIAS_CONSOLIDADO
    consolidado_synth = os.path.join(
        CFG.RUTA_RESULTADOS, "Prediccion",
        "anomalias_sinteticas_csv",
        "anomalias_comparacion_numerica_sinteticas.csv"
    )

    BATCH_SIZE = int(getattr(CFG, "EMB_BATCH_SIZE", 20))

    os.makedirs(out_root, exist_ok=True)

    print(f"[EMB_SYN][INFO] synth_dir         = {synth_dir}")
    print(f"[EMB_SYN][INFO] orig_dir          = {orig_dir}")
    print(f"[EMB_SYN][INFO] out_root          = {out_root}")
    print(f"[EMB_SYN][INFO] consolidado_orig  = {consolidado_orig}")
    print(f"[EMB_SYN][INFO] consolidado_synth = {consolidado_synth}")
    print(f"[EMB_SYN][INFO] BATCH_SIZE        = {BATCH_SIZE}")

    # ── Helper: encontrar PNG por id y fecha ──────────────────────────────────
    def _find_png(anom_id: int, start_date: str, directorio: str) -> str | None:
        base = Path(directorio)
        candidatos = [
            base / f"{anom_id:03d}_{start_date}.png",
            base / f"{anom_id}_{start_date}.png",
            base / f"{anom_id:04d}_{start_date}.png",
        ]
        for p in candidatos:
            if p.exists():
                return str(p)
        patron = f"{anom_id}_"
        for p in base.glob("*.png"):
            if p.name.startswith(patron) or p.name.startswith(f"{anom_id:03d}_"):
                return str(p)
        return None

    # ── Helper: embed con checkpoint por lotes ────────────────────────────────
    def _embed_list_batched(
        df_anom: pd.DataFrame,
        role: str,
        checkpoint_path: str,
        model,
        id_col: str,
        date_col: str,
        png_dir: str,
    ) -> pd.DataFrame:

        ids_pendientes = list(df_anom[id_col].astype(str))
        rows_previos   = []

        if os.path.isfile(checkpoint_path):
            try:
                df_ckpt        = pd.read_parquet(checkpoint_path)
                ids_ya         = set(df_ckpt["anomaly_id"].astype(str).tolist())
                ids_pendientes = [i for i in ids_pendientes if i not in ids_ya]
                rows_previos   = df_ckpt.to_dict("records")
                print(f"[EMB_SYN][CKPT] {role}: {len(rows_previos)} ya procesados, "
                      f"{len(ids_pendientes)} pendientes")
            except Exception as e:
                print(f"[EMB_SYN][CKPT][WARN] No se pudo leer checkpoint: {e} → recalculando todo")
                ids_pendientes = list(df_anom[id_col].astype(str))
                rows_previos   = []

        if not ids_pendientes:
            print(f"[EMB_SYN][CKPT] {role}: todos ya procesados desde checkpoint.")
            return pd.DataFrame(rows_previos)

        df_pendiente = df_anom[df_anom[id_col].astype(str).isin(ids_pendientes)].copy()

        rows    = list(rows_previos)
        missing = []
        total   = len(df_pendiente)

        for lote_inicio in range(0, total, BATCH_SIZE):
            lote     = df_pendiente.iloc[lote_inicio: lote_inicio + BATCH_SIZE]
            lote_num = lote_inicio // BATCH_SIZE + 1
            print(f"[EMB_SYN][INFO] {role} lote {lote_num} "
                  f"({lote_inicio + 1}-{min(lote_inicio + BATCH_SIZE, total)}/{total})")

            imagenes_abiertas = []

            for _, row in lote.iterrows():
                anom_id_raw = row[id_col]
                start_date  = str(row[date_col])[:10]

                # Intentar convertir a int para buscar PNG numérico
                try:
                    anom_id_int = int(anom_id_raw)
                except (ValueError, TypeError):
                    anom_id_int = None

                png_path = None
                if anom_id_int is not None:
                    png_path = _find_png(anom_id_int, start_date, png_dir)

                # Fallback: buscar por fecha en el nombre del archivo
                if png_path is None:
                    for p in Path(png_dir).glob("*.png"):
                        if start_date in p.name:
                            png_path = str(p)
                            break

                if png_path is None:
                    missing.append(anom_id_raw)
                    print(f"[EMB_SYN][WARN] PNG no encontrado id={anom_id_raw} fecha={start_date}")
                    continue

                img = Image.open(png_path).convert("RGB")
                imagenes_abiertas.append(img)

                emb = model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
                rows.append({
                    "file_path":  png_path,
                    "file_name":  Path(png_path).name,
                    "folder":     str(Path(png_dir).resolve()),
                    "vec":        emb.astype("float32").tolist(),
                    "anomaly_id": str(anom_id_raw),
                    "start_date": start_date,
                })
                del emb

            for img in imagenes_abiertas:
                try:
                    img.close()
                except Exception:
                    pass
            del imagenes_abiertas

            # Checkpoint parcial
            df_parcial = pd.DataFrame(rows)
            df_parcial.to_parquet(checkpoint_path, index=False)
            print(f"[EMB_SYN][CKPT] {role}: checkpoint guardado con {len(rows)} registros")

            del df_parcial
            gc.collect()

        print(f"[EMB_SYN][INFO] {role}: {len(rows)} indexados, {len(missing)} sin PNG")
        return pd.DataFrame(rows)

    # ── 1) Cargar anomalías originales ────────────────────────────────────────
    if not os.path.isfile(consolidado_orig):
        raise FileNotFoundError(f"[EMB_SYN] No encontrado: {consolidado_orig}")

    df_orig = pd.read_csv(consolidado_orig)
    print(f"[EMB_SYN][INFO] Anomalías originales: {len(df_orig)}")

    # ── 2) Cargar anomalías sintéticas ────────────────────────────────────────
    if not os.path.isfile(consolidado_synth):
        raise FileNotFoundError(f"[EMB_SYN] No encontrado: {consolidado_synth}")

    df_synth = pd.read_csv(consolidado_synth)
    print(f"[EMB_SYN][INFO] Anomalías sintéticas: {len(df_synth)}")

    # Detectar columna id y fecha en sintéticas
    id_col_synth   = "id_anomalia" if "id_anomalia" in df_synth.columns else "id"
    date_col_synth = "start_date"

    # ── 3) Calcular embeddings con checkpoint ─────────────────────────────────
    model = _load_img_model(CFG.CMP_MODEL_NAME)

    ckpt_q = os.path.join(out_root, "_ckpt_queries_syn.parquet")
    ckpt_c = os.path.join(out_root, "_ckpt_corpus_syn.parquet")

    # Queries = sintéticas
    df_q = _embed_list_batched(
        df_anom         = df_synth,
        role            = "queries_sinteticas",
        checkpoint_path = ckpt_q,
        model           = model,
        id_col          = id_col_synth,
        date_col        = date_col_synth,
        png_dir         = synth_dir,
    )

    # Corpus = originales
    df_c = _embed_list_batched(
        df_anom         = df_orig,
        role            = "corpus_originales",
        checkpoint_path = ckpt_c,
        model           = model,
        id_col          = "id",
        date_col        = "start_date",
        png_dir         = orig_dir,
    )

    # Liberar modelo
    del model
    gc.collect()
    print("[EMB_SYN][INFO] Modelo CLIP liberado de memoria")

    # ── 4) Guardar parquets finales ───────────────────────────────────────────
    q_parquet = os.path.join(out_root, "queries_synthetic.parquet")
    c_parquet = os.path.join(out_root, "corpus_original.parquet")
    df_q.to_parquet(q_parquet, index=False)
    df_c.to_parquet(c_parquet, index=False)

    # ── 5) Comparar ───────────────────────────────────────────────────────────
    out_csv = os.path.join(out_root, "similaridades_embeddings.csv")
    compare_synthetic_against_originals(
        queries_parquet    = q_parquet,
        corpus_parquet     = c_parquet,
        out_csv            = out_csv,
        cmp_min_sim        = CFG.CMP_MIN_SIM,
        topk_per_synthetic = CFG.CMP_TOPK_PER_SINTETICA,
        model_name         = CFG.CMP_MODEL_NAME,
        model_weights      = getattr(CFG, "CMP_MODEL_WEIGHTS", ""),
        consolidado_synth_csv = consolidado_synth,
        consolidado_orig_csv  = consolidado_orig,
    )

    # ── 6) Láminas ────────────────────────────────────────────────────────────
    if getattr(CFG, "CMP_GENERAR_LAMINAS_PER_SINTETICA", True):
        laminas_dir = CFG.CMP_LAMINAS_OUTPUT_DIR or out_root
        generate_contact_sheets_per_synthetic(
            results_csv        = out_csv,
            output_dir         = laminas_dir,
            topk_per_synthetic = CFG.CMP_TOPK_PER_SINTETICA,
            gap_px             = CFG.CMP_CANVAS_GAP_PX,
            thumb_width_px     = CFG.CMP_THUMB_WIDTH_PX,
            show_labels        = CFG.CMP_SHOW_LABELS,
            show_percent       = CFG.CMP_SHOW_PERCENT,
        )
    else:
        laminas_dir = out_root

    return {
        "queries_parquet": q_parquet,
        "corpus_parquet":  c_parquet,
        "csv":             out_csv,
        "laminas_dir":     laminas_dir,
        "n_queries":       len(df_q),
        "n_corpus":        len(df_c),
    }


def run_embeddings_comparison_and_sheetsV1(CFG) -> dict:
    """
    Orquesta:
    1) Indexado ORIGINALES (queries) y SINTÉTICAS (corpus)
    2) Comparación ORIGINAL -> SINTÉTICAS con umbral y topK
    3) Láminas por ORIGINAL

    Devuelve dict con rutas clave.
    """
    synth_dir = os.path.join(CFG.RUTA_RESULTADOS, "Prediccion", "anomalias_sinteticos_limpios")
    orig_dir  = os.path.join(CFG.RUTA_RESULTADOS, "GraficosConsolidado", "GraficosOriginalLimpios")

    out_root = os.path.join(CFG.RUTA_RESULTADOS, "ComparacionesEmbeddings")
    os.makedirs(out_root, exist_ok=True)

    q_parquet, c_parquet = build_corpus_and_queries_indices(
        synth_dir=synth_dir,
        orig_dir=orig_dir,
        out_dir=out_root,
        model_name=CFG.CMP_MODEL_NAME,
    )

    out_csv = os.path.join(out_root, "similaridades_embeddings.csv")
    compare_synthetic_against_originals(
        queries_parquet=q_parquet,
        corpus_parquet=c_parquet,
        out_csv=out_csv,
        cmp_min_sim=CFG.CMP_MIN_SIM,
        topk_per_synthetic=CFG.CMP_TOPK_PER_SINTETICA,
        model_name=CFG.CMP_MODEL_NAME,
        model_weights=getattr(CFG, "CMP_MODEL_WEIGHTS", ""),
    )

    if getattr(CFG, "CMP_GENERAR_LAMINAS_PER_SINTETICA", True):
        generate_contact_sheets_per_synthetic(
            results_csv=out_csv,
            output_dir=CFG.CMP_LAMINAS_OUTPUT_DIR or out_root,
            topk_per_synthetic=CFG.CMP_TOPK_PER_SINTETICA,
            gap_px=CFG.CMP_CANVAS_GAP_PX,
            thumb_width_px=CFG.CMP_THUMB_WIDTH_PX,
            show_labels=CFG.CMP_SHOW_LABELS,
            show_percent=CFG.CMP_SHOW_PERCENT,
        )

    return {
        "queries_parquet": q_parquet,
        "corpus_parquet": c_parquet,
        "csv": out_csv,
        "laminas_dir": CFG.CMP_LAMINAS_OUTPUT_DIR or out_root
    }



def _split_anomalies_by_cutoff(
    anomalias_csv: str,
    num_dias: int,
):
    """
    Divide el CSV de anomalías en dos grupos MUTUAMENTE EXCLUYENTES:
      - df_queries : anomalías recientes (últimos num_dias días)
      - df_corpus  : anomalías históricas (anteriores al corte)

    Garantiza que ningún id aparezca en ambos grupos.
    Retorna (df_queries, df_corpus, fecha_corte)
    """
    df = pd.read_csv(anomalias_csv)
    df["start_date"] = pd.to_datetime(df["start_date"])

    max_date    = df["start_date"].max()
    fecha_corte = max_date - pd.Timedelta(days=num_dias)

    df_queries = df[df["start_date"] >= fecha_corte].copy()
    df_corpus  = df[df["start_date"] <  fecha_corte].copy()

    # ── Garantizar separación limpia por ID ─────────────────────────────────
    # Si un mismo id cae en ambos grupos (ej: start_date == fecha_corte exacto
    # por redondeo o duplicados en el CSV), lo removemos del corpus para que
    # NUNCA se compare una anomalía contra sí misma.
    ids_queries = set(df_queries["id"].astype(int))
    ids_corpus  = set(df_corpus["id"].astype(int))
    overlap     = ids_queries & ids_corpus

    if overlap:
        print(f"[SPLIT][WARN] {len(overlap)} IDs presentes en ambos grupos "
              f"→ removiendo del corpus: {sorted(overlap)}")
        df_corpus = df_corpus[~df_corpus["id"].astype(int).isin(overlap)].copy()

    # ── Diagnóstico final ────────────────────────────────────────────────────
    print(f"[SPLIT] max_date    = {max_date.date()}")
    print(f"[SPLIT] fecha_corte = {fecha_corte.date()}")
    print(f"[SPLIT] queries (recientes)  : {len(df_queries)} anomalías")
    print(f"[SPLIT] corpus  (históricas) : {len(df_corpus)}  anomalías")
    print(f"[SPLIT] overlap tras fix     : {len(set(df_queries['id'].astype(int)) & set(df_corpus['id'].astype(int)))}")

    return df_queries, df_corpus, str(fecha_corte.date())

def _find_png_for_anomaly(
    anom_id: int,
    start_date: str,
    graficos_dir: str,
) -> str | None:
    """
    Busca el PNG de una anomalía en graficos_dir.
    Formato esperado: NNN_YYYY-MM-DD.png  (ej: 042_2023-05-01.png)
    Intenta varias combinaciones y devuelve la primera que exista.
    Retorna None si no encuentra nada.
    """
    base = Path(graficos_dir)

    candidatos = [
        base / f"{anom_id:03d}_{start_date}.png",
        base / f"{anom_id}_{start_date}.png",
        base / f"{anom_id:04d}_{start_date}.png",
    ]

    for p in candidatos:
        if p.exists():
            return str(p)

    # Búsqueda flexible: cualquier archivo que empiece con el id
    patron = f"{anom_id}_"
    for p in base.glob("*.png"):
        if p.name.startswith(patron) or p.name.startswith(f"{anom_id:03d}_"):
            return str(p)

    return None
        
        

def comparar_anomalias_embeddings_historicas_dtw(CFG) -> dict:
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
    compare_historical_against_historical , generate_contact_sheets_per_synthetic.
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

    # ── 2) Calcular embeddings con lotes + checkpoint + liberación de memoria ──
    #
    # Estrategia para 4 GB RAM:
    #   - BATCH_SIZE imágenes por vez → libera PIL + tensores entre lotes
    #   - Checkpoint en parquet parcial → si falla, reanuda desde el último lote
    #   - gc.collect() + cierre explícito de imágenes entre lotes
    #   - El modelo se carga UNA vez y se destruye al terminar ambos grupos

    BATCH_SIZE = int(getattr(CFG, "EMB_BATCH_SIZE", 20))  # ajustable en settings.py

    def _embed_anomaly_list_batched(
        df_anom: pd.DataFrame,
        role: str,
        checkpoint_path: str,
        model,
    ) -> pd.DataFrame:
        import gc

        # ── Checkpoint: si existe parquet parcial con todos los ids, reusar ──
        ids_pendientes = list(df_anom["id"].astype(int))
        rows_previos   = []

        if os.path.isfile(checkpoint_path):
            try:
                df_ckpt = pd.read_parquet(checkpoint_path)
                ids_ya  = set(df_ckpt["anomaly_id"].astype(int).tolist())
                ids_pendientes = [i for i in ids_pendientes if i not in ids_ya]
                rows_previos   = df_ckpt.to_dict("records")
                print(f"[EMB_HIST][CKPT] {role}: {len(rows_previos)} ya procesados, "
                      f"{len(ids_pendientes)} pendientes")
            except Exception as e:
                print(f"[EMB_HIST][CKPT][WARN] No se pudo leer checkpoint: {e} → recalculando todo")
                ids_pendientes = list(df_anom["id"].astype(int))
                rows_previos   = []

        if not ids_pendientes:
            print(f"[EMB_HIST][CKPT] {role}: todos ya procesados desde checkpoint.")
            return pd.DataFrame(rows_previos)

        # Filtrar solo los pendientes
        df_pendiente = df_anom[df_anom["id"].astype(int).isin(ids_pendientes)].copy()

        rows    = list(rows_previos)  # acumular sobre los ya procesados
        missing = []
        total   = len(df_pendiente)

        # Procesar en lotes
        for lote_inicio in range(0, total, BATCH_SIZE):
            lote = df_pendiente.iloc[lote_inicio : lote_inicio + BATCH_SIZE]
            lote_num = lote_inicio // BATCH_SIZE + 1
            print(f"[EMB_HIST][INFO] {role} lote {lote_num} "
                  f"({lote_inicio+1}-{min(lote_inicio+BATCH_SIZE, total)}/{total})")

            imagenes_abiertas = []  # para cerrar explícitamente al final del lote

            for _, row in lote.iterrows():
                anom_id    = int(row["id"])
                start_date = str(row["start_date"])[:10]
                png_path   = _find_png_for_anomaly(anom_id, start_date, graficos_dir)

                if png_path is None:
                    missing.append(anom_id)
                    print(f"[EMB_HIST][WARN] PNG no encontrado id={anom_id} fecha={start_date}")
                    continue

                img = Image.open(png_path).convert("RGB")
                imagenes_abiertas.append(img)

                emb = model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
                rows.append({
                    "file_path":  png_path,
                    "file_name":  Path(png_path).name,
                    "folder":     str(Path(graficos_dir).resolve()),
                    "vec":        emb.astype("float32").tolist(),
                    "anomaly_id": anom_id,
                    "start_date": start_date,
                })

                # Liberar referencia al embedding inmediatamente
                del emb

            # ── Cerrar imágenes PIL del lote ─────────────────────────────────
            for img in imagenes_abiertas:
                try:
                    img.close()
                except Exception:
                    pass
            del imagenes_abiertas

            # ── Checkpoint parcial tras cada lote ────────────────────────────
            df_parcial = pd.DataFrame(rows)
            df_parcial.to_parquet(checkpoint_path, index=False)
            print(f"[EMB_HIST][CKPT] {role}: checkpoint guardado con {len(rows)} registros")

            # ── Liberar memoria ──────────────────────────────────────────────
            del df_parcial
            gc.collect()

        print(f"[EMB_HIST][INFO] {role}: {len(rows)} indexados, {len(missing)} sin PNG")
        return pd.DataFrame(rows)

    # Cargar modelo UNA sola vez para ambos grupos
    model = _load_img_model(CFG.CMP_MODEL_NAME)

   # ckpt_q = os.path.join(out_root, "_ckpt_queries_hist.parquet")
   # ckpt_c = os.path.join(out_root, "_ckpt_corpus_hist.parquet")
    ckpt_q = os.path.join(out_root, f"_ckpt_queries_hist_{fecha_corte}.parquet")
    ckpt_c = os.path.join(out_root, f"_ckpt_corpus_hist_{fecha_corte}.parquet")

    df_q = _embed_anomaly_list_batched(df_queries, "queries", ckpt_q, model)
    df_c = _embed_anomaly_list_batched(df_corpus,  "corpus",  ckpt_c, model)

    # Liberar modelo de memoria antes de comparar
    import gc
    del model
    gc.collect()
    print("[EMB_HIST][INFO] Modelo CLIP liberado de memoria")

    # ── 3) Guardar parquets ──────────────────────────────────────────────────
    q_parquet = os.path.join(out_root, "queries_hist.parquet")
    c_parquet = os.path.join(out_root, "corpus_hist.parquet")
    df_q.to_parquet(q_parquet, index=False)
    df_c.to_parquet(c_parquet, index=False)

    # ── 4) Comparar (reutiliza la función ya existente) ──────────────────────
    out_csv = os.path.join(out_root, "similaridades_embeddings_hist.csv")
    compare_historical_against_historical (
        queries_parquet      = q_parquet,
        corpus_parquet       = c_parquet,
        out_csv              = out_csv,
        cmp_min_sim          = CFG.CMP_MIN_SIM,
        topk_per_query       = CFG.CMP_TOPK_PER_SINTETICA,  # ← nombre correcto
        model_name           = CFG.CMP_MODEL_NAME,
        model_weights        = getattr(CFG, "CMP_MODEL_WEIGHTS", ""),
        consolidado_orig_csv = CFG.ANOMALIAS_CONSOLIDADO,
    )

    # ── 5) Láminas (reutiliza la función ya existente) ───────────────────────
    laminas_dir = os.path.join(out_root, "laminas")    
    generate_contact_sheets_per_historical(
        results_csv    = out_csv,
        output_dir     = laminas_dir,
        topk_per_query = CFG.CMP_TOPK_PER_SINTETICA,
        gap_px         = CFG.CMP_CANVAS_GAP_PX,
        thumb_width_px = CFG.CMP_THUMB_WIDTH_PX,
        show_labels    = CFG.CMP_SHOW_LABELS,
        show_percent   = CFG.CMP_SHOW_PERCENT,
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


def generate_contact_sheets_per_historical(
    results_csv: str,
    output_dir: str,
    topk_per_query: int,
    gap_px: int,
    thumb_width_px: int,
    show_labels: bool,
    show_percent: bool,
):
    """
    Genera láminas visuales para el experimento de VALIDACIÓN histórica.
    1 lámina por anomalía RECIENTE con sus top-K HISTÓRICAS más similares.
    NO toca generate_contact_sheets_per_synthetic.
    """
    df = pd.read_csv(results_csv)
    if df.empty:
        return

    out_dir = Path(output_dir) / "laminas_per_historica"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_w          = int(thumb_width_px * 2.5)
    img_h_no_label = int(round(img_w * 0.75))
    label_h        = 60 if show_labels else 0
    tile_h         = img_h_no_label + label_h

    font_size = 30
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = None

    def _pick(row, keys, default=""):
        for k in keys:
            if k in row and pd.notna(row[k]) and str(row[k]).strip():
                return str(row[k]).strip()
        return default

    for syn_name, g in df.groupby("synthetic_name", sort=False):
        g = g.sort_values("similarity_cosine", ascending=False).head(int(topk_per_query))
        if g.empty:
            continue

        # ── Tile RECIENTE ─────────────────────────────────────────────────────
        syn_path = Path(g.iloc[0]["synthetic_path"])
        syn_img  = Image.open(syn_path).convert("RGB")
        syn_id   = _pick(g.iloc[0], ["synthetic_id"], "?")
        syn_ini  = _pick(g.iloc[0], ["synthetic_start_date"], "")
        syn_fin  = _pick(g.iloc[0], ["synthetic_end_date"],   "")

        syn_resized = ImageOps.contain(syn_img, (img_w, img_h_no_label))
        syn_tile    = Image.new("RGB", (img_w, img_h_no_label), "white")
        syn_tile.paste(
            syn_resized,
            ((img_w - syn_resized.width) // 2, (img_h_no_label - syn_resized.height) // 2),
        )
        thumbs = [syn_tile]

        if syn_ini and syn_fin:
            labels = [f"RECIENTE | id={syn_id} | {syn_ini}–{syn_fin}"]
        else:
            labels = [f"RECIENTE | id={syn_id} | {syn_name}"]

        # ── Tiles HISTÓRICAS (Top-K) ──────────────────────────────────────────
        for _, row in g.iterrows():
            im         = Image.open(row["original_path"]).convert("RGB")
            im_resized = ImageOps.contain(im, (img_w, img_h_no_label))
            tile       = Image.new("RGB", (img_w, img_h_no_label), "white")
            tile.paste(
                im_resized,
                ((img_w - im_resized.width) // 2, (img_h_no_label - im_resized.height) // 2),
            )
            thumbs.append(tile)

            orig_ini = _pick(row, ["original_start_date"], "")
            orig_fin = _pick(row, ["original_end_date"],   "")
            orig_id  = _pick(row, ["original_id"],         row["original_name"])

            if orig_ini and orig_fin:
                lbl = f"HISTÓRICA | id={orig_id} | {orig_ini}–{orig_fin}"
            else:
                lbl = f"HISTÓRICA | {row['original_name']}"

            if show_percent:
                pct = row.get("similarity_pct", "")
                if pd.notna(pct) and str(pct).strip():
                    #lbl = f"{lbl} | {pct}"
                    pct_fmt = str(pct)
                    lbl = f"{lbl} | {pct_fmt}"
            labels.append(lbl)

        # ── Layout ────────────────────────────────────────────────────────────
        k     = len(thumbs)
        cols  = min(3, k)
        nrows = math.ceil(k / cols)

        W = cols * img_w + (cols - 1) * int(gap_px)
        H = nrows * tile_h + (nrows - 1) * int(gap_px)

        canvas = Image.new("RGB", (W, H), "white")
        draw   = ImageDraw.Draw(canvas)

        for j in range(k):
            r = j // cols
            c = j % cols
            x = c * (img_w + int(gap_px))
            y = r * (tile_h + int(gap_px))

            canvas.paste(thumbs[j], (x, y))
            if show_labels:
                draw.rectangle([x, y + img_h_no_label, x + img_w, y + tile_h], fill=(240, 240, 240))
                draw.text((x + 10, y + img_h_no_label + 12), labels[j], fill=(0, 0, 0), font=font)

        out_path = out_dir / f"{Path(syn_name).stem}__top{len(g)}.png"
        canvas.save(out_path)