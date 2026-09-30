"""
componentes/news.py

Consulta noticias desde Fuseki (SPARQL) para pares de anomalías similares,
calcula similitud semántica entre los conjuntos de noticias usando embeddings
de texto, y guarda los resultados en un CSV.

Uso desde el DAG:
    from componentes.news import comparar_noticias_por_pares
    out_csv = comparar_noticias_por_pares(CFG)
"""

from __future__ import annotations

import os
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)

# ── Configuración ──────────────────────────────────────────────────────────────
FUSEKI_ENDPOINT = "http://localhost:3030/c22/sparql"
ANOM_PREFIX     = "http://w3id.org/anomaly-core#"
TEXT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
VENTANA_DIAS    = 5  # ±5 días alrededor de la anomalía

def _extraer_categoria(uri: str) -> str:
    """
    Convierte 'http://w3id.org/anomaly-core#cat_politica_gobierno'
    en 'politica_gobierno'. Retorna '' si el URI está vacío.
    """
    if not uri:
        return ""
    fragmento = uri.split("#")[-1].split("/")[-1]
    if fragmento.startswith("cat_"):
        fragmento = fragmento[4:]
    return fragmento
# ── 1) Consulta SPARQL ─────────────────────────────────────────────────────────

# ── 1) Consulta SPARQL ─────────────────────────────────────────────────────────

def _query_noticias(fecha_inicio: str, fecha_fin: str) -> list[dict]:
    sparql = f"""
        PREFIX anom: <{ANOM_PREFIX}>
        PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>

        SELECT ?title ?text ?source ?fecha ?categoria
        WHERE {{
            ?n a anom:NewsArticle ;
               anom:newsTitle   ?title ;
               anom:newsText    ?text ;
               anom:newsSource  ?source ;
               anom:publishedAt ?fecha .
            OPTIONAL {{ ?n anom:hasCategory ?categoria }}
            FILTER(?fecha >= "{fecha_inicio}"^^xsd:date &&
                   ?fecha <= "{fecha_fin}"^^xsd:date)
        }}
        ORDER BY ?fecha
    """
    resp = requests.post(
        FUSEKI_ENDPOINT,
        data={"query": sparql},
        headers={"Accept": "application/sparql-results+json"},
        timeout=30,
    )
    resp.raise_for_status()

    bindings = resp.json()["results"]["bindings"]
    return [
        {
            "title":     b["title"]["value"],
            "text":      b["text"]["value"],
            "source":    b["source"]["value"],
            "date":      b["fecha"]["value"],
            "categoria": _extraer_categoria(b.get("categoria", {}).get("value", "")),
        }
        for b in bindings
    ]


def _extraer_categoria(uri: str) -> str:
    """
    Convierte 'http://w3id.org/anomaly-core#cat_politica_gobierno'
    en 'politica_gobierno'. Retorna '' si el URI está vacío.
    """
    if not uri:
        return ""
    # Tomar la parte después de '#' o '/' y quitar prefijo 'cat_'
    fragmento = uri.split("#")[-1].split("/")[-1]
    if fragmento.startswith("cat_"):
        fragmento = fragmento[4:]
    return fragmento

# ── 2) Ventana temporal ────────────────────────────────────────────────────────

def _ventana(fecha_str: str) -> tuple[str, str]:
    """Devuelve (inicio, fin) en formato YYYY-MM-DD para ±VENTANA_DIAS."""
    d = date.fromisoformat(str(fecha_str)[:10])
    return (
        (d - timedelta(days=VENTANA_DIAS)).isoformat(),
        (d + timedelta(days=VENTANA_DIAS)).isoformat(),
    )


# ── 3) Embedding de un conjunto de noticias ────────────────────────────────────

def _embedding_centroide(noticias: list[dict], model: SentenceTransformer) -> np.ndarray | None:
    """
    Combina título + texto de cada noticia, calcula embeddings y devuelve
    el centroide (promedio normalizado). Retorna None si no hay noticias.
    """
    if not noticias:
        return None
    textos = [f"{n['title']}. {n['text']}".strip() for n in noticias]
    embs = model.encode(textos, normalize_embeddings=True, show_progress_bar=False)
    return embs.mean(axis=0)


# ── 4) Función principal ───────────────────────────────────────────────────────
#COMPARA LAS HOTICIAS POR PARES DE LAS SIMILITUDES ENCONTRADAS EN EL EXPERIMENTO

def comparar_noticias_por_pares_historicas(CFG) -> str:
    """
    Lee los pares de anomalías similares ya calculados (CSV de embeddings hist),
    consulta Fuseki por las noticias de cada período, calcula similitud
    semántica y guarda un CSV con los resultados.

    Parámetros esperados en CFG:
        CFG.RUTA_COMPARACION_HIST  → carpeta donde están los archivos de entrada y salida
        CFG.FUSEKI_ENDPOINT        → (opcional) sobreescribe el endpoint por defecto

    Archivos de entrada:
        similaridades_embeddings_hist.csv

    Archivo de salida:
        similitudes_noticias_hist.csv
    """
    out_root  = CFG.RUTA_COMPARACION_HIST
    endpoint  = getattr(CFG, "FUSEKI_ENDPOINT", FUSEKI_ENDPOINT)

    csv_similitudes = os.path.join(out_root, "similaridades_embeddings_hist.csv")
    csv_salida      = os.path.join(out_root, "similitudes_noticias_hist.csv")

    if not os.path.isfile(csv_similitudes):
        raise FileNotFoundError(f"[NEWS-HIST] Archivo no encontrado: {csv_similitudes}")

    df_pares = pd.read_csv(csv_similitudes)

    # ── Filtro defensivo: descartar auto-comparaciones ────────────────────────
    antes = len(df_pares)
    df_pares = df_pares[
        df_pares["synthetic_id"].astype(str) != df_pares["original_id"].astype(str)
    ].reset_index(drop=True)
    removidos = antes - len(df_pares)
    if removidos > 0:
        print(f"[NEWS-HIST][WARN] {removidos} auto-comparaciones removidas")
    print(f"[NEWS-HIST] Pares válidos a procesar: {len(df_pares)}")

    # ── Cargar modelo de texto ────────────────────────────────────────────────
    print(f"[NEWS-HIST] Cargando modelo: {TEXT_MODEL_NAME}")
    model = SentenceTransformer(TEXT_MODEL_NAME)

    resultados = []

    for _, par in df_pares.iterrows():

        # ── Extraer IDs ───────────────────────────────────────────────────────
        syn_parts  = str(par["synthetic_id"]).split("_", 1)
        orig_parts = str(par["original_id"]).split("_", 1)

        id_reciente  = int(syn_parts[0])
        id_historica = int(orig_parts[0])
        sim_visual   = float(par["similarity_cosine"])

        # ── Extraer fechas: primero desde columnas explícitas,
        #    si no existen o están vacías, desde el identificador ──────────────
        fecha_reciente = str(par.get("synthetic_start_date", ""))[:10]
        if not fecha_reciente or fecha_reciente == "nan":
            fecha_reciente = syn_parts[1] if len(syn_parts) > 1 else ""

        fecha_historica = str(par.get("original_start_date", ""))[:10]
        if not fecha_historica or fecha_historica == "nan":
            fecha_historica = orig_parts[1] if len(orig_parts) > 1 else ""

        fecha_fin_reciente  = str(par.get("synthetic_end_date", fecha_reciente))[:10]
        if not fecha_fin_reciente or fecha_fin_reciente == "nan":
            fecha_fin_reciente = fecha_reciente

        fecha_fin_historica = str(par.get("original_end_date", fecha_historica))[:10]
        if not fecha_fin_historica or fecha_fin_historica == "nan":
            fecha_fin_historica = fecha_historica

        print(f"[NEWS-HIST] Par reciente={id_reciente} ({fecha_reciente}) ↔ "
              f"historica={id_historica} ({fecha_historica}) | "
              f"sim_visual={sim_visual:.4f}")

        # ── Validaciones ──────────────────────────────────────────────────────
        if not fecha_reciente or not fecha_historica:
            print(f"[NEWS-HIST][WARN] Sin fecha para par {id_reciente} ↔ {id_historica} — saltando")
            continue

        if fecha_reciente == fecha_historica:
            print(f"[NEWS-HIST][WARN] Par {id_reciente} ↔ {id_historica} "
                  f"fechas idénticas ({fecha_reciente}) — saltando")
            continue

        inicio_r, fin_r = _ventana(fecha_reciente)
        inicio_h, fin_h = _ventana(fecha_historica)

        print(f"[NEWS-HIST]   ventana reciente : {inicio_r} → {fin_r}")
        print(f"[NEWS-HIST]   ventana historica: {inicio_h} → {fin_h}")

        # ── Consultar Fuseki ──────────────────────────────────────────────────
        try:
            noticias_reciente  = _query_noticias(inicio_r, fin_r)
            noticias_historica = _query_noticias(inicio_h, fin_h)
        except Exception as exc:
            log.error("[NEWS-HIST] Error Fuseki par %d ↔ %d: %s",
                      id_reciente, id_historica, exc)
            print(f"[NEWS-HIST][ERROR] Fuseki par {id_reciente} ↔ {id_historica}: {exc}")
            continue

        print(f"[NEWS-HIST]   noticias reciente={len(noticias_reciente)} "
              f"historica={len(noticias_historica)}")

        # ── Calcular similitud semántica ──────────────────────────────────────
        emb_r = _embedding_centroide(noticias_reciente,  model)
        emb_h = _embedding_centroide(noticias_historica, model)

        cats_reciente  = sorted(set(
            n.get("categoria", "") for n in noticias_reciente
            if n.get("categoria", "")
        ))
        cats_historica = sorted(set(
            n.get("categoria", "") for n in noticias_historica
            if n.get("categoria", "")
        ))

        if emb_r is None and emb_h is None:
            sim_noticias = None
            motivo = "sin_noticias_ambos"
        elif emb_r is None:
            sim_noticias = None
            motivo = "sin_noticias_reciente"
        elif emb_h is None:
            sim_noticias = None
            motivo = "sin_noticias_historica"
        else:
            sim_noticias = float(cosine_similarity([emb_r], [emb_h])[0][0])
            motivo = "ok"

        print(f"[NEWS-HIST]   similitud_noticias={sim_noticias}  motivo={motivo}")

        resultados.append({
            "id_reciente":            id_reciente,
            "id_historica":           id_historica,
            "fecha_inicio_reciente":  fecha_reciente,
            "fecha_fin_reciente":     fecha_fin_reciente,
            "fecha_inicio_historica": fecha_historica,
            "fecha_fin_historica":    fecha_fin_historica,
            "similitud_visual":       sim_visual,
            "similitud_noticias":     sim_noticias,
            "motivo":                 motivo,
            "categorias_reciente":    ", ".join(cats_reciente),
            "categorias_historica":   ", ".join(cats_historica),
            "n_noticias_reciente":    len(noticias_reciente),
            "n_noticias_historica":   len(noticias_historica),
            "noticias_reciente":      str(noticias_reciente),
            "noticias_historica":     str(noticias_historica),
        })

    # ── Guardar CSV ───────────────────────────────────────────────────────────
    df_out = pd.DataFrame(resultados)
    df_out.to_csv(csv_salida, index=False)

    print(f"[NEWS-HIST] CSV guardado: {csv_salida}")
    print(f"[NEWS-HIST] Total pares procesados: {len(df_out)}")

    if not df_out.empty:
        ok           = df_out[df_out["motivo"] == "ok"]
        sin_noticias = df_out[df_out["motivo"] != "ok"]
        print(f"[NEWS-HIST] Con similitud calculada  : {len(ok)}")
        print(f"[NEWS-HIST] Sin noticias suficientes : {len(sin_noticias)}")
        if not sin_noticias.empty:
            print(f"[NEWS-HIST] Motivos:\n"
                  f"{sin_noticias['motivo'].value_counts().to_string()}")
        if not ok.empty:
            print(f"[NEWS-HIST] sim_noticias máx : {ok['similitud_noticias'].max():.4f}")
            print(f"[NEWS-HIST] sim_noticias mín : {ok['similitud_noticias'].min():.4f}")
            print(f"[NEWS-HIST] sim_noticias prom: {ok['similitud_noticias'].mean():.4f}")

    return csv_salida

###GENERA NOTICIAS PARA ANOMALIAS SINTETICAS — COMPARACION  NUMERICA
def generar_noticias_comparacion_numerica_hist(CFG) -> str:
    """
    Para cada par de anomalías similares numéricamente (DTW holdout),
    consulta Fuseki por las noticias de la anomalía histórica similar
    y genera un CSV con el contexto textual transferible.

    Solo procesa pares que superen DTW_HIST_MIN_SIM en similarity_combined.

    Archivo de entrada:
        CFG.RUTA_COMPARACION_HIST/similaridades_dtw.csv

    Archivo de salida:
        CFG.RUTA_COMPARACION_HIST/noticias_comparacion_numerica.csv

    Retorna:
        Ruta al CSV generado.
    """
    import os
    import requests
    import pandas as pd
    from datetime import date, timedelta

    FUSEKI_ENDPOINT = getattr(CFG, "FUSEKI_ENDPOINT", "http://localhost:3030/c22/sparql")
    ANOM_PREFIX     = "http://w3id.org/anomaly-core#"
    VENTANA_DIAS    = int(getattr(CFG, "VENTANA_DIAS_NOTICIAS", 5))
    DTW_MIN_SIM     = float(getattr(CFG, "DTW_HIST_MIN_SIM", 0.85))

    # CORRECTO para predichas
    out_root    = CFG.RUTA_COMPARACION_HIST
    csv_entrada = os.path.join(out_root, "similaridades_dtw.csv")
    csv_salida  = os.path.join(out_root, "noticias_comparacion_numerica.csv")

    print(f"[DTW-NEWS] ========== Noticias comparación numérica DTW ==========")
    print(f"[DTW-NEWS] CSV entrada      : {csv_entrada} | existe={os.path.isfile(csv_entrada)}")
    print(f"[DTW-NEWS] Fuseki           : {FUSEKI_ENDPOINT}")
    print(f"[DTW-NEWS] Ventana días     : ±{VENTANA_DIAS}")
    print(f"[DTW-NEWS] DTW_HIST_MIN_SIM : {DTW_MIN_SIM}")
    print(f"[DTW-NEWS] CSV salida       : {csv_salida}")

    if not os.path.isfile(csv_entrada):
        raise FileNotFoundError(f"[DTW-NEWS] No encontrado: {csv_entrada}")

    # ── Helper: ejecutar SPARQL ───────────────────────────────────────────────
    def _sparql(query: str) -> list[dict]:
        resp = requests.post(
            FUSEKI_ENDPOINT,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["results"]["bindings"]

    # ── Helper: extraer categoría desde URI ──────────────────────────────────
    def _categoria(uri: str) -> str:
        if not uri:
            return ""
        fragmento = uri.split("#")[-1].split("/")[-1]
        if fragmento.startswith("cat_"):
            fragmento = fragmento[4:]
        return fragmento

    # ── Leer CSV de similitudes DTW ───────────────────────────────────────────
    df_pares = pd.read_csv(csv_entrada)

    total_antes = len(df_pares)

    # Filtro por umbral de similitud
    df_pares = df_pares[
        df_pares["similarity_combined"] >= DTW_MIN_SIM
    ].reset_index(drop=True)

    print(f"[DTW-NEWS] Total pares      : {total_antes}")
    print(f"[DTW-NEWS] Pares >= {DTW_MIN_SIM} : {len(df_pares)}")

    # Filtro defensivo autopares
    antes    = len(df_pares)
    df_pares = df_pares[
        df_pares["id_anomalia_original"].astype(str) != df_pares["id_anomalia_sintetica"].astype(str)
    ].reset_index(drop=True)
    removidos = antes - len(df_pares)
    if removidos > 0:
        print(f"[DTW-NEWS][WARN] {removidos} auto-comparaciones removidas")

    print(f"[DTW-NEWS] Pares válidos    : {len(df_pares)}")

    resultados = []

    for _, par in df_pares.iterrows():
        id_original  = int(par["id_anomalia_original"])
        id_sintetica = int(par["id_anomalia_sintetica"])
        sim_combined = float(par["similarity_combined"])
        sim_dtw      = float(par["similarity_dtw"])
        sim_feat     = float(par["similarity_feat"])
        rank         = int(par["rank"])

        start_orig   = str(par["start_date_original"])[:10]
        end_orig     = str(par["end_date_original"])[:10]
        start_sin    = str(par["start_date_sintetica"])[:10]
        end_sin      = str(par["end_date_sintetica"])[:10]

        method_orig  = str(par.get("method_original", ""))
        method_sin   = str(par.get("method_sintetica", ""))

        print(f"[DTW-NEWS] Par orig={id_original} ({start_orig}) ↔ "
              f"sin={id_sintetica} ({start_sin}) | "
              f"combined={sim_combined:.4f} rank={rank}")

        # Ventana de noticias para la anomalía original (histórica)
        d_ini = (date.fromisoformat(start_orig) - timedelta(days=VENTANA_DIAS)).isoformat()
        d_fin = (date.fromisoformat(end_orig)   + timedelta(days=VENTANA_DIAS)).isoformat()

        # Consultar noticias
        query_noticias = f"""
            PREFIX anom: <{ANOM_PREFIX}>
            PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>

            SELECT ?title ?text ?source ?fecha ?categoria
            WHERE {{
                ?n a anom:NewsArticle ;
                   anom:newsTitle   ?title ;
                   anom:newsText    ?text ;
                   anom:newsSource  ?source ;
                   anom:publishedAt ?fecha .
                OPTIONAL {{ ?n anom:hasCategory ?categoria }}
                FILTER(?fecha >= "{d_ini}"^^xsd:date &&
                       ?fecha <= "{d_fin}"^^xsd:date)
            }}
            ORDER BY ?fecha
        """

        try:
            noticias = _sparql(query_noticias)
        except Exception as exc:
            print(f"[DTW-NEWS][ERROR] Fuseki par {id_original} ↔ {id_sintetica}: {exc}")
            noticias = []

        print(f"[DTW-NEWS]   → {len(noticias)} noticias para anomalía original {id_original}")

        if not noticias:
            resultados.append({
                "id_anomalia_sintetica":     id_sintetica,
                "sin_fecha_inicio":          start_sin,
                "sin_fecha_fin":             end_sin,
                "sin_metodo_deteccion":      method_sin,
                "id_anomalia_original":      id_original,
                "orig_fecha_inicio":         start_orig,
                "orig_fecha_fin":            end_orig,
                "orig_metodo_deteccion":     method_orig,
                "similitud_combined":        round(sim_combined, 6),
                "similitud_dtw":             round(sim_dtw, 6),
                "similitud_feat":            round(sim_feat, 6),
                "rank_similitud":            rank,
                "titulo_noticia":            None,
                "texto_noticia":             None,
                "fuente_noticia":            None,
                "fecha_noticia":             None,
                "categoria_noticia":         None,
                "motivo":                    "sin_noticias",
            })
            continue

        for noticia in noticias:
            resultados.append({
                "id_anomalia_sintetica":     id_sintetica,
                "sin_fecha_inicio":          start_sin,
                "sin_fecha_fin":             end_sin,
                "sin_metodo_deteccion":      method_sin,
                "id_anomalia_original":      id_original,
                "orig_fecha_inicio":         start_orig,
                "orig_fecha_fin":            end_orig,
                "orig_metodo_deteccion":     method_orig,
                "similitud_combined":        round(sim_combined, 6),
                "similitud_dtw":             round(sim_dtw, 6),
                "similitud_feat":            round(sim_feat, 6),
                "rank_similitud":            rank,
                "titulo_noticia":            noticia["title"]["value"],
                "texto_noticia":             noticia["text"]["value"],
                "fuente_noticia":            noticia["source"]["value"],
                "fecha_noticia":             noticia["fecha"]["value"][:10],
                "categoria_noticia":         _categoria(noticia.get("categoria", {}).get("value", "")),
                "motivo":                    "ok",
            })

    # ── Guardar CSV ───────────────────────────────────────────────────────────
    df_out = pd.DataFrame(resultados)
    df_out.to_csv(csv_salida, index=False)

    print(f"[DTW-NEWS] ========== Resultado ==========")
    print(f"[DTW-NEWS] CSV guardado      : {csv_salida}")
    print(f"[DTW-NEWS] Total filas       : {len(df_out)}")

    if not df_out.empty:
        ok  = df_out[df_out["motivo"] == "ok"]
        sin = df_out[df_out["motivo"] != "ok"]
        print(f"[DTW-NEWS] Con noticias      : {len(ok)}")
        print(f"[DTW-NEWS] Sin noticias      : {len(sin)}")
        print(f"[DTW-NEWS] Sintéticas únicas : {df_out['id_anomalia_sintetica'].nunique()}")
        print(f"[DTW-NEWS] Originales únicas : {df_out['id_anomalia_original'].nunique()}")
        if not ok.empty:
            cats = sorted(ok['categoria_noticia'].dropna().unique().tolist())
            print(f"[DTW-NEWS] Categorías        : {cats}")
            print(f"[DTW-NEWS] sim_combined máx  : {ok['similitud_combined'].max():.4f}")
            print(f"[DTW-NEWS] sim_combined mín  : {ok['similitud_combined'].min():.4f}")
            print(f"[DTW-NEWS] sim_combined prom : {ok['similitud_combined'].mean():.4f}")

    return csv_salida
###GENERA NOTICIAS PARA ANOMALIAS SINTETICAS — COMPARACION GRAFICA LIP    

def generar_noticias_comparacion_grafica_hist(CFG) -> str:
    """
    Para cada par de anomalías similares visualmente (CLIP embeddings — holdout),
    consulta Fuseki por las noticias de la anomalía histórica similar
    y genera un CSV con el contexto textual.
 
    Archivo de entrada:
        CFG.RUTA_COMPARACION_HIST/similaridades_embeddings_hist.csv
 
    Archivo de salida:
        CFG.RUTA_COMPARACION_HIST/noticias_comparacion_grafica_hist.csv
    """
    import os
    import re
    import requests
    import pandas as pd
    from datetime import date, timedelta
 
    FUSEKI_ENDPOINT = getattr(CFG, "FUSEKI_ENDPOINT", "http://localhost:3030/v19/sparql")
    ANOM_PREFIX     = "http://w3id.org/anomaly-core#"
    VENTANA_DIAS    = int(getattr(CFG, "VENTANA_DIAS_NOTICIAS", 5))
 
    out_root    = CFG.RUTA_COMPARACION_HIST
    csv_entrada = os.path.join(out_root, "similaridades_embeddings_hist.csv")
    csv_salida  = os.path.join(out_root, "noticias_comparacion_grafica_hist.csv")
 
    print(f"[GRAF-NEWS-HIST] ========== Noticias comparación gráfica holdout ==========")
    print(f"[GRAF-NEWS-HIST] CSV entrada   : {csv_entrada} | existe={os.path.isfile(csv_entrada)}")
    print(f"[GRAF-NEWS-HIST] Fuseki        : {FUSEKI_ENDPOINT}")
    print(f"[GRAF-NEWS-HIST] Ventana días  : ±{VENTANA_DIAS}")
    print(f"[GRAF-NEWS-HIST] CSV salida    : {csv_salida}")
 
    if not os.path.isfile(csv_entrada):
        raise FileNotFoundError(f"[GRAF-NEWS-HIST] No encontrado: {csv_entrada}")
 
    def _sparql(query: str) -> list[dict]:
        resp = requests.post(
            FUSEKI_ENDPOINT,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["results"]["bindings"]
 
    def _categoria(uri: str) -> str:
        if not uri:
            return ""
        fragmento = uri.split("#")[-1].split("/")[-1]
        if fragmento.startswith("cat_"):
            fragmento = fragmento[4:]
        return fragmento
 
    def _fecha_valida(valor: str) -> bool:
        """Devuelve True si el valor es una fecha en formato YYYY-MM-DD válida."""
        if not valor or valor == "nan":
            return False
        try:
            date.fromisoformat(valor[:10])
            return True
        except ValueError:
            return False
 
    df_pares = pd.read_csv(csv_entrada)
 
    antes    = len(df_pares)
    df_pares = df_pares[
        df_pares["synthetic_id"].astype(str) != df_pares["original_id"].astype(str)
    ].reset_index(drop=True)
    removidos = antes - len(df_pares)
    if removidos > 0:
        print(f"[GRAF-NEWS-HIST][WARN] {removidos} auto-comparaciones removidas")
 
    print(f"[GRAF-NEWS-HIST] Pares válidos : {len(df_pares)}")
 
    resultados = []
 
    for _, par in df_pares.iterrows():
        syn_id     = str(par["synthetic_id"])
        orig_id    = str(par["original_id"])
        sim_cosine = float(par["similarity_cosine"])
        rank       = int(par["rank_in_corpus"])
        syn_name   = str(par.get("synthetic_name", ""))
        orig_name  = str(par.get("original_name", ""))
 
        # ── Extraer fechas: columna explícita primero, respaldo desde nombre ──
        syn_start = str(par.get("synthetic_start_date", ""))[:10]
        if not _fecha_valida(syn_start):
            fechas = re.findall(r'\d{4}-\d{2}-\d{2}', syn_name)
            syn_start = fechas[0] if fechas else ""
 
        syn_end = str(par.get("synthetic_end_date", ""))[:10]
        if not _fecha_valida(syn_end):
            syn_end = syn_start
 
        orig_start = str(par.get("original_start_date", ""))[:10]
        if not _fecha_valida(orig_start):
            fechas = re.findall(r'\d{4}-\d{2}-\d{2}', orig_name)
            orig_start = fechas[0] if fechas else ""
 
        orig_end = str(par.get("original_end_date", ""))[:10]
        if not _fecha_valida(orig_end):
            orig_end = orig_start
 
        print(f"[GRAF-NEWS-HIST] Par {syn_id} ↔ {orig_id} | "
              f"syn={syn_start} orig={orig_start} | "
              f"cosine={sim_cosine:.4f} rank={rank}")
 
        if not orig_start:
            print(f"[GRAF-NEWS-HIST][WARN] Sin fecha para original {orig_id} — saltando")
            continue
 
        d_ini = (date.fromisoformat(orig_start) - timedelta(days=VENTANA_DIAS)).isoformat()
        d_fin = (date.fromisoformat(orig_end)   + timedelta(days=VENTANA_DIAS)).isoformat()
 
        print(f"[GRAF-NEWS-HIST]   ventana historica: {d_ini} → {d_fin}")
 
        query_noticias = f"""
            PREFIX anom: <{ANOM_PREFIX}>
            PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
 
            SELECT ?title ?text ?source ?fecha ?categoria
            WHERE {{
                ?n a anom:NewsArticle ;
                   anom:newsTitle   ?title ;
                   anom:newsText    ?text ;
                   anom:newsSource  ?source ;
                   anom:publishedAt ?fecha .
                OPTIONAL {{ ?n anom:hasCategory ?categoria }}
                FILTER(?fecha >= "{d_ini}"^^xsd:date &&
                       ?fecha <= "{d_fin}"^^xsd:date)
            }}
            ORDER BY ?fecha
        """
 
        try:
            noticias = _sparql(query_noticias)
        except Exception as exc:
            print(f"[GRAF-NEWS-HIST][ERROR] Fuseki par {syn_id} ↔ {orig_id}: {exc}")
            noticias = []
 
        print(f"[GRAF-NEWS-HIST]   → {len(noticias)} noticias para anomalía histórica {orig_id}")
 
        if not noticias:
            resultados.append({
                "anomalia_reciente_id":   syn_id,
                "reciente_fecha_inicio":  syn_start,
                "reciente_fecha_fin":     syn_end,
                "reciente_nombre":        syn_name,
                "anomalia_historica_id":  orig_id,
                "historica_fecha_inicio": orig_start,
                "historica_fecha_fin":    orig_end,
                "historica_nombre":       orig_name,
                "similitud_cosine":       round(sim_cosine, 6),
                "rank_visual":            rank,
                "titulo_noticia":         None,
                "texto_noticia":          None,
                "fuente_noticia":         None,
                "fecha_noticia":          None,
                "categoria_noticia":      None,
                "motivo":                 "sin_noticias",
            })
            continue
 
        for noticia in noticias:
            resultados.append({
                "anomalia_reciente_id":   syn_id,
                "reciente_fecha_inicio":  syn_start,
                "reciente_fecha_fin":     syn_end,
                "reciente_nombre":        syn_name,
                "anomalia_historica_id":  orig_id,
                "historica_fecha_inicio": orig_start,
                "historica_fecha_fin":    orig_end,
                "historica_nombre":       orig_name,
                "similitud_cosine":       round(sim_cosine, 6),
                "rank_visual":            rank,
                "titulo_noticia":         noticia["title"]["value"],
                "texto_noticia":          noticia["text"]["value"],
                "fuente_noticia":         noticia["source"]["value"],
                "fecha_noticia":          noticia["fecha"]["value"][:10],
                "categoria_noticia":      _categoria(noticia.get("categoria", {}).get("value", "")),
                "motivo":                 "ok",
            })
 
    df_out = pd.DataFrame(resultados)
    df_out.to_csv(csv_salida, index=False)
 
    print(f"[GRAF-NEWS-HIST] ========== Resultado ==========")
    print(f"[GRAF-NEWS-HIST] CSV guardado     : {csv_salida}")
    print(f"[GRAF-NEWS-HIST] Total filas      : {len(df_out)}")
 
    if not df_out.empty:
        ok  = df_out[df_out["motivo"] == "ok"]
        sin = df_out[df_out["motivo"] != "ok"]
        print(f"[GRAF-NEWS-HIST] Con noticias     : {len(ok)}")
        print(f"[GRAF-NEWS-HIST] Sin noticias     : {len(sin)}")
        print(f"[GRAF-NEWS-HIST] Recientes únicas : {df_out['anomalia_reciente_id'].nunique()}")
        print(f"[GRAF-NEWS-HIST] Históricas únicas: {df_out['anomalia_historica_id'].nunique()}")
        if not ok.empty:
            cats = sorted(ok['categoria_noticia'].dropna().unique().tolist())
            print(f"[GRAF-NEWS-HIST] Categorías       : {cats}")
 
    return csv_salida
def generar_noticias_comparacion_numerica(CFG) -> str:
    """
    Para cada par de anomalías similares numéricamente (DTW sintéticas vs históricas),
    consulta Fuseki por las noticias de la anomalía histórica similar
    y genera un CSV con el contexto textual transferible.

    Solo procesa pares que superen DTW_MIN_SIM_NOTICIAS en similarity_combined.

    Archivo de entrada:
        CFG.RUTA_COMPARACION/similaridades_dtw.csv

    Archivo de salida:
        CFG.RUTA_COMPARACION/noticias_sinteticas_dtw.csv
    """
    import os
    import requests
    import pandas as pd
    from datetime import date, timedelta

    FUSEKI_ENDPOINT = getattr(CFG, "FUSEKI_ENDPOINT", "http://localhost:3030/c22/sparql")
    ANOM_PREFIX     = "http://w3id.org/anomaly-core#"
    VENTANA_DIAS    = int(getattr(CFG, "VENTANA_DIAS_NOTICIAS", 5))
    DTW_MIN_SIM     = float(getattr(CFG, "DTW_MIN_SIM_NOTICIAS", 0.85))

    out_root    = CFG.RUTA_COMPARACION
    csv_entrada = os.path.join(out_root, "similaridades_dtw.csv")
    csv_salida  = os.path.join(out_root, "noticias_sinteticas_dtw.csv")

    print(f"[SYN-DTW-NEWS] ========== Noticias comparación numérica sintéticas ==========")
    print(f"[SYN-DTW-NEWS] CSV entrada      : {csv_entrada} | existe={os.path.isfile(csv_entrada)}")
    print(f"[SYN-DTW-NEWS] Fuseki           : {FUSEKI_ENDPOINT}")
    print(f"[SYN-DTW-NEWS] Ventana días     : ±{VENTANA_DIAS}")
    print(f"[SYN-DTW-NEWS] DTW_MIN_SIM      : {DTW_MIN_SIM}")
    print(f"[SYN-DTW-NEWS] CSV salida       : {csv_salida}")

    if not os.path.isfile(csv_entrada):
        raise FileNotFoundError(f"[SYN-DTW-NEWS] No encontrado: {csv_entrada}")

    def _sparql(query: str) -> list[dict]:
        resp = requests.post(
            FUSEKI_ENDPOINT,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["results"]["bindings"]

    def _categoria(uri: str) -> str:
        if not uri:
            return ""
        fragmento = uri.split("#")[-1].split("/")[-1]
        if fragmento.startswith("cat_"):
            fragmento = fragmento[4:]
        return fragmento

    df_pares = pd.read_csv(csv_entrada)
    total_antes = len(df_pares)

    df_pares = df_pares[
        df_pares["similarity_combined"] >= DTW_MIN_SIM
    ].reset_index(drop=True)

    antes = len(df_pares)
    df_pares = df_pares[
        df_pares["id_anomalia_original"].astype(str) != df_pares["id_anomalia_sintetica"].astype(str)
    ].reset_index(drop=True)

    print(f"[SYN-DTW-NEWS] Total pares       : {total_antes}")
    print(f"[SYN-DTW-NEWS] Pares >= {DTW_MIN_SIM}  : {antes}")
    print(f"[SYN-DTW-NEWS] Pares válidos     : {len(df_pares)}")

    resultados = []

    for _, par in df_pares.iterrows():
        id_original  = int(par["id_anomalia_original"])
        id_sintetica = int(par["id_anomalia_sintetica"])
        sim_combined = float(par["similarity_combined"])
        sim_dtw      = float(par["similarity_dtw"])
        sim_feat     = float(par["similarity_feat"])
        rank         = int(par["rank"])

        start_orig = str(par["start_date_original"])[:10]
        end_orig   = str(par["end_date_original"])[:10]
        start_sin  = str(par["start_date_sintetica"])[:10]
        end_sin    = str(par["end_date_sintetica"])[:10]

        method_orig = str(par.get("method_original", ""))
        method_sin  = str(par.get("method_sintetica", ""))

        print(f"[SYN-DTW-NEWS] Par orig={id_original} ({start_orig}) ↔ "
              f"sin={id_sintetica} ({start_sin}) | combined={sim_combined:.4f} rank={rank}")

        d_ini = (date.fromisoformat(start_orig) - timedelta(days=VENTANA_DIAS)).isoformat()
        d_fin = (date.fromisoformat(end_orig)   + timedelta(days=VENTANA_DIAS)).isoformat()

        query_noticias = f"""
            PREFIX anom: <{ANOM_PREFIX}>
            PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
            SELECT ?title ?text ?source ?fecha ?categoria
            WHERE {{
                ?n a anom:NewsArticle ;
                   anom:newsTitle   ?title ;
                   anom:newsText    ?text ;
                   anom:newsSource  ?source ;
                   anom:publishedAt ?fecha .
                OPTIONAL {{ ?n anom:hasCategory ?categoria }}
                FILTER(?fecha >= "{d_ini}"^^xsd:date &&
                       ?fecha <= "{d_fin}"^^xsd:date)
            }}
            ORDER BY ?fecha
        """

        try:
            noticias = _sparql(query_noticias)
        except Exception as exc:
            print(f"[SYN-DTW-NEWS][ERROR] Fuseki par {id_original} ↔ {id_sintetica}: {exc}")
            noticias = []

        print(f"[SYN-DTW-NEWS]   → {len(noticias)} noticias para anomalía original {id_original}")

        if not noticias:
            resultados.append({
                "id_anomalia_sintetica": id_sintetica,
                "sin_fecha_inicio":      start_sin,
                "sin_fecha_fin":         end_sin,
                "sin_metodo_deteccion":  method_sin,
                "id_anomalia_original":  id_original,
                "orig_fecha_inicio":     start_orig,
                "orig_fecha_fin":        end_orig,
                "orig_metodo_deteccion": method_orig,
                "similitud_combined":    round(sim_combined, 6),
                "similitud_dtw":         round(sim_dtw, 6),
                "similitud_feat":        round(sim_feat, 6),
                "rank_similitud":        rank,
                "titulo_noticia":        None,
                "texto_noticia":         None,
                "fuente_noticia":        None,
                "fecha_noticia":         None,
                "categoria_noticia":     None,
                "motivo":                "sin_noticias",
            })
            continue

        for noticia in noticias:
            resultados.append({
                "id_anomalia_sintetica": id_sintetica,
                "sin_fecha_inicio":      start_sin,
                "sin_fecha_fin":         end_sin,
                "sin_metodo_deteccion":  method_sin,
                "id_anomalia_original":  id_original,
                "orig_fecha_inicio":     start_orig,
                "orig_fecha_fin":        end_orig,
                "orig_metodo_deteccion": method_orig,
                "similitud_combined":    round(sim_combined, 6),
                "similitud_dtw":         round(sim_dtw, 6),
                "similitud_feat":        round(sim_feat, 6),
                "rank_similitud":        rank,
                "titulo_noticia":        noticia["title"]["value"],
                "texto_noticia":         noticia["text"]["value"],
                "fuente_noticia":        noticia["source"]["value"],
                "fecha_noticia":         noticia["fecha"]["value"][:10],
                "categoria_noticia":     _categoria(noticia.get("categoria", {}).get("value", "")),
                "motivo":                "ok",
            })

    df_out = pd.DataFrame(resultados)
    df_out.to_csv(csv_salida, index=False)

    print(f"[SYN-DTW-NEWS] ========== Resultado ==========")
    print(f"[SYN-DTW-NEWS] CSV guardado      : {csv_salida}")
    print(f"[SYN-DTW-NEWS] Total filas       : {len(df_out)}")

    if not df_out.empty:
        ok  = df_out[df_out["motivo"] == "ok"]
        sin = df_out[df_out["motivo"] != "ok"]
        print(f"[SYN-DTW-NEWS] Con noticias      : {len(ok)}")
        print(f"[SYN-DTW-NEWS] Sin noticias      : {len(sin)}")
        print(f"[SYN-DTW-NEWS] Sintéticas únicas : {df_out['id_anomalia_sintetica'].nunique()}")
        print(f"[SYN-DTW-NEWS] Originales únicas : {df_out['id_anomalia_original'].nunique()}")
        if not ok.empty:
            cats = sorted(ok['categoria_noticia'].dropna().unique().tolist())
            print(f"[SYN-DTW-NEWS] Categorías        : {cats}")
            print(f"[SYN-DTW-NEWS] sim_combined máx  : {ok['similitud_combined'].max():.4f}")
            print(f"[SYN-DTW-NEWS] sim_combined mín  : {ok['similitud_combined'].min():.4f}")
            print(f"[SYN-DTW-NEWS] sim_combined prom : {ok['similitud_combined'].mean():.4f}")

    return csv_salida


###GENERA NOTICIAS PARA ANOMALIAS SINTETICAS — COMPARACION VISUAL CLIP
def generar_noticias_comparacion_grafica(CFG) -> str:
    """
    Para cada par de anomalías similares visualmente (CLIP embeddings sintéticas
    vs históricas), consulta Fuseki por las noticias de la anomalía histórica
    similar y genera un CSV con el contexto textual.
 
    Archivo de entrada:
        CFG.RUTA_COMPARACION/similaridades_embeddings.csv
 
    Archivo de salida:
        CFG.RUTA_COMPARACION/noticias_sinteticas_visual.csv
    """
    import os
    import re
    import requests
    import pandas as pd
    from datetime import date, timedelta
 
    FUSEKI_ENDPOINT = getattr(CFG, "FUSEKI_ENDPOINT", "http://localhost:3030/c22/sparql")
    ANOM_PREFIX     = "http://w3id.org/anomaly-core#"
    VENTANA_DIAS    = int(getattr(CFG, "VENTANA_DIAS_NOTICIAS", 5))
 
    out_root    = CFG.RUTA_COMPARACION
    csv_entrada = os.path.join(out_root, "similaridades_embeddings.csv")
    csv_salida  = os.path.join(out_root, "noticias_sinteticas_visual.csv")
 
    print(f"[SYN-GRAF-NEWS] ========== Noticias comparación visual sintéticas ==========")
    print(f"[SYN-GRAF-NEWS] CSV entrada   : {csv_entrada} | existe={os.path.isfile(csv_entrada)}")
    print(f"[SYN-GRAF-NEWS] Fuseki        : {FUSEKI_ENDPOINT}")
    print(f"[SYN-GRAF-NEWS] Ventana días  : ±{VENTANA_DIAS}")
    print(f"[SYN-GRAF-NEWS] CSV salida    : {csv_salida}")
 
    if not os.path.isfile(csv_entrada):
        raise FileNotFoundError(f"[SYN-GRAF-NEWS] No encontrado: {csv_entrada}")
 
    def _sparql(query: str) -> list[dict]:
        resp = requests.post(
            FUSEKI_ENDPOINT,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["results"]["bindings"]
 
    def _categoria(uri: str) -> str:
        if not uri:
            return ""
        fragmento = uri.split("#")[-1].split("/")[-1]
        if fragmento.startswith("cat_"):
            fragmento = fragmento[4:]
        return fragmento
 
    def _fecha_valida(valor: str) -> bool:
        """Devuelve True si el valor es una fecha en formato YYYY-MM-DD válida."""
        if not valor or valor == "nan":
            return False
        try:
            date.fromisoformat(valor[:10])
            return True
        except ValueError:
            return False
 
    df_pares = pd.read_csv(csv_entrada)
 
    antes = len(df_pares)
    df_pares = df_pares[
        df_pares["synthetic_id"].astype(str) != df_pares["original_id"].astype(str)
    ].reset_index(drop=True)
    removidos = antes - len(df_pares)
    if removidos > 0:
        print(f"[SYN-GRAF-NEWS][WARN] {removidos} auto-comparaciones removidas")
 
    print(f"[SYN-GRAF-NEWS] Pares válidos : {len(df_pares)}")
 
    resultados = []
 
    for _, par in df_pares.iterrows():
        syn_id     = str(par["synthetic_id"])
        orig_id    = str(par["original_id"])
        sim_cosine = float(par["similarity_cosine"])
        rank       = int(par["rank_in_corpus"])
        syn_name   = str(par.get("synthetic_name", ""))
        orig_name  = str(par.get("original_name", ""))
 
        # ── Extraer fechas: columna explícita primero, respaldo desde nombre ──
        syn_start = str(par.get("synthetic_start_date", ""))[:10]
        if not _fecha_valida(syn_start):
            fechas = re.findall(r'\d{4}-\d{2}-\d{2}', syn_name)
            syn_start = fechas[0] if fechas else ""
 
        syn_end = str(par.get("synthetic_end_date", ""))[:10]
        if not _fecha_valida(syn_end):
            syn_end = syn_start
 
        orig_start = str(par.get("original_start_date", ""))[:10]
        if not _fecha_valida(orig_start):
            fechas = re.findall(r'\d{4}-\d{2}-\d{2}', orig_name)
            orig_start = fechas[0] if fechas else ""
 
        orig_end = str(par.get("original_end_date", ""))[:10]
        if not _fecha_valida(orig_end):
            orig_end = orig_start
 
        print(f"[SYN-GRAF-NEWS] Par {syn_id} ↔ {orig_id} | "
              f"syn={syn_start} orig={orig_start} | "
              f"cosine={sim_cosine:.4f} rank={rank}")
 
        if not orig_start:
            print(f"[SYN-GRAF-NEWS][WARN] Sin fecha para original {orig_id} — saltando")
            continue
 
        d_ini = (date.fromisoformat(orig_start) - timedelta(days=VENTANA_DIAS)).isoformat()
        d_fin = (date.fromisoformat(orig_end)   + timedelta(days=VENTANA_DIAS)).isoformat()
 
        print(f"[SYN-GRAF-NEWS]   ventana historica: {d_ini} → {d_fin}")
 
        query_noticias = f"""
            PREFIX anom: <{ANOM_PREFIX}>
            PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
            SELECT ?title ?text ?source ?fecha ?categoria
            WHERE {{
                ?n a anom:NewsArticle ;
                   anom:newsTitle   ?title ;
                   anom:newsText    ?text ;
                   anom:newsSource  ?source ;
                   anom:publishedAt ?fecha .
                OPTIONAL {{ ?n anom:hasCategory ?categoria }}
                FILTER(?fecha >= "{d_ini}"^^xsd:date &&
                       ?fecha <= "{d_fin}"^^xsd:date)
            }}
            ORDER BY ?fecha
        """
 
        try:
            noticias = _sparql(query_noticias)
        except Exception as exc:
            print(f"[SYN-GRAF-NEWS][ERROR] Fuseki par {syn_id} ↔ {orig_id}: {exc}")
            noticias = []
 
        print(f"[SYN-GRAF-NEWS]   → {len(noticias)} noticias para anomalía histórica {orig_id}")
 
        if not noticias:
            resultados.append({
                "anomalia_sintetica_id":  syn_id,
                "sintetica_fecha_inicio": syn_start,
                "sintetica_fecha_fin":    syn_end,
                "sintetica_nombre":       syn_name,
                "anomalia_historica_id":  orig_id,
                "historica_fecha_inicio": orig_start,
                "historica_fecha_fin":    orig_end,
                "historica_nombre":       orig_name,
                "similitud_cosine":       round(sim_cosine, 6),
                "rank_visual":            rank,
                "titulo_noticia":         None,
                "texto_noticia":          None,
                "fuente_noticia":         None,
                "fecha_noticia":          None,
                "categoria_noticia":      None,
                "motivo":                 "sin_noticias",
            })
            continue
 
        for noticia in noticias:
            resultados.append({
                "anomalia_sintetica_id":  syn_id,
                "sintetica_fecha_inicio": syn_start,
                "sintetica_fecha_fin":    syn_end,
                "sintetica_nombre":       syn_name,
                "anomalia_historica_id":  orig_id,
                "historica_fecha_inicio": orig_start,
                "historica_fecha_fin":    orig_end,
                "historica_nombre":       orig_name,
                "similitud_cosine":       round(sim_cosine, 6),
                "rank_visual":            rank,
                "titulo_noticia":         noticia["title"]["value"],
                "texto_noticia":          noticia["text"]["value"],
                "fuente_noticia":         noticia["source"]["value"],
                "fecha_noticia":          noticia["fecha"]["value"][:10],
                "categoria_noticia":      _categoria(
                    noticia.get("categoria", {}).get("value", "")
                ),
                "motivo":                 "ok",
            })
 
    df_out = pd.DataFrame(resultados)
    df_out.to_csv(csv_salida, index=False)
 
    print(f"[SYN-GRAF-NEWS] ========== Resultado ==========")
    print(f"[SYN-GRAF-NEWS] CSV guardado     : {csv_salida}")
    print(f"[SYN-GRAF-NEWS] Total filas      : {len(df_out)}")
 
    if not df_out.empty:
        ok  = df_out[df_out["motivo"] == "ok"]
        sin = df_out[df_out["motivo"] != "ok"]
        print(f"[SYN-GRAF-NEWS] Con noticias     : {len(ok)}")
        print(f"[SYN-GRAF-NEWS] Sin noticias     : {len(sin)}")
        print(f"[SYN-GRAF-NEWS] Sintéticas únicas: {df_out['anomalia_sintetica_id'].nunique()}")
        print(f"[SYN-GRAF-NEWS] Históricas únicas: {df_out['anomalia_historica_id'].nunique()}")
        if not ok.empty:
            cats = sorted(ok['categoria_noticia'].dropna().unique().tolist())
            print(f"[SYN-GRAF-NEWS] Categorías       : {cats}")
 
    return csv_salida
def comparar_noticias_por_pares_dtw_hist(CFG) -> str:
    """
    Lee los pares de anomalías similares numéricamente (DTW histórico vs reciente),
    toma el top N por anomalía reciente, consulta Fuseki por las noticias de cada
    período, calcula similitud semántica y guarda los resultados en un CSV.

    Parámetros esperados en CFG:
        CFG.RUTA_COMPARACION_HIST  → carpeta donde están los archivos de entrada y salida
        CFG.DTW_TOP_N_NOTICIAS     → (opcional) cuántos top pares por anomalía reciente (default 10)

    Archivos de entrada:
        similaridades_dtw.csv
            id_anomalia_sintetica : ID de la anomalía reciente
            id_anomalia_original  : ID de la anomalía histórica
            rank                  : ranking de similitud por anomalía reciente
            start_date_sintetica  : fecha de inicio de la anomalía reciente
            start_date_original   : fecha de inicio de la anomalía histórica

    Archivo de salida:
        similitudes_noticias_dtw_hist.csv
    """
    out_root    = CFG.RUTA_COMPARACION_HIST
    endpoint    = getattr(CFG, "FUSEKI_ENDPOINT", FUSEKI_ENDPOINT)
    top_n       = int(getattr(CFG, "DTW_TOP_N_NOTICIAS", 10))

    csv_entrada = os.path.join(out_root, "similaridades_dtw.csv")
    csv_salida  = os.path.join(out_root, "similitudes_noticias_dtw_hist.csv")

    if not os.path.isfile(csv_entrada):
        raise FileNotFoundError(f"[NEWS-DTW-HIST] Archivo no encontrado: {csv_entrada}")

    df_pares = pd.read_csv(csv_entrada)

    # ── Filtro: top N por anomalía reciente ──────────────────────────────────
    antes = len(df_pares)
    df_pares = df_pares[df_pares["rank"] <= top_n].reset_index(drop=True)
    print(f"[NEWS-DTW-HIST] Total pares disponibles : {antes}")
    print(f"[NEWS-DTW-HIST] Pares top {top_n} seleccionados : {len(df_pares)}")

    # ── Filtro defensivo: descartar auto-comparaciones ───────────────────────
    antes = len(df_pares)
    df_pares = df_pares[
        df_pares["id_anomalia_sintetica"].astype(str) != df_pares["id_anomalia_original"].astype(str)
    ].reset_index(drop=True)
    removidos = antes - len(df_pares)
    if removidos > 0:
        print(f"[NEWS-DTW-HIST][WARN] {removidos} auto-comparaciones removidas")
    print(f"[NEWS-DTW-HIST] Pares válidos a procesar: {len(df_pares)}")

    # ── Cargar modelo de texto ───────────────────────────────────────────────
    print(f"[NEWS-DTW-HIST] Cargando modelo: {TEXT_MODEL_NAME}")
    model = SentenceTransformer(TEXT_MODEL_NAME)

    resultados = []

    for _, par in df_pares.iterrows():
        id_reciente     = int(par["id_anomalia_sintetica"])
        id_historica    = int(par["id_anomalia_original"])
        fecha_reciente  = str(par["start_date_sintetica"])[:10]
        fecha_historica = str(par["start_date_original"])[:10]
        sim_combined    = float(par["similarity_combined"])
        rank            = int(par["rank"])

        print(f"[NEWS-DTW-HIST] Par reciente={id_reciente} ({fecha_reciente}) ↔ "
              f"historica={id_historica} ({fecha_historica}) | "
              f"sim_combined={sim_combined:.4f} rank={rank}")

        if not fecha_reciente or not fecha_historica:
            print(f"[NEWS-DTW-HIST][WARN] Sin fecha para par {id_reciente} ↔ {id_historica} — saltando")
            continue

        if fecha_reciente == fecha_historica:
            print(f"[NEWS-DTW-HIST][WARN] Par {id_reciente} ↔ {id_historica} fechas idénticas — saltando")
            continue

        inicio_r, fin_r = _ventana(fecha_reciente)
        inicio_h, fin_h = _ventana(fecha_historica)

        # ── Consultar Fuseki ─────────────────────────────────────────────────
        try:
            noticias_reciente  = _query_noticias(inicio_r, fin_r)
            noticias_historica = _query_noticias(inicio_h, fin_h)
        except Exception as exc:
            log.error("[NEWS-DTW-HIST] Error Fuseki par %d ↔ %d: %s", id_reciente, id_historica, exc)
            print(f"[NEWS-DTW-HIST][ERROR] Fuseki par {id_reciente} ↔ {id_historica}: {exc}")
            continue

        print(f"[NEWS-DTW-HIST]   noticias reciente={len(noticias_reciente)} "
              f"historica={len(noticias_historica)}")

        # ── Calcular similitud semántica ─────────────────────────────────────
        emb_r = _embedding_centroide(noticias_reciente,  model)
        emb_h = _embedding_centroide(noticias_historica, model)

        cats_reciente  = sorted(set(n.get("categoria", "") for n in noticias_reciente  if n.get("categoria", "")))
        cats_historica = sorted(set(n.get("categoria", "") for n in noticias_historica if n.get("categoria", "")))

        if emb_r is None and emb_h is None:
            sim_noticias = None
            motivo = "sin_noticias_ambos"
        elif emb_r is None:
            sim_noticias = None
            motivo = "sin_noticias_reciente"
        elif emb_h is None:
            sim_noticias = None
            motivo = "sin_noticias_historica"
        else:
            sim_noticias = float(cosine_similarity([emb_r], [emb_h])[0][0])
            motivo = "ok"

        print(f"[NEWS-DTW-HIST]   similitud_noticias={sim_noticias}  motivo={motivo}")

        resultados.append({
            "id_reciente":            id_reciente,
            "id_historica":           id_historica,
            "fecha_inicio_reciente":  fecha_reciente,
            "fecha_inicio_historica": fecha_historica,
            "similitud_combined":     sim_combined,
            "rank":                   rank,
            "similitud_noticias":     sim_noticias,
            "motivo":                 motivo,
            "categorias_reciente":    ", ".join(cats_reciente),
            "categorias_historica":   ", ".join(cats_historica),
            "n_noticias_reciente":    len(noticias_reciente),
            "n_noticias_historica":   len(noticias_historica),
            "noticias_reciente":      str(noticias_reciente),
            "noticias_historica":     str(noticias_historica),
        })

    # ── Guardar CSV ──────────────────────────────────────────────────────────
    df_out = pd.DataFrame(resultados)
    df_out.to_csv(csv_salida, index=False)

    print(f"[NEWS-DTW-HIST] CSV guardado: {csv_salida}")
    print(f"[NEWS-DTW-HIST] Total pares procesados: {len(df_out)}")

    if not df_out.empty:
        ok           = df_out[df_out["motivo"] == "ok"]
        sin_noticias = df_out[df_out["motivo"] != "ok"]
        print(f"[NEWS-DTW-HIST] Con similitud calculada  : {len(ok)}")
        print(f"[NEWS-DTW-HIST] Sin noticias suficientes : {len(sin_noticias)}")
        if not sin_noticias.empty:
            print(f"[NEWS-DTW-HIST] Motivos:\n{sin_noticias['motivo'].value_counts().to_string()}")
        if not ok.empty:
            print(f"[NEWS-DTW-HIST] sim_noticias máx : {ok['similitud_noticias'].max():.4f}")
            print(f"[NEWS-DTW-HIST] sim_noticias mín : {ok['similitud_noticias'].min():.4f}")
            print(f"[NEWS-DTW-HIST] sim_noticias prom: {ok['similitud_noticias'].mean():.4f}")

    return csv_salida    