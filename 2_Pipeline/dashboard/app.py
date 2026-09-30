import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import os
import glob
import plotly.express as px

# ─────────────────────────────────────────────
# FUSEKI — consulta noticias por rango de fechas
# ─────────────────────────────────────────────
FUSEKI_URL = "http://localhost:3030/c22/sparql"

@st.cache_data(ttl=300)
def consultar_noticiasv(fecha_ini: str, fecha_fin: str) -> pd.DataFrame:
    """Consulta noticias en Fuseki para un rango ±5 días. Retorna DataFrame vacío si falla."""
    try:
        from datetime import datetime, timedelta
        import requests
        d1 = (datetime.strptime(fecha_ini[:10], "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")
        d2 = (datetime.strptime(fecha_fin[:10], "%Y-%m-%d") + timedelta(days=5)).strftime("%Y-%m-%d")
        query = f"""
PREFIX anom: <http://w3id.org/anomaly-core#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
SELECT ?titulo ?fecha ?url ?categoria
WHERE {{
    ?n a anom:NewsArticle ;
       anom:newsTitle   ?titulo ;
       anom:publishedAt ?fecha .
    OPTIONAL {{ ?n anom:newsUrl ?url }}
    OPTIONAL {{ ?n anom:hasCategory ?cat .
               ?cat anom:categoryName ?categoria }}
    FILTER(?fecha >= "{d1}"^^xsd:date && ?fecha <= "{d2}"^^xsd:date)
}}
ORDER BY ?fecha
"""
        resp = requests.post(
            FUSEKI_URL,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=5
        )
        if resp.status_code != 200:
            return pd.DataFrame()
        bindings = resp.json().get("results", {}).get("bindings", [])
        rows = []
        for b in bindings:
            rows.append({
                "Fecha":     b.get("fecha",    {}).get("value","")[:10],
                "Título":    b.get("titulo",   {}).get("value",""),
                "Categoría": b.get("categoria",{}).get("value","").replace("_"," ").capitalize(),
                "URL":       b.get("url",      {}).get("value",""),
            })
        return pd.DataFrame(rows).drop_duplicates(subset=["Fecha","Título"])
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def consultar_noticias(fecha_ini: str, fecha_fin: str) -> pd.DataFrame:
    """Consulta noticias en Fuseki para un rango ±5 días. Retorna DataFrame vacío si falla."""
    try:
        from datetime import datetime, timedelta
        import requests
        d1 = (datetime.strptime(fecha_ini[:10], "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")
        d2 = (datetime.strptime(fecha_fin[:10], "%Y-%m-%d") + timedelta(days=5)).strftime("%Y-%m-%d")
        query = f"""
PREFIX anom: <http://w3id.org/anomaly-core#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
SELECT ?titulo ?fecha ?url ?cat
WHERE {{
    ?n a anom:NewsArticle ;
       anom:newsTitle   ?titulo ;
       anom:publishedAt ?fecha .
    OPTIONAL {{ ?n anom:newsUrl ?url }}
    OPTIONAL {{ ?n anom:hasCategory ?cat }}
    FILTER(?fecha >= "{d1}"^^xsd:date && ?fecha <= "{d2}"^^xsd:date)
}}
ORDER BY ?fecha
"""
        resp = requests.post(
            FUSEKI_URL,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=5
        )
        if resp.status_code != 200:
            return pd.DataFrame()
        bindings = resp.json().get("results", {}).get("bindings", [])
        rows = []
        for b in bindings:
            cat_uri = b.get("cat", {}).get("value", "")
            cat_slug = cat_uri.split("#")[-1].replace("cat_", "") if cat_uri else ""
            rows.append({
                "Fecha":     b.get("fecha",  {}).get("value","")[:10],
                "Título":    b.get("titulo", {}).get("value",""),
                "Categoría": CATS_LEGIBLES.get(cat_slug, cat_slug.replace("_"," ").capitalize()) if cat_slug else "",
                "URL":       b.get("url",    {}).get("value",""),
            })
        return pd.DataFrame(rows).drop_duplicates(subset=["Fecha","Título"])
    except Exception:
        return pd.DataFrame()

st.set_page_config(
    page_title="Riesgo País Ecuador",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────
BASE              = "/home/jacky/DatosSerie/Resultados"
RUTA_SERIE        = "/home/jacky/DatosSerie/SerieOriginal.csv"
RUTA_METODOS      = f"{BASE}/DetectadoXMetodo/Original"
RUTA_CONSOLIDADO  = f"{BASE}/anomalias_consolidado.csv"
RUTA_PREDICCION   = f"{BASE}/Prediccion"
RUTA_HIST_EMB     = f"{BASE}/ComparacionesHist/similaridades_embeddings_hist.csv"
RUTA_HIST_NEWS    = f"{BASE}/ComparacionesHist/similitudes_noticias_hist.csv"
RUTA_LAMINAS_HIST = f"{BASE}/ComparacionesHist/laminas/laminas_per_historica"
RUTA_LAMINAS_SINT = f"{BASE}/ComparacionesEmbeddings/laminas_per_sintetica"
RUTA_EMB_SINT     = f"{BASE}/ComparacionesEmbeddings/similaridades_embeddings.csv"
RUTA_DTW_SINT     = f"{BASE}/ComparacionesEmbeddings/similaridades_dtw.csv"
RUTA_DTW_HIST     = f"{BASE}/ComparacionesHist/similaridades_dtw.csv"
RUTA_NEWS_DTW_HIST = f"{BASE}/ComparacionesHist/similitudes_noticias_dtw_hist.csv"


# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────
PARAMS_DETECCION = {
    "ARIMA":    {"Orden": "(2,1,1)", "Umbral": "percentile", "Percentil": "96.0"},
    "COUTA":    {"Delta": "50", "Lambda": "4000"},
    "DBSCAN":   {"Variable": "zresid", "Ventana de tendencia": "45", "Eps": "0.01", "Muestras mínimas": "3", "Fracción retenida": "0.75"},
    "DIF":      {"Umbral": "percentile", "Percentil": "96.0"},
    "IFOREST":  {"Contaminación": "0.005", "N.º estimadores": "300", "Muestras máx.": "auto", "Semilla aleatoria": "42"},
    "TIMEGPT":  {"Ventana": "12", "Ventana de tendencia": "11", "Percentil": "97.5", "Suavizado": "3"},
    "TIMESNET": {"Ventana": "48", "Épocas": "8", "Tamaño oculto": "32", "Percentil": "97.5", "Suavizado": "3"},
    "TRANAD":   {"Ventana": "24", "Épocas": "10", "Tamaño oculto": "32", "Percentil": "97.5", "Fracción retenida": "1.0"},
}
COLORES_DETECCION = {
    "ARIMA": "#378ADD", "COUTA": "#D85A30", "DBSCAN": "#1D9E75",
    "DIF": "#7F77DD", "IFOREST": "#BA7517", "TIMEGPT": "#D4537E",
    "TIMESNET": "#639922", "TRANAD": "#888780",
}
MODELOS_PREDICCION = [
    "ARIMA_DIRECT", "ARIMA_FEEDBACK", "SVR_DIRECT", "SVR_FEEDBACK",
    "RF_DIRECT", "RF_FEEDBACK", "PROPHET_DIRECT", "PROPHET_FEEDBACK",
    "LSTM_DIRECT", "LSTM_FEEDBACK", "NBEATS_DIRECT", "NBEATS_FEEDBACK",
    "DEEPAR_DIRECT", "DEEPAR_FEEDBACK",
]
METODOS_DETECCION = list(PARAMS_DETECCION.keys())

# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────
@st.cache_data
def cargar_serie():
    df = pd.read_csv(RUTA_SERIE, parse_dates=["fecha"])
    return df.sort_values("fecha").reset_index(drop=True)

@st.cache_data
def cargar_puntos_original(metodo):
    path = f"{RUTA_METODOS}/{metodo}_puntos.csv"
    if not os.path.exists(path): return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["fecha"]).sort_values("fecha")

@st.cache_data
def cargar_consolidado():
    return pd.read_csv(RUTA_CONSOLIDADO, parse_dates=["start_date","end_date"]).sort_values("start_date")

@st.cache_data
def cargar_serie_sintetica(modelo):
    path = f"{RUTA_PREDICCION}/{modelo}/1_{modelo}_sintetica.csv"
    if not os.path.exists(path): return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["fecha"]).sort_values("fecha")

@st.cache_data
def cargar_puntos_sinteticos(modelo, metodo):
    path = f"{RUTA_PREDICCION}/{modelo}/{metodo}_puntos.csv"
    if not os.path.exists(path): return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["fecha"]).sort_values("fecha")

@st.cache_data
def cargar_hist_emb():
    if not os.path.exists(RUTA_HIST_EMB): return pd.DataFrame()
    return pd.read_csv(RUTA_HIST_EMB)

@st.cache_data
def cargar_hist_news():
    if not os.path.exists(RUTA_HIST_NEWS): return pd.DataFrame()
    return pd.read_csv(RUTA_HIST_NEWS)

@st.cache_data
def cargar_emb_sint():
    if not os.path.exists(RUTA_EMB_SINT): return pd.DataFrame()
    return pd.read_csv(RUTA_EMB_SINT)

@st.cache_data
def cargar_dtw_sint():
    if not os.path.exists(RUTA_DTW_SINT): return pd.DataFrame()
    return pd.read_csv(RUTA_DTW_SINT)

@st.cache_data
def cargar_dtw_hist():
    if not os.path.exists(RUTA_DTW_HIST): return pd.DataFrame()
    return pd.read_csv(RUTA_DTW_HIST)

@st.cache_data
def cargar_news_dtw_hist():
    if not os.path.exists(RUTA_NEWS_DTW_HIST): return pd.DataFrame()
    return pd.read_csv(RUTA_NEWS_DTW_HIST)    

def fig_layout(fig, height=280):
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="bps", gridcolor="rgba(0,0,0,0.05)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig

# ─────────────────────────────────────────────
# CARGA INICIAL
# ─────────────────────────────────────────────
serie       = cargar_serie()
consolidado = cargar_consolidado()
df_emb_hist = cargar_hist_emb()
df_news     = cargar_hist_news()
df_emb_sint = cargar_emb_sint()
df_dtw_sint = cargar_dtw_sint()
df_dtw_hist = cargar_dtw_hist()
df_news_dtw = cargar_news_dtw_hist()

# ══════════════════════════════════════════════
# BANNER SUPERIOR — Serie original completa
# ══════════════════════════════════════════════
tab1, tab2 = st.tabs(["📈 Análisis de anomalías", "🔗 Ontología"])
with tab1:
    col_banner, col_desc = st.columns([3, 1], gap="large")

    with col_banner:
        fig_banner = go.Figure()
        fig_banner.add_trace(go.Scatter(
            x=serie["fecha"], y=serie["valor"],
            mode="lines", name="Riesgo País",
            line=dict(color="#378ADD", width=1),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y} bps<extra></extra>",
            fill="tozeroy", fillcolor="rgba(55,138,221,0.08)"
        ))
        fig_banner.update_layout(
            height=200, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(title="bps", gridcolor="rgba(0,0,0,0.05)"),
            showlegend=False, hovermode="x unified",
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.markdown("### Riesgo País Ecuador (2004–2026)")
        st.plotly_chart(fig_banner, use_container_width=True)

    with col_desc:
        st.markdown("#### ¿Qué es el Riesgo País?")
        st.markdown("""
    El Riesgo País corresponde al indicador publicado diariamente por el Banco Central del Ecuador
     que mide la percepción de riesgo de inversión del país, expresado en puntos básicos 
     (\textit{basis points}, bps). La serie histórica utilizada en esta investigación comprende 
     7,854 observaciones registradas entre los años 2004 y 2026. 
     
     La vista inicial de la interfaz presenta una representación gráfica de la trayectoria temporal
      completa de la serie, acompañada de indicadores generales que resumen las principales
       características del proceso experimental y proporcionan el contexto necesario para la 
       exploración de las etapas posteriores del \textit{framework}.
    """)
        m1, m2 = st.columns(2)
        m1.metric("Puntos totales", f"{len(serie):,}")
        m2.metric("Anomalías", len(consolidado))
        m3, m4 = st.columns(2)
        m3.metric("Métodos detección", len(METODOS_DETECCION))
        m4.metric("Modelos predicción", len(MODELOS_PREDICCION))

    st.divider()

    # ══════════════════════════════════════════════
    # SECCIÓN 1 — Detección sobre serie original
    # ══════════════════════════════════════════════
    st.markdown("### Detección de anomalías — Serie histórica")

    s1_izq, s1_der = st.columns([1, 4], gap="large")

    with s1_izq:
        metodo_sel = st.selectbox(
            "Método de detección",
            options=["— ninguno —"] + METODOS_DETECCION,
            key="metodo_orig"
        )
        mostrar_cons = st.checkbox("Mostrar consolidadas", value=False)

        st.markdown("---")

        if metodo_sel != "— ninguno —":
            st.markdown(f"**Parámetros — {metodo_sel}**")
            df_p = pd.DataFrame(
                list(PARAMS_DETECCION[metodo_sel].items()),
                columns=["Parámetro", "Valor"]
            )
            st.dataframe(df_p, hide_index=True, use_container_width=True)

            pts = cargar_puntos_original(metodo_sel)
            if not pts.empty:
                st.markdown("**Estadísticas**")
                df_s = pd.DataFrame([
                    ("Puntos detectados", len(pts)),
                    ("Mínimo (bps)",      int(pts["valor"].min())),
                    ("Máximo (bps)",      int(pts["valor"].max())),
                    ("Promedio (bps)",    int(pts["valor"].mean())),
                    ("Primera detección", pts["fecha"].min().strftime("%Y-%m-%d")),
                    ("Última detección",  pts["fecha"].max().strftime("%Y-%m-%d")),
                ], columns=["Métrica", "Valor"])
                st.dataframe(df_s, hide_index=True, use_container_width=True)
        else:
            st.caption("Selecciona un método para ver sus parámetros y estadísticas.")

    with s1_der:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=serie["fecha"], y=serie["valor"],
            mode="lines", name="Riesgo País",
            line=dict(color="#93C5FD", width=1),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y} bps<extra></extra>"
        ))
        if metodo_sel != "— ninguno —":
            pts = cargar_puntos_original(metodo_sel)
            if not pts.empty:
                fig1.add_trace(go.Scatter(
                    x=pts["fecha"], y=pts["valor"],
                    mode="markers", name=metodo_sel,
                    marker=dict(color="#2563EB", size=5),
                    hovertemplate=f"{metodo_sel}<br>%{{x|%Y-%m-%d}}<br>%{{y}} bps<extra></extra>"
                ))
        if mostrar_cons:
            cons_v = consolidado.merge(
                serie.rename(columns={"fecha": "start_date"})[["start_date", "valor"]],
                on="start_date", how="left"
            )
            fig1.add_trace(go.Scatter(
                x=consolidado["start_date"], y=cons_v["valor"],
                mode="markers", name="Consolidadas",
                marker=dict(color="#DC2626", size=8, symbol="triangle-up"),
                hovertemplate="Consolidada<br>%{x|%Y-%m-%d}<extra></extra>"
            ))
        fig1.update_layout(
            height=380, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(title="bps", gridcolor="rgba(0,0,0,0.05)"),
            legend=dict(
                orientation="h", yanchor="top", y=-0.12,
                xanchor="left", x=0, font=dict(size=11)
            ),
            hovermode="x unified",
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    # ══════════════════════════════════════════════
    # SECCIÓN 2 — Predicción + detección
    # ══════════════════════════════════════════════
    st.markdown("### Exploración de escenarios sintéticos")

    s2c1, s2c2 = st.columns([1, 3], gap="large")

    with s2c1:
        modelo_pred = st.selectbox("Modelo de predicción", MODELOS_PREDICCION, key="modelo_pred")
        metodo_pred = st.selectbox("Método de detección", ["— ninguno —"] + METODOS_DETECCION, key="metodo_pred")

        if metodo_pred != "— ninguno —":
            st.markdown(f"**Parámetros — {metodo_pred}**")
            df_pp = pd.DataFrame(list(PARAMS_DETECCION[metodo_pred].items()), columns=["Parámetro", "Valor"])
            st.dataframe(df_pp, hide_index=True, use_container_width=True)

            pts_sint = cargar_puntos_sinteticos(modelo_pred, metodo_pred)
            if not pts_sint.empty:
                st.markdown("**Estadísticas**")
                df_ss = pd.DataFrame([
                    ("Puntos detectados", len(pts_sint)),
                    ("Mínimo (bps)", round(pts_sint["valor"].min(),1)),
                    ("Máximo (bps)", round(pts_sint["valor"].max(),1)),
                    ("Promedio (bps)", round(pts_sint["valor"].mean(),1)),
                    ("Primera detección", pts_sint["fecha"].min().strftime("%Y-%m-%d")),
                    ("Última detección",  pts_sint["fecha"].max().strftime("%Y-%m-%d")),
                ], columns=["Métrica", "Valor"])
                st.dataframe(df_ss, hide_index=True, use_container_width=True)
            else:
                st.info(f"Sin anomalías con {metodo_pred} sobre {modelo_pred}")

    with s2c2:
        serie_sint = cargar_serie_sintetica(modelo_pred)
        fig2 = go.Figure()
        if not serie_sint.empty:
            fig2.add_trace(go.Scatter(
                x=serie_sint["fecha"], y=serie_sint["valor"],
                mode="lines", name=modelo_pred,
                line=dict(color="#1D9E75", width=1.5),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y} bps<extra></extra>"
            ))
            if metodo_pred != "— ninguno —":
                pts_sint = cargar_puntos_sinteticos(modelo_pred, metodo_pred)
                if not pts_sint.empty:
                    fig2.add_trace(go.Scatter(
                        x=pts_sint["fecha"], y=pts_sint["valor"],
                        mode="markers", name=metodo_pred,
                        marker=dict(color=COLORES_DETECCION.get(metodo_pred,"#E24B4A"), size=7),
                        hovertemplate=f"{metodo_pred}<br>%{{x|%Y-%m-%d}}<br>%{{y}} bps<extra></extra>"
                    ))
        else:
            fig2.add_annotation(text="Serie sintética no encontrada", showarrow=False, font=dict(size=13))
        st.plotly_chart(fig_layout(fig2, 320), use_container_width=True)

    st.divider()

    # ══════════════════════════════════════════════
    # SECCIÓN 3 — Comparación original vs sintética
    # ══════════════════════════════════════════════


    st.markdown("### Comparación anomalías sintéticas vs historicas")

    if df_dtw_sint.empty:
        st.info("No se encontraron datos de similitud DTW.")
    else:
        df_dtw_sint["modelo_pred"] = df_dtw_sint["anomaly_iri_sintetica"].str.extract(r"__PM_([A-Z_]+?)_\d{4}")

        # ── Selector de modelo: etiqueta a la izquierda ──────────────
        lbl1, sel1 = st.columns([1, 3])
        with lbl1:
            st.markdown(
                '<div style="padding-top:0.6rem; font-size:0.75rem; font-weight:600; '
                'color:#6B7280; text-transform:uppercase; letter-spacing:0.03em;">'
                'Modelo de predicción</div>',
                unsafe_allow_html=True
            )
        with sel1:
            modelo_s3 = st.selectbox("Modelo de predicción", options=MODELOS_PREDICCION,
                                      key="modelo_s3", label_visibility="collapsed")

        dtw_modelo = df_dtw_sint[df_dtw_sint["modelo_pred"] == modelo_s3].copy()

        sint_con_lamina = set()
        if not df_emb_sint.empty:
            emb_modelo = df_emb_sint[df_emb_sint["synthetic_id"].str.contains(modelo_s3, case=False, na=False)]
            for sid_name in emb_modelo["synthetic_id"].unique():
                id_num = int(sid_name.split("_")[0])
                sint_con_lamina.add(id_num)

        sint_ids = sorted([x for x in dtw_modelo["id_anomalia_sintetica"].unique().tolist()
                        if x in sint_con_lamina])

        # ── Selector de anomalía sintética: etiqueta a la izquierda ──
        lbl2, sel2 = st.columns([1, 3])
        with lbl2:
            st.markdown(
                '<div style="padding-top:0.6rem; font-size:0.75rem; font-weight:600; '
                'color:#6B7280; text-transform:uppercase; letter-spacing:0.03em;">'
                'Anomalía sintética</div>',
                unsafe_allow_html=True
            )
        with sel2:
            if not sint_ids:
                st.info("Sin anomalías con lámina para este modelo.")
                sint_id_sel = None
            else:
                sint_id_sel = st.selectbox(
                    "Anomalía sintética",
                    options=sint_ids,
                    format_func=lambda x: (
                        f"#{x} — {dtw_modelo[dtw_modelo['id_anomalia_sintetica']==x]['method_sintetica'].iloc[0]}"
                        f" ({dtw_modelo[dtw_modelo['id_anomalia_sintetica']==x]['start_date_sintetica'].iloc[0]})"
                    ),
                    key="sint_id_sel",
                    label_visibility="collapsed"
                )

        if sint_id_sel is None:
            st.stop()

        fila_sint = dtw_modelo[dtw_modelo["id_anomalia_sintetica"] == sint_id_sel].iloc[0]

        # ── Preparar ruta de la lámina ────────────────────────────────
        lam_path = None
        synthetic_id_exacto = None
        id_str = str(sint_id_sel).zfill(3)

        emb_fila = pd.DataFrame()
        if not df_emb_sint.empty:
            emb_fila = df_emb_sint[
                df_emb_sint["synthetic_id"].str.startswith(id_str) &
                df_emb_sint["synthetic_id"].str.contains(modelo_s3, case=False, na=False)
            ]

        if not emb_fila.empty:
            synthetic_id_exacto = emb_fila.iloc[0]["synthetic_id"]
            nombre_png = f"{synthetic_id_exacto}__top2.png"
            candidato  = os.path.join(RUTA_LAMINAS_SINT, nombre_png)
            if os.path.exists(candidato):
                lam_path = candidato

        # ── Fila final: izquierda = imagen + CLIP | derecha = slider + DTW ──
        f3_izq, f3_der = st.columns(2, gap="large")

        with f3_izq:
            st.markdown(
                '<div style="border-left:4px solid #378ADD; padding-left:10px; '
                'font-size:1.05rem; font-weight:700; margin-bottom:0.5rem;">'
                'Similitud visual</div>',
                unsafe_allow_html=True
            )
            tabla_emb = None
            if not df_emb_sint.empty:
                pares_emb = df_emb_sint[df_emb_sint["synthetic_id"].str.startswith(id_str)].copy()
                if not pares_emb.empty:
                    pares_emb["orig_id_num"] = pares_emb["original_id"].str.split("_").str[0].astype(int)
                    pares_emb = pares_emb.merge(
                        consolidado[["id", "method"]],
                        left_on="orig_id_num", right_on="id", how="left"
                    )
                    # Orden explícito por similitud desc — mismo orden usado al generar la lámina
                    pares_emb = pares_emb.sort_values("similarity_cosine", ascending=False)

                    tabla_emb = pares_emb[["orig_id_num", "original_start_date", "original_end_date",
                        "similarity_pct", "method"]].copy()
                    tabla_emb["original_start_date"] = pd.to_datetime(tabla_emb["original_start_date"]).dt.strftime("%Y-%m-%d")
                    tabla_emb["original_end_date"]   = pd.to_datetime(tabla_emb["original_end_date"]).dt.strftime("%Y-%m-%d")
                    tabla_emb.columns = ["ID", "Fecha Inicio", "Fecha Fin", "Similitud", "Métodos detección"]
                    st.dataframe(tabla_emb, hide_index=True, use_container_width=True)
                else:
                    st.info("Sin datos de similitud CLIP para esta anomalía.")
            else:
                st.info("Sin datos de similitud CLIP disponibles.")

            if lam_path:
                # ── Recorte de la banda de texto ilegible ────────────
                img_original = Image.open(lam_path)
                thumb_width_px = 480  # debe coincidir con CMP_THUMB_WIDTH_PX en settings.py
                img_w = int(thumb_width_px * 2.5)
                img_h_no_label = int(round(img_w * 0.75))

                img_cortada = img_original.crop((0, 0, img_original.width, img_h_no_label))
                st.image(img_cortada, use_container_width=True)

                # ── IDs alineados debajo de cada gráfica ─────────────
                id_sintetica = str(sint_id_sel)
                id_col1 = str(tabla_emb.iloc[0]["ID"]) if tabla_emb is not None and len(tabla_emb) > 0 else "—"
                id_col2 = str(tabla_emb.iloc[1]["ID"]) if tabla_emb is not None and len(tabla_emb) > 1 else "—"

                c1, c2, c3 = st.columns(3)
                c1.caption(f"Sintética · id={id_sintetica}")
                c2.caption(f"Histórica · id={id_col1}")
                c3.caption(f"Histórica · id={id_col2}")
            else:
                st.info("Lámina no disponible para esta anomalía.")

        with f3_der:
            st.markdown(
                '<div style="border-left:4px solid #1D9E75; padding-left:10px; '
                'font-size:1.05rem; font-weight:700; margin-bottom:0.5rem;">'
                'Similitud Numérica</div>',
                unsafe_allow_html=True
            )
            st.caption("Compara ventanas de contexto de ±10 puntos alrededor de cada anomalía, no solo su rango exacto de fechas.")

            tabla_placeholder = st.empty()

            umbral_dtw = st.slider("Similitud mínima numérica", min_value=0.80, max_value=1.0,
                                    value=0.85, step=0.01, format="%.2f")

            dtw_anom = dtw_modelo[
                (dtw_modelo["id_anomalia_sintetica"] == sint_id_sel) &
                (dtw_modelo["similarity_combined"] >= umbral_dtw)
            ].sort_values("similarity_combined", ascending=False)

            with tabla_placeholder.container():
                if not dtw_anom.empty:
                    cols_show = ["id_anomalia_original","start_date_original","end_date_original",
                                "similarity_combined","method_original"]
                    cols_ok = [c for c in cols_show if c in dtw_anom.columns]
                    tabla = dtw_anom[cols_ok].head(8).copy()
                    tabla["similarity_combined"] = (tabla["similarity_combined"]*100).round(1).astype(str) + "%"
                    tabla.rename(columns={
                        "id_anomalia_original": "ID",
                        "start_date_original":  "Fecha Inicio",
                        "end_date_original":    "Fecha Fin",
                        "similarity_combined":  "Similitud",
                        "method_original":      "Métodos detección",
                        "len_vector_original":  "Puntos comparados",
                    }, inplace=True)
                    st.dataframe(tabla, hide_index=True, use_container_width=True)
                else:
                    st.info(f"Sin pares con similitud ≥ {umbral_dtw:.0%}")
    st.divider()

    # ══════════════════════════════════════════════
    # SECCIÓN 4 — Experimento histórico PRUEBA 
    # ══════════════════════════════════════════════
    st.markdown("### Experimento de comprobación — Anomalías recientes vs históricas")

    CATS_LEGIBLES = {
        "economia_finanzas": "Economía y finanzas",
        "politica_gobierno": "Política y gobierno",
        "seguridad_crimen": "Seguridad y crimen",
        "relaciones_internacionales": "Relaciones internacionales",
        "problemas_energeticos": "Problemas energéticos",
        "medio_ambiente": "Medio ambiente",
        "drogas_narcotrafico": "Drogas y narcotráfico",
        "riesgo_pais_calidad_vida": "Riesgo país y calidad de vida",
    }

    def cat_legible(cat):
        return CATS_LEGIBLES.get(cat.strip(), cat.strip().replace("_"," ").capitalize())

    if df_emb_hist.empty:
        st.info("No se encontraron datos de similitudes históricas.")
    else:
        recientes = sorted(df_emb_hist["synthetic_id"].unique().tolist())

        # ── Selector de anomalía reciente: etiqueta a la izquierda ────
        lbl_rec, sel_rec = st.columns([1, 3])
        with lbl_rec:
            st.markdown(
                '<div style="padding-top:0.6rem; font-size:0.75rem; font-weight:600; '
                'color:#6B7280; text-transform:uppercase; letter-spacing:0.03em;">'
                'Anomalía reciente</div>',
                unsafe_allow_html=True
            )
        with sel_rec:
            anom_sel = st.selectbox(
                "Anomalía reciente",
                options=recientes,
                format_func=lambda x: f"Anomalía #{x.split('_')[0]} · {x.split('_',1)[1]}",
                key="anom_hist",
                label_visibility="collapsed"
            )

        id_rec = int(anom_sel.split("_")[0])
        pares = df_emb_hist[df_emb_hist["synthetic_id"] == anom_sel].copy()

        id_rec_str = anom_sel.split("_")[0].zfill(3)
        cons_rec = consolidado[consolidado["id"] == id_rec]
        metodos_rec = cons_rec["method"].values[0] if not cons_rec.empty else "—"

        fecha_ini_rec_hdr = pares.iloc[0]["synthetic_start_date"] if not pares.empty else "—"
        fecha_fin_rec_hdr = pares.iloc[0]["synthetic_end_date"]   if not pares.empty else "—"

        # ── Tarjeta de la anomalía reciente (todo el ancho) ───────────
        f_vacio_rec, f_datos_rec = st.columns([1, 3])
        with f_datos_rec:
            c1, c2, c3 = st.columns(3)
            c1.metric("ID anomalía", f"#{id_rec}")
            c2.metric("Método detección", metodos_rec)
            if fecha_ini_rec_hdr == fecha_fin_rec_hdr:
                c3.metric("Fecha de la anomalía", fecha_ini_rec_hdr)
            else:
                c3.metric("Fecha de la anomalía", f"{fecha_ini_rec_hdr} – {fecha_fin_rec_hdr}")

        col_izq, col_der = st.columns(2, gap="large")

        with col_izq:
            st.markdown(
                '<div style="border-left:4px solid #378ADD; padding-left:10px; '
                'font-size:1.05rem; font-weight:700; margin-bottom:0.5rem;">'
                'Similitud Visual</div>',
                unsafe_allow_html=True
            )
            tabla_vis = None
            if pares.empty:
                st.info("Sin datos de similitud visual para esta anomalía.")
            else:
                pares_vis = pares.copy()
                pares_vis["orig_id_num"] = pares_vis["original_id"].str.split("_").str[0].astype(int)
                pares_vis = pares_vis.merge(
                    consolidado[["id", "method"]],
                    left_on="orig_id_num", right_on="id", how="left"
                )
                # Orden explícito por similitud desc — mismo orden usado al generar la lámina
                pares_vis = pares_vis.sort_values("similarity_cosine", ascending=False)

                tabla_vis = pares_vis[["orig_id_num", "original_start_date", "original_end_date",
                                        "similarity_pct", "method"]].copy()
                tabla_vis["original_start_date"] = pd.to_datetime(tabla_vis["original_start_date"]).dt.strftime("%Y-%m-%d")
                tabla_vis["original_end_date"]   = pd.to_datetime(tabla_vis["original_end_date"]).dt.strftime("%Y-%m-%d")
                tabla_vis.columns = ["ID", "Fecha Inicio", "Fecha Fin", "Similitud", "Métodos detección"]
                st.dataframe(tabla_vis, hide_index=True, use_container_width=True)

            # ── Lámina recortada (sin banda de texto ilegible) ────────
            nombre_lam = f"{anom_sel}__top2.png"
            ruta_lam_h = f"{RUTA_LAMINAS_HIST}/{nombre_lam}"
            if os.path.exists(ruta_lam_h):
                img_original = Image.open(ruta_lam_h)
                thumb_width_px = 480  # debe coincidir con CMP_THUMB_WIDTH_PX en settings.py
                img_w = int(thumb_width_px * 2.5)
                img_h_no_label = int(round(img_w * 0.75))

                img_cortada = img_original.crop((0, 0, img_original.width, img_h_no_label))
                st.image(img_cortada, use_container_width=True)

                # ── IDs alineados debajo de cada gráfica ─────────────
                id_hist1 = str(tabla_vis.iloc[0]["ID"]) if tabla_vis is not None and len(tabla_vis) > 0 else "—"
                id_hist2 = str(tabla_vis.iloc[1]["ID"]) if tabla_vis is not None and len(tabla_vis) > 1 else "—"

                c1, c2, c3 = st.columns(3)
                c1.caption(f"Reciente · id={id_rec}")
                c2.caption(f"Histórica · id={id_hist1}")
                c3.caption(f"Histórica · id={id_hist2}")
            else:
                st.warning(f"Lámina no encontrada: {nombre_lam}")

        # ── COLUMNA DERECHA: tabla de Similitud Numérica (DTW) ────────────────
        with col_der:
            st.markdown(
                '<div style="border-left:4px solid #1D9E75; padding-left:10px; '
                'font-size:1.05rem; font-weight:700; margin-bottom:0.5rem;">'
                'Similitud Numérica</div>',
                unsafe_allow_html=True
            )

            if df_dtw_hist.empty:
                st.info("No se encontraron datos de similitud DTW histórica.")
            else:
                dtw_rec = df_dtw_hist[df_dtw_hist["id_anomalia_sintetica"] == id_rec].copy()

                tabla_dtw_placeholder = st.empty()

                umbral_dtw_hist = st.slider(
                    "Similitud mínima numerica", min_value=0.65, max_value=0.96,
                    value=0.75, step=0.01, format="%.2f", key="umbral_dtw_hist"
                )

                dtw_rec_filtrado = dtw_rec[dtw_rec["similarity_combined"] >= umbral_dtw_hist] \
                    .sort_values("similarity_combined", ascending=False)

                with tabla_dtw_placeholder.container():
                    if not dtw_rec_filtrado.empty:
                        tabla_dtw = dtw_rec_filtrado[["id_anomalia_original", "start_date_original",
                                             "end_date_original", "similarity_combined",
                                             "method_original"]].head(10).copy()
                        tabla_dtw["similarity_combined"] = (tabla_dtw["similarity_combined"] * 100).round(1).astype(str) + "%"
                        tabla_dtw.columns = ["ID", "Fecha Inicio", "Fecha Fin", "Similitud", "Métodos detección"]
                        st.dataframe(tabla_dtw, hide_index=True, use_container_width=True)
                    else:
                        st.info(f"Sin pares con similitud ≥ {umbral_dtw_hist:.0%}")


        col_news_vis, col_news_num = st.columns(2, gap="large")
        
        # ══════════════════════════════════════════════
        # SECCIÓN 5 — CONTEXTO ASOCIADO 
        # ══════════════════════════════════════════════        

        st.markdown("---")
        st.markdown("### Noticias asociadas")

        st.markdown(f"**Anomalía reciente #{id_rec}** · {fecha_ini_rec_hdr} – {fecha_fin_rec_hdr}")
        df_not_rec = consultar_noticias(fecha_ini_rec_hdr, fecha_fin_rec_hdr)
        if df_not_rec.empty:
            st.caption("Sin noticias disponibles o Fuseki no está activo.")
        else:
            st.dataframe(
                df_not_rec[["Fecha","Título","Categoría"]],
                hide_index=True, use_container_width=True
            )

        col_news_izq, col_news_der = st.columns(2, gap="large")

        # ── IZQUIERDA: Noticias — Comparación Visual (CLIP) ───────────
        with col_news_izq:
            st.markdown(
                '<div style="border-left:4px solid #378ADD; padding-left:10px; '
                'font-size:1.05rem; font-weight:700; margin-bottom:0.5rem;">'
                'Comparación Visual </div>',
                unsafe_allow_html=True
            )

            if pares.empty:
                st.info("Sin históricas visuales para consultar noticias.")
            else:
                opciones_hist_vis = pares["original_id"].str.split("_").str[0].astype(int).tolist()

                def fmt_hist_vis(x):
                    fila = pares[pares["original_id"].str.split("_").str[0].astype(int) == x].iloc[0]
                    fecha = str(fila["original_start_date"])[:10]
                    sim = fila["similarity_pct"]
                    return f"Histórica #{x} · {fecha} · sim. {sim}"

                hist_sel_vis = st.selectbox(
                    "",
                    #"Histórica a analizar",
                    options=opciones_hist_vis,
                    format_func=fmt_hist_vis,
                    key="hist_sel_news_vis"
                )

                fila_hist_vis = pares[pares["original_id"].str.split("_").str[0].astype(int) == hist_sel_vis].iloc[0]

               # st.markdown(f"**Histórica #{hist_sel_vis}** · {str(fila_hist_vis['original_start_date'])[:10]} · sim. {fila_hist_vis['similarity_pct']}")
                df_not_hist_vis = consultar_noticias(fila_hist_vis["original_start_date"], fila_hist_vis["original_end_date"])
                if df_not_hist_vis.empty:
                    st.caption("Sin noticias disponibles o Fuseki no está activo.")
                else:
                    st.dataframe(
                        df_not_hist_vis[["Fecha","Título","Categoría"]],
                        hide_index=True, use_container_width=True
                    )

                # ── Categorías en común (enlazado a la misma histórica) ──
                if df_news.empty:
                    st.caption("Sin datos precalculados de similitud de noticias.")
                else:
                    news_vis = df_news[
                        (df_news["id_reciente"] == id_rec) &
                        (df_news["id_historica"] == hist_sel_vis)
                    ].copy()

                    if news_vis.empty:
                        st.caption("Sin datos de similitud de noticias para esta histórica.")
                    else:
                        fila_vis = news_vis.iloc[0]

                        emb_fila = df_emb_hist[
                            (df_emb_hist["synthetic_id"] == anom_sel) &
                            (df_emb_hist["original_id"].str.split("_").str[0].astype(int) == hist_sel_vis)
                        ]
                        sim_v = emb_fila["similarity_pct"].values[0] if not emb_fila.empty else "—"

                        col_mv1, col_mv2 = st.columns(2)
                        col_mv1.metric("Sim. visual", sim_v)
                        col_mv2.metric("Sim. noticias", f"{fila_vis['similitud_noticias']*100:.1f}%")

                        cats_rec_v  = [c.strip() for c in str(fila_vis["categorias_reciente"]).split(",") if c.strip()]
                        cats_hist_v = [c.strip() for c in str(fila_vis["categorias_historica"]).split(",") if c.strip()]
                        comunes_v   = set(cats_rec_v) & set(cats_hist_v)

                        st.markdown(f"**Categorías en común: {len(comunes_v)}**")
                        st.caption(f"Noticias: {int(fila_vis.get('n_noticias_reciente',0))} rec · "
                                   f"{int(fila_vis.get('n_noticias_historica',0))} hist")

                        cav, cbv = st.columns(2)
                        with cav:
                            st.markdown("**Reciente:**")
                            for cat in cats_rec_v:
                                icono = "🟢" if cat in comunes_v else "⚪"
                                st.markdown(f"{icono} {cat_legible(cat)}")
                        with cbv:
                            st.markdown("**Histórica:**")
                            for cat in cats_hist_v:
                                icono = "🟢" if cat in comunes_v else "⚪"
                                st.markdown(f"{icono} {cat_legible(cat)}")

        # ── DERECHA: Noticias — Comparación Numérica (DTW) ────────────
        with col_news_der:
            st.markdown(
                '<div style="border-left:4px solid #1D9E75; padding-left:10px; '
                'font-size:1.05rem; font-weight:700; margin-bottom:0.5rem;">'
                'Comparación Numérica</div>',
                unsafe_allow_html=True
            )

            if dtw_rec_filtrado.empty:
                st.info("Sin históricas DTW por encima del umbral actual para consultar noticias.")
            else:
                opciones_hist_dtw = dtw_rec_filtrado["id_anomalia_original"].astype(int).head(10).tolist()

                def fmt_hist_dtw(x):
                    fila = dtw_rec_filtrado[dtw_rec_filtrado["id_anomalia_original"].astype(int) == x].iloc[0]
                    fecha = fila["start_date_original"]
                    sim = f"{fila['similarity_combined']*100:.1f}%"
                    return f"Histórica #{x} · {fecha} · sim. {sim}"

                hist_sel_dtw = st.selectbox(
                    "",
                    #"Histórica a analizar",
                    options=opciones_hist_dtw,
                    format_func=fmt_hist_dtw,
                    key="hist_sel_news_dtw"
                )

                fila_hist_dtw = dtw_rec_filtrado[dtw_rec_filtrado["id_anomalia_original"].astype(int) == hist_sel_dtw].iloc[0]

                #sim_sel = f"{fila_hist_dtw['similarity_combined']*100:.1f}%"
                #st.markdown(f"**Histórica #{hist_sel_dtw}** · {fila_hist_dtw['start_date_original']} · sim. {sim_sel}")
                df_not_hist_dtw = consultar_noticias(fila_hist_dtw["start_date_original"], fila_hist_dtw["end_date_original"])
                if df_not_hist_dtw.empty:
                    st.caption("Sin noticias disponibles o Fuseki no está activo.")
                else:
                    st.dataframe(
                        df_not_hist_dtw[["Fecha","Título","Categoría"]],
                        hide_index=True, use_container_width=True
                    )

                # ── Categorías en común (enlazado a la misma histórica) ──
                if df_news_dtw.empty:
                    st.caption("Sin datos precalculados de similitud de noticias.")
                else:
                    news_num = df_news_dtw[
                        (df_news_dtw["id_reciente"] == id_rec) &
                        (df_news_dtw["id_historica"] == hist_sel_dtw)
                    ].copy()

                    if news_num.empty:
                        st.caption("Sin datos de similitud de noticias para esta histórica.")
                    else:
                        fila_num = news_num.iloc[0]

                        sim_dtw_val = f"{fila_num['similitud_combined']*100:.1f}%"

                        col_mn1, col_mn2 = st.columns(2)
                        col_mn1.metric("Sim. DTW", sim_dtw_val)
                        col_mn2.metric("Sim. noticias", f"{fila_num['similitud_noticias']*100:.1f}%")

                        cats_rec_n  = [c.strip() for c in str(fila_num["categorias_reciente"]).split(",") if c.strip()]
                        cats_hist_n = [c.strip() for c in str(fila_num["categorias_historica"]).split(",") if c.strip()]
                        comunes_n   = set(cats_rec_n) & set(cats_hist_n)

                        st.markdown(f"**Categorías en común: {len(comunes_n)}**")
                        st.caption(f"Noticias: {int(fila_num.get('n_noticias_reciente',0))} rec · "
                                   f"{int(fila_num.get('n_noticias_historica',0))} hist")

                        can, cbn = st.columns(2)
                        with can:
                            st.markdown("**Reciente:**")
                            for cat in cats_rec_n:
                                icono = "🟢" if cat in comunes_n else "⚪"
                                st.markdown(f"{icono} {cat_legible(cat)}")
                        with cbn:
                            st.markdown("**Histórica:**")
                            for cat in cats_hist_n:
                                icono = "🟢" if cat in comunes_n else "⚪"
                                st.markdown(f"{icono} {cat_legible(cat)}")

        st.markdown("---")
with tab2:
    import requests

    FUSEKI_ONT = "http://localhost:3030/c22/sparql"

    def _sparql(query, timeout=30):
        try:
            resp = requests.post(
                FUSEKI_ONT,
                data={"query": query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=timeout
            )
            if resp.status_code != 200:
                st.error(f"Fuseki respondió con estado {resp.status_code}")
                return None
            bindings = resp.json().get("results", {}).get("bindings", [])
            if not bindings:
                return pd.DataFrame()
            return pd.DataFrame([{k: v.get("value","") for k,v in b.items()} for b in bindings])
        except Exception as e:
            st.error(f"Error Fuseki: {e}")
            return None

    st.markdown("### Exploración de la Ontología")
    st.caption("Resultados en tiempo real desde Apache Jena Fuseki")

    # ── DEBUG ─────────────────────────────────────────────────────────────
    with st.expander("🔧 Depuración Fuseki", expanded=False):
        try:
            resp_test = requests.post(
                FUSEKI_ONT,
                data={"query": "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"},
                headers={"Accept": "application/sparql-results+json"},
                timeout=10
            )
            st.write(f"**Estado:** {resp_test.status_code}")
            st.write(f"**Respuesta:** {resp_test.text[:300]}")
        except Exception as e:
            st.error(f"Error: {e}")

    # ── Fila 1: Métodos + Origen ──────────────────────────────────────────
    c1, c2 = st.columns([3, 2], gap="large")

    with c1:
        st.markdown("#### Comparación de métodos de detección")
        df_met = _sparql("""
PREFIX anom: <http://w3id.org/anomaly-core#>
SELECT ?methodName ?methodType (COUNT(?anomaly) AS ?totalAnomalies)
WHERE {
  ?anomaly a anom:Anomaly ; anom:detectedBy ?method .
  ?method anom:detectionMethodName ?methodName ;
          anom:detectionMethodType ?methodType .
}
GROUP BY ?methodName ?methodType
ORDER BY DESC(?totalAnomalies)
""")
        if df_met.empty:
            st.info("Sin datos disponibles.")
        else:
            df_met["totalAnomalies"] = df_met["totalAnomalies"].str.extract(r'(\d+)').astype(int)
            COLORES_TIPO = {"statistical":"#378ADD","ml":"#1D9E75","dl":"#D85A30","foundation":"#D4537E"}
            fig = go.Figure(go.Bar(
                x=df_met["methodName"], y=df_met["totalAnomalies"],
                marker_color=[COLORES_TIPO.get(t,"#888780") for t in df_met["methodType"]],
                text=df_met["totalAnomalies"], textposition="outside",
                hovertemplate="<b>%{x}</b><br>Anomalías: %{y}<extra></extra>",
            ))
            fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(showgrid=False), yaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
                showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            lc1,lc2,lc3,lc4 = st.columns(4)
            lc1.markdown("🔵 Estadístico"); lc2.markdown("🟢 Aprendizaje automático")
            lc3.markdown("🟠 Aprendizaje profundp"); lc4.markdown("🩷 Preentrenado")

    with c2:
        st.markdown("#### Origen de anomalías")
        df_orig = _sparql("""
PREFIX anom: <http://w3id.org/anomaly-core#>
SELECT ?sourceType (COUNT(?anomaly) AS ?total)
WHERE {
  ?anomaly a anom:Anomaly ; anom:locatedInSeries ?series .
  BIND(IF(EXISTS {?series anom:hasSource anom:src_synthetic},"Sintéticas","Observadas") AS ?sourceType)
}
GROUP BY ?sourceType
""")
        if df_orig.empty:
            st.info("Sin datos disponibles.")
        else:
            df_orig["total"] = df_orig["total"].str.extract(r'(\d+)').astype(int)
            fig2 = go.Figure(go.Pie(
                labels=df_orig["sourceType"], values=df_orig["total"],
                marker_colors=["#378ADD","#1D9E75"], hole=0.5,
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Total: %{value}<extra></extra>",
            ))
            fig2.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=30),
                paper_bgcolor="white", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
            n = len(df_orig)
            col_izq_pad, *cols_orig, col_der_pad = st.columns([1] + [2]*n + [1])
            for col, (_, row) in zip(cols_orig, df_orig.iterrows()):
                label = "🔵 Observadas" if row["sourceType"] == "Observadas" else "🟢 Sintéticas"
                col.metric(label, f"{int(row['total']):,}")

    st.divider()

    # ── Fila 2: Años + Puntos calientes ──────────────────────────────────
    c3, c4 = st.columns([2, 3], gap="large")

    with c3:
        st.markdown("#### Anomalías por año")
        df_años = _sparql("""
PREFIX anom: <http://w3id.org/anomaly-core#>
SELECT ?año (COUNT(?anomaly) AS ?totalAnomalias) (SUM(?pointCount) AS ?totalPuntos)
WHERE {
  ?anomaly a anom:Anomaly ;
           anom:count ?pointCount ;
           anom:startDate ?startDate ;
           anom:locatedInSeries anom:series_riesgo_pais_1 .
  BIND(YEAR(?startDate) AS ?año)
}
GROUP BY ?año ORDER BY DESC(?totalPuntos)
""")
        if df_años.empty:
            st.info("Sin datos disponibles.")
        else:
            df_años["año"]            = df_años["año"].astype(int)
            df_años["totalAnomalias"] = df_años["totalAnomalias"].str.extract(r'(\d+)').astype(int)
            df_años["totalPuntos"]    = df_años["totalPuntos"].str.extract(r'(\d+)').astype(int)
            df_ord = df_años.sort_values("año")
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=df_ord["año"].astype(str), y=df_ord["totalPuntos"],
                name="Puntos anómalos", marker_color="#378ADD",
                hovertemplate="<b>%{x}</b><br>Puntos: %{y}<extra></extra>"))
            fig3.add_trace(go.Scatter(
                x=df_ord["año"].astype(str), y=df_ord["totalAnomalias"],
                name="Nº anomalías", mode="lines+markers",
                line=dict(color="#D85A30", width=2), marker=dict(size=6),
                yaxis="y2",
                hovertemplate="<b>%{x}</b><br>Anomalías: %{y}<extra></extra>"))
            fig3.update_layout(
                height=320, margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(showgrid=False, tickangle=-45),
                yaxis=dict(title="Puntos", gridcolor="rgba(0,0,0,0.05)"),
                yaxis2=dict(title="Anomalías", overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h", y=-0.25), hovermode="x unified")
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("**Top 3 años más críticos**")
            for _, row in df_años.head(3).iterrows():
                st.markdown(f"**{int(row['año'])}** — {int(row['totalAnomalias'])} anomalías · {int(row['totalPuntos'])} puntos")

    with c4:
        st.markdown("#### Anomalías de mayor duración")
        df_hot = _sparql("""
PREFIX anom: <http://w3id.org/anomaly-core#>
SELECT ?anomaly ?startDate ?endDate ?pointCount
WHERE {
  ?anomaly a anom:Anomaly ;
           anom:count ?pointCount ;
           anom:startDate ?startDate ;
           anom:endDate ?endDate ;
           anom:locatedInSeries anom:series_riesgo_pais_1 .
  FILTER(?pointCount > 1)
}
ORDER BY DESC(?pointCount) LIMIT 20
""")
        if df_hot.empty:
            st.info("Sin datos disponibles.")
        else:
            df_hot["anomaly"]    = df_hot["anomaly"].str.split("#").str[-1]
            df_hot["pointCount"] = df_hot["pointCount"].str.extract(r'(\d+)').astype(int)
            df_hot["startDate"]  = df_hot["startDate"].str[:10]
            df_hot["endDate"]    = df_hot["endDate"].str[:10]
            df_hot["year"]       = df_hot["startDate"].str[:4]
            fig4 = go.Figure(go.Bar(
                x=df_hot["anomaly"].str[-25:],
                y=df_hot["pointCount"],
                marker_color=["#D85A30" if y=="2020" else "#378ADD" for y in df_hot["year"]],
                text=df_hot["pointCount"], textposition="outside",
                customdata=df_hot[["startDate","endDate"]].values,
                hovertemplate="<b>%{x}</b><br>Puntos: %{y}<br>%{customdata[0]} → %{customdata[1]}<extra></extra>",
            ))
            fig4.update_layout(
                height=320, margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(showgrid=False, tickangle=-45, tickfont=dict(size=9)),
                yaxis=dict(title="Puntos anómalos", gridcolor="rgba(0,0,0,0.05)"),
                showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)
            st.caption("🟠 2020 (COVID-19)  🔵 Otros años")

    st.divider()
#---------------------------
    st.markdown("#### Exploración del contexto periodístico por anomalía")
    st.caption("Explora cualquier anomalía registrada en la ontología, sin pasar por el flujo guiado de modelo/método. Útil para búsquedas puntuales.")

    df_ids_anom = _sparql("""
PREFIX anom: <http://w3id.org/anomaly-core#>
SELECT ?anomalyId ?startDate ?endDate ?methodName
WHERE {
  ?a a anom:Anomaly ; anom:anomalyId ?anomalyId ;
     anom:startDate ?startDate ; anom:endDate ?endDate .
  OPTIONAL { ?a anom:detectedBy ?method . ?method anom:detectionMethodName ?methodName }
}
""")

    if df_ids_anom.empty:
        st.info("Sin anomalías disponibles en la ontología.")
    else:
        df_ids_anom["anomalyId_num"] = pd.to_numeric(df_ids_anom["anomalyId"], errors="coerce")
        df_ids_anom["startDate"] = df_ids_anom["startDate"].str[:10]
        df_ids_anom["endDate"] = df_ids_anom["endDate"].str[:10]

        # ── Deduplicar: una fila por anomalía, agrupando métodos ──────
        df_metodos_agrupados = (
            df_ids_anom.groupby("anomalyId")["methodName"]
            .apply(lambda serie: ",".join(sorted(set(m for m in serie if m))))
            .reset_index()
            .rename(columns={"methodName": "metodos"})
        )
        df_ids_anom = (
            df_ids_anom.drop_duplicates(subset=["anomalyId"])
            .merge(df_metodos_agrupados, on="anomalyId", how="left")
            .sort_values("anomalyId_num")
            .reset_index(drop=True)
        )

        opciones_ids = df_ids_anom["anomalyId"].tolist()

        def fmt_anom(x):
            fila = df_ids_anom[df_ids_anom["anomalyId"] == x].iloc[0]
            metodo = fila["metodos"] or "sin método"
            if fila["startDate"] == fila["endDate"]:
                fecha_str = fila["startDate"]
            else:
                fecha_str = f"{fila['startDate']} – {fila['endDate']}"
            return f"#{x} · {fecha_str} · {metodo}"

        anom_input = st.selectbox(
            "Anomalía a consultar",
            options=opciones_ids,
            format_func=fmt_anom,
            help="Trae las noticias publicadas en un rango ±5 días alrededor de esta anomalía, directamente desde la ontología",
            key="anom_input_ont"
        )

        df_fechas = _sparql(f"""
PREFIX anom: <http://w3id.org/anomaly-core#>
SELECT ?startDate ?endDate WHERE {{
  ?a a anom:Anomaly ; anom:anomalyId "{anom_input}" ;
     anom:startDate ?startDate ; anom:endDate ?endDate .
}} LIMIT 1
""")
        if df_fechas.empty:
            st.warning(f"No se encontró la anomalía {anom_input}")
        else:
            from datetime import datetime, timedelta
            start_real = df_fechas.iloc[0]["startDate"][:10]
            end_real   = df_fechas.iloc[0]["endDate"][:10]
            start = (datetime.strptime(start_real, "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")
            end   = (datetime.strptime(end_real,   "%Y-%m-%d") + timedelta(days=5)).strftime("%Y-%m-%d")

            st.caption(f"Anomalía #{anom_input}: {start_real} → {end_real}  ·  Ventana de búsqueda: {start} a {end}")

            df_news2 = _sparql(f"""
PREFIX anom: <http://w3id.org/anomaly-core#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
SELECT ?newsTitle ?publishedAt ?cat
WHERE {{
  ?news a anom:NewsArticle ;
        anom:newsTitle ?newsTitle ;
        anom:publishedAt ?publishedAt .
  FILTER(?publishedAt >= "{start}"^^xsd:date && ?publishedAt <= "{end}"^^xsd:date)
  OPTIONAL {{ ?news anom:hasCategory ?cat }}
}}
ORDER BY ?publishedAt
""")
            if df_news2.empty:
                st.info("Sin noticias registradas en ese período.")
            else:
                df_news2["publishedAt"] = df_news2["publishedAt"].str[:10]
                df_news2["cat_slug"] = df_news2["cat"].str.split("#").str[-1].str.replace("cat_", "", regex=False)
                df_news2["Categoría"] = df_news2["cat_slug"].apply(
                    lambda c: CATS_LEGIBLES.get(c, c.replace("_"," ").capitalize()) if c else ""
                )
                df_news2 = df_news2.rename(columns={
                    "newsTitle":   "Título",
                    "publishedAt": "Fecha",
                })
                st.dataframe(df_news2[["Fecha","Título","Categoría"]],
                    hide_index=True, use_container_width=True, height=300)
                st.caption(f"{len(df_news2)} noticias encontradas — consulta directa a la ontología vía SPARQL.")

    # ══════════════════════════════════════════════════════════════
    # Red de similitud desde una anomalía sintética (con expansión 2 saltos)
    # ══════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("#### Red de similitud desde una anomalía sintética")
    st.caption("Selecciona una anomalía sintética para ver a cuántas históricas se parece, según similitud visual y/o numérica.")
    import networkx as nx
    df_sint_list = _sparql("""
PREFIX anom: <http://w3id.org/anomaly-core#>
SELECT DISTINCT ?a
WHERE {
  { ?sim a anom:Similarity ; anom:betweenAnomalyA ?a }
  UNION
  { ?sim a anom:Similarity ; anom:betweenAnomalyB ?a }
  FILTER(CONTAINS(STR(?a), "_syn_"))
}
""")

    if df_sint_list.empty:
        st.info("No se encontraron anomalías sintéticas con relaciones de similitud en la ontología.")
    else:
        df_sint_list["nombre_corto"] = df_sint_list["a"].str.split("#").str[-1]
        opciones_sint = sorted(df_sint_list["nombre_corto"].tolist())

        sint_sel_red = st.selectbox(
            "Anomalía sintética",
            options=opciones_sint,
            key="sint_sel_red_ont"
        )

        umbral_red_sint = st.slider(
            "Similitud mínima a mostrar", min_value=0.5, max_value=1.0,
            value=0.85, step=0.01, format="%.2f", key="umbral_red_sint_ont"
        )

        uri_centro_sint = df_sint_list[df_sint_list["nombre_corto"] == sint_sel_red].iloc[0]["a"]

        df_red_sint = _sparql(f"""
PREFIX anom: <http://w3id.org/anomaly-core#>
SELECT ?a ?b ?similarityValue ?methodUsed
WHERE {{
  ?sim a anom:Similarity ;
       anom:betweenAnomalyA ?a ;
       anom:betweenAnomalyB ?b ;
       anom:similarityValue ?similarityValue ;
       anom:fromRun ?run .
  ?run anom:runMethod ?method .
  ?method anom:similarityMethodName ?methodUsed .
  FILTER(?similarityValue >= {umbral_red_sint})
  FILTER(?a = <{uri_centro_sint}> || ?b = <{uri_centro_sint}>)
}}
""")

        if df_red_sint.empty:
            st.info(f"La sintética '{sint_sel_red}' no tiene relaciones de similitud ≥ {umbral_red_sint:.0%}. Prueba bajar el umbral.")
        else:
            df_red_sint["a"] = df_red_sint["a"].str.split("#").str[-1]
            df_red_sint["b"] = df_red_sint["b"].str.split("#").str[-1]
            df_red_sint["similarityValue"] = df_red_sint["similarityValue"].astype(float)
            nodo_centro_sint = uri_centro_sint.split("#")[-1]

            G2 = nx.Graph()
            for _, row in df_red_sint.iterrows():
                G2.add_edge(row["a"], row["b"], weight=row["similarityValue"], method=row["methodUsed"])

            pos2 = nx.spring_layout(G2, k=0.6, seed=42)

            edge_x2, edge_y2 = [], []
            for u, v in G2.edges():
                x0, y0 = pos2[u]
                x1, y1 = pos2[v]
                edge_x2 += [x0, x1, None]
                edge_y2 += [y0, y1, None]

            edge_trace2 = go.Scatter(
                x=edge_x2, y=edge_y2,
                line=dict(width=0.9, color="rgba(150,150,150,0.5)"),
                hoverinfo="none", mode="lines"
            )

            node_x2, node_y2, node_color2, node_text2, node_size2 = [], [], [], [], []
            for n in G2.nodes():
                x, y = pos2[n]
                node_x2.append(x)
                node_y2.append(y)
                if n == nodo_centro_sint:
                    node_color2.append("#D4537E")
                else:
                    node_color2.append("#1D9E75")
                grado = G2.degree(n)
                tam_base = 18 if n == nodo_centro_sint else 8
                node_size2.append(tam_base + grado * 2)
                node_text2.append(f"{n}<br>Conexiones: {grado}")

            node_trace2 = go.Scatter(
                x=node_x2, y=node_y2,
                mode="markers",
                hoverinfo="text", text=node_text2,
                marker=dict(color=node_color2, size=node_size2, line=dict(width=0.5, color="white"))
            )

            fig_red2 = go.Figure(data=[edge_trace2, node_trace2])
            fig_red2.update_layout(
                height=500, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
            st.plotly_chart(fig_red2, use_container_width=True)

            col_legs1, col_legs2, col_legs3 = st.columns(3)
            col_legs1.markdown("🔴 Sintética seleccionada")
            col_legs2.markdown("🟢 Anomalías históricas")
            col_legs3.metric("Anomalías encontradas", len(G2.edges()))

            # ── Expansión a 2 saltos ──────────────────────────────────
            st.markdown("##### Exploración de la red de similitud entre anomalías")
            st.caption("Desde la sintética seleccionada, busca sus originales similares, y desde esas originales, qué otras sintéticas también se les parecen.")

            expandir_2saltos = st.checkbox("Expandir a 2 saltos", value=False, key="expandir_2saltos_ont")

            if expandir_2saltos:
                umbral_2saltos = st.slider(
                    "Similitud mínima (ambos saltos)", min_value=0.5, max_value=1.0,
                    value=umbral_red_sint, step=0.01, format="%.2f", key="umbral_2saltos_ont"
                )

                df_salto1 = _sparql(f"""
PREFIX anom: <http://w3id.org/anomaly-core#>
SELECT ?a ?b ?similarityValue ?methodUsed
WHERE {{
  ?sim a anom:Similarity ;
       anom:betweenAnomalyA ?a ;
       anom:betweenAnomalyB ?b ;
       anom:similarityValue ?similarityValue ;
       anom:fromRun ?run .
  ?run anom:runMethod ?method .
  ?method anom:similarityMethodName ?methodUsed .
  FILTER(?similarityValue >= {umbral_2saltos})
  FILTER(?a = <{uri_centro_sint}> || ?b = <{uri_centro_sint}>)
}}
""")

                if df_salto1.empty:
                    st.info(f"La sintética seleccionada no tiene originales con similitud ≥ {umbral_2saltos:.0%}.")
                else:
                    df_salto1["a"] = df_salto1["a"].str.split("#").str[-1]
                    df_salto1["b"] = df_salto1["b"].str.split("#").str[-1]
                    df_salto1["similarityValue"] = df_salto1["similarityValue"].astype(float)

                    originales_encontradas = set(df_salto1["a"]).union(set(df_salto1["b"])) - {nodo_centro_sint}

                    if not originales_encontradas:
                        st.info("No hay originales de las cuales expandir.")
                    else:
                        condiciones = " || ".join([
                            f"?a = anom:{orig} || ?b = anom:{orig}" for orig in originales_encontradas
                        ])

                        df_salto2 = _sparql(f"""
PREFIX anom: <http://w3id.org/anomaly-core#>
SELECT ?a ?b ?similarityValue ?methodUsed
WHERE {{
  ?sim a anom:Similarity ;
       anom:betweenAnomalyA ?a ;
       anom:betweenAnomalyB ?b ;
       anom:similarityValue ?similarityValue ;
       anom:fromRun ?run .
  ?run anom:runMethod ?method .
  ?method anom:similarityMethodName ?methodUsed .
  FILTER(?similarityValue >= {umbral_2saltos})
  FILTER({condiciones})
}}
""")

                        if df_salto2.empty:
                            st.info("Ninguna de las originales encontradas tiene otras sintéticas similares.")
                        else:
                            df_salto2["a"] = df_salto2["a"].str.split("#").str[-1]
                            df_salto2["b"] = df_salto2["b"].str.split("#").str[-1]
                            df_salto2["similarityValue"] = df_salto2["similarityValue"].astype(float)

                            G3 = nx.Graph()
                            for _, row in df_salto1.iterrows():
                                G3.add_edge(row["a"], row["b"], weight=row["similarityValue"], hop=1)
                            for _, row in df_salto2.iterrows():
                                if not G3.has_edge(row["a"], row["b"]):
                                    G3.add_edge(row["a"], row["b"], weight=row["similarityValue"], hop=2)

                            pos3 = nx.spring_layout(G3, k=0.6, seed=42)

                            edge_x3, edge_y3 = [], []
                            for u, v in G3.edges():
                                x0, y0 = pos3[u]
                                x1, y1 = pos3[v]
                                edge_x3 += [x0, x1, None]
                                edge_y3 += [y0, y1, None]

                            edge_trace3 = go.Scatter(
                                x=edge_x3, y=edge_y3,
                                line=dict(width=0.9, color="rgba(150,150,150,0.5)"),
                                hoverinfo="none", mode="lines"
                            )

                            node_x3, node_y3, node_color3, node_text3, node_size3 = [], [], [], [], []
                            for n in G3.nodes():
                                x, y = pos3[n]
                                node_x3.append(x)
                                node_y3.append(y)
                                es_sintetica_n = "syn" in n.lower()
                                if n == nodo_centro_sint:
                                    node_color3.append("#D4537E")
                                elif es_sintetica_n:
                                    node_color3.append("#378ADD")
                                else:
                                    node_color3.append("#1D9E75")
                                grado = G3.degree(n)
                                tam_base = 18 if n == nodo_centro_sint else 8
                                node_size3.append(tam_base + grado * 2)
                                node_text3.append(f"{n}<br>Conexiones: {grado}")

                            node_trace3 = go.Scatter(
                                x=node_x3, y=node_y3,
                                mode="markers",
                                hoverinfo="text", text=node_text3,
                                marker=dict(color=node_color3, size=node_size3, line=dict(width=0.5, color="white"))
                            )

                            fig_red3 = go.Figure(data=[edge_trace3, node_trace3])
                            fig_red3.update_layout(
                                height=550, margin=dict(l=0, r=0, t=10, b=0),
                                plot_bgcolor="white", paper_bgcolor="white",
                                showlegend=False,
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            )
                            st.plotly_chart(fig_red3, use_container_width=True)

                            col_l1, col_l2, col_l3, col_l4 = st.columns(4)
                            col_l1.markdown("🔴 Sintética seleccionada")
                            col_l2.markdown("🟢 Originales (1er salto)")
                            col_l3.markdown("🔵 Otras sintéticas (2do salto)")
                            col_l4.metric("Nodos totales", G3.number_of_nodes())

                            sinteticas_indirectas = [n for n in G3.nodes() if "syn" in n.lower() and n != nodo_centro_sint]
                            if sinteticas_indirectas:
                                st.markdown(f"**{len(sinteticas_indirectas)} sintéticas conectadas indirectamente:**")
#-----------------


    st.divider()
    st.markdown("#### Red de similitud entre anomalías sintéticas y del corpus histórico")
    st.caption("Cada nodo es una anomalía; cada línea, una relación de similitud por encima del umbral elegido. Centrada en la anomalía seleccionada arriba.")

    import networkx as nx

    if "anom_input" not in dir() or anom_input is None:
        st.info("Selecciona una anomalía en 'Consulta libre — noticias por anomalía' para ver su red de similitud.")
    else:
        umbral_red = st.slider(
            "Similitud mínima a mostrar", min_value=0.5, max_value=1.0,
            value=0.85, step=0.01, format="%.2f", key="o"
        )

        # ── 1) Obtener la URI completa de la anomalía seleccionada ────
        df_uri_centro = _sparql(f"""
PREFIX anom: <http://w3id.org/anomaly-core#>
SELECT ?anomaly WHERE {{
  ?anomaly a anom:Anomaly ; anom:anomalyId "{anom_input}" .
}} LIMIT 1
""")

        if df_uri_centro.empty:
            st.warning(f"No se encontró la anomalía #{anom_input} en la ontología.")
        else:
            uri_centro = df_uri_centro.iloc[0]["anomaly"]

            # ── 2) Traer solo las relaciones donde participa esta anomalía ──
            df_red = _sparql(f"""
PREFIX anom: <http://w3id.org/anomaly-core#>
SELECT ?a ?b ?similarityValue ?methodUsed
WHERE {{
  ?sim a anom:Similarity ;
       anom:betweenAnomalyA ?a ;
       anom:betweenAnomalyB ?b ;
       anom:similarityValue ?similarityValue ;
       anom:fromRun ?run .
  ?run anom:runMethod ?method .
  ?method anom:similarityMethodName ?methodUsed .
  FILTER(?similarityValue >= {umbral_red})
  FILTER(?a = <{uri_centro}> || ?b = <{uri_centro}>)
}}
""")

            if df_red.empty:
                st.info(f"La anomalía #{anom_input} no tiene relaciones de similitud ≥ {umbral_red:.0%}. Prueba bajar el umbral.")
            else:
                df_red["a"] = df_red["a"].str.split("#").str[-1]
                df_red["b"] = df_red["b"].str.split("#").str[-1]
                df_red["similarityValue"] = df_red["similarityValue"].astype(float)
                nodo_centro = uri_centro.split("#")[-1]

                # ── Construir grafo ────────────────────────────────────
                G = nx.Graph()
                for _, row in df_red.iterrows():
                    G.add_edge(row["a"], row["b"], weight=row["similarityValue"], method=row["methodUsed"])

                pos = nx.spring_layout(G, k=0.6, seed=42)

                # ── Aristas ─────────────────────────────────────────────
                edge_x, edge_y = [], []
                for u, v in G.edges():
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]

                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.9, color="rgba(150,150,150,0.5)"),
                    hoverinfo="none", mode="lines"
                )

                # ── Nodos ───────────────────────────────────────────────
                node_x, node_y, node_color, node_text, node_size = [], [], [], [], []
                for n in G.nodes():
                    x, y = pos[n]
                    node_x.append(x)
                    node_y.append(y)
                    if n == nodo_centro:
                        node_color.append("#D4537E")  # destaca la anomalía seleccionada
                    else:
                        es_sintetica = "syn" in n.lower()
                        node_color.append("#378ADD" if es_sintetica else "#1D9E75")
                    grado = G.degree(n)
                    tam_base = 18 if n == nodo_centro else 8
                    node_size.append(tam_base + grado * 2)
                    node_text.append(f"{n}<br>Conexiones: {grado}")

                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode="markers",
                    hoverinfo="text", text=node_text,
                    marker=dict(color=node_color, size=node_size, line=dict(width=0.5, color="white"))
                )

                fig_red = go.Figure(data=[edge_trace, node_trace])
                fig_red.update_layout(
                    height=500, margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    showlegend=False,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                )
                st.plotly_chart(fig_red, use_container_width=True)

                col_leg1, col_leg2, col_leg3, col_leg4 = st.columns(4)
                col_leg1.markdown("🔴 Anomalia histórica seleccionada")
                col_leg3.markdown("🟢 Anomalias históticas")
                col_leg4.metric("Conexiones encontradas", len(G.edges()))

#---------------------------

    # ── Fila 3: Similitudes + Noticias ───────────────────────────────────
    st.divider()
    st.markdown("#### Red de similitud entre anomalías")
    st.caption("Cada nodo es una anomalía; cada línea, una relación de similitud por encima del umbral elegido.")

    import networkx as nx

    umbral_red = st.slider(
        "Similitud mínima a mostrar", min_value=0.5, max_value=1.0,
        value=0.85, step=0.01, format="%.2f", key="umbral_red_ont"
    )

    df_red = _sparql(f"""
PREFIX anom: <http://w3id.org/anomaly-core#>
SELECT ?a ?b ?similarityValue ?methodUsed
WHERE {{
  ?sim a anom:Similarity ;
       anom:betweenAnomalyA ?a ;
       anom:betweenAnomalyB ?b ;
       anom:similarityValue ?similarityValue ;
       anom:fromRun ?run .
  ?run anom:runMethod ?method .
  ?method anom:similarityMethodName ?methodUsed .
  FILTER(?similarityValue >= {umbral_red})
}}
""")

    if df_red.empty:
        st.info(f"Sin relaciones de similitud ≥ {umbral_red:.0%}. Prueba bajar el umbral.")
    else:
        df_red["a"] = df_red["a"].str.split("#").str[-1]
        df_red["b"] = df_red["b"].str.split("#").str[-1]
        df_red["similarityValue"] = df_red["similarityValue"].astype(float)

        # ── Construir grafo ────────────────────────────────────────
        G = nx.Graph()
        for _, row in df_red.iterrows():
            G.add_edge(row["a"], row["b"], weight=row["similarityValue"], method=row["methodUsed"])

        pos = nx.spring_layout(G, k=0.5, seed=42)

        # ── Aristas ─────────────────────────────────────────────────
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.7, color="rgba(150,150,150,0.4)"),
            hoverinfo="none", mode="lines"
        )

        # ── Nodos ───────────────────────────────────────────────────
        node_x, node_y, node_color, node_text, node_size = [], [], [], [], []
        for n in G.nodes():
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)
            es_sintetica = "syn" in n.lower()
            node_color.append("#378ADD" if es_sintetica else "#1D9E75")
            grado = G.degree(n)
            node_size.append(8 + grado * 2)
            node_text.append(f"{n}<br>Conexiones: {grado}")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers",
            hoverinfo="text", text=node_text,
            marker=dict(color=node_color, size=node_size, line=dict(width=0.5, color="white"))
        )

        fig_red = go.Figure(data=[edge_trace, node_trace])
        fig_red.update_layout(
            height=500, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        st.plotly_chart(fig_red, use_container_width=True)

        col_leg1, col_leg2, col_leg3 = st.columns(3)
        col_leg1.markdown("🔵 Sintética")
        col_leg2.markdown("🟢 Histórica")
        col_leg3.metric("Total conexiones mostradas", len(G.edges()))    


                           