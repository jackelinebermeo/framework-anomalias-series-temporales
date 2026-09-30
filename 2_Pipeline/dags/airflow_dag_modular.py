import os
import re
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from airflow.decorators import dag, task
from airflow.sensors.filesystem import FileSensor
from airflow.operators.python import PythonOperator

from componentes.semantic_exporter import generate_ttl
from componentes.settings import RUTA_ARCHIVO, NORMALIZAR_DATOS, MODELOS_ACTIVOS, ANOMALIAS_CONSOLIDADO
#from componentes.prediction import collect_sinteticas_json, process_all_sintetica, anomalies_by_prediction_method_all_models, anomalies_by_date_all_models
from componentes.prediction import  anomalies_by_prediction_method_all_models, anomalies_by_date_all_models,exportar_ventanas_desde_grouped,plot_ventanas_desde_csv,crear_csv_comparacion_numerica_sinteticas,consolidar_puntos_sinteticos_all_models
from componentes import settings as CFG
#from embeddings import run_embeddings_comparison_and_sheets

default_args = {
    "start_date": datetime(2020, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

@dag(
    dag_id="pipeline_analisis_serie",
    schedule=None,
    default_args=default_args,
    catchup=False,
    max_active_tasks=3,
    tags=["anomalias", "deteccion", "modular"],
)
def anomaly_flow():
    # --------- Sensor de archivo de entrada ----------
    wait_file = FileSensor(
        task_id="wait_file",
        fs_conn_id="fs_local",
        filepath=RUTA_ARCHIVO,
        poke_interval=4,
        timeout=30,
        mode="reschedule",
    )

    # --------- Tareas TaskFlow ----------

    @task
    def clean_results():
        import os, shutil        
        out_dir = CFG.RUTA_SALIDA
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[LIMPIAR] Carpeta limpia: {out_dir}")
    @task
    def read_file():
        from componentes.exploration import ExploracionSerie
        return ExploracionSerie().read_file()


    @task
    def describe_series(json_df: str):
        from componentes.exploration import ExploracionSerie
        ExploracionSerie().describe(json_df)

    @task
    def normalize_series(json_df: str):
        from componentes.exploration import ExploracionSerie
        return ExploracionSerie().normalize_series(json_df)

    @task
    def plot_kde_distribution(json_df: str):
        from componentes.exploration import ExploracionSerie
        ExploracionSerie().plot_kde(json_df)

    @task
    def plot_time_series(json_df: str):
        from componentes.exploration import ExploracionSerie
        ExploracionSerie().plot_series(json_df)

    @task
    def analyze_seasonality(json_df: str):
        from componentes.exploration import ExploracionSerie
        ExploracionSerie().analyze_seasonality(json_df)

    #------------Tareas para deteccion------------------
    @task
    def detect_by_method(modelo: str, json_df: str, fuente: str) -> str:
        from componentes.detection import AnomaliesDetector
        return AnomaliesDetector().execute_model(modelo, json_df,fuente)

    #@task
    #def plot_by_method(modelo: str) -> str | None:
    #    from componentes.plotting import plot_anomalies_by_method
    #    return plot_anomalies_by_method(modelo)
    @task
    def plot_by_method(modelo: str, json_df: str, fuente: str) -> str | None:
        import os
        os.environ["MPLBACKEND"] = "Agg"
        from componentes.plotting import plot_anomalies_by_method
        # Ahora el plot recibe la MISMA serie que se usó en la detección
        return plot_anomalies_by_method(modelo, fuente=fuente, json_df=json_df)
    
    #--------------------Tareas para consolidacion resultados
    @task(do_xcom_push=False)
    def consolidate_results():
        from componentes.result import _load_series_points, _points_to_intervals_by_method, consolidate_segments
        puntos = _load_series_points()
        print(f"[CONSOLIDAR] Puntos cargados: {len(puntos)}")
        tramos = _points_to_intervals_by_method(puntos)
        print(f"[CONSOLIDAR] Tramos generados: {len(tramos)}")
        consolidate_segments(tramos)
        print("[CONSOLIDAR] anomalias_consolidado.csv escrito")

    @task
    def plot_consolidado_consolidation() -> str:
        from componentes.plotting import plot_consolidated
        return plot_consolidated()

    @task
    def plot_anomaly_padding():
        import os
        if not os.path.isfile(ANOMALIAS_CONSOLIDADO):
            print(f"[VENTANAS] No existe {ANOMALIAS_CONSOLIDADO}. Se omite graficado por ventanas.")
            return None
        from componentes.plotting import plot_anomaly_padding
        return plot_anomaly_padding()

    #-----------------Tarea genera csv adicional
    @task
    def enriquecer_anomalias_task():
        from componentes.result import crear_csv_enriquecido
        out_path = crear_csv_enriquecido()
        print(f"[ENRIQUECIDO] CSV generado: {out_path}")
   
    @task
    def crear_catalogo_comparacion_numerica_task(csv_enriquecido_path: str):
        from componentes.result import crear_csv_comparacion_numerica
        out_path = crear_csv_comparacion_numerica(csv_enriquecido=csv_enriquecido_path)
        print(f"[NUMERIC-CATALOG] CSV generado: {out_path}")
        return out_path   
    #------------------Tareas para generar ttl y poblar ontologia
    @task
    def export_ttl_task():
        generate_ttl()

    #----------------- Tareas prediccion
    @task
    def prediction_task(modelo: str, json_df: str) -> str:
        from componentes import settings as CFG
        from componentes.prediction import run_prediction
        
        out = run_prediction(modelo, json_df, out_root=CFG.RUTA_PREDICCION)
        return out["json"]

       # Ejemplo: si igual quieres mantener explícito ARIMA, esto sigue válido:
    # t_pred_arima = prediction_task.override(task_id="predict_arima")("ARIMA", json_preparado)

    #-------------------------Tarea copia carpeta local
    @task
    def copy_results():
        import os, shutil, subprocess, sys, datetime as dt

        origen  = CFG.RUTA_SALIDA      
        destino = CFG.RUTA_RESPALDO #"/mnt/d/Jacky/Resultados"#      # p.ej. "/mnt/d/Jacky/Resultados" (Windows compartido)

        print(f"[COPIAR] Origen : {origen}")
        print(f"[COPIAR] Destino: {destino}")

        # 0) Chequeos de ruta
        if not os.path.isdir(origen):
            raise RuntimeError(f"[COPIAR][ERROR] Origen no existe o no es carpeta: {origen}")

        # Mostrar contenido de origen (hasta 100 entradas)
        try:
            contenido_origen = os.listdir(origen)[:100]
            print(f"[COPIAR] Contenido origen (primeros 100): {contenido_origen}")
        except Exception as e:
            print(f"[COPIAR][WARN] No se pudo listar origen: {e}")

        # 1) (Re)crear destino vacío
        if os.path.isdir(destino):
            print(f"[COPIAR] Borrando respaldo previo en {destino}")
            shutil.rmtree(destino, ignore_errors=True)

        try:
            os.makedirs(destino, exist_ok=True)
            print(f"[COPIAR] Carpeta destino creada: {destino}")
        except Exception as e:
            raise RuntimeError(f"[COPIAR][ERROR] No se pudo crear destino {destino}: {e}")

        # 2) Copiar SOLO el contenido: cp -r origen/. destino
        cmd = ["bash", "-lc", f'cp -r "{origen}/." "{destino}"']
        # Nota: usamos bash -lc para asegurar expansión correcta en WSL y rutas con espacios.
        print(f"[COPIAR] Ejecutando: {' '.join(cmd)}")
        try:
            res = subprocess.run(cmd, check=False, capture_output=True, text=True)
            print(f"[COPIAR] cp stdout:\n{res.stdout}")
            if res.returncode != 0:
                print(f"[COPIAR][ERROR] cp stderr:\n{res.stderr}")
                raise RuntimeError(f"[COPIAR][ERROR] cp falló con código {res.returncode}")
        except Exception as e:
            # Fallback Python puro si cp falla
            print(f"[COPIAR][WARN] cp falló ({e}). Intentando fallback con shutil.copytree...")
            # Copia recursiva con shutil: mover todo el contenido
            for nombre in os.listdir(origen):
                src = os.path.join(origen, nombre)
                dst = os.path.join(destino, nombre)
                try:
                    if os.path.isdir(src):
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
                except Exception as ee:
                    raise RuntimeError(f"[COPIAR][ERROR] Fallback shutil falló copiando {src} -> {dst}: {ee}")

        # 3) Escribir un sentinela para comprobar escritura en destino
        try:
            sentinel = os.path.join(destino, f"_ok_copia_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(sentinel, "w", encoding="utf-8") as f:
                f.write(f"OK {dt.datetime.now().isoformat()}\n")
            print(f"[COPIAR] Sentinela creado: {sentinel}")
        except Exception as e:
            raise RuntimeError(f"[COPIAR][ERROR] No se pudo escribir en destino (permiso?): {e}")

        # 4) Listar destino (primeros 100)
        try:
            contenido_dest = os.listdir(destino)[:100]
            print(f"[COPIAR] Contenido destino (primeros 100): {contenido_dest}")
        except Exception as e:
            print(f"[COPIAR][WARN] No se pudo listar destino: {e}")

        print(f"[COPIAR] Copia completada: {origen} -> {destino}")

    # ============================================
    # NUEVO: utilidades para procesar TODAS las sintéticas sin sobrescribir
    # ============================================

      # from airflow.decorators import task
    '''
    @task
    def process_all_synthetics(json_map):
        try:
            from airflow.decorators import get_current_context
            from airflow.models.xcom_arg import XComArg
            ctx = get_current_context()
            if isinstance(json_map, XComArg):
                json_map = json_map.resolve(ctx)
        except Exception:
            pass

        from componentes.prediction import exportar_ventanas_sinteticas
        return exportar_ventanas_sinteticas(json_map)
    '''
    @task(execution_timeout=timedelta(minutes=60))
    def process_all_synthetics(json_map):
        from componentes.prediction import exportar_ventanas_sinteticas
        return exportar_ventanas_sinteticas(json_map)
    
    exportar_ventanas = PythonOperator(
        task_id="exportar_ventanas_desde_grouped",
        python_callable=exportar_ventanas_desde_grouped,
        op_args=[CFG],
    )
    # === NUEVA TAREA: GRAFICAR LAS VENTANAS ===
    plot_ventanas = PythonOperator(
        task_id="plot_ventanas_desde_csv",
        python_callable=plot_ventanas_desde_csv,
        op_args=[CFG],
    )
    @task
    def crear_catalogo_comparacion_numerica_sinteticas_task():
        from componentes.prediction import crear_csv_comparacion_numerica_sinteticas
        out_path = crear_csv_comparacion_numerica_sinteticas()
        print(f"[NUMERIC-CATALOG-S] CSV generado: {out_path}")
        return out_path

    @task
    def comparar_anomalias_numericas_dtw_taskV1():
        """
        Compara numéricamente (DTW) las anomalías originales vs sintéticas
        y genera el CSV de similaridades numéricas.
        """
        from componentes import settings as CFG
        from componentes import result
        out_csv = result.comparar_anomalias_numericas_dtw(CFG)
        print(f"[NUM-DTW] CSV de comparación numérica generado: {out_csv}")
        return out_csv
    @task
    def comparar_anomalias_numericas_predichas():
        """
        Compara numéricamente (DTW) las anomalías originales vs sintéticas
        y genera el CSV de similaridades numéricas.
        """
        from componentes import settings as CFG
        from componentes import result
        out_csv = result.comparar_anomalias_numericas_predichas(CFG)
        print(f"[NUM-DTW] CSV de comparación numérica generado: {out_csv}")
        return out_csv

    @task
    def compare_embeddings_all2all():
        """
        Orquesta:
        1) Indexar sintéticas (queries) y originales (corpus)
        2) Comparar sintética -> originales (top-K por sintética) con umbral CMP_MIN_SIM
        3) Generar CSV legible + láminas por sintética si está activado en settings
        """
        #from embeddings import run_embeddings_comparison_and_sheets

        from componentes.embeddings import run_embeddings_comparison_and_sheets
        return run_embeddings_comparison_and_sheets(CFG)

    @task
    def collect_sinteticas_jsonV1() -> dict:
        import os, glob

        pred_dir = getattr(CFG, "RUTA_PREDICCION", "/tmp/DatosSerie/Resultados/Prediccion")
        out = {}

        pattern = os.path.join(pred_dir, "*", "1_*_sintetica.csv")
        for fp in sorted(glob.glob(pattern)):
            base = os.path.basename(fp)
            pred_model = base.replace("1_", "").replace("_sintetica.csv", "").upper()
            out[pred_model] = fp  # ← solo el path, no el contenido

        print(f"[COLLECT] Sintéticas encontradas: {list(out.keys())}")
        return out

    @task
    def collect_sinteticas_json() -> dict:
        import os, glob
        pred_dir = getattr(CFG, "RUTA_PREDICCION", "/tmp/DatosSerie/Resultados/Prediccion")
        out = {}
        for fp in sorted(glob.glob(os.path.join(pred_dir, "*", "1_*_sintetica.csv"))):
            base = os.path.basename(fp)
            pred_model = base.replace("1_", "").replace("_sintetica.csv", "").upper()
            out[pred_model] = fp  # ← solo el path, no el contenido
        print(f"[COLLECT] Sintéticas encontradas: {list(out.keys())}")
        return out        
#-------------------------------------------------------------------------
#---------------------------------ORQUESTACION
#-------------------------------------------------------------------------
    t_limpieza   = clean_results()
    json_serie   = read_file()

    # Dependencias iniciales
    t_limpieza >> json_serie
    wait_file >> json_serie

    describe_series(json_serie)
    json_preparado = normalize_series(json_serie) if NORMALIZAR_DATOS else json_serie

    plot_kde_distribution(json_preparado)
    plot_time_series(json_preparado)
    analyze_seasonality(json_preparado)

    # ============================================
    # 1) PREDICCIONES (se mantienen como ya tengas)
    #    Ejemplo mínimo: aseguramos al menos ARIMA
    # ============================================

    try:
        PRED_MODELOS = getattr(CFG, "PRED_MODELOS_ACTIVOS")
        if not PRED_MODELOS:
            PRED_MODELOS = ["ARIMA"]
    except Exception:
        PRED_MODELOS = ["ARIMA"]

    pred_tasks = {
        modelo: prediction_task.override(task_id=f"predict_{modelo.lower()}")(modelo, json_serie)
        for modelo in PRED_MODELOS
    }
    # UNE TODAS LAS ANOMALIAS DETECTADAS
    concat_anomalies_phase_a_all_models = PythonOperator(
        task_id="concat_anomalies_phase_a_all_models",
        python_callable=anomalies_by_prediction_method_all_models,
        op_kwargs={"include_provenance": False},  # deja solo las 5 columnas pedidas
    )

    consolidar_puntos_sinteticos_phase_all_models = PythonOperator(
        task_id="consolidar_puntos_sinteticos_phase_all_models",
        python_callable=consolidar_puntos_sinteticos_all_models,
    )    
    for t in pred_tasks.values():
        t >> consolidar_puntos_sinteticos_phase_all_models



    # --- NUEVA TAREA: AGRUPA ANOMALÍAS POR FECHAS CONSECUTIVAS ---
    concat_anomalies_phase_b_all_models = PythonOperator(
        task_id="concat_anomalies_phase_b_all_models",
        python_callable=anomalies_by_date_all_models,
        op_args=[CFG],        
    )

    # ============================================
    # 2) DETECCIÓN – FUENTE ORIGINAL (sin cambios)
    #    Exporta a .../DetectadoXMetodo/Original/...
    # ============================================
    results_det = []
    for modelo in MODELOS_ACTIVOS:
        t_det_o = detect_by_method.override(task_id=f"detect_{modelo.lower()}_orig")(
            modelo, json_preparado, "original"
        )
        t_plot_o = plot_by_method.override(task_id=f"plot_{modelo.lower()}_orig")(
            modelo, json_preparado, "original"
        )
        t_det_o >> t_plot_o
        results_det.append(t_det_o)

    # ============================================
    # 3) PUENTE SINTÉTICOS: recolecta TODAS las 2_*_sintetica.csv
    #    en RUTA_PREDICCION y arma JSON literal por serie
    # ============================================



   

    jsons_sint = collect_sinteticas_json()
    for t in pred_tasks.values():
        t >> jsons_sint
    #for t in pred_tasks.values():
    #    t >> concat_anomalies_phase_a_all_models

    # Procesa TODAS en un solo task que aísla ruta por 'series_tag'
    # Procesa TODAS en un solo task que aísla ruta por 'series_tag'
    t_proc_sint = process_all_synthetics(jsons_sint)

    jsons_sint >> t_proc_sint
    consolidar_puntos_sinteticos_phase_all_models >> t_proc_sint

    #t_proc_sint >> concat_anomalies_phase_a_all_models >> concat_anomalies_phase_b_all_models >> exportar_ventanas >> plot_ventanas
    t_proc_sint >> concat_anomalies_phase_a_all_models >> concat_anomalies_phase_b_all_models >> exportar_ventanas >> plot_ventanas
    
    # Catálogo numérico de anomalías SINTÉTICAS
    t_catalogo_sinteticas = crear_catalogo_comparacion_numerica_sinteticas_task()
    exportar_ventanas >> t_catalogo_sinteticas

    # ============================================
    # 5) CONSOLIDACIÓN Y GRAFICADO (solo ORIGINAL)
    # ============================================
    t_consolida = consolidate_results()
    for t in results_det:
        t >> t_consolida

    t_plot_consol = plot_consolidado_consolidation()
    t_consolida >> t_plot_consol

    # Ventanas de contexto
    t_windows = plot_anomaly_padding()
    t_consolida >> t_windows

    # CSV enriquecido
    t_enriq = enriquecer_anomalias_task()
    t_consolida >> t_enriq

    # Catálogo numérico (para comparación numérica)
    t_catalogo_numerico = crear_catalogo_comparacion_numerica_task(t_enriq)

    # ============================================
    # NUEVO: COMPARACIÓN NUMÉRICA (DTW)
    # Necesita:
    #   - catálogo numérico original (t_catalogo_numerico)
    #   - catálogo numérico sintético (t_catalogo_sinteticas)
    # ============================================
    t_cmp_numerico = comparar_anomalias_numericas_predichas()
    [t_catalogo_numerico, t_catalogo_sinteticas] >> t_cmp_numerico

    # ===================================================
    # 6) EMBEDDINGS: comparación SINTÉTICAS vs ORIGINAL
    # ===================================================
    t_cmp_embeddings = compare_embeddings_all2all()

    # Dependencias correctas:
    # - Sintéticos procesados (t_proc_sint)
    # - Ventanas sintéticas (plot_ventanas)
    # - Consolidado enriquecido (t_enriq)
    # - Gráficos consolidados existen (t_plot_consol)
    [
        t_proc_sint,
        plot_ventanas,
        t_enriq,
        t_plot_consol,
    ] >> t_cmp_embeddings

    # ============================================
    # 7) TTL y Copia de resultados
    # ============================================
    t_ttl = export_ttl_task()
    #t_cmp_embeddings >> t_ttl
    [t_cmp_embeddings, t_cmp_numerico] >> t_ttl

    t_copy = copy_results()
    [
        t_plot_consol,
        t_windows,
        t_ttl,
        t_proc_sint,
        t_cmp_embeddings,
        plot_ventanas,
        t_cmp_numerico,
    ] >> t_copy

    



# Instancia del DAG
dag = anomaly_flow()
