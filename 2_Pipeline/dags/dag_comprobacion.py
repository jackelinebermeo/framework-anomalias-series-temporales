import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from airflow.decorators import dag, task
from airflow.sensors.filesystem import FileSensor
from componentes.settings import RUTA_ARCHIVO
from componentes import settings as CFG

default_args = {
    "start_date": datetime(2020, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

@dag(
    dag_id="aa_verif_06",
    schedule=None,
    default_args=default_args,
    catchup=False,
    max_active_tasks=3,
    tags=["comparacion", "dtw", "historico"],
)
def hist_dtw_flow():

    # --------- Sensor ----------
    '''
    wait_file = FileSensor(
        task_id="wait_file",
        fs_conn_id="fs_local",
        filepath=RUTA_ARCHIVO,
        poke_interval=30,
        timeout=300,
        mode="reschedule",
    )
    '''
    @task
    def copy_results(out_csv_dtw: str, out_csv_emb: str, out_csv_news: str,
                    out_csv_num: str, out_csv_graf: str):
        import os, shutil, subprocess, datetime as dt
        from componentes import settings as CFG
        # ... resto igual        

        origen  = CFG.RUTA_SALIDA
        destino = CFG.RUTA_RESPALDO

        print(f"[COPIAR] ========== Copia de resultados ==========")
        print(f"[COPIAR] Origen : {origen} | existe={os.path.isdir(origen)}")
        print(f"[COPIAR] Destino: {destino}")
        print(f"[COPIAR] CSV DTW       : {out_csv_dtw} | existe={os.path.isfile(out_csv_dtw)}")
        print(f"[COPIAR] CSV Embeddings: {out_csv_emb} | existe={os.path.isfile(out_csv_emb)}")

        if not os.path.isdir(origen):
            raise RuntimeError(f"[COPIAR][ERROR] Origen no existe: {origen}")

        try:
            contenido = os.listdir(origen)
            print(f"[COPIAR] Contenido origen ({len(contenido)} items): {contenido[:20]}")
        except Exception as e:
            print(f"[COPIAR][WARN] No se pudo listar origen: {e}")

        if os.path.isdir(destino):
            shutil.rmtree(destino, ignore_errors=True)
        os.makedirs(destino, exist_ok=True)

        cmd = ["bash", "-lc", f'cp -r "{origen}/." "{destino}"']
        res = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[COPIAR][WARN] cp falló ({res.stderr}), usando fallback Python...")
            for nombre in os.listdir(origen):
                src = os.path.join(origen, nombre)
                dst = os.path.join(destino, nombre)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)

        sentinel = os.path.join(
            destino,
            f"_ok_copia_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(sentinel, "w") as f:
            f.write(f"OK {dt.datetime.now().isoformat()}\n")

        print(f"[COPIAR] Copia completada: {origen} -> {destino}")

    #VALIDACION DE EXPERIMENTO COMPARA ANOMALIAS
    @task
    def comparar_historico_vs_reciente_numerica_hist() -> str:
        """
        Divide anomalias_comparacion_numerica.csv en:
          - Históricas: start_date < (max_date - NUM_DIAS_ANOMALIAS)
          - Recientes:  start_date >= (max_date - NUM_DIAS_ANOMALIAS)
        Compara recientes vs históricas con DTW.
        Escribe en CFG.RUTA_COMPARACION_HIST
        """
        import os
        from componentes import settings as CFG
        from componentes import result

        # --- Diagnóstico de rutas ---
        num_dias      = int(getattr(CFG, "NUM_DIAS_ANOMALIAS", 270))
        ruta_cmp      = getattr(CFG, "RUTA_COMPARACION_HIST",
                                os.path.join(CFG.RUTA_RESULTADOS, "ComparacionesHist"))
        ruta_anom     = os.path.join(CFG.RUTA_RESULTADOS, "anomalias_comparacion_numerica.csv")
        ruta_anom_alt = os.path.join(CFG.RUTA_RESULTADOS, "anomalias_consolidado.csv")
        ruta_serie    = os.path.join(CFG.RUTA_SALIDA, "SerieOriginal.csv")

        print(f"[HIST-DTW] ========== Configuración ==========")
        print(f"[HIST-DTW] NUM_DIAS_ANOMALIAS   : {num_dias}")
        print(f"[HIST-DTW] RUTA_RESULTADOS      : {CFG.RUTA_RESULTADOS}")
        print(f"[HIST-DTW] RUTA_SALIDA          : {CFG.RUTA_SALIDA}")
        print(f"[HIST-DTW] RUTA_COMPARACION_HIST: {ruta_cmp}")
        print(f"[HIST-DTW] ========== Archivos de entrada ==========")
        print(f"[HIST-DTW] Anomalías (principal)  : {ruta_anom} | existe={os.path.isfile(ruta_anom)}")
        print(f"[HIST-DTW] Anomalías (alternativo) : {ruta_anom_alt} | existe={os.path.isfile(ruta_anom_alt)}")
        print(f"[HIST-DTW] SerieOriginal.csv       : {ruta_serie} | existe={os.path.isfile(ruta_serie)}")
        print(f"[HIST-DTW] ========== Iniciando comparación ==========")

        out_csv = result.comparar_anomalias_numericas_historicas_dtw(CFG)


        print(f"[HIST-DTW] ========== Resultado ==========")
        print(f"[HIST-DTW] CSV generado : {out_csv}")
        print(f"[HIST-DTW] Existe       : {os.path.isfile(out_csv)}")

        if os.path.isfile(out_csv):
            import pandas as pd
            df = pd.read_csv(out_csv)
            print(f"[HIST-DTW] Filas en CSV                 : {len(df)}")
            if not df.empty:
                print(f"[HIST-DTW] Score máx                : {df['similarity_combined'].max():.4f}")
                print(f"[HIST-DTW] Score mín                : {df['similarity_combined'].min():.4f}")
                print(f"[HIST-DTW] Score prom               : {df['similarity_combined'].mean():.4f}")
                print(f"[HIST-DTW] Anomalías recientes únicas  : {df['id_anomalia_sintetica'].nunique()}")
                print(f"[HIST-DTW] Anomalías históricas únicas : {df['id_anomalia_original'].nunique()}")

        return out_csv
    #VALIDACION DE EXPERIMENTO COMPARA ANOMALIAS — EMBEDDINGS CLIP
    @task
    def comparar_historico_vs_reciente_embeddings_hist(out_csv_dtw: str) -> str:
        """
        Comparación visual por embeddings CLIP:
          - queries : anomalías del periodo recortado (últimos NUM_DIAS_ANOMALIAS días)
          - corpus  : anomalías históricas anteriores al corte
        Lee PNGs desde CFG.RUTA_GRAFICOS (formato NNN_YYYY-MM-DD.png).
        Escribe parquets, CSV y láminas en CFG.RUTA_COMPARACION_HIST.
        Se ejecuta después del DTW (recibe su CSV como dependencia).
        """
        import os
        from componentes import settings as CFG
        from componentes.embeddings import comparar_anomalias_embeddings_historicas_dtw

        num_dias  = int(getattr(CFG, "NUM_DIAS_ANOMALIAS", 90))
        ruta_cmp  = getattr(CFG, "RUTA_COMPARACION_HIST",
                            os.path.join(CFG.RUTA_RESULTADOS, "ComparacionesHist"))

        print(f"[HIST-EMB] ========== Configuración ==========")
        print(f"[HIST-EMB] NUM_DIAS_ANOMALIAS   : {num_dias}")
        print(f"[HIST-EMB] ANOMALIAS_CONSOLIDADO: {CFG.ANOMALIAS_CONSOLIDADO} | existe={os.path.isfile(CFG.ANOMALIAS_CONSOLIDADO)}")
        print(f"[HIST-EMB] RUTA_GRAFICOS        : {CFG.RUTA_GRAFICOS} | existe={os.path.isdir(CFG.RUTA_GRAFICOS)}")
        print(f"[HIST-EMB] RUTA_COMPARACION_HIST: {ruta_cmp}")
        print(f"[HIST-EMB] CMP_MODEL_NAME       : {CFG.CMP_MODEL_NAME}")
        print(f"[HIST-EMB] CMP_MIN_SIM          : {CFG.CMP_MIN_SIM}")
        print(f"[HIST-EMB] CMP_TOPK             : {CFG.CMP_TOPK_PER_SINTETICA}")
        print(f"[HIST-EMB] ========== Iniciando comparación visual ==========")

        resultado = comparar_anomalias_embeddings_historicas_dtw(CFG)

        out_csv_emb = resultado.get("csv", "")

        print(f"[HIST-EMB] ========== Resultado ==========")
        print(f"[HIST-EMB] CSV embeddings  : {out_csv_emb}")
        print(f"[HIST-EMB] Existe          : {os.path.isfile(out_csv_emb)}")
        print(f"[HIST-EMB] fecha_corte     : {resultado.get('fecha_corte', '?')}")
        print(f"[HIST-EMB] n_queries       : {resultado.get('n_queries', '?')}")
        print(f"[HIST-EMB] n_corpus        : {resultado.get('n_corpus', '?')}")
        print(f"[HIST-EMB] laminas_dir     : {resultado.get('laminas_dir', '?')}")

        if os.path.isfile(out_csv_emb):
            import pandas as pd
            df = pd.read_csv(out_csv_emb)
            print(f"[HIST-EMB] Filas en CSV                    : {len(df)}")
            if not df.empty:
                print(f"[HIST-EMB] Score máx                   : {df['similarity_cosine'].max():.4f}")
                print(f"[HIST-EMB] Score mín                   : {df['similarity_cosine'].min():.4f}")
                print(f"[HIST-EMB] Score prom                  : {df['similarity_cosine'].mean():.4f}")
                print(f"[HIST-EMB] Queries únicas              : {df['synthetic_id'].nunique()}")
                print(f"[HIST-EMB] Históricas únicas           : {df['original_id'].nunique()}")

        return out_csv_emb

    # VALIDACION DE EXPERIMENTO COMPARA NOTICIAS DE ANOMALIAS SIMILARES
    @task
    def comparar_noticias_por_pares_historicas(out_csv_emb: str) -> str:
        """
        Para cada par de anomalías similares (ya calculado por embeddings),
        consulta Fuseki por las noticias de cada período y calcula
        similitud semántica entre ellas.
        Escribe CFG.RUTA_COMPARACION_HIST/similitudes_noticias_hist.csv
        """
        import os
        from componentes import settings as CFG
        from componentes.news import comparar_noticias_por_pares_historicas

        print(f"[NEWS] ========== Comparación de noticias ==========")
        print(f"[NEWS] CSV embeddings entrada: {out_csv_emb} | existe={os.path.isfile(out_csv_emb)}")

        out_csv = comparar_noticias_por_pares_historicas(CFG)

        print(f"[NEWS] CSV noticias generado: {out_csv}")
        return out_csv
    
    # GENERA NOTICIAS PARA ANOMALIAS SINTETICAS PREDICHAS
    @task
    def generar_contexto_sinteticas() -> str:
        import os
        from componentes import settings as CFG
        from componentes.news import generar_noticias_anomalias_sinteticas

        print(f"[SYN-NEWS] ========== Contexto anomalías sintéticas ==========")
        out_csv = generar_noticias_anomalias_sinteticas(CFG)
        print(f"[SYN-NEWS] CSV generado: {out_csv} | existe={os.path.isfile(out_csv)}")
        return out_csv

    # GENERA NOTICIAS PARA PARES SINTETICAS -COMPARACION numerica
    @task
    def generar_noticias_comparacion_num(out_csv_dtw: str) -> str:
        import os
        from componentes import settings as CFG
        from componentes.news import generar_noticias_comparacion_numerica
        out_csv = generar_noticias_comparacion_numerica(CFG)
        print(f"[DTW-NEWS] CSV generado: {out_csv} | existe={os.path.isfile(out_csv)}")
        return out_csv
    
    ###GENERA NOTICIAS PARA ANOMALIAS SINTETICAS — COMPARACION grafica
    @task
    def generar_noticias_comparacion_graf(out_csv_emb: str) -> str:
        import os
        from componentes import settings as CFG
        from componentes.news import generar_noticias_comparacion_grafica
        out_csv = generar_noticias_comparacion_grafica(CFG)
        print(f"[GRAF-NEWS] CSV generado: {out_csv} | existe={os.path.isfile(out_csv)}")
        return out_csv

    @task
    def comparar_noticias_dtw_hist(out_csv_dtw: str) -> str:
        """
        Para cada par de anomalías similares numéricamente (top N por anomalía reciente),
        consulta Fuseki por las noticias de cada período y calcula
        similitud semántica entre ellas.
        Escribe CFG.RUTA_COMPARACION_HIST/similitudes_noticias_dtw_hist.csv
        """
        import os
        from componentes import settings as CFG
        from componentes.news import comparar_noticias_por_pares_dtw_hist

        print(f"[NEWS-DTW] ========== Comparación de noticias DTW ==========")
        print(f"[NEWS-DTW] CSV DTW entrada: {out_csv_dtw} | existe={os.path.isfile(out_csv_dtw)}")

        out_csv = comparar_noticias_por_pares_dtw_hist(CFG)

        print(f"[NEWS-DTW] CSV noticias generado: {out_csv}")
        return out_csv  


   
# --------- Orquestación ----------
# --------- Orquestación ----------
    t_dtw      = comparar_historico_vs_reciente_numerica_hist()
    t_emb      = comparar_historico_vs_reciente_embeddings_hist(t_dtw)
    t_news     = comparar_noticias_por_pares_historicas(t_emb)
    t_news_dtw = comparar_noticias_dtw_hist(t_dtw)
   ## t_news_ind = comparar_noticias_ind(t_emb)
  # t_syn      = generar_contexto_sinteticas()
    t_graf     = generar_noticias_comparacion_graf(t_emb)
    t_num      = generar_noticias_comparacion_num(t_dtw)
    t_copy     = copy_results(t_dtw, t_emb, t_news, t_num, t_graf)
dag = hist_dtw_flow()