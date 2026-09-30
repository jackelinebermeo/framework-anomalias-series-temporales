import os

#-------------------------------------------
#DATOS SERIE
#-------------------------------------------
SERIE= "Riesgo País"
FRECUENCIA="DIARIA"
UNIDAD="PUNTOS"
IRI= "anom:series_riesgo_pais_1"

# ------- Archivos de entrada
ARCHIVO_ORIGEN = "/tmp/Datos.xlsx"
RUTA_ARCHIVO = ARCHIVO_ORIGEN
SERIE_LIMPIA = "/home/jacky/DatosSerie/SerieOriginal.csv"
DATOS_SERIE="/tmp/DatosSerie.csv"

#Exploracion inicial
GRAFICO_KDE = "/home/jacky/DatosSerie/distribucion_kde.png"
GRAFICO_SERIE = "/home/jacky/DatosSerie/serie_temporal.png"

#-------------------Carpetas de salida
RUTA_SALIDA="/home/jacky/DatosSerie"
RUTA_RESULTADOS="/home/jacky/DatosSerie/Resultados/"
RUTA_RESPALDO="/mnt/d/Jacky/Tesis/Resultados"
RUTA_PREDICCION="/home/jacky/DatosSerie/Resultados/Prediccion"
RUTA_COMPARACION="/home/jacky/DatosSerie/Resultados/ComparacionesEmbeddings"
RUTA_COMPARACION_HIST = "/home/jacky/DatosSerie/Resultados/ComparacionesHist"   # histórico vs reciente

#--------------------Resultados específicos
RUTA_ANOMALIAS_METODO = "/home/jacky/DatosSerie/Resultados/DetectadoXMetodo"
#---------------------Consolidado
RUTA_GRAFICOS_CONSOLIDADOS="/home/jacky/DatosSerie/Resultados/GraficosConsolidado/GraficosOriginalDatos"
ANOMALIAS_CONSOLIDADO = "/home/jacky/DatosSerie/Resultados/anomalias_consolidado.csv"
ANOMALIAS_PREDICCION_CONSOLIDADO = "/home/jacky/DatosSerie/Resultados/anomalias__prediccion_consolidado.csv"
RUTA_ANOMALIAS_LIMPIAS_CSV="/home/jacky/DatosSerie/Resultados/GraficosConsolidado/AnomaliasOriginalesLimpias"

#------------Graficos
RUTA_GRAFICOS="/home/jacky/DatosSerie/Resultados/GraficosConsolidado/GraficosOriginalLimpios"
RUTA_GRAFICOS_ANOMALIAS_SINTETICAS = "/home/jacky/DatosSerie/Resultados/AnomaliasSinteticas"
RUTA_Compara_PNG="/home/jacky/DatosSerie/Resultados/GraficosConsolidado/SimilitudPNG"
ANOMALIAS_SINTETICAS_LIMPIAS="/home/jacky/DatosSerie/Resultados/GraficosConsolidado/GraficosSinteticoLimpios"
ANOMALIAS_SINTETICAS = "/home/jacky/DatosSerie/Resultados/GraficosConsolidado/GraficosSinteticosDatos"

#------------------CSV
ANOMALIAS_SINTETICAS_CSV = "/home/jacky/DatosSerie/Resultados/Prediccion/anomalias_sinteticas_csv"
ANOMALIAS_SINTETICAS_CSV_LIMPIAS = "/home/jacky/DatosSerie/Resultados/Prediccion/anomalias_sinteticas_limpias_csv"
#------------------DTW

#-----------Ontologia
DATOS_ONTOLOGIA="/home/jacky/DatosSerie/Resultados/Ontologia"

#-----------------------ENTRENAMIENTO
TRAIN_WINDOW = 730  # None o 0 para usar toda la historia

#RUTA_GRAFICOS_ANOMALIAS = "/home/jacky/DatosSerie/Resultados/GraficoXMetodo"

#RUTA_ARIMA = "/home/jacky/DatosSerie/Resultados/PorMetodo/Arima"
EXPORTAR_CON_TIMESTAMP = False
DATOS_EXTRA=True
SERIE_ID='001'

# Gráficos
GRAFICOS_FORMAT = "png"
GRAFICOS_DPI = 150
GRAFICOS_OVERWRITE = False
GRAFICOS_YLIMS = "global"  # "global" | "auto"

# FECHAS Y ARCHIVOS
# Formato de salida para TODOS los payloads a detectores
OUT_DATE_FMT = "%d/%m/%Y"

# Encender/apagar la escritura del archivo DEBUG_payload_sintetica_*.json
SAVE_DEBUG_PAYLOAD = True
# ------------------------------------------
# ⚙️ Parámetros de procesamiento
# ------------------------------------------
NORMALIZAR_DATOS = False
IMPUTACION_KNN_K = 5
GRAFICOS = True
CONSENSO_MIN_ALGOS = 2
VENTANA_CTX_DIAS = 10
PRED_ROUND_DECIMALS = 2 

# ------------------------------------------
# 📊 Configuración de detección
# ------------------------------------------

ANOMALIAS_MIN_DIST = 0
# ARIMA
ARIMA_ORDER = (2, 1, 1)  # p, d, q
ARIMA_THRESHOLD_METHOD = "percentile"
ARIMA_PERCENTIL = 96.0   # 0..100 (float/int)


#RUN_DIF
DIF_THRESHOLD_METHOD = "percentile"   # "percentile" | "std"
DIF_PERCENTIL = 96.0                  # si usas percentil

# COUTA / CUSUM - Page-Hinkley
COUTA_DELTA = 50   # tolerancia al drift (en unidades de la serie)
COUTA_LAMBDA = 4000    # umbral de disparo: menor => más sensible, mayor => más conservador

#IsolationForest
IF_CONTAMINATION = 0.005
IF_KEEP_FRACTION = 1

IF_N_ESTIMATORS = 300
IF_MAX_SAMPLES  = "auto"  # o 256 en series cortas
IF_RANDOM_STATE = 42

#DBSCAN
DBSCAN_FEATURE       = 'zresid'
DBSCAN_TREND_WIN     = 45     # suaviza un poco más el residuo
DBSCAN_EPS           = 0.01
DBSCAN_MIN_SAMPLES   = 3
DBSCAN_KEEP_FRACTION = 0.75

#TRANAD
TRANAD_WINDOW        = 24     # 12–48 típico (según frecuencia)
TRANAD_EPOCHS        = 10
TRANAD_HIDDEN        = 32
TRANAD_LR            = 1e-3
TRANAD_BATCH         = 64
TRANAD_STANDARDIZE   = True   # robust scale (mediana/IQR)
TRANAD_PERCENTIL     = 96   # ↑ = menos anomalías, ↓ = más
TRANAD_KEEP_FRACTION = 0.20    # 0.5–1.0
TRANAD_SMOOTH        = 3      # 1=sin suavizado; 3–5 limpia ruido del score


#TIMESNET
# Ventaneo y red
TIMESNET_WINDOW       = 24     # 24–96 típico según frecuencia
TIMESNET_EPOCHS       = 8
TIMESNET_HIDDEN       = 32
TIMESNET_N_BLOCKS     = 3
TIMESNET_KERNEL_SIZE  = 7
TIMESNET_LR           = 1e-3
TIMESNET_BATCH        = 64

# Selección de anomalías
TIMESNET_STANDARDIZE  = True   # robust scale mediana/IQR
TIMESNET_PERCENTIL    = 96   # ↑ = menos anomalías, ↓ = más
TIMESNET_KEEP_FRACTION= 0.20    # 0.5–1.0
TIMESNET_SMOOTH       = 3      # mediana del score (1 = sin suavizado)


#TIMEGPT
TIMEGPT_WINDOW            = 12      # tamaño de ventana para calcular energía
TIMEGPT_TREND_WIN         = 9      # ventana para mediana móvil (tendencia)
TIMEGPT_SEASONAL_PERIOD   = 1       # 1 = sin estacionalidad, 7/12/24 según tu frecuencia
TIMEGPT_STANDARDIZE       = True    # escalar residuo con mediana/IQR (True recomendado)
TIMEGPT_PERCENTIL         = 96   # umbral (más bajo = más sensible)
TIMEGPT_KEEP_FRACTION     = 0.20    # fracción top por severidad (0..1]; 0.3=30%
TIMEGPT_SMOOTH            = 3  


# ------------------------------------------
# 🧪 Comparaciones entre anomalías (opcional)
# ------------------------------------------
EVALUAR_SIMILITUDES_DTW = True
EVALUAR_SIMILITUDES_EMBEDDING = True

# ---- Umbral de similitud ----
CMP_MIN_SIM = 0.50 # solo se consideran pares >= 0.75
DTW_MODE = "context_window"

# ---- Parámetros de generación de PNGs ----
PNG_LIMPIO_FIGSIZE = (6, 3.2)
PNG_LIMPIO_DPI = 150
PNG_LIMPIO_LINEWIDTH = 1.5
PNG_LIMPIO_TIGHTPAD = 0.0
PNG_LIMPIO_AXIS_OFF = True

# ---- Configuración de comparación Embedding → Gráficos ----
CMP_TOPK_PER_SINTETICA = 2             # cantidad de originales mostradas por cada sintética
TOPK_SIMILARES_CSV = 5                  # Número máximo de series sintéticas que se mostrarán por anomalía

CMP_GENERAR_LAMINAS_PER_ORIGINAL = False
CMP_GENERAR_LAMINAS_PER_SINTETICA = True
CMP_LAMINAS_OUTPUT_DIR  = "/home/jacky/DatosSerie/Resultados/ComparacionesEmbeddings"
CMP_CANVAS_GAP_PX = 24
CMP_THUMB_WIDTH_PX = 480
CMP_SHOW_LABELS = True
CMP_SHOW_PERCENT = True
CMP_MODEL_NAME = "clip-ViT-B-32"
CMP_MODEL_WEIGHTS = "laion2b_s34b_b79k"
SIMILARITY_MIN=0.85
# Modelo CLIP para extracción de embeddings


# ------------------------------------------
# 🆔 Formato de ID para anomalías
# ------------------------------------------
PREFIJO_ID_ANOMALIA = "AN"
FORMATO_ID = "{:03d}"


#----------------------------
#Prediccion
#------------------------

PRED_PLOT=100
PRED_HORIZON = 30   # horizonte de días por defecto para pronóstico
PRED_LEVEL = 0.9
ANOM_K_SIGMA = 3.0          # umbral robusto (3 σ-MAD)
PRED_BACKTEST_STEPS = 30    # cuántos puntos recientes usar para el backtest
PRED_FUENTE = "sintetica"   # etiqueta de fuente para los CSV de puntos
WINDOW_LEN = 32
ARIMA_ORDER_predict = (2, 1, 1)       # parte no estacional 2,1,1
ARIMA_TREND_predict = "n"             # o "t", según prefieras drift/pendiente
ARIMA_SEASONAL_ORDER_predict = (0, 0, 0, 0)  # (P,D,Q,m) → m=7 si hay patrón semanal
ARIMA_AUTO_ORDER = True   # False para usar ARIMA_ORDER_predict fijo
ARIMA_TRAIN_YEARS = 5

#ARIMA
MAXITER_ARIMA = 200


# SVR
SVR_KERNEL = "rbf"
SVR_C = 100.0
SVR_EPSILON = 0.03
SVR_GAMMA = "scale"   # si sigue plano, prueba 0.1 o 0.05
SVR_TOL = 1e-3
SVR_MAX_ITER = 2000
N_JOBS = -1
SVR_STRIDE = 1        # déjalo en 1 para no perder eventos raros
SVR_CACHE_MB = 1000.0

# RF
RF_FEAT_AUG = True            # activar enriquecimiento de features
RF_N_ESTIMATORS = 500         # 400–600 si el tiempo lo permite
RF_MAX_DEPTH = None           # sin tope → captura no-linealidades
RF_MIN_SAMPLES_LEAF = 1       # hojas pequeñas → más detalle
RF_MAX_FEATURES = "sqrt"      # evita promediar en exceso
RF_BOOTSTRAP = True           # aleatoriedad útil para diversidad


# ─────────────────────────────────────────────────────────
# LSTM — Modo "SIN FRENOS" (Predicción real de picos/anomalías)
# ─────────────────────────────────────────────────────────

# Arquitectura y entrenamiento base
LSTM_HIDDEN = 64                 # neuronas en la capa LSTM
LSTM_DROPOUT = 0.2               # dropout estándar
LSTM_EPOCHS = 100                # número de épocas
LSTM_BATCH_SIZE = 32             # tamaño de batch
LSTM_EARLY_STOP_PATIENCE = 10    # paciencia para early stopping
LSTM_LOSS = "huber"              # pérdida robusta (tolera picos)
LSTM_OPTIMIZER = "adam"          # optimizador base
LSTM_DENSE = 32                  # capa densa intermedia para mayor capacidad no lineal
LSTM_REC_DROPOUT = 0.0           # sin recurrent dropout (mayor libertad temporal)
LSTM_CLIPNORM = 1.0              # clipping de gradiente básico (estabilidad numérica)
LSTM_USE_RLR = True              # reduce learning rate on plateau
LSTM_RLR_FACTOR = 0.5            # factor de reducción
LSTM_RLR_PATIENCE = 6            # paciencia para reducir LR



# Escalado (robusto, conserva picos)
LSTM_SCALER = "robust"           # usa mediana y rango intercuartílico
LSTM_MINMAX_RANGE = (0.0, 1.0)   # no se usa en robust, se mantiene por compatibilidad

# Preprocesado robusto (sin clipping ni detrend)
LSTM_DETREND_MEDIAN_WIN = 0      # sin detrend (mantiene forma real de la serie)
LSTM_CLIP_TARGETS_MAD_K = 0      # sin recorte por MAD
LSTM_CLIP_TARGETS_QTL = None     # sin recorte por cuantiles

# Estacionarización
LSTM_USE_DIFF1 = False           # aprende nivel real, no diferencia (permite picos)

# Validación
LSTM_VAL_SPLIT = 0.2             # fracción de validación
LSTM_EARLY_STOP_MONITOR = "val_loss"

# Feedback y frenos (completamente liberado)
LSTM_FEEDBACK_DAMPING = 1.0      # sin amortiguación (permite saltos naturales)
LSTM_RESID_SHRINK = 0.0          # sin encogimiento del paso
LSTM_PERSISTENCE_MIX = 0.30       # no mezcla con persistencia (aplana)
LSTM_MAX_STEP_SIGMAS = 3.0     # sin límite efectivo de paso (sin clamp)
LSTM_CLAMP_USE_MAD = False       # desactiva clamp robusto (deja oscilar libremente)
LSTM_TRAIN_USE_LAST_FRAC = 0.4       # usa ~40% final para entrenar
LSTM_TRAIN_DECAY_GAMMA = 0.97        # pondera más ventanas recientes (0.97–0.995)




# ── Salida estándar ──
PRED_FUENTE_ORIG = "original"   # histórico
# PRED_FUENTE ya definido = "sintetica"

# Semilla opcional (para reproducibilidad)
RANDOM_SEED_PREDICT = 123

# ─────────────────────────────────────────────────────────
# Prophet – Parámetros sugeridos en settings.py
# ─────────────────────────────────────────────────────────
PROPHET_N_CHANGEPOINTS    = 25        # ↑ más quiebres posibles
PROPHET_CHANGEPOINT_PRIOR = 0.10      # ↑ permite curvar más
PROPHET_CHANGEPOINT_RANGE = 0.9      # ↑ coloca changepoints también cerca del final

# Estacionalidad
PROPHET_YEARLY_SEASONALITY = True
PROPHET_WEEKLY_SEASONALITY = True     # solo si tu serie es diaria; si no, pon False
PROPHET_DAILY_SEASONALITY  = False
PROPHET_SEASONALITY_MODE   = "additive"

# Menos suavizado (más “vida” en la curva)
PROPHET_SEASONALITY_PRIOR  = 2.0      # ↓ de 10 → deja ondular
PROPHET_HOLIDAYS_PRIOR     = 2.0

# Intervalo (si lo usas)
PROPHET_INTERVAL_WIDTH     = 0.8


# N-BEATS (ajustado para detectar más anomalías y correr más rápido)
NBEATS_STACKS = 2
NBEATS_BLOCKS_PER_STACK = 3
NBEATS_WIDTH = 512          # más capacidad para picos
NBEATS_FC_LAYERS = 4
NBEATS_ACTIVATION = "relu"
NBEATS_DROPOUT = 0.05       # evita sobre-suavizar
NBEATS_LOSS = "huber"  # sensible a anomalías
NBEATS_OPTIMIZER = "adam"
NBEATS_EPOCHS = 150          # reduce tiempo, usa early stop
NBEATS_BATCH_SIZE = 64
NBEATS_VAL_SPLIT = 0.10
NBEATS_EARLY_STOP_PATIENCE = 5


# DeepAR
DEEPAR_UNITS = 64
DEEPAR_LAYERS = 2
DEEPAR_DROPOUT = 0.1
DEEPAR_EPOCHS = 30
DEEPAR_BATCH_SIZE = 64
DEEPAR_VAL_SPLIT = 0.15
DEEPAR_EARLY_STOP_PATIENCE = 4  # <-- cambia de 6 a 4

# --- Similaridad combinada ---
SIM_USE_COMBINED = True              # Activa el uso de la métrica combinada
SIM_COMB_METHOD = "weighted_mean"    # Opciones: "weighted_mean", "product", "geometric"
SIM_ALPHA_DTW = 0.7                  # Peso de DTW en la combinación (0–1)
#_THRESHOLD = 0.30                 # Umbral mínimo aplicado sobre similarity_combined

# --- DTW ---
DTW_CTX_WINDOW = 10          # Días/puntos de contexto antes y después
DTW_BAND_RATIO = 0.10        # Banda Sakoe-Chiba como % de longitud
DTW_NORMALIZATION = "zscore" # "zscore", "l2", "none"

# --- Features ---
FEAT_USE = True
FEAT_NORMALIZATION = "zscore"    # "zscore", "minmax", "none"
FEAT_DISTANCE = "euclidean"      # "euclidean" o "cosine"
FEAT_THRESHOLD = 0.0             # Umbral mínimo (0 si no quieres filtro previo)
# ------------------------------------------
# 🔍 Modelos activos habilitados
# ------------------------------------------
MODELOS_ACTIVOS = [
    
    "ARIMA",
    "DIF",
    "IFOREST",
    "COUTA",
    "DBSCAN",
    "TRANAD",
    "TIMESNET",
    "TIMEGPT",
]

# (Opcional) Activar runners LSTM
PRED_MODELOS_ACTIVOS = [
  "ARIMA_DIRECT", "ARIMA_FEEDBACK",
  "SVR_FEEDBACK","SVR_DIRECT",
  "RF_FEEDBACK", "RF_DIRECT",
  "PROPHET_DIRECT", "PROPHET_FEEDBACK",
  "LSTM_FEEDBACK", "LSTM_DIRECT",
  "NBEATS_DIRECT","NBEATS_FEEDBACK",
  "DEEPAR_DIRECT", "DEEPAR_FEEDBACK"
]
#----------------------------------------
#RUTAS VALIDACION
#----------------------------------------
NUM_DIAS_ANOMALIAS=270
num_news:30
VENTANA_DIAS_NOTICIAS = 5 #— margen de búsqueda de noticias
TOP_SIMILITUDES_SINTETICAS = 5 #— cuántos similares por sintética
DTW_MIN_SIM_NOTICIAS = 0.85
DTW_TOP_N_NOTICIAS = 10


RUTA_PREDICCION_BT = "/home/jacky/DatosSerie_BT/Resultados/Prediccion"
RUTA_SALIDA_ANALISIS      ="/home/jacky/DatosSerie_BT"
RUTA_RESULTADOS_ANALISIS  ="/home/jacky/DatosSerie_BT/Resultados/"
RUTA_PREDICCION_ANALISIS  ="/home/jacky/DatosSerie_BT/Resultados/Prediccion"
RUTA_RESPALDO_ANALISIS    ="mnt/d/Jacky/Tesis/Resultados_BT"
RUTA_RESULTADOS_BT= "/home/jacky/DatosSerie_BT/Resultados"
#DIAS_ANALISIS_PREDICCION =30