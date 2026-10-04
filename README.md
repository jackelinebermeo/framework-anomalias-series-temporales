# Framework para análisis y comparación de anomalías en series temporales univariantes

Repositorio del trabajo de titulación de la Maestría en Ciencia de Datos de la Universidad de Cuenca.

- **Autora:** Ing. Lorena Jackeline Bermeo Pacheco
- **Director:** Ing. Víctor Hugo Saquicela Galarza, PhD

El framework integra cuatro componentes en un mismo proceso:

1. Detección retrospectiva de anomalías por consenso entre ocho métodos.
2. Generación exploratoria de escenarios sintéticos de corto plazo con siete modelos de predicción.
3. Recuperación de precedentes históricos mediante similitud numérica (DTW) y visual (CLIP).
4. Contextualización con noticias categorizadas, almacenadas en la ontología AnomalyContext (OWL/RDF).

Como ejemplo de aplicación se utiliza la serie diaria de riesgo país del Ecuador (2004–2026), publicada por el Banco Central del Ecuador.

---

## Contenido

1. [Estructura del repositorio](#1-estructura-del-repositorio)
2. [Requisitos](#2-requisitos)
3. [Instalación](#3-instalación)
4. [Configuración](#4-configuración)
5. [Ejecución](#5-ejecución)
6. [Descripción de los componentes](#6-descripción-de-los-componentes)
7. [Archivo de configuración `settings.py`](#7-archivo-de-configuración-settingspy)
8. [Cómo citar](#8-cómo-citar)

---

## 1. Estructura del repositorio

```
framework-anomalias-series-temporales/
├── 1_Recursos/          Serie original, ontología AnomalyContext y corpus de noticias (TTL)
├── 2_Pipeline/
│   ├── componentes/     Módulos de Python del framework
│   ├── dags/            DAGs de Apache Airflow
│   └── Datos/           Resultados de la exploración inicial de la serie
├── 3_Interfaz/          Interfaz web en Streamlit (app.py)
├── 4_DatosSerie/        Resultados completos de la ejecución reportada en la tesis
├── 5_Denuncia/          Denuncia del trabajo de titulación (PDF)
├── 6_Publicaciones/     Artículo derivado de la tesis (CLEI 2026)
├── environment.yml      Entorno de Python (conda)
├── requirements.txt     Paquetes de Python (pip)
└── README.md            Este documento
```

| Carpeta | Contenido |
|---|---|
| `1_Recursos/` | Datos de entrada y base de conocimiento: `Datos.xlsx` (serie original publicada por el Banco Central del Ecuador), `DatosSerie.csv` (serie procesada), `anomaly_core.ttl` (esquema de la ontología AnomalyContext) y `ont_news.ttl` (corpus de noticias). El corpus incluye solo los metadatos de cada noticia (título, fuente, fecha, enlace y categoría) y un resumen breve, sin reproducir el texto completo de los artículos. Se distribuye exclusivamente con fines académicos y de reproducibilidad. |
| `2_Pipeline/` | Código del pipeline: componentes de detección, predicción, comparación y contextualización, y los DAGs que los orquestan. La subcarpeta `Datos/` contiene los resultados de la exploración inicial (distribución, descomposición y gráfico de la serie) y una copia de la serie utilizada en la ejecución. |
| `3_Interfaz/` | Interfaz web para explorar la serie, las anomalías, los escenarios sintéticos, los precedentes recuperados y la información de la ontología. |
| `4_DatosSerie/` | Resultados generados en la ejecución reportada en la tesis: anomalías consolidadas, series sintéticas, archivos de comparación numérica y visual, gráficos y artefactos usados para poblar la base de conocimiento. |
| `5_Denuncia/` | Denuncia del trabajo de titulación, con el problema, la justificación, la pregunta de investigación, la hipótesis y los objetivos. |
| `6_Publicaciones/` | Artículo *Contextualized Anomaly Analysis in Country-Risk Time Series Using Historical Precedents* (CLEI 2026). |

---

## 2. Requisitos

| Componente | Versión utilizada |
|---|---|
| Sistema operativo | Ubuntu 24.04 sobre WSL2 (Windows) |
| Python | 3.10.19 (entorno conda) |
| Apache Airflow | 3.0.1 |
| Streamlit | 1.56.0 |
| Apache Jena Fuseki | 4.8.0 |
| Java (OpenJDK) | 17 |

Todas las librerías de Python, con sus versiones exactas, están en `environment.yml`. Las principales son:

| Uso | Librerías |
|---|---|
| Predicción | prophet, pmdarima, statsmodels, scikit-learn, tensorflow, torch |
| Embeddings visuales y de texto | sentence-transformers, transformers |
| Ontología | rdflib |
| Datos y gráficos | pandas, numpy, pyarrow, matplotlib, plotly |

> El entorno incluye paquetes de CUDA (`nvidia-*-cu12`). Se recomienda instalarlo en Linux o WSL2; en Windows nativo o macOS estos paquetes pueden fallar.

> La primera ejecución descarga desde Hugging Face los modelos `clip-ViT-B-32` (comparación visual) y `paraphrase-multilingual-MiniLM-L12-v2` (comparación semántica), por lo que se requiere conexión a internet.

---

## 3. Instalación

### 3.1 Clonar el repositorio

```bash
git clone https://github.com/jackelinebermeo/framework-anomalias-series-temporales.git
cd framework-anomalias-series-temporales
```

### 3.2 Crear el entorno de Python

```bash
conda env create -f environment.yml
conda activate framework-anomalias
```

### 3.3 Instalar Java y Apache Jena Fuseki

```bash
sudo apt install openjdk-17-jre
```

Descargar Apache Jena Fuseki 4.8.0 desde <https://jena.apache.org/download/> y descomprimirlo.

### 3.4 Configurar Apache Airflow

```bash
export AIRFLOW_HOME=~/airflow
mkdir -p $AIRFLOW_HOME/dags
cp -r 2_Pipeline/dags/* $AIRFLOW_HOME/dags/
cp -r 2_Pipeline/componentes $AIRFLOW_HOME/dags/
```

---

## 4. Configuración

### 4.1 Rutas del archivo `settings.py`

El archivo `2_Pipeline/componentes/settings.py` contiene rutas absolutas del equipo en el que se desarrolló el framework (por ejemplo, `/home/jacky/DatosSerie/`). Antes de ejecutar, ajústelas a su equipo en las secciones:

- **Archivos de entrada:** `ARCHIVO_ORIGEN`, `SERIE_LIMPIA`, `DATOS_SERIE`. La serie original de entrada es `1_Recursos/Datos.xlsx`; `ARCHIVO_ORIGEN` debe apuntar a ese archivo.
- **Carpetas de salida:** `RUTA_SALIDA`, `RUTA_RESULTADOS` y las rutas que dependen de ellas.

Los demás parámetros (umbrales, ventanas y modelos activos) corresponden a la configuración del experimento reportado en la tesis y pueden mantenerse para reproducir los resultados.

### 4.2 Base de conocimiento en Fuseki

1. Iniciar el servidor:

   ```bash
   cd apache-jena-fuseki-4.8.0
   ./fuseki-server
   ```

2. Abrir <http://localhost:3030>.
3. Crear un dataset llamado **`c22`** (Manage → New dataset).
4. Cargar en `c22` (opción *add data*) los dos archivos de `1_Recursos/`:
   - `anomaly_core.ttl`: esquema de la ontología AnomalyContext.
   - `ont_news.ttl`: corpus de noticias.

El framework y la interfaz consultan el endpoint:

```
http://localhost:3030/c22/sparql
```

Este endpoint se define en `settings.py` mediante el parámetro `FUSEKI_ENDPOINT`.

---

## 5. Ejecución

Fuseki debe estar en ejecución antes de iniciar el pipeline o la interfaz.

### 5.1 Pipeline

```bash
conda activate framework-anomalias
airflow standalone
```

Abrir <http://localhost:8080> y ejecutar los DAGs:

| Archivo | Función |
|---|---|
| `airflow_dag_modular.py` | Pipeline principal: exploración, detección, consolidación, generación de escenarios sintéticos, comparación numérica y visual, contextualización y exportación a la ontología. |
| `dag_comprobacion.py` | Experimento de comprobación del framework: compara las 10 anomalías recientes (2025) con el corpus de 152 anomalías históricas (2004–2024), mediante comparación numérica, visual y semántica. Corresponde a la evaluación retrospectiva del Capítulo 4 de la tesis. |

Los resultados se guardan en la carpeta definida por `RUTA_RESULTADOS`. Los resultados de la ejecución reportada en la tesis están en `4_DatosSerie/`.

### 5.2 Interfaz web

```bash
conda activate framework-anomalias
cd 3_Interfaz
streamlit run app.py
```

Abrir <http://localhost:8501>.

---

## 6. Descripción de los componentes

Los módulos están en `2_Pipeline/componentes/`:

| Archivo | Función |
|---|---|
| `settings.py` | Configuración central del framework (ver sección 7). |
| `exploration.py` | Exploración inicial de la serie: limpieza, interpolación de días sin registro, estadísticas descriptivas y descomposición. |
| `detection.py` | Implementación de los ocho métodos de detección: ARIMA, DIF, COUTA, DBSCAN, Isolation Forest, TranAD, TimesNet y TimeGPT. Los nombres DIF, COUTA y TimeGPT son identificadores internos y no corresponden a las arquitecturas homónimas de la literatura. |
| `result.py` | Consolidación de anomalías por consenso, comparación numérica mediante DTW y características estadísticas, y generación del archivo TTL. |
| `prediction.py` | Generación de las series sintéticas con siete modelos (ARIMA, SVR, Random Forest, Prophet, LSTM, N-BEATS y DeepAR), en inferencia directa e iterativa. |
| `embeddings.py` | Comparación visual: embeddings con CLIP, similitud coseno, filtro de tendencia y láminas comparativas. |
| `news.py` | Consulta de noticias en Fuseki, asignación de contexto a las anomalías y cálculo de la similitud semántica entre conjuntos de noticias. |
| `semantic_exporter.py` | Exportación de los resultados a RDF para poblar la ontología AnomalyContext. |
| `plotting.py` | Generación de gráficos de series y anomalías. |

---

## 7. Archivo de configuración `settings.py`

**Ubicación:** `2_Pipeline/componentes/settings.py`

Concentra todos los parámetros del framework. Permite adaptar el framework a otra serie temporal o modificar el experimento sin cambiar el código de los componentes. Los valores actuales son los utilizados en la tesis (Tabla 3.19).

| Sección | Parámetros principales | Valor en la tesis | Descripción |
|---|---|---|---|
| Datos de la serie | `SERIE`, `FRECUENCIA`, `UNIDAD` | Riesgo País, diaria, puntos | Identificación de la serie analizada. |
| Archivos de entrada y salida | `ARCHIVO_ORIGEN`, `RUTA_SALIDA`, `RUTA_RESULTADOS` | Rutas locales | **Deben ajustarse al equipo** (sección 4.1). |
| Consolidación | `CONSENSO_MIN_ALGOS` | 2 | Número mínimo de métodos que deben detectar una anomalía para considerarla válida. |
| | `VENTANA_CTX_DIAS` | 10 | Días de contexto añadidos alrededor de cada anomalía. |
| Detección | `MODELOS_ACTIVOS` | 8 métodos | Métodos de detección que se ejecutan. |
| | `ARIMA_*`, `DIF_*`, `COUTA_*`, `IF_*`, `DBSCAN_*`, `TRANAD_*`, `TIMESNET_*`, `TIMEGPT_*` | Ver archivo | Parámetros de cada método (percentiles, ventanas, umbrales). |
| Predicción | `PRED_MODELOS_ACTIVOS` | 14 configuraciones | Siete modelos en inferencia directa e iterativa. |
| | `PRED_HORIZON` | 30 días | Horizonte de los escenarios sintéticos. |
| | `ARIMA_*`, `SVR_*`, `RF_*`, `LSTM_*`, `PROPHET_*`, `NBEATS_*`, `DEEPAR_*` | Ver archivo | Parámetros de cada modelo. |
| Comparación numérica | `CMP_MIN_SIM` | 0,50 | Umbral de similitud combinada para registrar pares. |
| | `SIM_ALPHA_DTW` | 0,7 | Peso de DTW en la similitud combinada (0,3 para las características). |
| | `DTW_CTX_WINDOW`, `DTW_BAND_RATIO`, `DTW_NORMALIZATION` | 10, 0,10, z-score | Ventana de contexto, banda de Sakoe–Chiba y normalización. |
| Comparación visual | `SIMILARITY_MIN` | 0,85 | Umbral de similitud coseno entre imágenes. |
| | `CMP_MODEL_NAME` | clip-ViT-B-32 | Modelo CLIP utilizado. |
| | `CMP_TOPK_PER_SINTETICA` | 2 | Precedentes conservados por anomalía sintética. |
| Contextualización | `VENTANA_DIAS_NOTICIAS` | 5 días | Margen de búsqueda de noticias alrededor de cada anomalía. |
| | `DTW_MIN_SIM_NOTICIAS` | 0,85 | Umbral de similitud combinada para asignar noticias por la vía numérica. |
| | `FUSEKI_ENDPOINT` | `http://localhost:3030/c22/sparql` | Endpoint SPARQL de la base de conocimiento. |
| Evaluación retrospectiva | `NUM_DIAS_ANOMALIAS` | 270 días | Horizonte que define las anomalías recientes. |

---

## 8. Cómo citar

**Trabajo de titulación:**

> L. J. Bermeo Pacheco, *Framework para análisis y comparación de anomalías en series temporales univariantes*, Trabajo de titulación de Maestría en Ciencia de Datos, Universidad de Cuenca, Cuenca, Ecuador, 2026.

**Artículo:**

> J. Bermeo, E. Sánchez y V. Saquicela, "Contextualized Anomaly Analysis in Country-Risk Time Series Using Historical Precedents", trabajo en progreso, 52.ª Conferencia Latinoamericana de Informática (CLEI 2026).
