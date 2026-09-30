import os
import re
import sys
from datetime import datetime, timedelta
import csv
from pathlib import Path
from componentes import settings as CFG  # Mantengo tu enfoque: no cambio nombres de variables


# componentes/contexto.py
from pathlib import Path
import pandas as pd

SERIE_ORIGINAL_NOMBRE = None

def cargar_nombre_serie_original_desde_csv(serie_limpia_path: Path) -> str:
    """
    Recupera el nombre de la serie original desde el CSV limpio.

    Prioridad:
    1) Columna 'serie', 'serie_nombre' o 'nombre_serie' (si existe)
    2) Comentario especial en la primera fila (# Serie: ...)
    3) Nombre del archivo (fallback seguro)
    """
    global SERIE_ORIGINAL_NOMBRE

    serie_limpia_path = Path(serie_limpia_path)

    # 1) Intentar desde columnas
    try:
        df = pd.read_csv(serie_limpia_path, nrows=5)
        for col in ("serie", "serie_nombre", "nombre_serie"):
            if col in df.columns:
                val = df[col].dropna()
                if not val.empty:
                    SERIE_ORIGINAL_NOMBRE = str(val.iloc[0])
                    return SERIE_ORIGINAL_NOMBRE
    except Exception:
        pass

    # 2) Intentar desde comentario en primera línea
    try:
        with open(serie_limpia_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line.startswith("#") and "serie" in first_line.lower():
                SERIE_ORIGINAL_NOMBRE = first_line.lstrip("#").strip()
                return SERIE_ORIGINAL_NOMBRE
    except Exception:
        pass

    # 3) Fallback: nombre del archivo
    SERIE_ORIGINAL_NOMBRE = serie_limpia_path.stem
    return SERIE_ORIGINAL_NOMBRE

def _export_series():
    import csv
    import os

    print("[SERIES] Iniciando _export_series()")

    input_csv = getattr(CFG, "DATOS_SERIE", None)
    datos_onto = getattr(CFG, "DATOS_ONTOLOGIA", None)

    print(f"[SERIES] CFG.DATOS_SERIE = {input_csv}")
    print(f"[SERIES] CFG.DATOS_ONTOLOGIA = {datos_onto}")

    if not input_csv:
        raise ValueError("[SERIES][ERROR] CFG.DATOS_SERIE no está definido o está vacío")
    if not datos_onto:
        raise ValueError("[SERIES][ERROR] CFG.DATOS_ONTOLOGIA no está definido o está vacío")

    # Normalizar por si vinieran comillas desde settings/ENV
    input_csv = str(input_csv).strip().strip('"')
    datos_onto = str(datos_onto).strip().strip('"')

    print(f"[SERIES] DATOS_SERIE (normalizado) = {input_csv}")
    print(f"[SERIES] DATOS_ONTOLOGIA (normalizado) = {datos_onto}")

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"[SERIES][ERROR] No existe el archivo: {input_csv}")

    try:
        size_bytes = os.path.getsize(input_csv)
        print(f"[SERIES] Tamaño CSV = {size_bytes} bytes")
        if size_bytes == 0:
            raise ValueError("[SERIES][ERROR] El CSV existe pero está vacío (0 bytes). No hay nada que exportar.")
    except Exception as e:
        print(f"[SERIES][WARN] No pude obtener tamaño del CSV: {e}")

    def _pick_delimiter(header_line: str, sample: str) -> str:
        if ";" in header_line and header_line.count(";") >= header_line.count(","):
            return ";"
        if "\t" in header_line:
            return "\t"
        try:
            return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]).delimiter
        except Exception:
            if ";" in sample and sample.count(";") >= sample.count(","):
                return ";"
            if "\t" in sample:
                return "\t"
            return ","

    def _normalize_key(k: str) -> str:
        return (k or "").strip().lower()

    def _norm_row_keys(row: dict) -> dict:
        return {_normalize_key(k): (v if v is not None else "") for k, v in (row or {}).items()}

    def _escape_ttl_literal(s: str) -> str:
        return str(s).replace("\\", "\\\\").replace('"', '\\"').strip()

    # === FIX: usar el source canónico definido en tu ontología ===
    # Por defecto: anom:src_original (no anom:OriginalSource)
    ORIGINAL_SOURCE_IRI = getattr(CFG, "ORIGINAL_SOURCE_IRI", "anom:src_original")

    # Labels/atributos alineados a tu vocabulario
    ORIGINAL_SOURCE_LABEL_ES = getattr(CFG, "ORIGINAL_SOURCE_LABEL_ES", "Fuente original")
    ORIGINAL_SOURCE_LABEL_EN = getattr(CFG, "ORIGINAL_SOURCE_LABEL_EN", "Original source")
    ORIGINAL_SOURCE_TYPE = getattr(CFG, "ORIGINAL_SOURCE_TYPE", "observada")

    # detectar delimitador
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        header_line = f.readline()
        sample_rest = f.read(4096)
        sample = (header_line or "") + (sample_rest or "")

    print(f"[SERIES] Header (raw) = {repr(header_line[:200])}")
    delimiter = _pick_delimiter(header_line, sample)
    print(f"[SERIES] Delimitador detectado = {repr(delimiter)}")
    print(f"[SERIES] Muestra inicial (primeros 200 chars) = {repr(sample[:200])}")

    ttl_blocks = []
    row_num = 1
    previews = 0

    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        # si quedó pegado con ';' adentro, reintentar
        if reader.fieldnames and len(reader.fieldnames) == 1 and ";" in (reader.fieldnames[0] or ""):
            print("[SERIES][WARN] Fieldnames pegados; reintentando con delimitador ';'")
            f.seek(0)
            delimiter = ";"
            reader = csv.DictReader(f, delimiter=delimiter)

        print(f"[SERIES] Columnas detectadas ({len(reader.fieldnames or [])}): {reader.fieldnames}")

        fieldnames = [_normalize_key(c) for c in (reader.fieldnames or [])]

        required = {"serie_id", "iri", "nombre_es", "frecuencia", "unidad"}
        missing = [c for c in required if c not in set(fieldnames)]
        if missing:
            raise ValueError(
                f"[SERIES][ERROR] Faltan columnas requeridas en {input_csv}: {missing}. "
                f"Columnas detectadas: {reader.fieldnames}. Delimitador usado: {repr(delimiter)}"
            )

        for raw_row in reader:
            row_num += 1

            # Preview de filas (para confirmar lectura)
            if previews < 3:
                print(f"[SERIES][ROW-PREVIEW {row_num}] {raw_row}")
                previews += 1

            row = _norm_row_keys(raw_row)

            serie_id = (row.get("serie_id") or "").strip()
            iri = (row.get("iri") or "").strip()
            nombre_es = (row.get("nombre_es") or "").strip()
            frecuencia = (row.get("frecuencia") or "").strip()
            unidad = (row.get("unidad") or "").strip()

            if not serie_id or not iri or not nombre_es or not frecuencia or not unidad:
                raise ValueError(
                    f"[SERIES][ERROR] Fila incompleta en {input_csv} (línea aprox. {row_num}). "
                    f"Valores: serie_id={serie_id!r}, iri={iri!r}, nombre_es={nombre_es!r}, "
                    f"frecuencia={frecuencia!r}, unidad={unidad!r}. Fila cruda: {raw_row}"
                )

            # Sujeto: IRI absoluta entre <>, o prefijo si ya viene como anom:...
            subj = f"<{iri}>" if iri.startswith(("http://", "https://")) else iri

            series_id_literal = _escape_ttl_literal(serie_id)
            nombre_es_ttl = _escape_ttl_literal(nombre_es)
            frecuencia_ttl = _escape_ttl_literal(frecuencia)
            unidad_ttl = _escape_ttl_literal(unidad)

            ttl_blocks.append(
                f"""{subj}
  rdf:type anom:Series ;
  anom:seriesId "{series_id_literal}" ;
  anom:hasSource {ORIGINAL_SOURCE_IRI} ;
  anom:seriesName "{nombre_es_ttl}" ;
  rdfs:label "{nombre_es_ttl}"@es ;
  anom:frequency "{frecuencia_ttl}" ;
  anom:unit "{unidad_ttl}" .
"""
            )

    print(f"[SERIES] Total TTL blocks generados = {len(ttl_blocks)}")

    if len(ttl_blocks) == 0:
        raise ValueError("[SERIES][ERROR] No se generó ninguna serie. Revisa el CSV y los previews de filas.")

    # === FIX: Source block alineado al vocabulario (incluye ES/EN y sourceType) ===
    source_label_es_ttl = _escape_ttl_literal(ORIGINAL_SOURCE_LABEL_ES)
    source_label_en_ttl = _escape_ttl_literal(ORIGINAL_SOURCE_LABEL_EN)
    source_type_ttl = _escape_ttl_literal(ORIGINAL_SOURCE_TYPE)

    source_block = f"""{ORIGINAL_SOURCE_IRI}
  rdf:type anom:Source ;
  rdfs:label "{source_label_es_ttl}"@es ;
  rdfs:label "{source_label_en_ttl}"@en ;
  anom:sourceType "{source_type_ttl}" .
"""

    ttl = (
        "@prefix anom: <http://w3id.org/anomaly-core#> .\n"
        "@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
        "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .\n\n"
        + source_block
        + "\n"
        + "\n".join(ttl_blocks)
    )

    os.makedirs(datos_onto, exist_ok=True)
    output_path = os.path.join(datos_onto, "1series.ttl")
    print(f"[SERIES] Ruta de salida = {output_path}")

    # sobrescribe siempre
    with open(output_path, "w", encoding="utf-8", newline="") as out:
        out.write(ttl.rstrip() + "\n")

    try:
        out_size = os.path.getsize(output_path)
        print(f"[SERIES] Archivo TTL escrito. Tamaño = {out_size} bytes")
    except Exception as e:
        print(f"[SERIES][WARN] No pude obtener tamaño del TTL: {e}")

    print(f"[SERIES] OK - Series exportadas a {output_path}")
    return output_path

def _export_points():
    """
    Exporta los puntos (fecha, valor) desde SERIE_LIMPIA a 2datos.ttl
    Ontología v9:
      - Point -> anom:isPointOf (Series)
      - anom:pointDate (xsd:date)
      - anom:pointValue (xsd:decimal)
    NOTA: Source se modela en Series (anom:hasSource), no en Point.
    """
    import os
    import csv
    from pathlib import Path

    # --- Config desde settings (SIEMPRE via CFG)
    serie_limpia_path = Path(getattr(CFG, "SERIE_LIMPIA", "/tmp/DatosSerie/SerieOriginal.csv"))
    iri_series = getattr(CFG, "IRI")
    datos_onto = getattr(CFG, "DATOS_ONTOLOGIA")

    if not iri_series:
        raise ValueError("[POINTS][ERROR] CFG.IRI no está definido o está vacío (IRI de la serie)")
    if not datos_onto:
        raise ValueError("[POINTS][ERROR] CFG.DATOS_ONTOLOGIA no está definido o está vacío")
    if not serie_limpia_path.exists():
        raise FileNotFoundError(f"[POINTS][ERROR] No existe SERIE_LIMPIA: {serie_limpia_path}")

    def _escape_ttl_literal(s: str) -> str:
        return str(s).replace("\\", "\\\\").replace('"', '\\"').strip()

    def _series_ref(iri: str) -> str:
        iri = str(iri).strip()
        if iri.startswith(("http://", "https://")):
            return f"<{iri}>"
        return iri  # ej: anom:series_riesgo_pais_1

    def _pick_delimiter(header_line: str, sample: str) -> str:
        if ";" in header_line and header_line.count(";") >= header_line.count(","):
            return ";"
        if "\t" in header_line:
            return "\t"
        try:
            return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]).delimiter
        except Exception:
            if ";" in sample and sample.count(";") >= sample.count(","):
                return ";"
            if "\t" in sample:
                return "\t"
            return ","

    # Obtener un sufijo estable para nombrar los puntos a partir del IRI de la serie
    series_suffix = str(iri_series).strip()
    if series_suffix.startswith("anom:"):
        series_suffix = series_suffix.replace("anom:", "", 1)
    else:
        series_suffix = series_suffix.replace(":", "_").replace("/", "_").replace("#", "_")

    header = (
        "@prefix anom: <http://w3id.org/anomaly-core#> .\n"
        "@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
        "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .\n\n"
    )

    lines = [header]
    series_obj = _series_ref(iri_series)

    # Detectar delimitador (por si SERIE_LIMPIA viene con ';')
    with open(str(serie_limpia_path), "r", encoding="utf-8-sig", newline="") as f:
        header_line = f.readline()
        sample_rest = f.read(4096)
        sample = (header_line or "") + (sample_rest or "")

    delimiter = _pick_delimiter(header_line, sample)

    # Lee CSV de serie limpia (espera fecha, valor)
    with open(str(serie_limpia_path), "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        # si quedó pegado (caso típico cuando viene ';' pero se leyó ',')
        if reader.fieldnames and len(reader.fieldnames) == 1 and ";" in (reader.fieldnames[0] or ""):
            f.seek(0)
            delimiter = ";"
            reader = csv.DictReader(f, delimiter=delimiter)

        expected_min = {"fecha", "valor"}
        fieldnames = [c.strip().lower() for c in (reader.fieldnames or [])]
        if not expected_min.issubset(set(fieldnames)):
            raise ValueError(
                f"[POINTS][ERROR] SERIE_LIMPIA debe tener columnas {sorted(expected_min)}. "
                f"Columnas actuales: {reader.fieldnames} | Delimitador usado: {repr(delimiter)}"
            )

        for row in reader:
            row_norm = {k.strip().lower(): (v if v is not None else "") for k, v in row.items()}

            fecha = (row_norm.get("fecha") or "").strip()   # YYYY-MM-DD
            valor = (row_norm.get("valor") or "").strip()

            if not fecha or not valor:
                continue

            # Normaliza decimal (coma -> punto)
            valor_norm = valor.replace(",", ".").strip()

            # Validación mínima para xsd:decimal (evita NaN/inf/científica si aparece)
            # Si tu serie puede traer notación científica, dímelo y lo adaptamos.
            bad = {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
            if valor_norm.lower() in bad:
                continue

            # IRI del punto estable: anom:pt_<serie>_<YYYY_MM_DD>
            fecha_id = fecha.replace("-", "_")
            pt_iri = f"anom:pt_{series_suffix}_{fecha_id}"

            fecha_ttl = _escape_ttl_literal(fecha)
            valor_ttl = _escape_ttl_literal(valor_norm)

            lines.append(
                f"""{pt_iri}
  rdf:type anom:Point ;
  anom:isPointOf {series_obj} ;
  anom:pointDate "{fecha_ttl}"^^xsd:date ;
  anom:pointValue "{valor_ttl}"^^xsd:decimal .
"""
            )

    os.makedirs(datos_onto, exist_ok=True)
    output_path = os.path.join(datos_onto, "2datos.ttl")
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write("".join(lines))

    print(f"[POINTS] OK - Points exportados a {output_path}")
    return output_path

def _export_anomalies():
    """
    Exporta anomalías desde el CSV consolidado a 3anomalies.ttl (compatible con tu ontología final).

    Reglas para Anomaly:
      - anom:locatedInSeries (exactly 1) -> IRI de la Series
      - anom:hasAnomalousPoint (min 1) -> IRI(s) de Point(s)
      - anom:detectedBy (min 1) -> individuos del catálogo anom:DetectionMethod (ej. anom:dm_arima)
      - anom:startDate (1) xsd:date
      - anom:endDate (1) xsd:date
      - anom:count (1) xsd:integer  (número de puntos)
      - anom:anomalyId (0..1) -> ID operativo (xsd:string)
      - rdfs:label (0..1) -> etiqueta humana (@es)

    NOTA:
      - count se toma desde la columna num_values del CSV.
      - Por defecto se enlaza al Point de startDate y, si endDate != startDate, también al Point de endDate.
    """
    import os
    import csv
    import re
    from datetime import datetime

    print("[ANOM] Iniciando _export_anomalies()")

    ruta_resultados = getattr(CFG, "RUTA_RESULTADOS", None)
    datos_onto = getattr(CFG, "DATOS_ONTOLOGIA", None)

    print(f"[ANOM] CFG.RUTA_RESULTADOS = {ruta_resultados}")
    print(f"[ANOM] CFG.DATOS_ONTOLOGIA = {datos_onto}")

    if not ruta_resultados:
        raise ValueError("[ANOM][ERROR] RUTA_RESULTADOS no está definida en settings")
    if not datos_onto:
        raise ValueError("[ANOM][ERROR] DATOS_ONTOLOGIA no está definida en settings")

    input_csv = getattr(CFG, "ANOMALIAS_CONSOLIDADO", None)
    if not input_csv:
        input_csv = os.path.join(ruta_resultados, "anomalias_consolidado_completo.csv")

    print(f"[ANOM] CSV entrada = {input_csv}")

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"[ANOM][ERROR] No existe el archivo: {input_csv}")

    try:
        size_bytes = os.path.getsize(input_csv)
        print(f"[ANOM] Tamaño CSV = {size_bytes} bytes")
        if size_bytes == 0:
            raise ValueError("[ANOM][ERROR] El CSV existe pero está vacío (0 bytes). No hay nada que exportar.")
    except Exception as e:
        print(f"[ANOM][WARN] No pude obtener tamaño del CSV: {e}")

    os.makedirs(datos_onto, exist_ok=True)
    out_ttl = os.path.join(datos_onto, "3anomalies.ttl")
    print(f"[ANOM] TTL salida = {out_ttl}")

    def _detect_delimiter(sample: str) -> str:
        if "\t" in sample:
            return "\t"
        if ";" in sample and sample.count(";") >= sample.count(","):
            return ";"
        return ","

    def _norm(s):
        return re.sub(r"\s+", "_", str(s).strip().lower())

    def _pick_col(fieldnames, candidates):
        """Devuelve el primer nombre de columna existente (case-insensitive)."""
        if not fieldnames:
            return None
        norm_map = {_norm(c): c for c in fieldnames}
        for cand in candidates:
            key = _norm(cand)
            if key in norm_map:
                return norm_map[key]
        return None

    def _sanitize_local(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]+", "_", str(s).strip())

    def _escape_ttl_literal(s: str) -> str:
        return str(s).replace("\\", "\\\\").replace('"', '\\"').strip()

    def _parse_date(s: str) -> str:
        """Devuelve fecha ISO YYYY-MM-DD para TTL (xsd:date)."""
        if s is None:
            return None
        s = str(s).strip()
        if not s:
            return None

        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return s

        for fmt in ("%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(s, fmt).date().isoformat()
            except Exception:
                pass

        m = re.match(r"^(\d{4}-\d{2}-\d{2})[T\s].*$", s)
        if m:
            return m.group(1)

        raise ValueError(f"[ANOM][ERROR] No pude interpretar la fecha: '{s}'")

    def _method_token_to_dm_iri(token: str) -> str:
        """
        Convierte un token de método a un individuo del catálogo estandarizado: anom:dm_<token_en_minusculas>.
        """
        t = (token or "").strip()
        if not t:
            return None

        if t.startswith("anom:"):
            t = t.split("anom:", 1)[1].strip()

        t = t.upper().replace("-", "_").replace(" ", "_")
        if t.startswith("DM_"):
            t = t[3:]
        elif t == "DM":
            return None

        local = _sanitize_local(t).lower()
        if not local:
            return None

        return f"anom:dm_{local}"

    def _as_iri_ref(v: str) -> str:
        """Si viene URL, lo envuelve en <>. Si viene prefijo (anom:...), lo deja."""
        v = str(v or "").strip()
        if not v:
            return v
        if v.startswith(("http://", "https://")):
            return f"<{v}>"
        return v

    # Inferencias desde CFG
    serie_id_cfg = getattr(CFG, "SERIE_ID", None)
    serie_iri_cfg = getattr(CFG, "IRI", None)

    print(f"[ANOM] serie_id_cfg (CFG.SERIE_ID) = {serie_id_cfg}")
    print(f"[ANOM] serie_iri_cfg (CFG.IRI) = {serie_iri_cfg}")

    # Sufijo estable para construir IRIs de Point (debe coincidir con _export_points)
    series_suffix_cfg = None
    if serie_iri_cfg:
        sfx = str(serie_iri_cfg).strip()
        if sfx.startswith("anom:"):
            sfx = sfx.replace("anom:", "", 1)
        else:
            sfx = sfx.replace(":", "_").replace("/", "_").replace("#", "_")
        series_suffix_cfg = sfx

    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        delimiter = _detect_delimiter(sample)
        print(f"[ANOM] Delimitador detectado = {repr(delimiter)}")
        print(f"[ANOM] Muestra inicial (primeros 200 chars) = {repr(sample[:200])}")
        f.seek(0)

        reader = csv.DictReader(f, delimiter=delimiter)
        cols = reader.fieldnames or []
        print(f"[ANOM] Columnas detectadas ({len(cols)}): {cols}")

        # Aliases soportados
        col_serie_id = _pick_col(cols, ["serie_id", "id_serie", "series_id"])
        col_start = _pick_col(cols, ["fecha_inicio", "fecha", "start_date", "start", "inicio", "date"])
        col_end = _pick_col(cols, ["fecha_fin", "end_date", "end", "fin"])
        col_method = _pick_col(cols, ["metodo", "method"])
        col_anom_id = _pick_col(cols, ["anomalia_id", "anomaly_id", "id"])
        # === CAMBIO ÚNICO: tomar count desde num_values ===
        col_num_values = _pick_col(cols, ["num_values"])

        print("[ANOM] Mapeo de columnas seleccionado:")
        print(f"       col_serie_id   = {col_serie_id}")
        print(f"       col_start      = {col_start}")
        print(f"       col_end        = {col_end}")
        print(f"       col_method     = {col_method}")
        print(f"       col_anom_id    = {col_anom_id}")
        print(f"       col_num_values = {col_num_values}")

        if not col_start:
            raise ValueError(
                "[ANOM][ERROR] El CSV de anomalías debe tener 'fecha_inicio' (o 'fecha' o 'start_date'). "
                f"Columnas detectadas: {cols}"
            )

        if not col_serie_id and (serie_id_cfg is None and serie_iri_cfg is None):
            raise ValueError(
                "[ANOM][ERROR] El CSV de anomalías no trae 'serie_id' y tampoco hay CFG.SERIE_ID / CFG.IRI para inferirlo. "
                f"Columnas detectadas: {cols}"
            )

        # Si el CSV no trae num_values, paramos (porque pediste que count salga de ahí)
        if not col_num_values:
            raise ValueError(
                "[ANOM][ERROR] El CSV de anomalías no trae la columna 'num_values' requerida para anom:count. "
                f"Columnas detectadas: {cols}"
            )

        ttl_lines = [
            '@prefix anom: <http://w3id.org/anomaly-core#> .',
            '@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .',
            '@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .',
            '@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .',
            ''
        ]

        total = 0
        errores = 0
        previews = 0

        for idx, row in enumerate(reader, start=1):
            if previews < 3:
                print(f"[ANOM][ROW-PREVIEW {idx}] {row}")
                previews += 1

            try:
                serie_id_val = row.get(col_serie_id) if col_serie_id else None
                if serie_id_val is None or str(serie_id_val).strip() == "":
                    serie_id_val = serie_id_cfg
                serie_id_val = str(serie_id_val).strip() if serie_id_val is not None else ""

                start_raw = row.get(col_start)
                start_iso = _parse_date(start_raw)

                end_iso = None
                if col_end:
                    end_raw = row.get(col_end)
                    if end_raw and str(end_raw).strip():
                        end_iso = _parse_date(end_raw)

                # Métodos -> detectedBy (mínimo 1)
                method_val = row.get(col_method) if col_method else None
                method_val = str(method_val).strip() if method_val is not None else ""

                method_tokens = [t.strip() for t in re.split(r"[,\|;/]+", method_val) if t.strip()] if method_val else []
                detected_by_iris = []
                for t in method_tokens:
                    dm = _method_token_to_dm_iri(t)
                    if dm:
                        detected_by_iris.append(dm)

                # quitar duplicados preservando orden
                seen = set()
                detected_by_iris = [x for x in detected_by_iris if not (x in seen or seen.add(x))]

                if not detected_by_iris:
                    raise ValueError(
                        "[ANOM][ERROR] La anomalía no tiene métodos de detección. "
                        "La ontología requiere anom:detectedBy (1..*). Revisa columna 'method/metodo'."
                    )

                anom_id_val = row.get(col_anom_id) if col_anom_id else None
                anom_id_val = str(anom_id_val).strip() if anom_id_val is not None else ""

                # anomalyId (operativo) y local IRI
                if anom_id_val:
                    anomaly_id_literal = anom_id_val
                    local_id = f"anomaly_{_sanitize_local(serie_id_val)}_{_sanitize_local(anom_id_val)}"
                else:
                    anomaly_id_literal = f"{start_iso}"
                    local_id = f"anomaly_{_sanitize_local(serie_id_val)}_{start_iso}"

                local_id = _sanitize_local(local_id)
                subj = f"anom:{local_id}"

                # --- IRI de la serie
                if serie_iri_cfg:
                    series_iri = _as_iri_ref(serie_iri_cfg)
                else:
                    if not serie_id_val:
                        raise ValueError("[ANOM][ERROR] No hay serie_id para construir la IRI de la serie.")
                    series_iri = f"anom:series_{_sanitize_local(serie_id_val)}"

                # --- IRIs de Point(s)
                if series_suffix_cfg:
                    series_suffix = series_suffix_cfg
                else:
                    sfx = series_iri
                    if sfx.startswith("anom:"):
                        sfx = sfx.replace("anom:", "", 1)
                    else:
                        sfx = sfx.replace("<", "").replace(">", "").replace(":", "_").replace("/", "_").replace("#", "_")
                    series_suffix = sfx

                start_id = start_iso.replace("-", "_")
                pt_start = f"anom:pt_{series_suffix}_{start_id}"

                points = [pt_start]
                if end_iso and end_iso != start_iso:
                    end_id = end_iso.replace("-", "_")
                    points.append(f"anom:pt_{series_suffix}_{end_id}")

                # === CAMBIO ÚNICO: count_int desde num_values (no desde len(points)) ===
                raw_nv = row.get(col_num_values)
                raw_nv = str(raw_nv).strip() if raw_nv is not None else ""
                if raw_nv == "":
                    raise ValueError(f"[ANOM][ERROR] 'num_values' vacío en la fila {idx}.")
                try:
                    count_int = int(float(raw_nv))
                except Exception:
                    raise ValueError(f"[ANOM][ERROR] 'num_values' inválido: {raw_nv!r} en la fila {idx}.")

                # label (@es, sin duplicados)
                if anom_id_val:
                    label_es = f"Anomalía {anom_id_val} ({start_iso})" if (end_iso is None or end_iso == start_iso) else f"Anomalía {anom_id_val} ({start_iso} a {end_iso})"
                else:
                    label_es = f"Anomalía ({start_iso})" if (end_iso is None or end_iso == start_iso) else f"Anomalía ({start_iso} a {end_iso})"

                label_es = _escape_ttl_literal(label_es)
                anomaly_id_literal = _escape_ttl_literal(anomaly_id_literal)

                # --- Construcción TTL
                ttl_lines.append(f"{subj}")
                ttl_lines.append("  rdf:type anom:Anomaly ;")
                ttl_lines.append(f"  anom:anomalyId \"{anomaly_id_literal}\"^^xsd:string ;")
                ttl_lines.append(f"  rdfs:label \"{label_es}\"@es ;")
                ttl_lines.append(f"  anom:locatedInSeries {series_iri} ;")
                ttl_lines.append(f"  anom:startDate \"{start_iso}\"^^xsd:date ;")
                ttl_lines.append(f"  anom:endDate \"{(end_iso or start_iso)}\"^^xsd:date ;")

                for p in points:
                    ttl_lines.append(f"  anom:hasAnomalousPoint {p} ;")

                for dm in detected_by_iris:
                    ttl_lines.append(f"  anom:detectedBy {dm} ;")

                ttl_lines.append(f"  anom:count \"{count_int}\"^^xsd:integer .")
                ttl_lines.append("")

                total += 1

            except Exception as e:
                errores += 1
                if errores <= 20:
                    print(f"[ANOM][ERROR][ROW {idx}] {e}")
                    print(f"[ANOM][ERROR][ROW {idx}] Row completa: {row}")
                else:
                    print(f"[ANOM][ERROR] Demasiados errores; ocultando detalles adicionales. Último error: {e}")
                    break

        print(f"[ANOM] Filas procesadas total = {idx if 'idx' in locals() else 0}")
        print(f"[ANOM] Filas exportadas (OK) = {total}")
        print(f"[ANOM] Filas con error = {errores}")
        print(f"[ANOM] Líneas TTL en memoria = {len(ttl_lines)}")

        if total == 0:
            raise ValueError(
                "[ANOM][ERROR] No se exportó ninguna anomalía. "
                "Revisa: CSV vacío, delimitador incorrecto, fechas no parseables o métodos vacíos."
            )

        with open(out_ttl, "w", encoding="utf-8", newline="") as w:
            w.write("\n".join(ttl_lines).rstrip() + "\n")

        try:
            out_size = os.path.getsize(out_ttl)
            print(f"[ANOM] Archivo TTL escrito. Tamaño = {out_size} bytes")
        except Exception as e:
            print(f"[ANOM][WARN] No pude obtener tamaño del TTL: {e}")

        print(f"[ANOM] OK - {total} anomalías exportadas a {out_ttl}")
        return out_ttl


def _export_series_sinteticas():
    """
    Genera 4series_sinteticas.ttl a partir de:
      <CFG.RUTA_RESULTADOS>/puntos_sinteticos_consolidados.csv

    Crea 1 anom:Series por cada metodo_prediccion detectado.

    Reglas (ONT v9):
      - hasSource     = anom:src_synthetic
      - predictedBy   = individuo anom:pm_* (ej. anom:pm_arima_direct)
      - basedOnSeries = <IRI de la serie base> (trazabilidad)
      - IRI sintética: anom:series_<BASE_LOCAL>__PM_<TOKEN>   (BASE_LOCAL incluye 'series_')

    Cambio solicitado:
      - anom:seriesId de la serie sintética debe ser ÚNICO por serie base.
        Se construye con el sufijo del IRI base (base_local) + PM token:
          seriesId = "<base_local>__<PM_TOKEN>"
        Ej: "series_riesgo_pais_1__PM_ARIMA_FEEDBACK"
    """
    import os
    import csv
    import re

    print("[SERIES-SINT] Iniciando _export_series_sinteticas()")

    ruta_resultados = getattr(CFG, "RUTA_RESULTADOS", None)
    datos_onto = getattr(CFG, "DATOS_ONTOLOGIA", None)

    if not ruta_resultados:
        raise ValueError("[SERIES-SINT][ERROR] CFG.RUTA_RESULTADOS no está definido")
    if not datos_onto:
        raise ValueError("[SERIES-SINT][ERROR] CFG.DATOS_ONTOLOGIA no está definido")

    ruta_resultados = str(ruta_resultados).strip().strip('"')
    datos_onto = str(datos_onto).strip().strip('"')

    input_csv = os.path.join(ruta_resultados, "puntos_sinteticos_consolidados.csv")
    print(f"[SERIES-SINT] CSV entrada = {input_csv}")

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"[SERIES-SINT][ERROR] No existe el archivo: {input_csv}")

    def _detect_delimiter(sample: str) -> str:
        if "\t" in sample:
            return "\t"
        if ";" in sample and sample.count(";") >= sample.count(","):
            return ";"
        return ","

    def _norm_key(k: str) -> str:
        return (k or "").strip().lower()

    def _sanitize_local(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]+", "_", str(s).strip())

    def _escape_ttl_literal(s: str) -> str:
        return str(s).replace("\\", "\\\\").replace('"', '\\"').strip()

    def _as_iri_ref(v: str) -> str:
        """Si viene URL, lo envuelve en <>. Si viene prefijo (anom:...), lo deja."""
        v = str(v or "").strip()
        if not v:
            return v
        if v.startswith(("http://", "https://")):
            return f"<{v}>"
        return v

    def _series_local_from_iri(iri: str) -> str:
        """
        Convierte:
          anom:series_riesgo_pais_1 -> series_riesgo_pais_1
          URL -> último segmento (best-effort)
        """
        s = str(iri or "").strip()
        if not s:
            return ""
        if s.startswith("anom:"):
            return s.replace("anom:", "", 1)
        if "#" in s:
            return s.split("#")[-1]
        if "/" in s:
            return s.rstrip("/").split("/")[-1]
        return s

    # --- Token PM del CSV (solo para naming/seriesId/series IRI)
    def _to_pm_token(method_raw: str) -> str:
        """
        Devuelve token canonical PM_* para naming.
        Ej:
          ARIMA -> PM_ARIMA_DIRECT
          PM_ARIMA_DIRECT -> PM_ARIMA_DIRECT
          arima_direct -> PM_ARIMA_DIRECT
        """
        m = (method_raw or "").strip().upper()
        if not m:
            return ""
        if m == "ARIMA":
            m = "ARIMA_DIRECT"
        m = m.replace("-", "_").replace(" ", "_")
        m = _sanitize_local(m)
        if not m.startswith("PM_"):
            m = "PM_" + m
        return m

    # --- IRI del individuo PredictionMethod (catálogo v9): anom:pm_*
    def _to_pm_individual_iri(pm_token: str) -> str:
        """
        Convierte:
          PM_ARIMA_DIRECT -> anom:pm_arima_direct
        """
        t = (pm_token or "").strip()
        if not t:
            return ""
        t = t.upper().replace("-", "_").replace(" ", "_")
        t = _sanitize_local(t)
        if t.startswith("PM_"):
            t = t.replace("PM_", "", 1)
        return f"anom:pm_{t.lower()}"

    # Leer CSV y construir combinaciones únicas (serie base + método)
    combos = set()
    previews = 0

    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        delim = _detect_delimiter(sample)
        f.seek(0)

        reader = csv.DictReader(f, delimiter=delim)
        cols = reader.fieldnames or []
        print(f"[SERIES-SINT] Columnas detectadas ({len(cols)}): {cols}")
        norm_map = {_norm_key(c): c for c in cols}

        col_method = norm_map.get("metodo_prediccion")
        col_series_iri = norm_map.get("series_iri")
        col_series_id = norm_map.get("series_id")  # se conserva lectura, pero ya NO define el seriesId sintético

        if not col_method:
            raise ValueError("[SERIES-SINT][ERROR] El CSV no tiene columna 'metodo_prediccion'")

        if not col_series_iri:
            cfg_iri = getattr(CFG, "IRI", None)
            if not cfg_iri:
                raise ValueError("[SERIES-SINT][ERROR] El CSV no tiene 'series_iri' y CFG.IRI no está definido")
            print("[SERIES-SINT][WARN] No hay 'series_iri' en CSV; usaré CFG.IRI como serie base")

        if not col_series_id:
            cfg_sid = getattr(CFG, "SERIE_ID", None)
            if not cfg_sid:
                print("[SERIES-SINT][WARN] No hay 'series_id' en CSV y CFG.SERIE_ID no está definido; continuaré igual (no lo usaré para seriesId sintético)")

        for i, row in enumerate(reader, start=1):
            if previews < 3:
                print(f"[SERIES-SINT][ROW-PREVIEW {i}] {row}")
                previews += 1

            method_raw = (row.get(col_method) or "").strip()
            pm_token = _to_pm_token(method_raw)
            if not pm_token:
                continue

            base_series_iri_raw = (row.get(col_series_iri) or "").strip() if col_series_iri else ""
            if not base_series_iri_raw:
                base_series_iri_raw = str(getattr(CFG, "IRI")).strip()

            # se mantiene en la tupla solo para unicidad de combos; no lo usaremos en seriesId sintético
            base_series_id = (row.get(col_series_id) or "").strip() if col_series_id else ""
            if not base_series_id:
                sid = getattr(CFG, "SERIE_ID", "")
                base_series_id = str(sid).strip() if sid is not None else ""

            combos.add((base_series_iri_raw, base_series_id, pm_token))

    if not combos:
        raise ValueError("[SERIES-SINT][ERROR] No se encontraron combinaciones (series_iri, metodo_prediccion) en el CSV.")

    os.makedirs(datos_onto, exist_ok=True)
    out_ttl = os.path.join(datos_onto, "4series_sinteticas.ttl")
    print(f"[SERIES-SINT] TTL salida = {out_ttl}")

    ttl_lines = [
        "@prefix anom: <http://w3id.org/anomaly-core#> .",
        "@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .",
        ""
    ]

    # Generar TTL
    for (base_series_iri_raw, _base_series_id_unused, pm_token) in sorted(combos):
        base_series_iri = _as_iri_ref(base_series_iri_raw)  # URL o anom:
        base_local = _series_local_from_iri(base_series_iri_raw)  # ej: series_riesgo_pais_1

        pm_token_sane = _sanitize_local(pm_token)  # PM_ARIMA_DIRECT
        pm_ind_iri = _to_pm_individual_iri(pm_token_sane)  # anom:pm_arima_direct

        # IRI sintética (mantener base_local completo para quedar como tu ejemplo final)
        sint_series_iri = f"anom:{base_local}__{pm_token_sane}"

        # seriesId sintética ÚNICA por serie base (usa base_local, no base_series_id numérico)
        series_id_sint = f"{base_local}__{pm_token_sane}"

        # seriesName (texto humano)
        series_name = f"{base_local} ({pm_token_sane})"

        ttl_lines.append(f"{sint_series_iri}")
        ttl_lines.append("  rdf:type anom:Series ;")
        ttl_lines.append(f"  anom:seriesId \"{_escape_ttl_literal(series_id_sint)}\"^^xsd:string ;")
        ttl_lines.append("  anom:hasSource anom:src_synthetic ;")
        ttl_lines.append(f"  anom:predictedBy {pm_ind_iri} ;")
        ttl_lines.append(f"  anom:basedOnSeries {base_series_iri} ;")
        ttl_lines.append(f"  anom:seriesName \"{_escape_ttl_literal(series_name)}\"^^xsd:string ;")
        ttl_lines.append("  anom:frequency \"daily\"^^xsd:string ;")
        ttl_lines.append("  anom:unit \"points\"^^xsd:string .")
        ttl_lines.append("")

    with open(out_ttl, "w", encoding="utf-8") as w:
        w.write("\n".join(ttl_lines).rstrip() + "\n")

    try:
        out_size = os.path.getsize(out_ttl)
        print(f"[SERIES-SINT] Archivo TTL escrito. Tamaño = {out_size} bytes")
    except Exception as e:
        print(f"[SERIES-SINT][WARN] No pude obtener tamaño del TTL: {e}")

    print(f"[SERIES-SINT] OK - {len(combos)} series sintéticas exportadas a {out_ttl}")
    return out_ttl


def _export_points_sinteticos():
    """
    Genera 5points_sinteticos.ttl a partir de:
      <CFG.RUTA_RESULTADOS>/puntos_sinteticos_consolidados.csv

    Ontología final:
      - Point -> isPointOf (Series)
      - pointDate (xsd:date)
      - pointValue (xsd:decimal)

    Convención de IRI (coherente con points originales):
      anom:pt_<series_suffix>_<YYYY_MM_DD>

    Para cada fila del CSV:
      1) Determina la IRI de la serie sintética (MISMA regla que _export_series_sinteticas ACTUAL):
         anom:<base_local>__<PM_TOKEN>
         donde base_local incluye 'series_' (ej. series_riesgo_pais_1)
         y PM_TOKEN es PM_* (ej. PM_ARIMA_FEEDBACK)
      2) Crea un anom:Point único por (serie sintética + fecha)
      3) Enlaza: anom:isPointOf -> serie sintética

    Requiere columnas mínimas (aliases tolerados):
      - metodo_prediccion
      - fecha | date | ds | timestamp | datetime
      - valor | value | yhat | pred | prediccion | prediction

    Escribe: <CFG.DATOS_ONTOLOGIA>/5points_sinteticos.ttl
    """
    print("[POINTS-SINT] Iniciando _export_points_sinteticos()")

    ruta_resultados = getattr(CFG, "RUTA_RESULTADOS", None)
    datos_onto = getattr(CFG, "DATOS_ONTOLOGIA", None)

    if not ruta_resultados:
        raise ValueError("[POINTS-SINT][ERROR] CFG.RUTA_RESULTADOS no está definido")
    if not datos_onto:
        raise ValueError("[POINTS-SINT][ERROR] CFG.DATOS_ONTOLOGIA no está definido")

    ruta_resultados = str(ruta_resultados).strip().strip('"')
    datos_onto = str(datos_onto).strip().strip('"')

    input_csv = os.path.join(ruta_resultados, "puntos_sinteticos_consolidados.csv")
    print(f"[POINTS-SINT] CSV entrada = {input_csv}")

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"[POINTS-SINT][ERROR] No existe el archivo: {input_csv}")

    def _detect_delimiter(sample: str) -> str:
        if "\t" in sample:
            return "\t"
        if ";" in sample and sample.count(";") >= sample.count(","):
            return ";"
        return ","

    def _norm_key(k: str) -> str:
        return (k or "").strip().lower()

    def _sanitize_local(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]+", "_", str(s).strip())

    def _escape_ttl_literal(s: str) -> str:
        return str(s).replace("\\", "\\\\").replace('"', '\\"').strip()

    def _series_local_from_iri(iri: str) -> str:
        """
        Convierte:
          anom:series_riesgo_pais_1 -> series_riesgo_pais_1
          URL -> último segmento (best-effort)
        """
        s = str(iri or "").strip()
        if not s:
            return ""
        if s.startswith("anom:"):
            return s.replace("anom:", "", 1)
        if "#" in s:
            return s.split("#")[-1]
        if "/" in s:
            return s.rstrip("/").split("/")[-1]
        return s

    def _iri_to_suffix(iri: str) -> str:
        """
        Sufijo estable (para point IRI):
        - anom:xxx -> xxx
        - URL -> normaliza con reemplazos
        """
        s = str(iri or "").strip()
        if s.startswith("anom:"):
            return s.replace("anom:", "", 1)
        return (
            s.replace("http://", "")
             .replace("https://", "")
             .replace(":", "_")
             .replace("/", "_")
             .replace("#", "_")
        )

    # Token PM del CSV (para naming y para coincidir con series sintéticas ya cargadas)
    # Regla: si viene "ARIMA" (sin DIRECT/FEEDBACK), asumimos DIRECT.
    def _to_pm_token(method_raw: str) -> str:
        """
        Devuelve token canonical PM_*.
        Ej:
          ARIMA -> PM_ARIMA_DIRECT
          PM_ARIMA_DIRECT -> PM_ARIMA_DIRECT
          arima_direct -> PM_ARIMA_DIRECT
        """
        m = (method_raw or "").strip().upper()
        if not m:
            return ""
        if m == "ARIMA":
            m = "ARIMA_DIRECT"
        m = m.replace("-", "_").replace(" ", "_")
        m = _sanitize_local(m)
        if not m.startswith("PM_"):
            m = "PM_" + m
        return m

    def _build_sint_series_iri(base_series_iri: str, pm_token: str) -> str:
        """
        MISMA regla que _export_series_sinteticas ACTUAL:
          anom:<base_local>__<PM_TOKEN>
        donde base_local incluye 'series_'.
        """
        base_local = _series_local_from_iri(base_series_iri)  # ej: series_riesgo_pais_1
        base_local = _sanitize_local(base_local)
        pm_token = _sanitize_local(pm_token)
        if not base_local or not pm_token:
            return ""
        return f"anom:{base_local}__{pm_token}"

    def _extract_xsd_date(raw: str) -> str:
        """
        Devuelve 'YYYY-MM-DD' (xsd:date) desde entradas como:
          - YYYY-MM-DD
          - YYYY/MM/DD
          - YYYY-MM-DDTHH:MM:SSZ
          - YYYY-MM-DD HH:MM:SS
        """
        s = (raw or "").strip()
        if not s:
            return None
        if "T" in s:
            s = s.split("T", 1)[0].strip()
        if " " in s:
            s = s.split(" ", 1)[0].strip()
        if "/" in s:
            s = s.replace("/", "-")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return s
        m = re.search(r"(\d{4})[-/](\d{2})[-/](\d{2})", s)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        return None

    def _parse_decimal(raw: str) -> str:
        """
        Normaliza número a string decimal con punto.
        Devuelve None si no es parseable.
        """
        s = (raw or "").strip()
        if not s:
            return None
        s = s.replace(" ", "")

        # coma decimal simple
        if s.count(",") == 1 and s.count(".") == 0:
            s = s.replace(",", ".")

        # heurística separadores de miles
        if "," in s and "." in s:
            if re.search(r",\d{1,6}$", s):   # coma decimal al final
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")

        try:
            float(s)
            return s
        except Exception:
            return None

    ttl_lines = []
    n_rows = 0
    n_points = 0
    previews = 0

    # Deduplicación por (serieSinteticaIRI, YYYY-MM-DD)
    seen = set()

    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        delim = _detect_delimiter(sample)
        f.seek(0)

        reader = csv.DictReader(f, delimiter=delim)
        cols = reader.fieldnames or []
        print(f"[POINTS-SINT] Columnas detectadas ({len(cols)}): {cols}")
        norm_map = {_norm_key(c): c for c in cols}

        col_method = norm_map.get("metodo_prediccion")
        col_series_iri = norm_map.get("series_iri")

        col_date = (
            norm_map.get("fecha") or norm_map.get("date") or norm_map.get("ds") or
            norm_map.get("timestamp") or norm_map.get("datetime")
        )
        col_value = (
            norm_map.get("valor") or norm_map.get("value") or norm_map.get("yhat") or
            norm_map.get("pred") or norm_map.get("prediccion") or norm_map.get("prediction")
        )

        if not col_method:
            raise ValueError("[POINTS-SINT][ERROR] El CSV no tiene columna 'metodo_prediccion'")
        if not col_date:
            raise ValueError("[POINTS-SINT][ERROR] No se detectó columna de fecha (fecha/date/ds/timestamp/datetime)")
        if not col_value:
            raise ValueError("[POINTS-SINT][ERROR] No se detectó columna de valor (valor/value/yhat/pred/...)")

        if not col_series_iri:
            cfg_iri = getattr(CFG, "IRI", None)
            if not cfg_iri:
                raise ValueError("[POINTS-SINT][ERROR] Falta 'series_iri' y CFG.IRI no está definido")
            print("[POINTS-SINT][WARN] No hay 'series_iri' en CSV; usaré CFG.IRI como serie base")

        for i, row in enumerate(reader, start=1):
            n_rows += 1
            if previews < 3:
                print(f"[POINTS-SINT][ROW-PREVIEW {i}] {row}")
                previews += 1

            method_raw = (row.get(col_method) or "").strip()
            pm_token = _to_pm_token(method_raw)
            if not pm_token:
                continue

            base_series_iri = (row.get(col_series_iri) or "").strip() if col_series_iri else ""
            if not base_series_iri:
                base_series_iri = str(getattr(CFG, "IRI")).strip()

            sint_series_iri = _build_sint_series_iri(base_series_iri, pm_token)
            if not sint_series_iri:
                continue

            date_xsd = _extract_xsd_date(row.get(col_date))
            if not date_xsd:
                continue

            val_dec = _parse_decimal(row.get(col_value))
            if val_dec is None:
                continue

            # Dedup por serie+fecha
            key = (sint_series_iri, date_xsd)
            if key in seen:
                continue
            seen.add(key)

            # IRI del punto: anom:pt_<seriesSuffix>_<YYYY_MM_DD>
            series_suffix = _iri_to_suffix(sint_series_iri)  # sint_series_iri es anom:...
            date_id = date_xsd.replace("-", "_")
            point_iri = f"anom:pt_{series_suffix}_{date_id}"

            ttl_lines.append(
                f"""{point_iri}
  rdf:type anom:Point ;
  anom:isPointOf {sint_series_iri} ;
  anom:pointDate "{_escape_ttl_literal(date_xsd)}"^^xsd:date ;
  anom:pointValue "{_escape_ttl_literal(val_dec)}"^^xsd:decimal .

"""
            )
            n_points += 1

    if n_points == 0:
        raise ValueError("[POINTS-SINT][ERROR] No se generaron puntos. Revisa columnas de fecha/valor y contenido del CSV.")

    ttl = (
        "@prefix anom: <http://w3id.org/anomaly-core#> .\n"
        "@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
        "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .\n\n"
        + "".join(ttl_lines).rstrip()
        + "\n"
    )

    os.makedirs(datos_onto, exist_ok=True)
    output_path = os.path.join(datos_onto, "5points_sinteticos.ttl")
    with open(output_path, "w", encoding="utf-8", newline="") as out:
        out.write(ttl)

    print(f"[POINTS-SINT] OK - Puntos sintéticos exportados a {output_path} (rows={n_rows}, points={n_points})")
    return output_path

def _export_anomalies_sinteticas():
    import os
    import csv
    import re
    from datetime import datetime, date, timedelta

    def _log(msg: str):
        print(msg)

    ruta_anom_syn = getattr(CFG, "ANOMALIAS_SINTETICAS_CSV", None)
    datos_onto = getattr(CFG, "DATOS_ONTOLOGIA", None)

    if not ruta_anom_syn or not datos_onto:
        raise ValueError("Rutas no definidas en CFG")

    input_csv = getattr(
        CFG,
        "ANOMALIAS_SINTETICAS_CONSOLIDADO",
        os.path.join(ruta_anom_syn, "anomalias_comparacion_numerica_sinteticas.csv"),
    )

    out_ttl = os.path.join(datos_onto, "6anomalies_sinteticas.ttl")
    os.makedirs(os.path.dirname(out_ttl), exist_ok=True)

    def _sanitize(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]+", "_", str(s).strip())

    def _parse_date(s: str) -> str:
        """
        Soporta:
          - YYYY-MM-DD
          - M/D/YYYY o MM/DD/YYYY (ej: 4/10/2025)
          - ISO datetime
        """
        s = "" if s is None else str(s).strip()
        if not s:
            raise ValueError("Fecha vacía")

        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return s

        for fmt in ("%m/%d/%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(s, fmt).date().isoformat()
            except Exception:
                pass

        try:
            return datetime.fromisoformat(s).date().isoformat()
        except Exception:
            pass

        raise ValueError(f"No pude interpretar la fecha: {s}")

    def _yyyy_mm_dd(d: str) -> str:
        return d.replace("-", "_")

    def _dates_inclusive(start_iso, end_iso):
        s = date.fromisoformat(start_iso)
        e = date.fromisoformat(end_iso)
        cur = s
        out = []
        while cur <= e:
            out.append(cur.isoformat())
            cur += timedelta(days=1)
        return out

    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []

        def col(*names):
            for n in names:
                if n in cols:
                    return n
            return None

        # CAMBIO CLAVE: tomar la base desde 'serie_base' (nombre de la serie origen)
        col_serie_base = col("serie_base", "serie", "serie_origen", "series_base")

        col_start = col("start_date", "fecha_inicio", "fecha")
        col_end = col("end_date", "fecha_fin")
        col_mp = col("metodo_prediccion")
        col_md = col("metodo_deteccion")

        if not col_serie_base:
            raise ValueError("Falta columna 'serie_base' en el CSV (requerida para formar IRIs correctas).")
        if not col_start:
            raise ValueError("Falta columna start_date/fecha_inicio/fecha.")
        if not col_mp:
            raise ValueError("Falta columna metodo_prediccion.")
        if not col_md:
            raise ValueError("Falta columna metodo_deteccion.")

        ttl = [
            "@prefix anom: <http://w3id.org/anomaly-core#> .",
            "@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .",
            "",
        ]

        total = 0

        for row_idx, row in enumerate(reader, start=1):
            # --- base series (desde CSV: ej. "series_riesgo_pais_1")
            base_local = _sanitize(row.get(col_serie_base, ""))
            if not base_local:
                _log(f"[ANOM-SYN][WARN][ROW {row_idx}] serie_base vacía; se omite fila.")
                continue

            # --- fechas
            start_iso = _parse_date(row.get(col_start))
            end_iso = _parse_date(row.get(col_end)) if col_end and row.get(col_end) else start_iso

            # --- método predicción (sin normalizar; solo sanitize)
            mp = _sanitize(row.get(col_mp, ""))
            if not mp:
                _log(f"[ANOM-SYN][WARN][ROW {row_idx}] metodo_prediccion vacío; se omite fila.")
                continue

            # --- detectores (pueden ser varios)
            md_raw = row.get(col_md, "")
            md_raw = "" if md_raw is None else str(md_raw)
            detectors = [
                _sanitize(d.lower())
                for d in re.split(r"[,\|;/]+", md_raw)
                if d.strip()
            ]
            if not detectors:
                _log(f"[ANOM-SYN][WARN][ROW {row_idx}] metodo_deteccion vacío; se omite fila.")
                continue

            # --- IRIs
            series_local = f"{base_local}__PM_{mp}"
            series_iri = f"anom:{series_local}"

            dates = _dates_inclusive(start_iso, end_iso)
            points = [f"anom:pt_{series_local}_{_yyyy_mm_dd(d)}" for d in dates]

            for det in detectors:
                # IRI estable y descriptivo (Corrección B)
                if start_iso == end_iso:
                    anom_local = f"anomaly_syn_{series_local}_{_yyyy_mm_dd(start_iso)}__dm_{det}"
                else:
                    anom_local = (
                        f"anomaly_syn_{series_local}_"
                        f"{_yyyy_mm_dd(start_iso)}_{_yyyy_mm_dd(end_iso)}__dm_{det}"
                    )

                subj = f"anom:{anom_local}"

                ttl.append(subj)
                ttl.append("  rdf:type anom:Anomaly ;")
                ttl.append(f"  anom:locatedInSeries {series_iri} ;")
                ttl.append(f"  anom:startDate \"{start_iso}\"^^xsd:date ;")
                ttl.append(f"  anom:endDate \"{end_iso}\"^^xsd:date ;")

                for p in points:
                    ttl.append(f"  anom:hasAnomalousPoint {p} ;")

                ttl.append(f"  anom:detectedBy anom:dm_{det} ;")
                ttl.append(f"  anom:count \"{len(points)}\"^^xsd:integer .")
                ttl.append("")

                total += 1

    with open(out_ttl, "w", encoding="utf-8") as w:
        w.write("\n".join(ttl).rstrip() + "\n")

    _log(f"[ANOM-SYN] OK – {total} anomalías exportadas a {out_ttl}")
    return out_ttl


def _export_similarities_dtw():
    """
    Exporta similitudes NUMÉRICAS desde similaridades_dtw.csv a 7similarities_dtw.ttl
    conforme al MODELO UNIFICADO v9 (Ontv9_simil), soportando 3 métricas + score final.

    Nota importante para TU CSV:
    - score_final_used NO es un número; es un selector ("combined" / "dtw").
      Por tanto, similarityValue se toma desde similarity_combined o similarity_dtw según ese selector.
    """
    import os
    import csv

    print("[SIM-NUM] Iniciando exportación de similitudes numéricas (DTW/Feat/Combined + final)")

    base_res = getattr(CFG, "RUTA_RESULTADOS", None)
    datos_onto = getattr(CFG, "DATOS_ONTOLOGIA", None)

    if not base_res:
        raise ValueError("[SIM-NUM] CFG.RUTA_RESULTADOS no está definida.")
    if not datos_onto:
        raise ValueError("[SIM-NUM] CFG.DATOS_ONTOLOGIA no está definida.")

    # ----------------------------
    # Paths (FUENTE CORRECTA)
    # ----------------------------
    input_candidates = [
        os.path.join(base_res, "ComparacionesNumericas", "similaridades_dtw.csv"),
        os.path.join(base_res, "Comparaciones", "similaridades_dtw.csv"),
        os.path.join(base_res, "similaridades_dtw.csv"),
    ]

    ruta_comparacion = getattr(CFG, "RUTA_COMPARACION", None)
    if ruta_comparacion:
        input_candidates.insert(0, os.path.join(ruta_comparacion, "similaridades_dtw.csv"))

    input_csv = next((p for p in input_candidates if os.path.exists(p)), None)
    if not input_csv:
        raise FileNotFoundError(f"[SIM-NUM] No existe similaridades_dtw.csv en rutas candidatas: {input_candidates}")

    cat_o_candidates = [
        os.path.join(base_res, "anomalias_comparacion_numerica.csv"),
        os.path.join(base_res, "Anomalias", "anomalias_comparacion_numerica.csv"),
    ]
    cat_original = next((p for p in cat_o_candidates if os.path.exists(p)), None)
    if not cat_original:
        raise FileNotFoundError(
            "[SIM-NUM] No encuentro el catálogo de originales anomalias_comparacion_numerica.csv "
            f"en rutas candidatas: {cat_o_candidates}"
        )

    cat_s_candidates = [
        os.path.join(base_res, "Prediccion", "anomalias_sinteticas_csv", "anomalias_comparacion_numerica_sinteticas.csv"),
        os.path.join(base_res, "anomalias_comparacion_numerica_sinteticas.csv"),
        os.path.join(base_res, "Anomalias", "anomalias_comparacion_numerica_sinteticas.csv"),
    ]
    cat_sint = next((p for p in cat_s_candidates if os.path.exists(p)), None)
    if not cat_sint:
        raise FileNotFoundError(
            "[SIM-NUM] No encuentro el catálogo de sintéticas anomalias_comparacion_numerica_sinteticas.csv "
            f"en rutas candidatas: {cat_s_candidates}"
        )

    out_ttl = os.path.join(datos_onto, "7similarities_dtw.ttl")
    os.makedirs(os.path.dirname(out_ttl), exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    def _detect_delimiter(sample: str) -> str:
        if "\t" in sample:
            return "\t"
        if ";" in sample and sample.count(";") >= sample.count(","):
            return ";"
        return ","

    def _get_first(row: dict, keys: list, default=None):
        for k in keys:
            if k in row and row[k] is not None and str(row[k]).strip() != "":
                return row[k]
        return default

    def _safe_int(v):
        return int(float(str(v).strip()))

    def _safe_float(v):
        return float(str(v).strip())

    def _dec(v):
        return str(float(str(v).strip()))

    def _to_similarity_from_normdist(normdist):
        nd = _safe_float(normdist)
        return 1.0 / (1.0 + nd)

    def _ttl_term(s: str) -> str:
        if s is None:
            return ""
        t = str(s).strip()
        if not t:
            return ""
        if t.startswith("<") and t.endswith(">"):
            return t
        if t.startswith("http://") or t.startswith("https://"):
            return f"<{t}>"
        return t  # asume prefijo (anom:)

    def _derive_synth_iri_from_row(row: dict) -> str:
        # Solo para cat_sint si hiciera falta; tu CSV principal ya trae IRIs completas.
        serie_base = str(row.get("serie_base", "")).strip()
        mp = str(row.get("metodo_prediccion", "")).strip()
        sd = str(row.get("start_date", "")).strip()
        md = str(row.get("metodo_deteccion", "")).strip().lower()
        if not (serie_base and mp and sd and md):
            return ""
        date_u = sd.replace("-", "_")
        return f"anom:anomaly_syn_{serie_base}__PM_{mp}_{date_u}__dm_{md}"

    def _load_id_to_iri(
        csv_path: str,
        id_keys=("id_anomalia", "id"),
        iri_keys=("anomaly_iri",),
        encoding="utf-8-sig",
        allow_derive: bool = False,
    ):
        out = {}
        with open(csv_path, "r", encoding=encoding, newline="") as f:
            sample = f.read(4096)
            delim = _detect_delimiter(sample)
            f.seek(0)

            r = csv.DictReader(f, delimiter=delim)
            if not r.fieldnames:
                return out

            fn = {c.strip(): c for c in r.fieldnames}

            id_col = None
            for k in id_keys:
                if k in fn:
                    id_col = fn[k]
                    break

            iri_col = None
            for k in iri_keys:
                if k in fn:
                    iri_col = fn[k]
                    break

            if not id_col:
                return out

            for row in r:
                raw_id = row.get(id_col, "")
                if raw_id is None or str(raw_id).strip() == "":
                    continue
                try:
                    i = _safe_int(raw_id)
                except Exception:
                    continue

                raw_iri = row.get(iri_col, "") if iri_col else ""
                iri = str(raw_iri).strip() if raw_iri is not None else ""

                if (not iri) and allow_derive:
                    iri = _derive_synth_iri_from_row(row)

                if not iri:
                    continue

                out[i] = iri

        return out

    # ----------------------------
    # Cargar mapas canónicos de IRIs (fallback)
    # ----------------------------
    map_o = _load_id_to_iri(cat_original, id_keys=("id_anomalia", "id"), iri_keys=("anomaly_iri",))
    map_s = _load_id_to_iri(cat_sint, id_keys=("id_anomalia", "id"), iri_keys=("anomaly_iri",), allow_derive=True)

    if not map_o:
        raise ValueError(f"[SIM-NUM] No pude construir mapa id->anomaly_iri desde catálogo original: {cat_original}")
    if not map_s:
        raise ValueError(
            f"[SIM-NUM] No pude construir mapa id->anomaly_iri desde catálogo sintético: {cat_sint} "
            "(no existe anomaly_iri y no pude derivarla; revisa columnas serie_base/metodo_prediccion/start_date/metodo_deteccion)"
        )

    # ----------------------------
    # Recursos fijos v9
    # ----------------------------
    run_iri = "anom:SR_DTW_AllPairs"
    run_method_iri = "anom:SM_DTW"
    scale_0_1_iri = "anom:Scale_0_1"

    ttl_lines = [
        "@prefix anom: <http://w3id.org/anomaly-core#> .",
        "@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .",
        "",
        f"{run_iri}",
        "  rdf:type anom:SimilarityRun ;",
        f"  anom:runMethod {run_method_iri} ;",
        "  anom:selectionPolicy anom:ALL_PAIRS ;",
        f"  anom:sourceFile \"{os.path.basename(input_csv)}\"^^xsd:string .",
        "",
    ]

    processed = 0
    exported = 0
    skipped_no_metric = 0
    skipped_bad_ids = 0
    skipped_not_found_in_catalog = 0

    # ----------------------------
    # Leer similaridades_dtw.csv y exportar
    # ----------------------------
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        delimiter = _detect_delimiter(sample)
        f.seek(0)

        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = set(reader.fieldnames or [])

        # IDs requeridos (TU CSV usa id_anomalia_original/sintetica)
        req_map = {
            "original_id": ["original_id", "originalId", "id_original", "id_anomalia_original"],
            "synthetic_id": ["synthetic_id", "syntheticId", "id_sintetica", "id_anomalia_sintetica"],
        }
        missing = []
        for logical, options in req_map.items():
            if not any(opt in fieldnames for opt in options):
                missing.append(logical)
        if missing:
            raise ValueError(f"[SIM-NUM] Faltan columnas requeridas: {missing}")

        # IRIs directas (TU CSV trae betweenAnomalyA/B perfecto)
        iriA_keys = ["betweenAnomalyA", "anomaly_iri_original", "original_anomaly_iri", "original_iri", "anomalyA_iri"]
        iriB_keys = ["betweenAnomalyB", "anomaly_iri_sintetica", "synthetic_anomaly_iri", "synthetic_iri", "anomalyB_iri"]

        dtw_keys = ["similarity_dtw", "dtw_similarity"]
        feat_keys = ["similarity_feat", "feat_similarity", "similarity_features"]
        comb_keys = ["similarity_combined", "combined_similarity"]
        selector_keys = ["score_final_used"]  # selector: "combined" / "dtw" (en tu CSV)
        normdist_keys = ["dtw_distance_norm", "distance_norm_dtw"]

        rank_keys = ["rank_for_original", "rank"]
        ts_keys = ["run_timestamp", "computed_at", "timestamp"]

        for row in reader:
            processed += 1

            original_id = _get_first(row, req_map["original_id"])
            synthetic_id = _get_first(row, req_map["synthetic_id"])
            if original_id is None or synthetic_id is None:
                skipped_bad_ids += 1
                continue

            try:
                orig_n = _safe_int(original_id)
                syn_n = _safe_int(synthetic_id)
            except Exception:
                skipped_bad_ids += 1
                continue

            # ---- DTW (directo o derivado)
            dtw_out = None
            dtw_val = _get_first(row, dtw_keys)
            if dtw_val not in (None, ""):
                try:
                    dtw_out = _dec(dtw_val)
                except Exception:
                    dtw_out = None
            else:
                nd = _get_first(row, normdist_keys)
                if nd not in (None, ""):
                    try:
                        dtw_out = _dec(_to_similarity_from_normdist(nd))
                    except Exception:
                        dtw_out = None

            # ---- FEAT
            feat_out = None
            feat_val = _get_first(row, feat_keys)
            if feat_val not in (None, ""):
                try:
                    feat_out = _dec(feat_val)
                except Exception:
                    feat_out = None

            # ---- COMBINED
            comb_out = None
            comb_val = _get_first(row, comb_keys)
            if comb_val not in (None, ""):
                try:
                    comb_out = _dec(comb_val)
                except Exception:
                    comb_out = None

            # ---- FINAL (tu CSV: score_final_used es selector, NO número)
            final_out = None
            selector = _get_first(row, selector_keys)
            selector_norm = str(selector).strip().lower() if selector not in (None, "") else ""

            if selector_norm in ("combined", "comb", "ensemble"):
                if comb_out is not None:
                    final_out = comb_out
                elif dtw_out is not None:
                    final_out = dtw_out
                elif feat_out is not None:
                    final_out = feat_out
            elif selector_norm in ("dtw", "distance", "sim_dtw"):
                if dtw_out is not None:
                    final_out = dtw_out
                elif comb_out is not None:
                    final_out = comb_out
                elif feat_out is not None:
                    final_out = feat_out
            else:
                # Si algún día score_final_used viene numérico, lo intentamos parsear
                if selector_norm:
                    try:
                        final_out = _dec(selector_norm)
                    except Exception:
                        final_out = None

            # Fallback general: max(dtw, feat) o lo disponible
            if final_out is None:
                candidates = []
                if dtw_out is not None:
                    candidates.append(float(dtw_out))
                if feat_out is not None:
                    candidates.append(float(feat_out))
                if candidates:
                    final_out = str(max(candidates))
                elif comb_out is not None:
                    final_out = comb_out

            if final_out is None:
                skipped_no_metric += 1
                continue

            # ---- IRIs (prioridad CSV; fallback catálogos)
            iri_original = _get_first(row, iriA_keys)
            iri_syn = _get_first(row, iriB_keys)

            if not iri_original:
                iri_original = map_o.get(orig_n)
            if not iri_syn:
                iri_syn = map_s.get(syn_n)

            if not iri_original or not iri_syn:
                skipped_not_found_in_catalog += 1
                continue

            iri_original_t = _ttl_term(iri_original)
            iri_syn_t = _ttl_term(iri_syn)

            sim_iri = f"anom:sim_NUM_{orig_n}_{syn_n}"
            rank = _get_first(row, rank_keys)
            ts = _get_first(row, ts_keys)

            ttl_lines.append(f"{sim_iri}")
            ttl_lines.append("  rdf:type anom:Similarity ;")
            ttl_lines.append(f"  anom:betweenAnomalyA {iri_original_t} ;")
            ttl_lines.append(f"  anom:betweenAnomalyB {iri_syn_t} ;")
            ttl_lines.append(f"  anom:fromRun {run_iri} ;")
            ttl_lines.append(f"  anom:hasScale {scale_0_1_iri} ;")

            if dtw_out is not None:
                ttl_lines.append(f"  anom:similarityDTW \"{dtw_out}\"^^xsd:decimal ;")
            if feat_out is not None:
                ttl_lines.append(f"  anom:similarityFeat \"{feat_out}\"^^xsd:decimal ;")
            if comb_out is not None:
                ttl_lines.append(f"  anom:similarityCombined \"{comb_out}\"^^xsd:decimal ;")

            ttl_lines.append(f"  anom:similarityValue \"{final_out}\"^^xsd:decimal ;")

            if ts not in (None, ""):
                ttl_lines.append(f"  anom:computedAt \"{str(ts).strip()}\"^^xsd:dateTime ;")

            if rank not in (None, ""):
                try:
                    ttl_lines.append(f"  anom:rank \"{int(float(rank))}\"^^xsd:integer ;")
                except Exception:
                    pass

            ttl_lines[-1] = ttl_lines[-1].rstrip(" ;") + " ."
            ttl_lines.append("")
            exported += 1

    with open(out_ttl, "w", encoding="utf-8") as out:
        out.write("\n".join(ttl_lines))

    print(f"[SIM-NUM] Filas procesadas = {processed}")
    print(f"[SIM-NUM] Filas exportadas = {exported}")
    print(f"[SIM-NUM] Omitidas (sin métricas/final) = {skipped_no_metric}")
    print(f"[SIM-NUM] Omitidas (ids inválidos) = {skipped_bad_ids}")
    print(f"[SIM-NUM] Omitidas (id no encontrado en catálogos) = {skipped_not_found_in_catalog}")
    print(f"[SIM-NUM] Fuente usada: {input_csv}")
    print(f"[SIM-NUM] Catálogo original usado: {cat_original}")
    print(f"[SIM-NUM] Catálogo sintético usado: {cat_sint}")
    print(f"[SIM-NUM] TTL escrito en: {out_ttl}")
    return out_ttl

def _export_similarities_embeddings():
    """
    Exporta similitudes por embeddings desde similaridades_embeddings.csv a 8similarities_embeddings.ttl
    conforme al MODELO UNIFICADO v9 (Ontv9_simil).

    Cada fila => anom:Similarity (UNA SOLA métrica canónica)
      - betweenAnomalyA (original)   -> IRI REAL (CSV si existe, si no catálogo original)
      - betweenAnomalyB (synthetic)  -> IRI REAL (CSV si existe, si no catálogo sintético; o derivado)
      - computedBy anom:SM_ImageEmbeddings
      - fromRun anom:SR_Visual_TopK
      - hasScale anom:Scale_0_1
      - similarityValue (decimal) [obligatoria]
      - rank (opcional)

    Nota v9:
      - No exportar propiedades que NO existan en Ontv9_simil.ttl.
      - NO inventar IRIs.
    """
    import os
    import csv
    import re

    print("[SIM-EMB] Iniciando exportación de similitudes embeddings (v9)")

    base_res = getattr(CFG, "RUTA_RESULTADOS", None)
    ruta_comp = getattr(CFG, "RUTA_COMPARACION", None)
    datos_onto = getattr(CFG, "DATOS_ONTOLOGIA", None)

    if not datos_onto:
        raise ValueError("[SIM-EMB] CFG.DATOS_ONTOLOGIA no está definida.")
    if not (ruta_comp or base_res):
        raise ValueError("[SIM-EMB] Necesito CFG.RUTA_COMPARACION o CFG.RUTA_RESULTADOS definida.")

    # ----------------------------
    # Paths: input embeddings
    # ----------------------------
    input_candidates = []
    if ruta_comp:
        input_candidates.append(os.path.join(ruta_comp, "similaridades_embeddings.csv"))
    if base_res:
        input_candidates.extend([
            os.path.join(base_res, "ComparacionesEmbeddings", "similaridades_embeddings.csv"),
            os.path.join(base_res, "Comparaciones", "similaridades_embeddings.csv"),
        ])

    input_csv = next((p for p in input_candidates if p and os.path.exists(p)), None)
    if not input_csv:
        raise FileNotFoundError(f"[SIM-EMB] No existe similaridades_embeddings.csv en candidatos: {input_candidates}")

    out_ttl = os.path.join(datos_onto, "8similarities_embeddings.ttl")
    os.makedirs(os.path.dirname(out_ttl), exist_ok=True)

    # ----------------------------
    # Catálogos para resolver IRIs reales
    # ----------------------------
    if not base_res:
        raise ValueError("[SIM-EMB] CFG.RUTA_RESULTADOS es requerida para resolver catálogos (IRIs reales).")

    cat_o_candidates = [
        os.path.join(base_res, "anomalias_comparacion_numerica.csv"),
        os.path.join(base_res, "Anomalias", "anomalias_comparacion_numerica.csv"),
    ]
    cat_original = next((p for p in cat_o_candidates if os.path.exists(p)), None)
    if not cat_original:
        raise FileNotFoundError(
            "[SIM-EMB] No encuentro el catálogo de originales anomalias_comparacion_numerica.csv "
            f"en rutas candidatas: {cat_o_candidates}"
        )

    cat_s_candidates = [
        os.path.join(base_res, "Prediccion", "anomalias_sinteticas_csv", "anomalias_comparacion_numerica_sinteticas.csv"),
        os.path.join(base_res, "anomalias_comparacion_numerica_sinteticas.csv"),
        os.path.join(base_res, "Anomalias", "anomalias_comparacion_numerica_sinteticas.csv"),
    ]
    cat_sint = next((p for p in cat_s_candidates if os.path.exists(p)), None)
    if not cat_sint:
        raise FileNotFoundError(
            "[SIM-EMB] No encuentro el catálogo de sintéticas anomalias_comparacion_numerica_sinteticas.csv "
            f"en rutas candidatas: {cat_s_candidates}"
        )

    # ----------------------------
    # Helpers
    # ----------------------------
    def _detect_delimiter(sample: str) -> str:
        if "\t" in sample:
            return "\t"
        if ";" in sample and sample.count(";") >= sample.count(","):
            return ";"
        return ","

    def _get_first(row: dict, keys: list, default=None):
        for k in keys:
            if k in row and row[k] is not None and str(row[k]).strip() != "":
                return row[k]
        return default

    def _safe_int(v):
        return int(float(str(v).strip()))

    def _safe_decimal_str(v):
        return str(float(str(v).strip()))

    def _ttl_term(s: str) -> str:
        if s is None:
            return ""
        t = str(s).strip()
        if not t:
            return ""
        if t.startswith("<") and t.endswith(">"):
            return t
        if t.startswith("http://") or t.startswith("https://"):
            return f"<{t}>"
        return t  # asume prefijo (anom:)

    def _derive_synth_iri_from_catalog_row(row: dict) -> str:
        """
        Deriva el IRI sintético según el patrón REAL que confirmaste:

        anom:anomaly_syn_{serie_base}__PM_{metodo_prediccion}_{YYYY_MM_DD}__dm_{metodo_deteccion_lower}
        """
        serie_base = str(row.get("serie_base", "")).strip()
        mp = str(row.get("metodo_prediccion", "")).strip()
        sd = str(row.get("start_date", "")).strip()
        md = str(row.get("metodo_deteccion", "")).strip().lower()

        if not (serie_base and mp and sd and md):
            return ""
        date_u = sd.replace("-", "_")
        return f"anom:anomaly_syn_{serie_base}__PM_{mp}_{date_u}__dm_{md}"

    def _load_id_to_iri(csv_path: str, allow_derive: bool = False):
        """
        Retorna dict[int] -> anomaly_iri

        - Si existe columna anomaly_iri, la usa
        - Si allow_derive=True y no hay anomaly_iri (o está vacía), deriva (solo para sintéticas)
        """
        out = {}
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
            delim = _detect_delimiter(sample)
            f.seek(0)

            r = csv.DictReader(f, delimiter=delim)
            if not r.fieldnames:
                return out

            fn = {c.strip(): c for c in r.fieldnames}

            id_col = fn.get("id_anomalia") or fn.get("id") or fn.get("anom_id")
            iri_col = fn.get("anomaly_iri") or fn.get("iri") or fn.get("anom_iri")

            if not id_col:
                return out

            for row in r:
                raw_id = row.get(id_col, "")
                if raw_id is None or str(raw_id).strip() == "":
                    continue
                try:
                    i = _safe_int(raw_id)
                except Exception:
                    continue

                iri = ""
                if iri_col:
                    raw_iri = row.get(iri_col, "")
                    iri = str(raw_iri).strip() if raw_iri is not None else ""

                if (not iri) and allow_derive:
                    iri = _derive_synth_iri_from_catalog_row(row)

                if iri:
                    out[i] = iri

        return out

    # Mapas canónicos
    map_o = _load_id_to_iri(cat_original, allow_derive=False)
    map_s = _load_id_to_iri(cat_sint, allow_derive=True)

    if not map_o:
        raise ValueError(f"[SIM-EMB] No pude construir mapa id->anomaly_iri desde catálogo original: {cat_original}")
    if not map_s:
        raise ValueError(
            f"[SIM-EMB] No pude construir mapa id->anomaly_iri desde catálogo sintético: {cat_sint} "
            "(no existe anomaly_iri y no pude derivarla; revisa columnas serie_base/metodo_prediccion/start_date/metodo_deteccion)"
        )

    # ----------------------------
    # Recursos fijos v9
    # ----------------------------
    run_iri = "anom:SR_Visual_TopK"
    method_iri = "anom:SM_ImageEmbeddings"
    scale_iri = "anom:Scale_0_1"

    topk = getattr(CFG, "SIM_TOPK_VISUAL", None)
    try:
        topk_int = int(topk) if topk is not None else 2
    except Exception:
        topk_int = 2

    ttl_lines = [
        "@prefix anom: <http://w3id.org/anomaly-core#> .",
        "@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .",
        "",
        f"{run_iri}",
        "  rdf:type anom:SimilarityRun ;",
        f"  anom:runMethod {method_iri} ;",
        "  anom:selectionPolicy anom:TOP_K ;",
        f"  anom:topK \"{topk_int}\"^^xsd:integer ;",
        f"  anom:sourceFile \"{os.path.basename(input_csv)}\"^^xsd:string .",
        "",
    ]

    processed = 0
    exported = 0
    skipped_bad = 0
    skipped_no_score = 0
    skipped_not_found = 0

    # ----------------------------
    # Leer CSV embeddings y exportar
    # ----------------------------
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        delimiter = _detect_delimiter(sample)
        f.seek(0)

        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = set(reader.fieldnames or [])

        # IDs + score (aliases amplios)
        req_map = {
            "original_id": ["original_id", "originalId", "orig_id", "id_original", "idA", "anomalyA_id"],
            "synthetic_id": ["synthetic_id", "syntheticId", "syn_id", "id_synthetic", "idB", "anomalyB_id"],
            "score": ["similarity_cosine", "cosine_similarity", "similarity", "score", "sim", "cosine"],
        }
        missing = []
        for logical, options in req_map.items():
            if not any(opt in fieldnames for opt in options):
                missing.append(logical)
        if missing:
            raise ValueError(f"[SIM-EMB] Faltan columnas requeridas (según tu CSV): {missing}. Cols={sorted(fieldnames)}")

        # IRIs opcionales (si ya vienen en el CSV)
        iriA_keys = ["original_anomaly_iri", "original_iri", "anomalyA_iri", "original_anomalyIRI"]
        iriB_keys = ["synthetic_anomaly_iri", "synthetic_iri", "anomalyB_iri", "synthetic_anomalyIRI"]

        for row in reader:
            processed += 1

            original_id = _get_first(row, req_map["original_id"])
            synthetic_id = _get_first(row, req_map["synthetic_id"])
            score_val = _get_first(row, req_map["score"])

            if original_id is None or synthetic_id is None:
                skipped_bad += 1
                continue
            if score_val in (None, ""):
                skipped_no_score += 1
                continue

            try:
                orig_n = _safe_int(original_id)
                syn_n = _safe_int(synthetic_id)
            except Exception:
                skipped_bad += 1
                continue

            # Resolver IRIs reales (CSV -> catálogos)
            iri_original = _get_first(row, iriA_keys) or map_o.get(orig_n)
            iri_syn = _get_first(row, iriB_keys) or map_s.get(syn_n)

            if not iri_original or not iri_syn:
                skipped_not_found += 1
                continue

            iri_original_t = _ttl_term(iri_original)
            iri_syn_t = _ttl_term(iri_syn)

            # Similarity IRI estable
            sim_iri = f"anom:sim_GRAF_emb_{orig_n}_{syn_n}"

            rank = _get_first(row, ["rank_in_corpus", "rank_for_original", "rank"])

            ttl_lines.append(f"{sim_iri}")
            ttl_lines.append("  rdf:type anom:Similarity ;")
            ttl_lines.append(f"  anom:betweenAnomalyA {iri_original_t} ;")
            ttl_lines.append(f"  anom:betweenAnomalyB {iri_syn_t} ;")
            ttl_lines.append(f"  anom:computedBy {method_iri} ;")
            ttl_lines.append(f"  anom:fromRun {run_iri} ;")
            ttl_lines.append(f"  anom:hasScale {scale_iri} ;")
            ttl_lines.append(f"  anom:similarityValue \"{_safe_decimal_str(score_val)}\"^^xsd:decimal ;")

            if rank not in (None, ""):
                try:
                    ttl_lines.append(f"  anom:rank \"{int(float(rank))}\"^^xsd:integer ;")
                except Exception:
                    pass

            # IMPORTANTE: evidenceFile solo si tu Ontv9_simil lo define.
            # Si confirmas que existe, lo activamos. Por defecto, NO lo exporto para no desalinear v9.
            # synthetic_path = _get_first(row, ["synthetic_path", "synthetic_png", "png_sintetico"])
            # if synthetic_path:
            #     base = os.path.basename(str(synthetic_path).strip())
            #     if re.search(r"\.(png|jpg|jpeg)$", base, flags=re.IGNORECASE):
            #         ttl_lines.append(f"  anom:evidenceFile \"{base}\"^^xsd:string ;")

            ttl_lines[-1] = ttl_lines[-1].rstrip(" ;") + " ."
            ttl_lines.append("")
            exported += 1

    with open(out_ttl, "w", encoding="utf-8") as out:
        out.write("\n".join(ttl_lines))

    print(f"[SIM-EMB] Filas procesadas = {processed}")
    print(f"[SIM-EMB] Filas exportadas = {exported}")
    print(f"[SIM-EMB] Omitidas (bad ids) = {skipped_bad}")
    print(f"[SIM-EMB] Omitidas (sin score) = {skipped_no_score}")
    print(f"[SIM-EMB] Omitidas (no resueltas en catálogos) = {skipped_not_found}")
    print(f"[SIM-EMB] Catálogo original usado: {cat_original}")
    print(f"[SIM-EMB] Catálogo sintético usado: {cat_sint}")
    print(f"[SIM-EMB] TTL escrito en: {out_ttl}")
    return out_ttl




def generate_ttl():
    """
    Genera archivos TTL para exportar la ontología de resultados.
    Retorna un dict con las rutas generadas.
    """
    out = {}

    out["1series"] = _export_series()
    out["2points"] = _export_points()
    out["3anomalies"] = _export_anomalies()
    out["4series_sinteticas"] = _export_series_sinteticas()
    out["5points_sinteticos"] = _export_points_sinteticos()
    out["6anomalies_sinteticas"] = _export_anomalies_sinteticas()
    out["7similarities_dtw"] = _export_similarities_dtw()
    out["8similarities_embeddings"] = _export_similarities_embeddings()

    return out


