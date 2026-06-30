import streamlit as st
import pandas as pd
import numpy as np
import io
import unicodedata
from datetime import datetime, time

# ── Fuzzy matching con normalización ──────────────────────────────────────────
try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

def remove_accents(text: str) -> str:
    """Remueve tildes y normaliza el texto a minúsculas limpias."""
    if not text:
        return ""
    text = str(text).strip().lower()
    return "".join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def fuzzy_match(value: str, patterns: list, threshold: int = 75) -> bool:
    if not value or not patterns:
        return False
    val_norm = remove_accents(value)
    for p in patterns:
        pat_norm = remove_accents(p)
        if not pat_norm:
            continue
        # Coincidencia exacta o contención directa
        if pat_norm in val_norm or val_norm in pat_norm:
            return True
        # Coincidencia aproximada si rapidfuzz está disponible
        if FUZZY_AVAILABLE:
            score = max(
                fuzz.ratio(val_norm, pat_norm),
                fuzz.partial_ratio(val_norm, pat_norm),
                fuzz.token_sort_ratio(val_norm, pat_norm),
            )
            if score >= threshold:
                return True
    return False


# ── Constantes y Mapeos ──────────────────────────────────────────────────────
FRANJA_MAP = [
    ((0, 0), (6, 0),  "0h a 6h"),
    ((6, 0), (12, 0), "6h a 12h"),
    ((12,0), (18, 0), "12h a 18h"),
    ((18,0), (24, 0), "18h a 0h"),
]

DIAS_ES = {
    "Monday":"Lunes","Tuesday":"Martes","Wednesday":"Miércoles",
    "Thursday":"Jueves","Friday":"Viernes","Saturday":"Sábado","Sunday":"Domingo",
}

NUMERIC_COLS = [
    "comments","shares","wow","love","like","haha","sad","angry",
    "thankful","views","retweet","favs","hearts","likes","dislikes",
    "fans","followers","Interacciones totales",
]

FINAL_COLS = [
    "ID","Título","Descripción","url","Fecha","Hora","Franja","Día",
    "Red Social","Tono","Autor","cumulative_reach","Interacciones",
    "Tipo específico","Alcance","Vistas",
]


# ── Helpers de formato y conversión ──────────────────────────────────────────
def get_franja(t):
    if pd.isnull(t):
        return ""
    try:
        cur = t.hour * 60 + t.minute
    except AttributeError:
        return ""
    for (sh,sm),(eh,em),label in FRANJA_MAP:
        if sh*60+sm <= cur < eh*60+em:
            return label
    return "18h a 0h"

def safe_int(v):
    try:
        if pd.isna(v):
            return 0
        f = float(v)
        return 0 if np.isnan(f) else int(f)
    except (TypeError, ValueError):
        return 0


# ── Limpieza principal de Datos ──────────────────────────────────────────────
def clean_df(df, own_authors, exclude_authors, exclude_keywords, fuzzy_threshold=75):
    # 1. Quitar Título original si existe para recrearlo de forma uniforme
    if "Título" in df.columns:
        df = df.drop(columns=["Título"])

    # 2. Convertir columnas numéricas de manera segura a enteros
    for col in [c for c in NUMERIC_COLS if c in df.columns]:
        df[col] = df[col].apply(safe_int)

    # 3. Renombrar columnas clave
    df = df.rename(columns={
        "id": "ID",
        "Link para la fuente": "url",
        "Grupo de dominio": "Red Social",
        "Creado": "FechaHora",
        "Interacciones totales": "Interacciones",
    })

    # 4. Homologar Twitter a X
    if "Red Social" in df.columns:
        df["Red Social"] = df["Red Social"].str.replace("Twitter", "X", regex=False)

    # 5. Generar Título y Descripción basados en el Contenido
    col_c = "Contenido de la publicación"
    if col_c in df.columns:
        df["Título"]      = df[col_c].fillna("").astype(str)
        df["Descripción"] = df["Título"]
        df = df.drop(columns=[col_c])
    else:
        df["Título"] = df.get("Título", "")
        df["Descripción"] = df["Título"]

    # 6. Procesar fechas, horas, franjas y días en español
    if "FechaHora" in df.columns:
        df["FechaHora"] = pd.to_datetime(df["FechaHora"], errors="coerce")
        df["Fecha"]  = df["FechaHora"].dt.date
        df["Hora"]   = df["FechaHora"].dt.time
        df["Franja"] = df["FechaHora"].apply(get_franja)
        df["Día"]    = df["FechaHora"].dt.day_name().map(DIAS_ES).fillna("")
        df = df.drop(columns=["FechaHora"])
    else:
        for c in ["Fecha","Hora","Franja","Día"]:
            if c not in df.columns:
                df[c] = ""

    # 7. Calcular cumulative_reach y Alcance (Suma de fans + followers)
    fans_series = df["fans"].apply(safe_int) if "fans" in df.columns else pd.Series(0, index=df.index)
    followers_series = df["followers"].apply(safe_int) if "followers" in df.columns else pd.Series(0, index=df.index)
    
    reach_sum = fans_series + followers_series
    df["cumulative_reach"] = reach_sum
    df["Alcance"]          = reach_sum

    # 8. Calcular Vistas (Basado en views)
    df["Vistas"] = df["views"].apply(safe_int) if "views" in df.columns else pd.Series(0, index=df.index)

    # 9. Formatear interacciones a enteros
    if "Interacciones" in df.columns:
        df["Interacciones"] = df["Interacciones"].apply(safe_int)
    else:
        df["Interacciones"] = 0

    # 10. Clasificación inteligente de Tono (Fuzzy con autores)
    def assign_tono(row):
        autor = str(row.get("Autor","") or "")
        sentimiento = str(row.get("Sentimiento","") or "").upper()
        if fuzzy_match(autor, own_authors, fuzzy_threshold):
            return "Positivo"
        return {"NEUTRAL": "Neutro", "POSITIVE": "Positivo", "NEGATIVE": "Negativo"}.get(sentimiento, "Neutro")

    df["Tono"] = df.apply(assign_tono, axis=1)

    # 11. Filtros de exclusión selectivos
    if "Autor" in df.columns and exclude_authors:
        df = df[~df["Autor"].apply(lambda a: fuzzy_match(str(a), exclude_authors, fuzzy_threshold))]

    if "Título" in df.columns and exclude_keywords:
        df = df[~df["Título"].apply(
            lambda t: any(kw.lower() in str(t).lower() for kw in exclude_keywords if kw)
        )]

    # 12. Garantizar ID como formato texto plano
    if "ID" in df.columns:
        df["ID"] = df["ID"].astype(str)

    # 13. Mantener y reordenar solo las columnas finales requeridas
    df = df[[c for c in FINAL_COLS if c in df.columns]]
    return df.reset_index(drop=True)


def df_to_excel(df):
    """Escribe el DataFrame a un buffer en formato Excel aplicando formato numérico nativo de miles."""
    buf = io.BytesIO()
    cols_numericas = ["cumulative_reach", "Interacciones", "Alcance", "Vistas"]
    
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Datos")
        ws = writer.sheets["Datos"]
        
        # Obtener los índices de las columnas que queremos formatear (base 1 para openpyxl)
        col_indices = {col_name: idx + 1 for idx, col_name in enumerate(df.columns)}
        
        # Aplicar formato de miles a las celdas de datos numéricos
        for r_idx in range(2, ws.max_row + 1):
            for col_name in cols_numericas:
                if col_name in col_indices:
                    cell = ws.cell(row=r_idx, column=col_indices[col_name])
                    if cell.value is not None:
                        try:
                            # Asegurar valor numérico puro
                            cell.value = int(float(cell.value))
                            # Formato numérico estándar de Excel para separación de miles sin decimales
                            cell.number_format = '#,##0'
                        except (ValueError, TypeError):
                            pass

        # Auto-ajustar dinámicamente el ancho de las columnas
        for col_cells in ws.columns:
            max_len = max((len(str(c.value)) if c.value else 0) for c in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 4, 60)
            
    return buf.getvalue()


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA DE STREAMLIT (ESTILO INTERFAZ CLAUDE)
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Procesador de Monitoreo RRSS",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Hojas de Estilo CSS Personalizadas ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ─ Fondo del canvas general ─ */
.stApp { background: #f8fafc !important; }
.main .block-container {
    background: #f8fafc;
    padding: 2.5rem 3.5rem !important;
    max-width: 1300px;
}

/* ─ Menú lateral minimalista (Blanco absoluto) ─ */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 2rem 1.5rem !important;
}

/* Elementos de entrada del Sidebar */
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] input {
    background: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] textarea:focus,
[data-testid="stSidebar"] input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.15) !important;
}

/* Cabecera del Panel Principal */
.app-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding-bottom: 1.8rem;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 2rem;
}
.app-logo {
    width: 38px; height: 38px;
    background: #0f172a;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; color: #ffffff; flex-shrink: 0;
}
.app-header-text h1 {
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    margin: 0 !important; padding: 0 !important;
    letter-spacing: -0.5px;
}
.app-header-text p {
    font-size: 0.82rem !important;
    color: #64748b !important;
    margin: 2px 0 0 0 !important;
}

/* Etiquetas de sección */
.section-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #94a3b8;
    font-weight: 600;
    margin-bottom: 8px;
}

/* Tarjetas de métricas fluidas */
.metric-row {
    display: flex;
    gap: 12px;
    margin: 1.5rem 0;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 130px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.2rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}
.metric-card .val {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.5px;
}
.metric-card .lbl {
    font-size: 0.68rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: .5px;
    margin-top: 4px;
    font-weight: 500;
}

/* Etiquetas visuales en forma de píldoras */
.pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 500;
    margin: 2px;
}
.pill-pos { background: #f0fdf4; color: #166534; border: 1px solid #dcfce7; }

/* Mensajes informativos integrados */
.callout {
    background: #f8fafc;
    border-left: 3px solid #64748b;
    border-radius: 0 6px 6px 0;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #334155;
    margin: 1rem 0;
}
.callout.ok { background: #f0fdf4; border-color: #22c55e; color: #166534; }
.callout.warn { background: #fffbeb; border-color: #f59e0b; color: #78350f; }

/* Botón de acción principal */
div.stButton > button {
    background: #0f172a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.6rem 1.8rem !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    transition: background 0.15s ease;
}
div.stButton > button:hover {
    background: #1e293b !important;
}

/* Descargas */
[data-testid="stDownloadButton"] > button {
    background: #4f46e5 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.6rem 1.8rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #4338ca !important;
}

.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    background: #ffffff;
    border-radius: 12px;
    border: 1px dashed #cbd5e1;
    margin-top: 1rem;
}
.empty-state .icon { font-size: 1.8rem; margin-bottom: 0.5rem; color: #94a3b8; }
.empty-state .title { font-weight: 600; color: #334155; font-size: 0.95rem; }
.empty-state .sub   { font-size: 0.82rem; color: #64748b; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Cabecera de la aplicación ────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-logo">✦</div>
  <div class="app-header-text">
    <h1>Procesador de Monitoreo RRSS</h1>
    <p>Estandarización y filtrado dinámico para archivos de reporte de redes sociales</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PANEL DE CONTROL (SIDEBAR)
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="section-label">Configuración de Marcas</div>', unsafe_allow_html=True)

    st.markdown("**Asignación: Tono Positivo**")
    st.caption("Escribe un autor o palabra clave por línea. Se evaluará de forma flexible (sin distinguir acentos).")
    own_authors_raw = st.text_area(
        "Autores propios",
        height=140,
        placeholder="@fenavi\nFenavi Bogotá\nfenavi colombia",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Exclusiones (Autores)**")
    st.caption("Un autor por línea a descartar.")
    exclude_authors_raw = st.text_area(
        "Excluir autores",
        height=90,
        placeholder="@falsonoticias\nBot_Spam",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Exclusiones (Palabras Clave en Título)**")
    st.caption("Una frase o término por línea a descartar.")
    exclude_kw_raw = st.text_area(
        "Excluir palabras clave",
        height=90,
        placeholder="sorteo nacional\ncomprar seguidores",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Umbral de Precisión (Fuzzy)**")
    fuzzy_threshold = st.slider(
        "Sensibilidad de coincidencia",
        min_value=50, max_value=100, value=75, step=5,
        help="Un valor más bajo es más flexible con diferencias de escritura. 100 requiere coincidencia exacta.",
    )
    if not FUZZY_AVAILABLE:
        st.markdown(
            '<div class="callout warn">⚠️ <b>rapidfuzz</b> no está instalado en el entorno actual. '
            'Usando búsqueda de subcadena tradicional.</div>',
            unsafe_allow_html=True,
        )


# ── Extracción limpia de entradas de texto ──────────────────────────────────
def parse_lines(raw):
    return [l.strip() for l in (raw or "").splitlines() if l.strip()]

own_authors      = parse_lines(own_authors_raw)
exclude_authors  = parse_lines(exclude_authors_raw)
exclude_keywords = parse_lines(exclude_kw_raw)


# ════════════════════════════════════════════════════════════════════════════
# CONTENEDOR PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════
col_up, col_guide = st.columns([2, 1], gap="large")

with col_up:
    st.markdown('<div class="section-label">Archivos de entrada</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Carga tus archivos en formato .xlsx",
        type=["xlsx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

with col_guide:
    st.markdown('<div class="section-label">Guía de uso</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.82rem; color: #475569; line-height: 1.5;">
        1. Configura tus marcas y reglas de tono en la barra lateral.<br>
        2. Arrastra y suelta tus archivos Excel.<br>
        3. Presiona el botón <b>"Procesar Datos"</b>.<br>
        4. Comprueba los resultados y descarga los archivos limpios.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PROCESAMIENTO ACTIVO DE DATOS
# ════════════════════════════════════════════════════════════════════════════
if uploaded_files:
    st.markdown('<div class="section-label" style="margin-top:2rem;">Acción</div>', unsafe_allow_html=True)
    
    # Se añade un botón manual para activar la ejecución de la limpieza
    if st.button("✨ Procesar Datos", use_container_width=True):
        st.session_state["processed_results"] = {}
        
        for uploaded in uploaded_files:
            try:
                # Leer todo como texto para evitar pérdida de enteros o IDs numéricos inicialmente
                df_raw = pd.read_excel(uploaded, dtype=str)
            except Exception as e:
                st.error(f"Error al leer el archivo {uploaded.name}: {e}")
                continue

            # Determinación inteligente del Nombre de Proyecto
            project_name = uploaded.name.replace(".xlsx", "")
            if "Nombre del proyecto" in df_raw.columns:
                valid_names = df_raw["Nombre del proyecto"].dropna().unique()
                if len(valid_names) > 0 and str(valid_names[0]).strip():
                    project_name = str(valid_names[0]).strip()

            # Procesamiento de limpieza principal
            df_clean = clean_df(
                df_raw.copy(),
                own_authors=own_authors,
                exclude_authors=exclude_authors,
                exclude_keywords=exclude_keywords,
                fuzzy_threshold=fuzzy_threshold,
            )
            
            # Guardado en estado de sesión para persistencia
            st.session_state["processed_results"][uploaded.name] = {
                "df_clean": df_clean,
                "project_name": project_name,
                "raw_count": len(df_raw),
                "clean_count": len(df_clean)
            }

    # Despliegue de resultados si existen en caché de sesión
    if "processed_results" in st.session_state and st.session_state["processed_results"]:
        for name, data in st.session_state["processed_results"].items():
            st.divider()
            st.markdown(f"#### 📄 `{name}`")
            
            df_clean = data["df_clean"]
            n_raw = data["raw_count"]
            n_clean = data["clean_count"]
            n_removed = n_raw - n_clean
            
            tono_counts = df_clean["Tono"].value_counts() if "Tono" in df_clean.columns else pd.Series()
            pos = int(tono_counts.get("Positivo", 0))
            neu = int(tono_counts.get("Neutro", 0))
            neg = int(tono_counts.get("Negativo", 0))

            # Renderizado de Tarjetas de Métricas en el Dashboard
            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-card">
                <div class="val">{n_raw:,}</div>
                <div class="lbl">Entradas iniciales</div>
              </div>
              <div class="metric-card">
                <div class="val">{n_clean:,}</div>
                <div class="lbl">Resultados finales</div>
              </div>
              <div class="metric-card">
                <div class="val">{n_removed:,}</div>
                <div class="lbl">Filtrados</div>
              </div>
              <div class="metric-card">
                <div class="val" style="color:#166534">{pos:,}</div>
                <div class="lbl">Positivo</div>
              </div>
              <div class="metric-card">
                <div class="val" style="color:#475569">{neu:,}</div>
                <div class="lbl">Neutro</div>
              </div>
              <div class="metric-card">
                <div class="val" style="color:#991b1b">{neg:,}</div>
                <div class="lbl">Negativo</div>
              </div>
            </div>
            """.replace(",", "."), unsafe_allow_html=True)

            # Autores propios detectados
            if own_authors and "Autor" in df_clean.columns:
                matched = df_clean[df_clean["Tono"] == "Positivo"]["Autor"].dropna().unique()
                if len(matched) > 0:
                    with st.expander(f"✦ {len(matched)} Autores propios validados como Positivo"):
                        pills_html = " ".join(f'<span class="pill pill-pos">{a}</span>' for a in sorted(matched))
                        st.markdown(pills_html, unsafe_allow_html=True)

            # Vista Previa Dinámica
            with st.expander("Ver muestra estructurada del resultado (Primeras 50 filas)"):
                preview_cols = [c for c in [
                    "Fecha", "Red Social", "Tono", "Autor", "Título", 
                    "Interacciones", "Alcance", "Vistas", "Tipo específico"
                ] if c in df_clean.columns]
                
                # Renderizar números con formato de coma/punto de miles legible en pantalla
                st.dataframe(
                    df_clean[preview_cols].head(50), 
                    use_container_width=True
                )

            # Generación de Descargas
            final_filename = f"{data['project_name']}.xlsx"
            excel_bytes = df_to_excel(df_clean)

            st.markdown(
                f'<div class="callout ok">El archivo ha sido formateado correctamente con el formato numérico '
                f'estándar. Se han conservado <b>{n_clean}</b> registros válidos de un total original de <b>{n_raw}</b>.</div>',
                unsafe_allow_html=True,
            )
            
            st.download_button(
                label=f"⬇ Descargar {final_filename}",
                data=excel_bytes,
                file_name=final_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_{name}",
            )
else:
    st.markdown("""
    <div class="empty-state">
      <div class="icon">✦</div>
      <div class="title">Bandeja vacía</div>
      <div class="sub">Por favor, carga uno o varios archivos en la zona designada para iniciar el procesamiento.</div>
    </div>
    """, unsafe_allow_html=True)
