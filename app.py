import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from datetime import datetime, time

# ── Fuzzy matching (graceful fallback si no hay rapidfuzz) ──────────────────
try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

def fuzzy_match(value: str, patterns: list[str], threshold: int = 75) -> bool:
    """Devuelve True si 'value' coincide con alguno de los patrones al nivel de threshold."""
    if not value or not patterns:
        return False
    val = str(value).strip().lower()
    for p in patterns:
        pat = str(p).strip().lower()
        if not pat:
            continue
        # Coincidencia exacta siempre gana
        if pat in val or val in pat:
            return True
        # Fuzzy solo si está disponible
        if FUZZY_AVAILABLE:
            score = max(
                fuzz.ratio(val, pat),
                fuzz.partial_ratio(val, pat),
                fuzz.token_sort_ratio(val, pat),
            )
            if score >= threshold:
                return True
        else:
            # Fallback: coincidencia parcial simple
            if pat in val:
                return True
    return False


# ── Lógica de limpieza (equivalente al código M) ────────────────────────────

FRANJA_MAP = [
    ((0, 0), (6, 0), "0h a 6h"),
    ((6, 0), (12, 0), "6h a 12h"),
    ((12, 0), (18, 0), "12h a 18h"),
    ((18, 0), (24, 0), "18h a 0h"),
]

DIAS_ES = {
    "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
    "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo",
}

NUMERIC_COLS = [
    "comments", "shares", "wow", "love", "like", "haha", "sad", "angry",
    "thankful", "views", "retweet", "favs", "hearts", "likes", "dislikes",
    "fans", "followers", "Interacciones totales",
]

FINAL_COLS = [
    "ID", "Título", "Descripción", "url", "Fecha", "Hora", "Franja", "Día",
    "Red Social", "Tono", "Autor", "cumulative_reach", "Interacciones",
    "Tipo específico", "Alcance", "Vistas",
]

def get_franja(t):
    if pd.isnull(t):
        return ""
    try:
        h, m = t.hour, t.minute
    except AttributeError:
        return ""
    for (sh, sm), (eh, em), label in FRANJA_MAP:
        start = sh * 60 + sm
        end = eh * 60 + em
        cur = h * 60 + m
        if start <= cur < end:
            return label
    return "18h a 0h"

def fmt_co(n):
    """Formato colombiano con punto como separador de miles."""
    try:
        v = int(float(n))
        return f"{v:,}".replace(",", ".")
    except (ValueError, TypeError):
        return "0"

def safe_int(v):
    try:
        f = float(v)
        if np.isnan(f):
            return 0
        return int(f)
    except (TypeError, ValueError):
        return 0

def clean_df(
    df: pd.DataFrame,
    own_authors: list[str],
    exclude_authors: list[str],
    exclude_keywords: list[str],
    fuzzy_threshold: int = 75,
) -> pd.DataFrame:

    # ── 1. Eliminar Título si existe (se recrea luego) ──────────────────────
    if "Título" in df.columns:
        df = df.drop(columns=["Título"])

    # ── 2. Tipos y nulos numéricos ──────────────────────────────────────────
    present_num = [c for c in NUMERIC_COLS if c in df.columns]
    for col in present_num:
        df[col] = df[col].apply(safe_int)

    # ── 3. Renombrar ────────────────────────────────────────────────────────
    rename_map = {
        "id": "ID",
        "Link para la fuente": "url",
        "Grupo de dominio": "Red Social",
        "Creado": "FechaHora",
        "Interacciones totales": "Interacciones",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # ── 4. Twitter → X ──────────────────────────────────────────────────────
    if "Red Social" in df.columns:
        df["Red Social"] = df["Red Social"].str.replace("Twitter", "X", regex=False)

    # ── 5. Título y Descripción desde contenido ──────────────────────────────
    contenido_col = "Contenido de la publicación"
    if contenido_col in df.columns:
        df["Título"] = df[contenido_col].fillna("").astype(str)
        df["Descripción"] = df["Título"]
        df = df.drop(columns=[contenido_col])
    else:
        df["Título"] = ""
        df["Descripción"] = ""

    # ── 6. Fecha, Hora, Franja, Día ─────────────────────────────────────────
    if "FechaHora" in df.columns:
        df["FechaHora"] = pd.to_datetime(df["FechaHora"], errors="coerce")
        df["Fecha"] = df["FechaHora"].dt.date
        df["Hora"] = df["FechaHora"].dt.time
        df["Franja"] = df["FechaHora"].apply(get_franja)
        df["Día"] = df["FechaHora"].dt.day_name().map(DIAS_ES).fillna("")
        df = df.drop(columns=["FechaHora"])
    else:
        for col in ["Fecha", "Hora", "Franja", "Día"]:
            if col not in df.columns:
                df[col] = ""

    # ── 7. cumulative_reach ─────────────────────────────────────────────────
    fans_col = "fans" if "fans" in df.columns else None
    followers_col = "followers" if "followers" in df.columns else None
    reach_raw = (
        (df[fans_col].apply(safe_int) if fans_col else 0) +
        (df[followers_col].apply(safe_int) if followers_col else 0)
    )
    df["cumulative_reach"] = reach_raw.apply(fmt_co)

    # ── 8. Alcance (suma fans+followers, numérico para la columna final) ────
    df["Alcance"] = reach_raw.apply(fmt_co)

    # ── 9. Vistas ───────────────────────────────────────────────────────────
    views_col = "views" if "views" in df.columns else None
    df["Vistas"] = (df[views_col].apply(safe_int) if views_col else 0).apply(fmt_co)

    # ── 10. Interacciones formateadas ───────────────────────────────────────
    if "Interacciones" in df.columns:
        df["Interacciones"] = df["Interacciones"].apply(safe_int).apply(fmt_co)

    # ── 11. Tono ────────────────────────────────────────────────────────────
    def assign_tono(row):
        autor = str(row.get("Autor", "") or "")
        sentimiento = str(row.get("Sentimiento", "") or "")
        # Autores propios → siempre Positivo (fuzzy)
        if fuzzy_match(autor, own_authors, fuzzy_threshold):
            return "Positivo"
        if sentimiento == "NEUTRAL":
            return "Neutro"
        if sentimiento == "POSITIVE":
            return "Positivo"
        if sentimiento == "NEGATIVE":
            return "Negativo"
        return sentimiento

    df["Tono"] = df.apply(assign_tono, axis=1)

    # ── 12. Filtros de exclusión ─────────────────────────────────────────────
    if "Autor" in df.columns and exclude_authors:
        mask_autor = df["Autor"].apply(
            lambda a: fuzzy_match(str(a), exclude_authors, fuzzy_threshold)
        )
        df = df[~mask_autor]

    if "Título" in df.columns and exclude_keywords:
        mask_kw = df["Título"].apply(
            lambda t: any(
                kw.lower() in str(t).lower() for kw in exclude_keywords if kw
            )
        )
        df = df[~mask_kw]

    # ── 13. ID como texto ────────────────────────────────────────────────────
    if "ID" in df.columns:
        df["ID"] = df["ID"].astype(str)

    # ── 14. Columnas finales ─────────────────────────────────────────────────
    available = [c for c in FINAL_COLS if c in df.columns]
    df = df[available]

    return df.reset_index(drop=True)


def df_to_excel(df: pd.DataFrame) -> bytes:
    """Exporta el DataFrame a bytes XLSX con formato básico."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Datos")
        ws = writer.sheets["Datos"]
        # Ancho automático de columnas
        for col_cells in ws.columns:
            max_len = max(
                (len(str(cell.value)) if cell.value is not None else 0)
                for cell in col_cells
            )
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 4, 60)
    return output.getvalue()


# ════════════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Limpiador RRSS",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f11;
    border-right: 1px solid #1e1e24;
}
[data-testid="stSidebar"] * {
    color: #e8e8f0 !important;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea {
    background: #18181f !important;
    border: 1px solid #2a2a35 !important;
    color: #e8e8f0 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stSlider > div { color: #e8e8f0 !important; }

/* Main */
.main { background: #fafafa; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1200px; }

/* Header strip */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 0 0 1.8rem 0;
    border-bottom: 1px solid #e2e2e8;
    margin-bottom: 1.8rem;
}
.app-header .logo {
    width: 40px; height: 40px;
    background: #0f0f11;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; color: #fff;
}
.app-header h1 {
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    color: #0f0f11 !important;
    margin: 0 !important; padding: 0 !important;
    letter-spacing: -0.3px;
}
.app-header p {
    font-size: 0.8rem !important;
    color: #888 !important;
    margin: 0 !important;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #d0d0db !important;
    border-radius: 12px !important;
    background: #f7f7fb !important;
    padding: 1rem !important;
}

/* Metric tiles */
.metric-row { display: flex; gap: 12px; margin: 1.2rem 0; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 120px;
    background: #fff;
    border: 1px solid #e8e8ee;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,.04);
}
.metric-card .val {
    font-size: 1.7rem;
    font-weight: 700;
    color: #0f0f11;
    letter-spacing: -1px;
    line-height: 1;
}
.metric-card .lbl {
    font-size: 0.72rem;
    color: #999;
    text-transform: uppercase;
    letter-spacing: .7px;
    margin-top: 4px;
}

/* Pill tags */
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 500;
    margin: 2px;
}
.pill-pos { background: #e6f9f0; color: #1a7a4a; }
.pill-neg { background: #fdecea; color: #b91c1c; }
.pill-neu { background: #f0f0f5; color: #555; }

/* Section headers */
.section-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #aaa;
    font-weight: 600;
    margin-bottom: 8px;
}

/* Download button */
[data-testid="stDownloadButton"] button {
    background: #0f0f11 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.4rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: -0.2px;
    width: 100%;
    transition: opacity .2s;
}
[data-testid="stDownloadButton"] button:hover { opacity: 0.85 !important; }

/* Expander */
[data-testid="stExpander"] { border: 1px solid #e8e8ee !important; border-radius: 10px !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* Step badges */
.step-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 24px; height: 24px;
    background: #0f0f11; color: #fff;
    border-radius: 50%;
    font-size: 0.7rem; font-weight: 700;
    margin-right: 8px; flex-shrink: 0;
}
.step-row { display: flex; align-items: center; margin-bottom: .5rem; font-size: .9rem; color: #333; }

/* Info callout */
.callout {
    background: #f0f4ff;
    border-left: 3px solid #3b5bdb;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 0.83rem;
    color: #333;
    margin: .8rem 0;
}
.callout.warn { background: #fffbeb; border-color: #d97706; }
.callout.ok   { background: #e6f9f0; border-color: #1a7a4a; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="logo">✦</div>
    <div>
        <h1>Limpiador RRSS</h1>
        <p>Procesador de datos de redes sociales · Formato estándar de análisis</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — configuración de marca
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="section-label">Configuración de marca</div>', unsafe_allow_html=True)

    st.markdown("**Autores propios → Tono Positivo**")
    st.caption("Un autor por línea. Acepta fragmentos o variantes (fuzzy matching).")
    own_authors_raw = st.text_area(
        "Autores propios",
        height=130,
        placeholder="@fenavi\nFenavi Bogotá\nfenavi colombia",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Autores a excluir**")
    st.caption("Un autor por línea.")
    exclude_authors_raw = st.text_area(
        "Excluir autores",
        height=100,
        placeholder="@fenavideve\nFederación Venezuela",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Palabras clave a excluir del Título**")
    st.caption("Una palabra/frase por línea (insensible a mayúsculas).")
    exclude_kw_raw = st.text_area(
        "Excluir palabras clave",
        height=100,
        placeholder="#Fenavidevenezuela\nFederación Nacional de Avicultura de Venezuela",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Umbral fuzzy matching**")
    fuzzy_threshold = st.slider(
        "Similitud mínima (%)",
        min_value=50, max_value=100, value=75, step=5,
        help="75 = tolerante a errores pequeños · 95 = casi exacto",
    )
    if not FUZZY_AVAILABLE:
        st.markdown('<div class="callout warn">⚠️ <b>rapidfuzz</b> no instalado. Se usa coincidencia parcial simple. Instala: <code>pip install rapidfuzz</code></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Acerca de</div>', unsafe_allow_html=True)
    st.caption("Aplica las mismas reglas del query M de Power Query: tipos, nulos, renombrado, Tono, franjas horarias, Alcance, Vistas.")


# ── Parse inputs ─────────────────────────────────────────────────────────────
def parse_lines(raw: str) -> list[str]:
    return [l.strip() for l in raw.splitlines() if l.strip()]

own_authors = parse_lines(own_authors_raw)
exclude_authors = parse_lines(exclude_authors_raw)
exclude_keywords = parse_lines(exclude_kw_raw)

# ════════════════════════════════════════════════════════════════════════════
# MAIN — carga de archivos
# ════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.markdown('<div class="section-label">Archivos de entrada</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Sube uno o más archivos .xlsx",
        type=["xlsx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

with col_right:
    st.markdown('<div class="section-label">Guía rápida</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="step-row"><span class="step-badge">1</span> Configura la marca en el panel izquierdo</div>
<div class="step-row"><span class="step-badge">2</span> Sube el xlsx de monitoreo</div>
<div class="step-row"><span class="step-badge">3</span> Revisa la vista previa</div>
<div class="step-row"><span class="step-badge">4</span> Descarga el archivo limpio</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PROCESAMIENTO
# ════════════════════════════════════════════════════════════════════════════
if uploaded_files:
    for uploaded in uploaded_files:
        st.markdown("---")
        st.markdown(f"#### 📄 `{uploaded.name}`")

        try:
            df_raw = pd.read_excel(uploaded, dtype=str)
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
            continue

        # Nombre del proyecto
        project_name = uploaded.name.replace(".xlsx", "")
        if "Nombre del proyecto" in df_raw.columns:
            vals = df_raw["Nombre del proyecto"].dropna().unique()
            if len(vals) > 0:
                project_name = str(vals[0]).strip()

        # ── Procesar ────────────────────────────────────────────────────────
        with st.spinner("Procesando…"):
            df_clean = clean_df(
                df_raw.copy(),
                own_authors=own_authors,
                exclude_authors=exclude_authors,
                exclude_keywords=exclude_keywords,
                fuzzy_threshold=fuzzy_threshold,
            )

        n_raw = len(df_raw)
        n_clean = len(df_clean)
        n_removed = n_raw - n_clean

        # ── Métricas ─────────────────────────────────────────────────────────
        # Tono counts
        tono_counts = df_clean["Tono"].value_counts() if "Tono" in df_clean.columns else pd.Series()
        pos = tono_counts.get("Positivo", 0)
        neu = tono_counts.get("Neutro", 0)
        neg = tono_counts.get("Negativo", 0)

        st.markdown(f"""
<div class="metric-row">
    <div class="metric-card"><div class="val">{n_raw}</div><div class="lbl">Filas entrada</div></div>
    <div class="metric-card"><div class="val">{n_clean}</div><div class="lbl">Filas resultado</div></div>
    <div class="metric-card"><div class="val">{n_removed}</div><div class="lbl">Filas excluidas</div></div>
    <div class="metric-card"><div class="val" style="color:#1a7a4a">{pos}</div><div class="lbl">Positivo</div></div>
    <div class="metric-card"><div class="val" style="color:#555">{neu}</div><div class="lbl">Neutro</div></div>
    <div class="metric-card"><div class="val" style="color:#b91c1c">{neg}</div><div class="lbl">Negativo</div></div>
</div>
""", unsafe_allow_html=True)

        # ── Autores reconocidos como propios ────────────────────────────────
        if own_authors and "Autor" in df_clean.columns:
            matched_own = df_clean[df_clean["Tono"] == "Positivo"]["Autor"].dropna().unique()
            if len(matched_own):
                with st.expander(f"✦ {len(matched_own)} autores clasificados como Positivo"):
                    st.markdown(
                        " ".join(f'<span class="pill pill-pos">{a}</span>' for a in sorted(matched_own)),
                        unsafe_allow_html=True,
                    )

        # ── Vista previa ────────────────────────────────────────────────────
        with st.expander("Vista previa (primeras 50 filas)"):
            preview_cols = [c for c in ["Fecha", "Red Social", "Tono", "Autor", "Título", "Interacciones", "Alcance", "Vistas"] if c in df_clean.columns]
            st.dataframe(df_clean[preview_cols].head(50), use_container_width=True)

        # ── Descarga ─────────────────────────────────────────────────────────
        clean_filename = f"{project_name}.xlsx"
        excel_bytes = df_to_excel(df_clean)

        st.markdown(f'<div class="callout ok">✓ Archivo listo: <b>{clean_filename}</b> · {n_clean} filas · {df_clean.shape[1]} columnas</div>', unsafe_allow_html=True)

        st.download_button(
            label=f"⬇ Descargar {clean_filename}",
            data=excel_bytes,
            file_name=clean_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"dl_{uploaded.name}",
        )

else:
    st.markdown("""
<div style="
    text-align:center;
    padding: 3.5rem 2rem;
    background: #f7f7fb;
    border-radius: 16px;
    border: 1.5px dashed #d8d8e4;
    color: #aaa;
    margin-top: 1rem;
">
    <div style="font-size:2rem;margin-bottom:.5rem">✦</div>
    <div style="font-weight:600;color:#555;font-size:1rem">Sin archivos cargados</div>
    <div style="font-size:.83rem;margin-top:.3rem">Sube un xlsx para comenzar · Puedes cargar varios a la vez</div>
</div>
""", unsafe_allow_html=True)
