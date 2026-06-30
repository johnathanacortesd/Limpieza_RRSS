import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime, time

# ── Fuzzy matching ──────────────────────────────────────────────────────────
try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

def fuzzy_match(value: str, patterns: list, threshold: int = 75) -> bool:
    if not value or not patterns:
        return False
    val = str(value).strip().lower()
    for p in patterns:
        pat = str(p).strip().lower()
        if not pat:
            continue
        if pat in val or val in pat:
            return True
        if FUZZY_AVAILABLE:
            score = max(
                fuzz.ratio(val, pat),
                fuzz.partial_ratio(val, pat),
                fuzz.token_sort_ratio(val, pat),
            )
            if score >= threshold:
                return True
    return False


# ── Constantes ──────────────────────────────────────────────────────────────
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


# ── Helpers ──────────────────────────────────────────────────────────────────
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

def fmt_co(n):
    try:
        v = int(float(n))
        return f"{v:,}".replace(",",".")
    except (ValueError, TypeError):
        return "0"

def safe_int(v):
    try:
        f = float(v)
        return 0 if np.isnan(f) else int(f)
    except (TypeError, ValueError):
        return 0


# ── Limpieza principal ───────────────────────────────────────────────────────
def clean_df(df, own_authors, exclude_authors, exclude_keywords, fuzzy_threshold=75):

    # 1. Quitar Título original (se recrea)
    if "Título" in df.columns:
        df = df.drop(columns=["Título"])

    # 2. Numéricos: nulos → 0
    for col in [c for c in NUMERIC_COLS if c in df.columns]:
        df[col] = df[col].apply(safe_int)

    # 3. Renombrar
    df = df.rename(columns={
        "id": "ID",
        "Link para la fuente": "url",
        "Grupo de dominio": "Red Social",
        "Creado": "FechaHora",
        "Interacciones totales": "Interacciones",
    })

    # 4. Twitter → X
    if "Red Social" in df.columns:
        df["Red Social"] = df["Red Social"].str.replace("Twitter","X",regex=False)

    # 5. Título + Descripción desde Contenido
    col_c = "Contenido de la publicación"
    if col_c in df.columns:
        df["Título"]      = df[col_c].fillna("").astype(str)
        df["Descripción"] = df["Título"]
        df = df.drop(columns=[col_c])
    else:
        df["Título"] = df.get("Título","")
        df["Descripción"] = df["Título"]

    # 6. Fecha / Hora / Franja / Día
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

    # 7. cumulative_reach + Alcance
    reach_raw = (
        (df["fans"].apply(safe_int)      if "fans"      in df.columns else pd.Series(0, index=df.index)) +
        (df["followers"].apply(safe_int) if "followers" in df.columns else pd.Series(0, index=df.index))
    )
    df["cumulative_reach"] = reach_raw.apply(fmt_co)
    df["Alcance"]          = reach_raw.apply(fmt_co)

    # 8. Vistas
    df["Vistas"] = (
        df["views"].apply(safe_int) if "views" in df.columns
        else pd.Series(0, index=df.index)
    ).apply(fmt_co)

    # 9. Interacciones formateadas
    if "Interacciones" in df.columns:
        df["Interacciones"] = df["Interacciones"].apply(safe_int).apply(fmt_co)

    # 10. Tono
    def assign_tono(row):
        autor      = str(row.get("Autor","") or "")
        sentimiento = str(row.get("Sentimiento","") or "")
        if fuzzy_match(autor, own_authors, fuzzy_threshold):
            return "Positivo"
        return {"NEUTRAL":"Neutro","POSITIVE":"Positivo","NEGATIVE":"Negativo"}.get(sentimiento, sentimiento)

    df["Tono"] = df.apply(assign_tono, axis=1)

    # 11. Filtros exclusión
    if "Autor" in df.columns and exclude_authors:
        df = df[~df["Autor"].apply(lambda a: fuzzy_match(str(a), exclude_authors, fuzzy_threshold))]

    if "Título" in df.columns and exclude_keywords:
        df = df[~df["Título"].apply(
            lambda t: any(kw.lower() in str(t).lower() for kw in exclude_keywords if kw)
        )]

    # 12. ID texto
    if "ID" in df.columns:
        df["ID"] = df["ID"].astype(str)

    # 13. Columnas finales
    df = df[[c for c in FINAL_COLS if c in df.columns]]
    return df.reset_index(drop=True)


def df_to_excel(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Datos")
        ws = writer.sheets["Datos"]
        for col_cells in ws.columns:
            max_len = max((len(str(c.value)) if c.value else 0) for c in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 4, 60)
    return buf.getvalue()


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Limpiador RRSS",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: 100% tema claro ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ─ Fondo general ─ */
.stApp { background: #f5f5f7 !important; }
.main .block-container {
    background: #f5f5f7;
    padding: 2rem 2.5rem !important;
    max-width: 1200px;
}

/* ─ Sidebar: blanco total ─ */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e4e4e9 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1.2rem !important;
}
/* Todos los textos del sidebar: oscuros */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: #1a1a2e !important;
}
/* Inputs del sidebar */
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] input {
    background: #f7f7fa !important;
    border: 1.5px solid #dcdce6 !important;
    color: #1a1a2e !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] textarea:focus,
[data-testid="stSidebar"] input:focus {
    border-color: #4361ee !important;
    box-shadow: 0 0 0 3px rgba(67,97,238,0.1) !important;
}
/* Slider en sidebar */
[data-testid="stSidebar"] .stSlider span { color: #1a1a2e !important; }
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] { color: #666 !important; }

/* Separadores sidebar */
[data-testid="stSidebar"] hr {
    border-color: #e8e8f0 !important;
    margin: 1rem 0 !important;
}

/* ─ Header de la app ─ */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding-bottom: 1.5rem;
    border-bottom: 1.5px solid #e4e4e9;
    margin-bottom: 1.8rem;
}
.app-logo {
    width: 42px; height: 42px;
    background: #1a1a2e;
    border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; color: #fff; flex-shrink: 0;
}
.app-header-text h1 {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: #1a1a2e !important;
    margin: 0 !important; padding: 0 !important;
    letter-spacing: -0.4px;
    line-height: 1.2;
}
.app-header-text p {
    font-size: 0.78rem !important;
    color: #6b7280 !important;
    margin: 2px 0 0 0 !important;
}

/* ─ Section labels ─ */
.section-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    color: #9ca3af;
    font-weight: 600;
    margin-bottom: 10px;
    margin-top: 4px;
}

/* ─ Cards de métricas ─ */
.metric-row {
    display: flex;
    gap: 10px;
    margin: 1.2rem 0 1rem;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 100px;
    background: #ffffff;
    border: 1px solid #e4e4e9;
    border-radius: 12px;
    padding: 1rem 1.1rem 0.9rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.metric-card .val {
    font-size: 1.65rem;
    font-weight: 700;
    color: #1a1a2e;
    letter-spacing: -1.5px;
    line-height: 1;
}
.metric-card .lbl {
    font-size: 0.68rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: .8px;
    margin-top: 5px;
    font-weight: 500;
}

/* ─ Pills de tono ─ */
.pill {
    display: inline-block;
    padding: 3px 11px;
    border-radius: 99px;
    font-size: 0.73rem;
    font-weight: 500;
    margin: 2px 3px;
}
.pill-pos { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
.pill-neg { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
.pill-neu { background: #f3f4f6; color: #374151; border: 1px solid #e5e7eb; }

/* ─ Steps guía ─ */
.step-badge {
    display: inline-flex;
    align-items: center; justify-content: center;
    width: 22px; height: 22px;
    background: #1a1a2e;
    color: #fff;
    border-radius: 50%;
    font-size: 0.68rem;
    font-weight: 700;
    margin-right: 9px;
    flex-shrink: 0;
}
.step-row {
    display: flex;
    align-items: center;
    margin-bottom: 8px;
    font-size: 0.86rem;
    color: #374151;
    font-weight: 400;
}

/* ─ Callouts ─ */
.callout {
    background: #eff6ff;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: #1e3a5f;
    margin: 0.8rem 0;
    line-height: 1.5;
}
.callout.warn { background: #fffbeb; border-color: #f59e0b; color: #78350f; }
.callout.ok   { background: #f0fdf4; border-color: #22c55e; color: #14532d; }

/* ─ Upload zone ─ */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 1.5px dashed #d1d5db !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"] label { color: #374151 !important; }
[data-testid="stFileUploader"] span  { color: #6b7280 !important; }

/* ─ Botón descarga ─ */
[data-testid="stDownloadButton"] > button {
    background: #1a1a2e !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    width: 100% !important;
    letter-spacing: -0.1px;
    transition: background .2s;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #2d2d52 !important;
}

/* ─ Expander ─ */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e4e4e9 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #1a1a2e !important; }

/* ─ Dataframe ─ */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ─ Títulos sección main ─ */
h4 { color: #1a1a2e !important; font-weight: 600 !important; }

/* ─ Spinner ─ */
[data-testid="stSpinner"] p { color: #6b7280 !important; }

/* ─ Empty state ─ */
.empty-state {
    text-align: center;
    padding: 3.5rem 2rem;
    background: #ffffff;
    border-radius: 16px;
    border: 1.5px dashed #d1d5db;
    margin-top: 1rem;
}
.empty-state .icon { font-size: 2rem; margin-bottom: 0.5rem; }
.empty-state .title { font-weight: 600; color: #374151; font-size: 1rem; }
.empty-state .sub   { font-size: 0.82rem; color: #9ca3af; margin-top: 4px; }

/* ─ Sidebar: label de campos ─ */
[data-testid="stSidebar"] .stTextArea label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stSlider label {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #374151 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-logo">✦</div>
  <div class="app-header-text">
    <h1>Limpiador RRSS</h1>
    <p>Procesador de datos de redes sociales · Estandarización y análisis</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="section-label">Configuración de marca</div>', unsafe_allow_html=True)

    st.markdown("**Autores propios → Tono Positivo**")
    st.caption("Un autor por línea. Acepta fragmentos o variantes.")
    own_authors_raw = st.text_area(
        "Autores propios",
        height=130,
        placeholder="@fenavi\nFenavi Bogotá\nfenavi colombia",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Autores a excluir**")
    st.caption("Un autor por línea.")
    exclude_authors_raw = st.text_area(
        "Excluir autores",
        height=90,
        placeholder="@fenavideve\nFederación Venezuela",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Palabras clave a excluir del Título**")
    st.caption("Una palabra/frase por línea.")
    exclude_kw_raw = st.text_area(
        "Excluir palabras clave",
        height=90,
        placeholder="#Fenavidevenezuela\nFederación Venezuela",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Umbral fuzzy matching**")
    fuzzy_threshold = st.slider(
        "Similitud mínima (%)",
        min_value=50, max_value=100, value=75, step=5,
        help="75 = tolerante · 95 = casi exacto",
    )
    if not FUZZY_AVAILABLE:
        st.markdown(
            '<div class="callout warn">⚠️ <b>rapidfuzz</b> no instalado. '
            'Instala: <code>pip install rapidfuzz</code></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown('<div class="section-label">Acerca de</div>', unsafe_allow_html=True)
    st.caption(
        "Aplica las mismas reglas del query M: tipos, nulos → 0, "
        "Twitter → X, Tono fuzzy, franjas horarias, Alcance, Vistas."
    )


# ── Parse inputs ─────────────────────────────────────────────────────────────
def parse_lines(raw):
    return [l.strip() for l in (raw or "").splitlines() if l.strip()]

own_authors      = parse_lines(own_authors_raw)
exclude_authors  = parse_lines(exclude_authors_raw)
exclude_keywords = parse_lines(exclude_kw_raw)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
col_up, col_guide = st.columns([2, 1], gap="large")

with col_up:
    st.markdown('<div class="section-label">Archivos de entrada</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Sube uno o más archivos .xlsx",
        type=["xlsx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

with col_guide:
    st.markdown('<div class="section-label">Guía rápida</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="step-row"><span class="step-badge">1</span>Configura la marca en el panel lateral</div>
<div class="step-row"><span class="step-badge">2</span>Sube el xlsx de monitoreo</div>
<div class="step-row"><span class="step-badge">3</span>Revisa métricas y vista previa</div>
<div class="step-row"><span class="step-badge">4</span>Descarga el archivo limpio</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PROCESAMIENTO
# ════════════════════════════════════════════════════════════════════════════
if uploaded_files:
    for uploaded in uploaded_files:
        st.divider()
        st.markdown(f"#### 📄 `{uploaded.name}`")

        try:
            df_raw = pd.read_excel(uploaded, dtype=str)
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
            continue

        # Nombre del proyecto
        project_name = uploaded.name.replace(".xlsx","")
        if "Nombre del proyecto" in df_raw.columns:
            vals = df_raw["Nombre del proyecto"].dropna().unique()
            if len(vals):
                project_name = str(vals[0]).strip()

        with st.spinner("Procesando…"):
            df_clean = clean_df(
                df_raw.copy(),
                own_authors=own_authors,
                exclude_authors=exclude_authors,
                exclude_keywords=exclude_keywords,
                fuzzy_threshold=fuzzy_threshold,
            )

        n_raw    = len(df_raw)
        n_clean  = len(df_clean)
        n_removed = n_raw - n_clean

        tono_counts = df_clean["Tono"].value_counts() if "Tono" in df_clean.columns else pd.Series()
        pos = int(tono_counts.get("Positivo", 0))
        neu = int(tono_counts.get("Neutro", 0))
        neg = int(tono_counts.get("Negativo", 0))

        # Métricas
        st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="val">{n_raw}</div>
    <div class="lbl">Entrada</div>
  </div>
  <div class="metric-card">
    <div class="val">{n_clean}</div>
    <div class="lbl">Resultado</div>
  </div>
  <div class="metric-card">
    <div class="val">{n_removed}</div>
    <div class="lbl">Excluidas</div>
  </div>
  <div class="metric-card">
    <div class="val" style="color:#166534">{pos}</div>
    <div class="lbl">Positivo</div>
  </div>
  <div class="metric-card">
    <div class="val" style="color:#374151">{neu}</div>
    <div class="lbl">Neutro</div>
  </div>
  <div class="metric-card">
    <div class="val" style="color:#991b1b">{neg}</div>
    <div class="lbl">Negativo</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Autores propios detectados
        if own_authors and "Autor" in df_clean.columns:
            matched = df_clean[df_clean["Tono"]=="Positivo"]["Autor"].dropna().unique()
            if len(matched):
                with st.expander(f"✦ {len(matched)} autores clasificados como Positivo"):
                    st.markdown(
                        " ".join(f'<span class="pill pill-pos">{a}</span>' for a in sorted(matched)),
                        unsafe_allow_html=True,
                    )

        # Vista previa
        with st.expander("Vista previa — primeras 50 filas"):
            preview_cols = [c for c in
                ["Fecha","Red Social","Tono","Autor","Título","Interacciones","Alcance","Vistas","Tipo específico"]
                if c in df_clean.columns]
            st.dataframe(df_clean[preview_cols].head(50), use_container_width=True)

        # Descarga
        clean_filename = f"{project_name}.xlsx"
        excel_bytes = df_to_excel(df_clean)

        st.markdown(
            f'<div class="callout ok">✓ Listo: <b>{clean_filename}</b> · '
            f'{n_clean} filas · {df_clean.shape[1]} columnas</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            label=f"⬇  Descargar  {clean_filename}",
            data=excel_bytes,
            file_name=clean_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"dl_{uploaded.name}",
        )

else:
    st.markdown("""
<div class="empty-state">
  <div class="icon">✦</div>
  <div class="title">Sin archivos cargados</div>
  <div class="sub">Sube un xlsx para comenzar · Puedes cargar varios a la vez</div>
</div>
""", unsafe_allow_html=True)
