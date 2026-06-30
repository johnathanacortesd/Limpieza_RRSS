import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import unicodedata
from datetime import datetime, time

# ── Fuzzy matching con normalización y precisión ─────────────────────────────
try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

def remove_accents(text: str) -> str:
    """
    Normaliza un texto para comparación robusta:
    - minúsculas y sin espacios sobrantes
    - sin tildes
    - separadores comunes de handles en redes (@, _, -, .) convertidos a espacio
      (ej. '@Fenavi_Oficial' -> 'fenavi oficial', 'Fenavi.Bogota' -> 'fenavi bogota')
    - espacios múltiples colapsados
    """
    if not text:
        return ""
    text = str(text).strip().lower()
    text = "".join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    text = re.sub(r'[@_\-\.]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _tokenize(text: str) -> list:
    return text.split() if text else []


# Palabras de control geográfico y regional para evitar cruces falsos entre marcas de distintos territorios
REGIONAL_WORDS = {
    "valle", "antioquia", "santander", "bogota", "quindio", "tolima", 
    "cartagena", "atlantico", "cundinamarca", "huila", "nariño", "meta",
    "boyaca", "caldas", "risaralda", "cesar", "cordoba", "sucre",
    "colombia", "venezuela", "ecuador", "peru", "mexico", "panama", "espana", "usa"
}

# Sufijos y variaciones típicas unidas a nombres de usuario en redes sociales
SOCIAL_MODIFIERS = {
    "colombia", "col", "oficial", "ofic", "co", "valle", "delagente",
    "oficialcol", "oficialcolombia", "coop", "caja", "nal", "nacional"
}

def get_regions(tokens: list) -> set:
    """Identifica de manera precisa si dentro de los tokens hay menciones geográficas."""
    regions = set()
    for tok in tokens:
        for r in REGIONAL_WORDS:
            if tok == r:
                regions.add(r)
            elif len(r) >= 3:
                if tok.startswith(r) or tok.endswith(r):
                    regions.add(r)
    return regions

def precise_match(value: str, patterns: list, threshold: int = 75) -> bool:
    """
    Evalúa coincidencia de marcas de manera altamente precisa, tolerando
    variaciones leves de un mismo Autor entre redes sociales
    (ej. '@fenavi', 'Fenavi_Oficial', 'Fenavi Bogotá', 'fenavi.colombia').

    Evita falsos positivos entre acrónimos similares (ej. fenavi vs fenalco)
    o federaciones de distintos países (ej. Colombia vs Venezuela)
    mediante validación de conflictos geográficos y umbrales dinámicos por longitud de palabra.
    """
    if not value or not patterns:
        return False
    val_norm = remove_accents(value)
    val_tokens = _tokenize(val_norm)
    val_regions = get_regions(val_tokens)

    for p in patterns:
        pat_norm = remove_accents(p)
        if not pat_norm:
            continue
        pat_tokens = _tokenize(pat_norm)
        pat_regions = get_regions(pat_tokens)

        # Evitar falsos positivos si hay conflicto geográfico (ej. Colombia vs Venezuela, Valle vs Antioquia)
        if pat_regions and val_regions and pat_regions != val_regions:
            continue

        # 1. Coincidencia exacta directa (cadena completa ya normalizada)
        if val_norm == pat_norm:
            return True

        # 2. Coincidencia por token completo o con variaciones de redes sociales unidas (ej. 'fenavicolombia' -> patrón 'fenavi')
        if len(pat_tokens) == 1:
            pat_single = pat_tokens[0]
            for tok in val_tokens:
                if tok == pat_single:
                    return True
                # Caso de usuario concatenado al final (ej. "fenavicolombia" empieza por "fenavi" + sufijo "colombia")
                if tok.startswith(pat_single):
                    remainder = tok[len(pat_single):]
                    if remainder in SOCIAL_MODIFIERS or remainder == "":
                        return True
                # Caso inverso al inicio (ej. "oficialfenavi")
                if tok.endswith(pat_single):
                    prefix = tok[:-len(pat_single)]
                    if prefix in SOCIAL_MODIFIERS or prefix == "":
                        return True

        # 3. Coincidencia de frase completa como subsecuencia contigua de tokens
        if len(pat_tokens) > 1:
            n = len(pat_tokens)
            if any(val_tokens[i:i + n] == pat_tokens for i in range(len(val_tokens) - n + 1)):
                return True

        # 4. Umbral de seguridad para siglas/palabras cortas del patrón completo (7 caracteres o menos):
        effective_threshold = threshold
        if len(pat_norm) <= 7 or len(val_norm) <= 7:
            effective_threshold = max(threshold, 92)

        # 5. Evaluación de aproximación si está habilitada
        if FUZZY_AVAILABLE:
            if len(pat_tokens) == 1:
                # Comparación token a token: más precisa para acrónimos/marcas
                # cortas, evita que palabras extra del Autor distorsionen el score
                for tok in val_tokens:
                    # Determinar umbral específico basado en el tamaño real del token comparado
                    tok_threshold = threshold
                    if len(tok) <= 7 or len(pat_norm) <= 7:
                        tok_threshold = max(threshold, 92)
                    
                    if fuzz.ratio(tok, pat_norm) >= tok_threshold:
                        return True
            else:
                # token_set_ratio tolera mejor el orden distinto de palabras
                # y texto adicional alrededor del patrón compuesto
                score = fuzz.token_set_ratio(val_norm, pat_norm)
                if score >= effective_threshold:
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

# Perfiles de Marcas predefinidos con sus respectivos autores e inhabilitadores
PREDEFINED_PROFILES = {
    "Personalizado (Ingreso manual)": {
        "own_authors": "",
        "exclude_authors": "",
        "exclude_keywords": ""
    },
    "FENAVI (Federación Nacional de Avicultores)": {
        "own_authors": (
            "@FenaviColombia\n"
            "Fenavi Colombia\n"
            "Federación Nacional de Avicultores de Colombia\n"
            "Fenavi\n"
            "@elpoderdelhuevo\n"
            "El poder del huevo\n"
            "@Acomerpollo\n"
            "A comer pollo\n"
            "Fenavi Bogotá\n"
            "Fenavi Santander\n"
            "Fenavi Antioquia\n"
            "Fenavi Valle"
        ),
        "exclude_authors": (
            "FEDERACIÓN NACIONAL DE AVICULTURA DE VENEZUELA (@Fenavideve)"
        ),
        "exclude_keywords": ""
    },
    "Comfenalco Valle Delagente": {
        "own_authors": (
            "@ComfenalcoValle\n"
            "Comfenalco Valle\n"
            "Comfenalco Valle Delagente\n"
            "@ComfenalcoValleDelagente\n"
            "Caja de Compensación Familiar Comfenalco Valle\n"
            "delagente"
        ),
        "exclude_authors": "",
        "exclude_keywords": ""
    }
}


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
        # Formato de fecha corta solicitado: DD/MM/YYYY (ej. 23/06/2026)
        df["Fecha"]  = df["FechaHora"].dt.strftime("%d/%m/%Y").fillna("")
        df["Hora"]   = df["FechaHora"].dt.strftime("%H:%M:%S").fillna("")
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

    # 10. Clasificación inteligente de Tono (Usa precise_match) [1]
    def assign_tono(row):
        autor = str(row.get("Autor","") or "")
        sentimiento = str(row.get("Sentimiento","") or "").upper()
        if bool(precise_match(autor, own_authors, fuzzy_threshold)):
            return "Positivo"
        return {"NEUTRAL": "Neutro", "POSITIVE": "Positivo", "NEGATIVE": "Negativo"}.get(sentimiento, "Neutro")

    df["Tono"] = df.apply(assign_tono, axis=1)

    # 11. Filtros de exclusión selectivos (Usa precise_match) [1]
    if "Autor" in df.columns and exclude_authors:
        df = df[~df["Autor"].apply(lambda a: bool(precise_match(str(a), exclude_authors, fuzzy_threshold)))]

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


# ── Extracción limpia de entradas de texto ──────────────────────────────────
def parse_lines(raw):
    return [l.strip() for l in (raw or "").splitlines() if l.strip()]


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA DE STREAMLIT (ESTILO GOOGLE GEMINI)
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Procesador de Monitoreo RRSS",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Hojas de Estilo CSS Personalizadas ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Fondo principal */
.stApp { background: #f8fafc !important; }
.main .block-container {
    background: #f8fafc;
    padding: 2.5rem 3.5rem !important;
    max-width: 1300px;
}

/* Panel lateral (Blanco puro con sombreado tenue) */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 2rem 1.5rem !important;
}

/* Campos de entrada del Sidebar */
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] div[data-baseweb="select"] {
    background: #f1f5f9 !important;
    border-radius: 12px !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] textarea:focus,
[data-testid="stSidebar"] input:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.15) !important;
}

/* Cabecera Estilo Gemini (Gradiente inteligente) */
.app-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding-bottom: 1.8rem;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 2rem;
}
.app-logo {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 30%, #8b5cf6 70%, #ec4899 100%);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; color: #ffffff; flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.2);
}
.app-header-text h1 {
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #1e293b 30%, #4f46e5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
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
    letter-spacing: 1.2px;
    color: #64748b;
    font-weight: 700;
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
    min-width: 140px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-top: 3px solid #8b5cf6; /* Acento en la parte superior */
    border-radius: 12px;
    padding: 1.2rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.01);
}
.metric-card .val {
    font-size: 1.6rem;
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
    font-weight: 600;
}

/* Etiquetas visuales en forma de píldoras */
.pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 2px;
}
.pill-pos { background: #f0fdf4; color: #166534; border: 1px solid #dcfce7; }

/* Mensajes informativos integrados */
.callout {
    background: #f8fafc;
    border-left: 4px solid #8b5cf6;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #334155;
    margin: 1rem 0;
}
.callout.ok { background: #f0fdf4; border-color: #10b981; color: #166534; }
.callout.warn { background: #fffbeb; border-color: #f59e0b; color: #78350f; }

/* Botones con estilo gradiente */
div.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: -0.2px;
    box-shadow: 0 4px 10px rgba(139, 92, 246, 0.2) !important;
    transition: all 0.2s ease;
}
div.stButton > button:hover {
    opacity: 0.95;
    transform: translateY(-1px);
    box-shadow: 0 6px 14px rgba(139, 92, 246, 0.3) !important;
}

/* Botón de descarga con estilo premium */
[data-testid="stDownloadButton"] > button {
    background: #0f172a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    box-shadow: 0 4px 10px rgba(15, 23, 42, 0.1) !important;
    transition: all 0.2s ease;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #1e293b !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 14px rgba(15, 23, 42, 0.15) !important;
}

/* Zonas desplegables y visualizadores */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.01);
}

.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    background: #ffffff;
    border-radius: 16px;
    border: 1px dashed #cbd5e1;
    margin-top: 1rem;
}
.empty-state .icon { font-size: 2rem; margin-bottom: 0.5rem; color: #a855f7; }
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

    # Selector de perfil predefinido
    selected_profile_name = st.selectbox(
        "Perfil de marca predefinido",
        options=list(PREDEFINED_PROFILES.keys()),
        index=0,
        help="Carga automática de cuentas oficiales para FENAVI o Comfenalco Valle."
    )
    
    current_profile = PREDEFINED_PROFILES[selected_profile_name]

    st.divider()
    st.markdown("**Asignación: Tono Positivo**")
    st.caption(
        "Escribe un autor o palabra clave por línea. Se evaluará de forma flexible "
        "(sin distinguir acentos, mayúsculas, ni separadores como @, _, - o . entre redes)."
    )
    
    own_authors_raw = st.text_area(
        "Autores propios",
        value=current_profile["own_authors"],
        height=140,
        placeholder="@fenavi\nFenavi Bogotá\nfenavi colombia",
    )

    st.divider()
    st.markdown("**Exclusiones (Autores)**")
    st.caption("Un autor por línea a descartar del reporte final.")
    exclude_authors_raw = st.text_area(
        "Excluir autores",
        value=current_profile["exclude_authors"],
        height=90,
        placeholder="@falsonoticias\nBot_Spam",
    )

    st.divider()
    st.markdown("**Exclusiones (Palabras Clave en Título)**")
    st.caption("Una frase o término por línea a descartar.")
    exclude_kw_raw = st.text_area(
        "Excluir palabras clave",
        value=current_profile["exclude_keywords"],
        height=90,
        placeholder="sorteo nacional\ncomprar seguidores",
    )

    st.divider()
    st.markdown("**Umbral de Precisión (Fuzzy)**")
    fuzzy_threshold = st.slider(
        "Sensibilidad de coincidencia",
        min_value=50, max_value=100, value=75, step=5,
        help="Un valor más bajo es más flexible con diferencias de escritura. 100 requiere coincidencia exacta. "
             "Para marcas o siglas de 7 caracteres o menos se aplica automáticamente un mínimo de 92% "
             "para evitar falsos positivos entre acrónimos parecidos (ej. fenavi vs fenalco).",
    )
    if not FUZZY_AVAILABLE:
        st.markdown(
            '<div class="callout warn">⚠️ <b>rapidfuzz</b> no está instalado en el entorno actual. '
            'Usando búsqueda de subcadena tradicional.</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    with st.expander("🔍 Probar una coincidencia"):
        st.caption("Verifica cómo clasificaría la app un Autor sin necesidad de subir un archivo.")
        test_value = st.text_input(
            "Autor o texto de prueba",
            placeholder="@Fenavi_Oficial",
        )
        if test_value:
            _own = parse_lines(own_authors_raw)
            _excl = parse_lines(exclude_authors_raw)
            if precise_match(test_value, _own, fuzzy_threshold):
                st.markdown(
                    '<div class="callout ok">✔ Coincide con <b>Autores propios</b> → Tono: <b>Positivo</b></div>',
                    unsafe_allow_html=True,
                )
            elif precise_match(test_value, _excl, fuzzy_threshold):
                st.markdown(
                    '<div class="callout warn">✘ Coincide con <b>Exclusiones</b> → la fila sería descartada</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="callout">— No coincide con ninguna lista configurada</div>',
                    unsafe_allow_html=True,
                )


# ── Extracción limpia de entradas de texto (listas para el procesamiento) ───
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
        1. Selecciona un perfil predefinido de marca (FENAVI o Comfenalco Valle) o configúralo manualmente en el menú lateral.<br>
        2. Arrastra y suelta tus archivos Excel de monitoreo.<br>
        3. Presiona el botón <b>"Procesar Datos"</b>.<br>
        4. Comprueba los resultados y descarga los archivos limpios.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PROCESAMIENTO ACTIVO DE DATOS
# ════════════════════════════════════════════════════════════════════════════
if uploaded_files:
    st.markdown('<div class="section-label" style="margin-top:2rem;">Acción</div>', unsafe_allow_html=True)
    
    # Botón manual con estilo acentuado para activar la ejecución de la limpieza
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

            # Renderizado de Tarjetas de Métricas
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
              <div class="metric-card" style="border-top-color: #64748b;">
                <div class="val">{n_removed:,}</div>
                <div class="lbl">Filtrados</div>
              </div>
              <div class="metric-card" style="border-top-color: #10b981;">
                <div class="val" style="color:#166534">{pos:,}</div>
                <div class="lbl">Positivo</div>
              </div>
              <div class="metric-card" style="border-top-color: #64748b;">
                <div class="val" style="color:#475569">{neu:,}</div>
                <div class="lbl">Neutro</div>
              </div>
              <div class="metric-card" style="border-top-color: #ef4444;">
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
                
                # Renderizar muestra de los datos en formato DataFrame
                st.dataframe(
                    df_clean[preview_cols].head(50), 
                    use_container_width=True
                )

            # Generación de Descargas
            final_filename = f"{data['project_name']}.xlsx"
            excel_bytes = df_to_excel(df_clean)

            st.markdown(
                f'<div class="callout ok">El archivo ha sido procesado de manera correcta. '
                f'Se han conservado <b>{n_clean}</b> registros de un total original de <b>{n_raw}</b>.</div>',
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
