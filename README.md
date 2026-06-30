# Limpiador RRSS · Streamlit

Procesador de datos de redes sociales basado en las reglas del query M de Power Query.

---

## Estructura del repositorio

```
/
├── app.py            ← App principal
├── requirements.txt  ← Dependencias
└── README.md
```

---

## Despliegue en Streamlit Community Cloud (sin usar git en terminal)

### Paso 1 — Crear repositorio en GitHub desde el navegador

1. Ir a [github.com/new](https://github.com/new)
2. Nombre: `limpiador-rrss` (o el que prefieras)
3. Visibilidad: **Public** (requerido en el plan gratuito de Streamlit Cloud)
4. Marcar ✅ *Add a README file*
5. Clic en **Create repository**

### Paso 2 — Subir los archivos desde el navegador

En tu repositorio recién creado:

1. Clic en **Add file → Upload files**
2. Arrastra `app.py` y `requirements.txt`
3. Clic en **Commit changes**

Para actualizar la app en el futuro: mismo proceso — GitHub detecta el cambio y Streamlit Cloud redespliega automáticamente.

### Paso 3 — Conectar a Streamlit Community Cloud

1. Ir a [share.streamlit.io](https://share.streamlit.io)
2. **Sign in with GitHub**
3. Clic en **New app**
4. Seleccionar: repositorio `limpiador-rrss`, rama `main`, archivo `app.py`
5. Clic en **Deploy**

La app queda pública en una URL tipo:
`https://limpiador-rrss-tuusuario.streamlit.app`

---

## Uso local (opcional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Qué hace la app

| Regla del código M | Equivalente en la app |
|---|---|
| `RemoveColumns Título` → recrear desde Contenido | ✅ Automático |
| `ChangedType` numéricos + nulos → 0 | ✅ Automático |
| `ReplaceValue Twitter → X` | ✅ Automático |
| `AddedTono` con lista de autores propios | ✅ Configurable en sidebar + fuzzy matching |
| `FilteredRows` por autor y palabras clave | ✅ Configurable en sidebar |
| Franja horaria 6 tramos | ✅ Automático |
| Día en español | ✅ Automático |
| `cumulative_reach` = fans + followers | ✅ Automático |
| **Alcance** (fans + followers formateado CO) | ✅ Columna nueva |
| **Vistas** (views formateado CO) | ✅ Columna nueva |
| **Tipo específico** | ✅ Se conserva del xlsx original |
| Nombre archivo = columna "Nombre del proyecto" | ✅ Automático |

---

## Notas sobre fuzzy matching

El umbral configurable en el sidebar (por defecto 75%) controla qué tan estricta es la coincidencia para autores propios:

- **95%+** → casi exacto (sirve para nombres únicos)
- **75%** → tolera errores tipográficos, variantes con/sin @, etc.
- **50%** → muy permisivo (puede dar falsos positivos)

Siempre se evalúa también coincidencia por **contención** (si el patrón está contenido en el valor o viceversa), independiente del umbral.
