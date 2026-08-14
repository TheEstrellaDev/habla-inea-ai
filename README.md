# 🎙️ PrimarIA INEA AI (HablaINEA AI)

**Asistente de Aprendizaje Interactivo por Voz para Adultos Mayores del INEA**

PrimarIA INEA AI es un sistema educativo accesible basado en Inteligencia Artificial y síntesis de voz, diseñado especialmente para personas adultas mayores que cursan su educación primaria en el **Instituto Nacional para la Educación de los Adultos (INEA)**.

El sistema elimina por completo las barreras de lectura y escritura complejas mediante un **botón gigante de voz interactivo**, consultando de manera estricta los libros de texto oficiales en formato PDF almacenados localmente en la carpeta `data/`.

---

## 🌟 Características Principales

- **👵 100% Accesible para Adultos Mayores**:
  - Interfaz visual con botón táctil gigante y transiciones cromáticas intuitivas:
    - 🔵 **Azul**: Listo para escuchar.
    - 🔴 **Rojo pulsante**: Escuchando tu voz con calma.
    - 🟡 **Ámbar**: Buscando en los libros de texto.
    - 🟢 **Verde con ondas**: Explicando en voz alta con tono pausado.
  - Subtítulos en pantalla con tamaño de letra ajustable ($A+$).
  - Retroalimentación auditiva suave mediante campanillas generadas con Web Audio.
- **⚡ Respuestas Rápidas, Claras y Sin Fatiga**:
  - Explicaciones ultracortas (máximo 1 o 2 párrafos breves, menos de 70 palabras).
  - Vocabulario hiper-sencillo con analogías de la vida cotidiana (el mercado, la cocina, el hogar, la familia, el trabajo).
  - Cero formato markdown o símbolos raros para garantizar una pronunciación limpia en el sintetizador de voz.
- **📚 Basado 100% en los Libros del INEA**:
  - Escaneo e indexación automática de todos los archivos `.pdf` dentro de `data/`.
  - Fallback elegante si una pregunta está fuera del material de estudio.
- **🛡️ Arquitectura Resiliente**:
  - SDK oficial de Google GenAI (`google-genai`).
  - Fallback inteligente entre modelos (`gemini-2.5-flash`, `gemini-3-flash-preview`, `gemini-flash-latest`, etc.).
  - Reintentos automáticos ante códigos de error 503 o cuellos de botella temporales.

---

## 📂 Estructura del Proyecto

```text
primarIA/
├── app/
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py               # Configuración con pydantic-settings
│   ├── services/
│   │   ├── __init__.py
│   │   └── ai_service.py           # Servicio de IA con Google GenAI y escaneo de PDFs
│   ├── static/
│   │   └── index.html              # Frontend accesible con botón gigante de voz
│   ├── __init__.py
│   └── main.py                     # Backend FastAPI y endpoints REST
├── data/
│   ├── .gitkeep
│   ├── lengua_comunicacion_1.pdf   # Libros de texto INEA
│   ├── lengua_comunicacion_2.pdf
│   ├── pensamiento_matematico_1.pdf
│   ├── pensamiento_matematico_2.pdf
│   └── vida_comunidad_1.pdf
├── .env                            # Variables de entorno con clave de API
├── .env.example                    # Plantilla de variables de entorno
├── .gitignore
├── requirements.txt                # Dependencias del proyecto
└── README.md
```

---

## 🚀 Instalación y Puesta en Marcha

### 1. Requisitos Previos
- Python 3.10 o superior (compatible con Python 3.14).
- Una clave de API de Google Gemini ([Google AI Studio](https://aistudio.google.com/)).

### 2. Clonar o descargar el proyecto
Ubícate en la carpeta raíz del proyecto:
```bash
cd primarIA
```

### 3. Instalar Dependencias
```bash
py -m pip install -r requirements.txt
```

### 4. Configurar Variables de Entorno
Copia el archivo `.env.example` a `.env` si aún no lo has creado:
```bash
cp .env.example .env
```
Edita `.env` con tu clave de API de Gemini:
```env
GEMINI_API_KEY=tu_api_key_aqui
GEMINI_MODEL=gemini-2.5-flash
APP_NAME="PrimarIA INEA AI"
DATA_DIR=data
```

### 5. Colocar los Libros en PDF
Coloca los libros de estudio del INEA (como `lengua_comunicacion_1.pdf`, `pensamiento_matematico_1.pdf`, etc.) dentro de la carpeta `data/`.

### 6. Ejecutar el Servidor
Inicia la aplicación con Uvicorn:
```bash
py -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🌐 Uso de la Aplicación

1. Abre tu navegador web en:
   ```text
   http://localhost:8000
   ```
2. Da permisos de acceso al **Micrófono** cuando el navegador lo solicite.
3. Presiona el **botón azul gigante** del centro y haz tu pregunta sobre cualquier tema de los libros de texto.
4. El asistente responderá de inmediato en voz alta con explicaciones claras, amables y breves.

---

## 🔌 Endpoints de la API

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/` | Retorna la interfaz web interactiva de voz. |
| `GET` | `/health` | Estado del servicio, modelo activo y cantidad de PDFs indexados. |
| `POST` | `/api/v1/chat` | Envía una consulta en formato JSON: `{"message": "¿Cómo se suman fracciones?"}` y recibe la respuesta explicada para voz. |
| `POST` | `/api/v1/reload-books` | Vuelve a escanear y registrar los archivos PDF de `data/` en caliente. |

---

## ♿ Lineamientos de Accesibilidad Implementados

- **Diseño Ergonómico**: Área de toque del botón central mayor a 200px con contraste reforzado.
- **Tipografía**: Fuentes *Lexend* y *Outfit* para máxima legibilidad en dislexia y dificultades visuales.
- **Sintetizador Web Speech**: Configurado a velocidad pausada (`rate = 0.88`) en español mexicano (`es-MX`).
- **Alternativa Textual**: Entrada de texto accesible para usuarios que no puedan utilizar micrófono.
