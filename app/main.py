import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.ai_service import ai_service

# Configuración de logging
logger = logging.getLogger("main")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# Definición del ciclo de vida (Lifespan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Evento de inicio y cierre de la aplicación FastAPI.
    En el arranque, escanea y precarga los libros PDF en memoria.
    """
    logger.info(f"=== Iniciando {settings.APP_NAME} ===")
    logger.info(f"Modelo configurado: {settings.GEMINI_MODEL}")
    
    # Escaneo y precarga asíncrona de PDFs
    try:
        indexed_books = await asyncio.to_thread(ai_service.scan_and_upload_pdfs)
        logger.info(f"Libros INEA indexados exitosamente: {indexed_books}")
    except Exception as e:
        logger.error(f"Error durante el escaneo inicial de PDFs: {e}")

    yield

    logger.info(f"=== Apagando {settings.APP_NAME} ===")


# Inicialización de la aplicación FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    description="Asistente de aprendizaje interactivo por voz para adultos mayores del INEA",
    version="1.0.0",
    lifespan=lifespan
)

# Configuración de CORS para permitir acceso desde navegadores
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definición de rutas estáticas
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Esquemas Pydantic para el API
class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Pregunta o mensaje de voz transcrito del estudiante adulto mayor"
    )
    selected_book: Optional[str] = Field(
        default=None,
        description="Nombre del archivo PDF seleccionado por el estudiante (ej: lengua_comunicacion_1.pdf)"
    )


class ChatResponse(BaseModel):
    response: str = Field(description="Respuesta educativa en lenguaje claro y estructurada para voz")
    model_used: str = Field(description="Modelo de IA utilizado")
    success: bool = Field(default=True, description="Estado de éxito de la operación")
    referenced_books: List[str] = Field(default_factory=list, description="Lista de libros consultados")
    error_detail: Optional[str] = Field(default=None, description="Detalle del error si ocurrió alguno")


class PreloadBookRequest(BaseModel):
    selected_book: str = Field(..., description="Nombre del archivo PDF a precargar")


# Endpoints
@app.get("/", summary="Página principal de voz para estudiantes")
async def get_index():
    """Retorna la interfaz web interactiva con el botón gigante de voz."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="El archivo index.html no fue encontrado en app/static."
        )
    return FileResponse(str(index_file))


@app.get("/health", summary="Verificación de estado y conectividad")
async def health_check():
    """Retorna el estado de salud del sistema, modelo activo y libros indexados."""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "model_configured": settings.GEMINI_MODEL,
        "fallback_models": settings.FALLBACK_MODELS,
        "total_books_indexed": len(ai_service.indexed_filenames),
        "indexed_books": ai_service.indexed_filenames
    }


@app.get("/api/v1/books", summary="Lista de libros disponibles para la UI de selección")
async def get_books():
    """Retorna los metadatos de los libros indexados para mostrar en la pantalla de selección."""
    return {
        "books": ai_service.get_books_info(),
        "total": len(ai_service.indexed_filenames)
    }


@app.post("/api/v1/chat", response_model=ChatResponse, summary="Procesar consulta educativa por voz")
async def process_chat(request: ChatRequest):
    """
    Recibe la pregunta transcrita del estudiante, consulta los PDFs del INEA y
    retorna una respuesta corta, empática y optimizada para síntesis de voz.
    """
    user_query = request.message.strip()
    if not user_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El mensaje no puede estar vacío."
        )

    logger.info(f"Nueva consulta recibida: '{user_query}'")

    try:
        # Ejecutar la generación en un thread separado para no bloquear FastAPI
        result = await asyncio.to_thread(
            ai_service.generate_educational_response,
            user_query,
            request.selected_book
        )
        return ChatResponse(**result)

    except Exception as e:
        logger.error(f"Error procesando la consulta: {e}")
        return ChatResponse(
            response="Disculpa, ocurrió un inconveniente al consultar tus libros. Por favor, vuelve a presionar el botón.",
            model_used="error",
            success=False,
            referenced_books=ai_service.indexed_filenames,
            error_detail=str(e)
        )


@app.post("/api/v1/preload-book", summary="Precargar un libro concreto para responder solo con ese material")
async def preload_book(request: PreloadBookRequest):
    """Carga en memoria el contenido del libro elegido para analizarlo antes del primer chat."""
    try:
        context = await asyncio.to_thread(ai_service.prepare_book_context, request.selected_book)
        return {
            "success": bool(context),
            "selected_book": request.selected_book,
            "message": "Libro precargado y listo para responder." if context else "No se encontró ese libro para precargar."
        }
    except Exception as e:
        logger.error(f"Error precargando libro: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/reload-books", summary="Volver a escanear y subir los PDFs de la carpeta data")
async def reload_books():
    """Permite recargar e indexar los libros PDF sin reiniciar el servidor."""
    try:
        indexed_books = await asyncio.to_thread(ai_service.scan_and_upload_pdfs)
        return {
            "success": True,
            "message": f"Se indexaron {len(indexed_books)} libros exitosamente.",
            "indexed_books": indexed_books
        }
    except Exception as e:
        logger.error(f"Error recargando libros: {e}")
        raise HTTPException(status_code=500, detail=str(e))
