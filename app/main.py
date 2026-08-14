from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.core.config import settings
from app.services.ai_service import generate_educational_response

app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url=None
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "online",
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV
    }

@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Education"])
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint para procesar dudas educativas con el asistente HablaINEA AI.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")

    try:
        reply = generate_educational_response(request.message)
        return ChatResponse(response=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la solicitud: {str(e)}")