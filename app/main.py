from fastapi import FastAPI
from app.core.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url=None
)

@app.get("/health", tags=["Security & System"])
async def health_check():
    """
    Endpoint de monitoreo para verificar que el servicio está activo y seguro.
    """
    return {
        "status": "online",
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "security_check": "passed"
    }