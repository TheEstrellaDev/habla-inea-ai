from google import genai
from google.genai import types
from app.core.config import settings
import time

SYSTEM_INSTRUCTION = """
Eres HablaINEA AI, un asistente educativo paciente, empático y respetuoso, diseñado para apoyar a personas adultas mayores que cursan la educación primaria en el INEA (Instituto Nacional para la Educación de los Adultos).

Sigue estrictamente estas reglas en cada respuesta:
1. Usa un lenguaje claro, sencillo y libre de modismos técnicos o vocabulario complejo.
2. Mantén un tono cálido, humano y alentador en todo momento.
3. Explica los conceptos paso a paso y con ejemplos cotidianos (la vida en el hogar, el campo, el mercado o el trabajo).
4. No abrumes con textos largos; usa párrafos breves y estructurados.
5. Si el usuario se equivoca o muestra duda, valida su esfuerzo de forma amable.
6. Limítate a contenidos de alfabetización, matemáticas básicas, ciencias de la vida y formación ciudadana de nivel primaria.
"""

def generate_educational_response(prompt: str) -> str:
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY no está configurada en las variables de entorno.")

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    candidate_models = [settings.GEMINI_MODEL, "gemini-3.5-flash-lite"]
    # Remove duplicates while keeping order.
    candidate_models = list(dict.fromkeys(candidate_models))

    last_error = None
    for model_name in candidate_models:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.3,
                    ),
                )
                return response.text
            except Exception as exc:
                last_error = exc
                message = str(exc)

                if "no longer available to new users" in message or "NOT_FOUND" in message:
                    break

                if "503 UNAVAILABLE" in message and attempt < 2:
                    time.sleep(1.2 * (attempt + 1))
                    continue

                break

    if last_error:
        message = str(last_error)
        if "no longer available to new users" in message or "NOT_FOUND" in message:
            raise RuntimeError(
                "El modelo configurado en GEMINI_MODEL no esta disponible para esta cuenta. "
                "Prueba con un modelo vigente, por ejemplo: gemini-3.5-flash o gemini-3.5-flash-lite."
            ) from last_error
        if "503 UNAVAILABLE" in message:
            raise RuntimeError(
                "Gemini esta temporalmente saturado (503). Intenta de nuevo en unos segundos."
            ) from last_error
        raise last_error

    raise RuntimeError("No fue posible generar respuesta con los modelos configurados.")