import glob
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError
import pypdf

from app.core.config import settings

logger = logging.getLogger("ai_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

SYSTEM_INSTRUCTION = """Eres PrimarIA INEA AI (HablaINEA AI), un tutor educativo de voz paciente, empático, cálido y muy respetuoso para adultos mayores que cursan la educación primaria en el INEA.

REGLAS INVIOLABLES:
1. Basate ÚNICAMENTE en la información de los libros de estudio del INEA provistos en el contexto.
2. Si el estudiante dice 'este libro', 'ese libro', 'del libro que elegimos', 'la página 50', 'página 50', 'pag 50', o menciona un número de página, interpreta que se refiere al libro actualmente seleccionado en la interfaz y usa ese libro como contexto principal, aunque la frase no tenga palabras clave exactas.
3. Si el usuario eligió un libro al inicio, prioriza ese libro por encima de otros y no respondas que no está en los libros solo porque la pregunta no usa palabras exactas del texto.
4. Usa un lenguaje sumamente sencillo, claro, directo y amigable. Trata al estudiante con el máximo respeto y cariño.
5. Explica usando palabras cotidianas y ejemplos prácticos de la vida diaria: el hogar, la cocina, el mercado, la familia o el trabajo.
6. Responde SIEMPRE en 1 o 2 oraciones muy breves (máximo 50 palabras). El texto será leído por voz en el celular.
7. NO uses asteriscos (*), almohadillas (#), guiones de viñeta, emojis ni ningún símbolo de formato Markdown. Solo texto limpio.
8. Cuando la actividad sea visual, como completar un cuadro, recuadro, imagen, ilustración, dibujar, señalar, colorear, encerrar o rellenar una casilla, interpreta la instrucción como una tarea visual del libro, no como una ausencia de contenido. Si la página muestra una actividad visual, explícalo con claridad y menciona que se trata de la instrucción visual del libro.
9. Si definitivamente la pregunta no tiene relación con los libros del INEA y no se eligió ningún libro ni se menciona una referencia clara al material del INEA, responde exactamente: "Disculpa, esa información no viene en tus libros de estudio, pero dime si puedo ayudarte con otra duda."
10. Si la pregunta es sobre qué es un libro o materia del INEA, explica brevemente de qué trata ese libro.
"""


class AIService:
    """Servicio de IA para procesar consultas educativas sobre libros de texto INEA."""

    VISUAL_ACTIVITY_KEYWORDS = (
        "cuadro",
        "recuadro",
        "casilla",
        "imagen",
        "ilustracion",
        "ilustración",
        "figura",
        "dibujo",
        "observa",
        "completa",
        "rellena",
        "escribe",
        "encierra",
        "subraya",
        "colorea",
        "señala",
        "actividad",
        "ejercicio",
        "responde",
        "pregunta",
        "tarea",
    )

    # Mapeo de nombre de archivo a información del libro
    BOOK_INFO = {
        "lengua_comunicacion_1": {
            "label": "Lengua y Comunicación 1",
            "emoji": "📖",
            "description": "Aprendo a leer, escribir y comunicarme",
            "color": "blue",
        },
        "lengua_comunicacion_2": {
            "label": "Lengua y Comunicación 2",
            "emoji": "📝",
            "description": "Textos, cartas y lectura avanzada",
            "color": "indigo",
        },
        "pensamiento_matematico_1": {
            "label": "Pensamiento Matemático 1",
            "emoji": "🔢",
            "description": "Números, sumas, restas y medidas",
            "color": "green",
        },
        "pensamiento_matematico_2": {
            "label": "Pensamiento Matemático 2",
            "emoji": "➗",
            "description": "Fracciones, geometría y problemas",
            "color": "emerald",
        },
        "vida_comunidad_1": {
            "label": "Vida en Comunidad 1",
            "emoji": "🏘️",
            "description": "Salud, familia, naturaleza y derechos",
            "color": "amber",
        },
    }

    def __init__(self):
        self.api_key: str = settings.GEMINI_API_KEY
        self.client: Optional[genai.Client] = None
        self.uploaded_files: List[Any] = []
        self.indexed_filenames: List[str] = []
        self.book_texts: Dict[str, str] = {}  # {filename: extracted_text}
        self.book_page_texts: Dict[str, List[str]] = {}  # {filename: [page_texts]}
        self.book_page_visual_flags: Dict[str, List[bool]] = {}  # {filename: [page_has_visual_content]}
        self.preloaded_book_contexts: Dict[str, str] = {}
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Inicializa el cliente de Google GenAI con la API Key configurada."""
        try:
            if not self.api_key or self.api_key.startswith("tu_api_key"):
                logger.warning("GEMINI_API_KEY no está configurada adecuadamente.")
                return
            self.client = genai.Client(api_key=self.api_key)
            logger.info("Cliente Google GenAI inicializado correctamente.")
        except Exception as e:
            logger.error(f"Error al inicializar el cliente Google GenAI: {e}")
            self.client = None

    def scan_and_upload_pdfs(self) -> List[str]:
        """Escanea data/ buscando PDFs, extrae su texto con caché en disco."""
        if not self.client:
            self._initialize_client()

        data_dir = settings.DATA_DIR
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)

        pdf_pattern = str(data_dir / "*.pdf")
        pdf_paths = glob.glob(pdf_pattern)

        if not pdf_paths:
            logger.warning(f"No se encontraron archivos PDF en '{pdf_pattern}'.")
            self.uploaded_files = []
            self.indexed_filenames = []
            self.book_texts = {}
            return []

        logger.info(f"Se encontraron {len(pdf_paths)} libros PDF. Extrayendo texto e indexando...")

        cache_file = data_dir / ".books_text_cache.json"
        import json
        text_cache: Dict[str, Any] = {}
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    text_cache = json.load(f)
                logger.info(f"Caché local cargado con {len(text_cache)} libros.")
            except Exception as e:
                logger.warning(f"No se pudo leer la caché local: {e}")

        filenames_list = []
        extracted_texts = {}
        extracted_pages = {}
        extracted_visual_flags = {}
        cache_updated = False

        for pdf_path_str in pdf_paths:
            pdf_path = Path(pdf_path_str)
            filename = pdf_path.name
            filenames_list.append(filename)
            mtime = pdf_path.stat().st_mtime

            cached_item = text_cache.get(filename, {})
            cached_pages = cached_item.get("pages", [])
            has_valid_pages = isinstance(cached_pages, list) and len(cached_pages) > 0

            if filename in text_cache and cached_item.get("mtime") == mtime and has_valid_pages:
                extracted_texts[filename] = cached_item.get("text", "")
                extracted_pages[filename] = cached_pages
                extracted_visual_flags[filename] = cached_item.get("visual_flags", [])
                logger.info(f"Libro desde caché: {filename} ({len(extracted_texts[filename]):,} chars, {len(cached_pages)} pgs)")
                continue

            if filename in text_cache and cached_item.get("mtime") == mtime and not has_valid_pages:
                logger.info(f"Caché incompleta detectada en {filename}; reextrayendo páginas para habilitar contexto por página.")

            try:
                reader = pypdf.PdfReader(str(pdf_path))
                page_texts = []
                visual_flags = []
                for page in reader.pages:
                    txt = page.extract_text()
                    visual_flags.append(bool(page.images))
                    if txt:
                        page_texts.append(txt)
                    else:
                        page_texts.append("")
                full_text = "\n".join(page_texts)
                extracted_texts[filename] = full_text
                extracted_pages[filename] = page_texts
                extracted_visual_flags[filename] = visual_flags
                text_cache[filename] = {"mtime": mtime, "text": full_text, "pages": page_texts, "visual_flags": visual_flags}
                cache_updated = True
                logger.info(f"Libro extraído: {filename} ({len(reader.pages)} pgs, {len(full_text):,} chars)")
            except Exception as e:
                logger.error(f"Error extrayendo texto de '{filename}': {e}")
                extracted_texts[filename] = ""
                extracted_pages[filename] = []
                extracted_visual_flags[filename] = []

        if cache_updated:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(text_cache, f, ensure_ascii=False)
                logger.info("Caché local guardada.")
            except Exception as e:
                logger.warning(f"No se pudo guardar la caché: {e}")

        # Verificar archivos remotos activos en Gemini
        existing_remote_files: Dict[str, Any] = {}
        if self.client:
            try:
                for remote_f in self.client.files.list():
                    if remote_f.display_name and str(remote_f.state.name).upper() == "ACTIVE":
                        existing_remote_files[remote_f.display_name] = remote_f
            except Exception as e:
                logger.warning(f"Aviso al consultar Files API: {e}")

        uploaded_list = [existing_remote_files[f] for f in filenames_list if f in existing_remote_files]

        self.uploaded_files = uploaded_list
        self.indexed_filenames = filenames_list
        self.book_texts = extracted_texts
        self.book_page_texts = extracted_pages
        self.book_page_visual_flags = extracted_visual_flags
        logger.info(f"Indexación completa: {len(self.indexed_filenames)} libros listos.")
        return self.indexed_filenames

    def _has_visual_task_hint(self, user_message: str) -> bool:
        """Detecta preguntas sobre actividades visuales del libro, no solo texto."""
        lowered = user_message.lower()
        visual_keywords = [
            "cuadro", "recuadro", "casilla", "imagen", "ilustracion", "ilustración",
            "dibujar", "colorear", "encierra", "rellena", "subraya", "señala", "observa",
            "completa el cuadro", "escribe en el cuadro", "actividad visual", "imagen del libro"
        ]
        return any(keyword in lowered for keyword in visual_keywords)

    def get_books_info(self) -> List[Dict[str, Any]]:
        """Retorna metadatos de los libros disponibles para la UI de selección."""
        books = []
        for filename in self.indexed_filenames:
            stem = Path(filename).stem  # e.g. "lengua_comunicacion_1"
            info = self.BOOK_INFO.get(stem, {
                "label": stem.replace("_", " ").title(),
                "emoji": "📚",
                "description": "Libro de estudio INEA",
                "color": "slate",
            })
            books.append({
                "filename": filename,
                "stem": stem,
                **info
            })
        return books

    def _extract_page_reference(self, user_message: str) -> Optional[int]:
        """Extrae un número de página cuando el usuario menciona 'página 50', 'pag 50' o 'la ilustración 16'."""
        match = re.search(
            r"(?:p[aá]gina|pagina|pag(?:ina)?|p\.|ilustraci[oó]n|numero|número)\s*(?:de\s+la\s+|de\s+|del\s+)?(\d{1,3})",
            user_message,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _book_hint_from_query(self, user_message: str) -> Optional[str]:
        """Detecta si el usuario menciona explícitamente un libro por nombre."""
        normalized = user_message.lower()
        candidates = {
            "lengua_comunicacion": ["lengua", "comunicacion", "comunicación", "lectura", "escritura", "lengua y comunicacion"],
            "pensamiento_matematico": ["matematicas", "matemáticas", "matematico", "pensamiento matematico", "numeros", "sumas"],
            "vida_comunidad": ["vida y comunidad", "vida en comunidad", "comunidad", "salud", "familia", "naturaleza"],
        }
        for stem, keywords in candidates.items():
            if any(k in normalized for k in keywords):
                for filename in self.indexed_filenames:
                    if stem in filename.lower():
                        return filename
        return None

    def _page_candidates(self, ref_number: int) -> List[int]:
        """Lista de páginas razonables cuando el lector marca una página con un desfasaje pequeño."""
        return sorted({ref_number + delta for delta in (-2, -1, 0, 1, 2)})

    def _query_keywords(self, user_message: str) -> List[str]:
        """Extrae palabras útiles de la pregunta para puntuar la mejor página candidata."""
        words = re.findall(r"\w+", user_message.lower())
        stop = {
            "de", "la", "que", "el", "en", "y", "a", "los", "se", "del", "las", "un",
            "por", "con", "no", "una", "su", "para", "es", "al", "lo", "como", "mas",
            "pero", "sus", "le", "ya", "o", "fue", "este", "ha", "si", "porque", "esta",
            "cuando", "muy", "sin", "sobre", "ser", "tiene", "tambien", "me", "hasta",
            "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos", "uno",
            "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos", "e", "esto",
            "mi", "antes", "algunos", "unos", "yo", "otro", "otras", "otra", "tanto",
            "esa", "estos", "mucho", "quienes", "nada", "muchos", "cual", "poco",
            "ella", "estar", "estas", "algunas", "algo", "nosotros", "mis", "tu", "te",
            "ti", "que", "cual", "es", "son", "qué", "cuál", "pagina", "página", "pag"
        }
        return [w for w in words if w not in stop and len(w) > 2]

    def _score_page_candidate(self, page_text: str, requested_page: int, candidate_page: int, query_keywords: List[str]) -> int:
        """Asigna puntaje a una página según coincidencia visual, temática y cercanía al número solicitado."""
        if not page_text:
            return -10_000

        normalized = re.sub(r"\s+", " ", page_text).replace("\xa0", " ").strip().lower()
        score = 0

        distance = abs(candidate_page - requested_page)
        score += max(0, 25 - (distance * 8))

        if re.search(rf"(?<!\d){requested_page}(?!\d)", normalized):
            score += 80
        if re.search(rf"(?:p[aá]gina|pagina|pag(?:ina)?)\s*{requested_page}", normalized, flags=re.IGNORECASE):
            score += 100

        visual_hits = sum(1 for kw in self.VISUAL_ACTIVITY_KEYWORDS if kw in normalized)
        score += min(visual_hits, 6) * 10

        query_hits = sum(1 for kw in query_keywords if kw in normalized)
        score += min(query_hits, 8) * 12

        if "actividad" in normalized and "definiciones" in normalized:
            score += 20

        return score

    def _matches_page_number(self, page_text: str, ref_number: int) -> bool:
        """Comprueba si la numeración de la página coincide con la referencia solicitada."""
        if not page_text:
            return False

        normalized = re.sub(r"\s+", " ", page_text).replace("\xa0", " ").strip()
        lowered = normalized.lower()

        visual_activity_hits = [keyword for keyword in self.VISUAL_ACTIVITY_KEYWORDS if keyword in lowered]
        if visual_activity_hits and re.search(rf"(?<!\d){ref_number}(?!\d)", normalized):
            return True

        page_context_markers = [
            "primaria", "lengua", "comunicación", "comunicacion", "matematicas",
            "matemáticas", "vida", "comunidad", "actividad", "tema", "unidad",
            "modelo de educación para la vida", "aprendeinea"
        ]

        visible_patterns = [
            rf"\|\s*{ref_number}\s*\|",
            rf"\|\s*{ref_number}\s*$",
            rf"^\s*{ref_number}\s*\|",
            rf"^\s*{ref_number}\s*$",
            rf"(?:p[aá]gina|pagina|pag(?:ina)?)\s*{ref_number}",
        ]
        for pattern in visible_patterns:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                if any(marker in lowered for marker in page_context_markers):
                    return True
                if re.search(rf"\|\s*{ref_number}\s*\|", normalized, flags=re.IGNORECASE):
                    return True

        if re.search(rf"(?<!\d){ref_number}(?!\d)", normalized):
            if any(marker in lowered for marker in page_context_markers):
                return True

        return False

    def _find_best_page_match(self, filename: str, ref_number: int, user_message: Optional[str] = None) -> Optional[Tuple[int, str]]:
        """Busca la mejor coincidencia de página, aceptando un desfase de 2 páginas."""
        page_pages = self.book_page_texts.get(filename, [])
        if not page_pages:
            return None

        query_keywords = self._query_keywords(user_message or "")
        best_candidate: Optional[Tuple[int, str]] = None
        best_score = -10_000

        # Buscar en una ventana cercana a la página pedida. En estos libros suele existir
        # un desfase pequeño entre numeración visible y posición física del PDF.
        window_start = max(1, ref_number - 5)
        window_end = min(len(page_pages), ref_number + 8)
        for candidate in range(window_start, window_end + 1):
            page_text = page_pages[candidate - 1]
            score = self._score_page_candidate(page_text, ref_number, candidate, query_keywords)
            if score > best_score:
                best_score = score
                best_candidate = candidate, page_text

        if best_candidate and best_score >= 20:
            return best_candidate

        for candidate in self._page_candidates(ref_number):
            if candidate < 1 or candidate > len(page_pages):
                continue
            page_text = page_pages[candidate - 1]
            if self._matches_page_number(page_text, candidate):
                return candidate, page_text

        if best_candidate and best_score >= 15:
            return best_candidate

        # Si no apareció texto visible, usamos la página física equivalente del PDF.
        if 1 <= ref_number <= len(page_pages):
            return ref_number, page_pages[ref_number - 1]

        return None

    def _build_context_note(self, user_message: str, selected_book: Optional[str]) -> str:
        """Crea una nota de contexto que refuerza la referencia al libro seleccionado y a páginas."""
        if not user_message:
            return ""

        normalized = user_message.lower()
        selected_label = ""
        if selected_book:
            stem = Path(selected_book).stem if "." in selected_book else selected_book
            selected_label = self.BOOK_INFO.get(stem, {}).get("label", selected_book)

        page_ref = self._extract_page_reference(user_message)
        book_reference = any(phrase in normalized for phrase in [
            "este libro", "ese libro", "este libro", "ese libro", "del libro",
            "de este libro", "de ese libro", "el libro que elegimos", "el libro seleccionado",
            "la pagina", "la página", "pagina", "página"
        ])

        pieces = []
        if selected_book and selected_label:
            pieces.append(f"El estudiante eligió el libro '{selected_label}'.")
        if page_ref:
            pieces.append(f"La pregunta hace referencia a la página {page_ref}.")
        if book_reference and selected_book:
            pieces.append("Cuando dice 'este libro', 'ese libro' o 'la página', debe entenderse como el libro que ya fue seleccionado por el estudiante.")

        if not pieces:
            return ""
        return "Contexto adicional: " + " ".join(pieces)

    def prepare_book_context(self, selected_book: Optional[str]) -> str:
        """Precarga el contenido de un libro concreto para que la IA solo analice ese archivo."""
        if not selected_book:
            return ""
        if selected_book not in self.book_texts:
            return ""
        if selected_book not in self.preloaded_book_contexts:
            text = self.book_texts[selected_book]
            self.preloaded_book_contexts[selected_book] = text[:20000]
        return self.preloaded_book_contexts[selected_book]

    def _get_relevant_context(
        self,
        user_message: str,
        selected_book: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """Obtiene el contexto exacto del libro activo y, si existe, de la página pedida."""
        if not self.book_texts:
            return "", []

        import unicodedata

        def normalize(s: str) -> str:
            return ''.join(
                c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn'
            ).lower()

        query = user_message.lower()
        query_norm = normalize(query)
        words = re.findall(r"\w+", query_norm)
        stop_words = {
            "de", "la", "que", "el", "en", "y", "a", "los", "se", "del", "las", "un",
            "por", "con", "no", "una", "su", "para", "es", "al", "lo", "como", "mas",
            "pero", "sus", "le", "ya", "o", "fue", "este", "ha", "si", "porque", "esta",
            "cuando", "muy", "sin", "sobre", "ser", "tiene", "tambien", "me", "hasta",
            "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos", "uno",
            "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos", "e", "esto",
            "mi", "antes", "algunos", "unos", "yo", "otro", "otras", "otra", "tanto",
            "esa", "estos", "mucho", "quienes", "nada", "muchos", "cual", "poco",
            "ella", "estar", "estas", "algunas", "algo", "nosotros", "mis", "tu", "te",
            "ti", "que", "cual", "es", "son", "qué", "cuál", "pagina", "página", "pag"
        }
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        page_ref = self._extract_page_reference(user_message)
        book_hint = self._book_hint_from_query(user_message)
        target_book = selected_book or book_hint

        if target_book:
            book_items = [
                (fname, txt)
                for fname, txt in self.book_texts.items()
                if target_book in fname or fname == target_book
            ]
        else:
            book_items = list(self.book_texts.items())

        context_snippets = []
        referenced_books = []

        for filename, text in book_items:
            referenced_books.append(filename)

            if self._has_visual_task_hint(user_message):
                page_ref = self._extract_page_reference(user_message) or 1
                match_result = self._find_best_page_match(filename, page_ref, user_message)
                if match_result:
                    matched_page_number, matched_page_text = match_result
                    visual_hint = "La pregunta parece referirse a una actividad visual del libro (cuadro, imagen, casilla, ilustración o completar una tarea visual). No respondas que faltó ese elemento cuando la página lo pide."
                    context_snippets.append(
                        f"--- {filename} (página {matched_page_number}, actividad visual probable) ---\n{matched_page_text}\n\nNota: {visual_hint}"
                    )
                    continue

            if page_ref is not None:
                match_result = self._find_best_page_match(filename, page_ref, user_message)
                if match_result:
                    matched_page_number, matched_page_text = match_result
                    context_snippets.append(
                        f"--- {filename} (página {matched_page_number}, aproximación de lectura) ---\n{matched_page_text}\n\nNota: se corrigió el desfase de lectura del estudiante para ubicar la página más cercana del libro."
                    )
                    continue

            # Si tiene un libro activo, usamos el contenido del libro completo, pero priorizamos
            # frases relevantes y evita la respuesta genérica.
            if target_book:
                text_clean = text.strip()
                if len(text_clean) > 4000:
                    text_clean = text_clean[:4000]
                context_snippets.append(
                    f"--- {filename} ---\n{text_clean}\n\nNota: El estudiante eligió este libro como contexto principal."
                )
                continue

            chunk_size = 500
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            scored_chunks = []
            for chunk in chunks:
                chunk_clean = chunk.strip()
                if len(chunk_clean) < 50:
                    continue
                chunk_norm = normalize(chunk_clean)
                score = sum(1 for kw in keywords if kw in chunk_norm)
                if score > 0:
                    scored_chunks.append((score, chunk_clean))

            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            top_chunks = [c[1] for c in scored_chunks[:8]]
            if top_chunks:
                context_snippets.append(f"--- {filename} ---\n" + "\n".join(top_chunks))
            else:
                total = len(text)
                step = max(1, total // 6)
                samples = [text[i:i+800] for i in range(step, min(total, step * 5), step)]
                sample_text = "\n".join(samples)
                context_snippets.append(f"--- {filename} ---\n{sample_text}")

        context_str = "\n\n".join(context_snippets)
        if len(context_str) > 20000:
            context_str = context_str[:20000]

        return context_str, referenced_books

    def _clean_tts_text(self, text: str) -> str:
        """Limpia texto para que el sintetizador de voz no lea caracteres extraños."""
        if not text:
            return ""
        cleaned = re.sub(r"[\*#_`~>\[\]\(\)]", "", text)
        cleaned = re.sub(r"\n+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def generate_educational_response(
        self,
        user_message: str,
        selected_book: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Genera una respuesta educativa basada en los libros INEA.
        Si selected_book está definido, solo consulta ese libro.
        """
        if not self.client:
            self._initialize_client()
            if not self.client:
                return {
                    "response": "Disculpa, el servicio de inteligencia artificial no está configurado correctamente.",
                    "model_used": "none",
                    "success": False,
                    "referenced_books": []
                }

        if not self.book_texts:
            logger.info("Sin libros en memoria. Escaneando data/...")
            self.scan_and_upload_pdfs()

        if selected_book:
            self.prepare_book_context(selected_book)

        # Obtener fragmentos relevantes del libro seleccionado o los más pertinentes
        context_text, referenced_books = self._get_relevant_context(user_message, selected_book)

        user_lower = user_message.lower()
        has_visual_request = any(keyword in user_lower for keyword in [
            "cuadro", "recuadro", "casilla", "imagen", "ilustracion", "ilustración",
            "dibujar", "colorear", "encierra", "rellena", "subraya", "señala",
            "escribe en el cuadro", "completa el cuadro", "observa la imagen"
        ])

        book_label = ""
        if selected_book:
            stem = Path(selected_book).stem if "." in selected_book else selected_book
            info = self.BOOK_INFO.get(stem, {})
            book_label = info.get("label", selected_book)

        context_note = self._build_context_note(user_message, selected_book)

        # Construir el prompt
        book_note = f" El estudiante está estudiando el libro '{book_label}'." if book_label else ""
        prompt = (
            f"EXTRACTO DEL LIBRO DE TEXTO DEL INEA:{book_note}\n"
            f"{context_note}\n"
            f"{context_text}\n\n"
            f"PREGUNTA DEL ESTUDIANTE: {user_message}\n\n"
            f"Responde SOLO con información que aparezca en este contenido del libro. "
            f"Si la actividad es visual o pide completar un cuadro, imagen, recuadro, casilla o ilustración, identifica esa instrucción visual en la página y responde a partir de ello. "
            f"No digas que 'no hay un cuadro' ni que no aparece en la página cuando la actividad del libro claramente pide completar o observar una imagen o cuadro. "
            f"Si la pregunta menciona una página, usa la numeración visible del libro y no la numeración del archivo PDF. Ignora la portada, índice y páginas preliminares. "
            f"Si la persona eligió un libro al inicio, ese libro es el contexto principal aunque la pregunta sea breve o imprecisa. "
            f"No respondas con frases pregrabadas ni con mensajes del tipo 'no viene en tus libros' cuando el contenido del libro sí lo explica. "
            f"Responde de forma concreta: menciona exactamente qué pide la actividad y cómo resolverla paso a paso en palabras simples. "
            f"Si hay una tabla o cuadro, indica qué va en cada columna o casilla usando los términos del ejercicio. "
            f"Explica de manera muy simple, amable y útil para adultos mayores, en 1 o 2 oraciones muy cortas y claras. "
            f"Sin asteriscos, sin formato, solo texto limpio para leer en voz alta."
        )

        # Modelos a intentar
        models_to_try = [settings.GEMINI_MODEL]
        for fallback in settings.FALLBACK_MODELS:
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.3,
            max_output_tokens=300
        )

        last_error: Optional[Exception] = None

        for model_name in models_to_try:
            for attempt in range(settings.MAX_RETRIES):
                try:
                    logger.info(f"Generando con '{model_name}' (Intento {attempt + 1}/{settings.MAX_RETRIES})...")
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=config
                    )

                    raw_text = response.text or ""
                    cleaned_text = self._clean_tts_text(raw_text)

                    if not cleaned_text:
                        cleaned_text = "Disculpa, esa información no viene en tus libros de estudio, pero dime si puedo ayudarte con otra duda."

                    return {
                        "response": cleaned_text,
                        "model_used": model_name,
                        "success": True,
                        "referenced_books": referenced_books
                    }

                except ServerError as e:
                    logger.warning(f"Error 503 en '{model_name}': Reintentando...")
                    last_error = e
                    time.sleep(settings.RETRY_DELAY * (attempt + 1))
                except ClientError as e:
                    logger.warning(f"Error de cliente en '{model_name}': {e}. Siguiente modelo...")
                    last_error = e
                    break
                except APIError as e:
                    logger.warning(f"Error API en '{model_name}': Reintentando...")
                    last_error = e
                    time.sleep(settings.RETRY_DELAY)
                except Exception as e:
                    logger.error(f"Error inesperado con '{model_name}': {e}")
                    last_error = e
                    break

        if last_error is not None and selected_book:
            return {
                "response": "Disculpa, no pude encontrar una explicación clara en la página elegida. Te puedo ayudar a revisar otra parte del mismo libro.",
                "model_used": model_name,
                "success": False,
                "referenced_books": referenced_books,
                "error_detail": str(last_error) if last_error else "Error desconocido"
            }

        logger.error(f"Todos los modelos fallaron. Último error: {last_error}")
        return {
            "response": "Disculpa, no pude consultar tus libros en este momento. Por favor vuelve a intentarlo.",
            "model_used": "fallback_error",
            "success": False,
            "referenced_books": referenced_books,
            "error_detail": str(last_error) if last_error else "Error desconocido"
        }


# Instancia singleton del servicio de IA
ai_service = AIService()
