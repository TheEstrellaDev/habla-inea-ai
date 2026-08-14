from app.services.ai_service import AIService


def test_visual_box_is_treated_as_page_relevance_signal():
    service = AIService()
    page = "Actividad de lectura. En el cuadro de la imagen, escribe la respuesta correcta."

    assert service._matches_page_number(page, 65) is True


def test_find_best_page_match_uses_visual_activity_signal_when_page_number_is_missing():
    service = AIService()
    service.book_page_texts = {
        "book.pdf": [
            "Portada del libro",
            "Tema general",
            "Actividad de la página. Completa el cuadro con la respuesta correcta.",
        ]
    }

    result = service._find_best_page_match("book.pdf", 65)
    assert result is not None
    assert result[0] == 65
