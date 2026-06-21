from deep_translator import GoogleTranslator


def translate_to_urdu(english_text: str) -> str:
    """Translate English caption to Urdu."""
    try:
        translator = GoogleTranslator(source="en", target="ur")
        return translator.translate(english_text)
    except Exception as e:
        return "ترجمہ دستیاب نہیں"