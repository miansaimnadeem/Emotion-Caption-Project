from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

_processor = None
_model = None

def _load_model():
    """Load BLIP processor and model once (cached for performance)."""
    global _processor, _model

    if _processor is None or _model is None:
        print("Loading BLIP model from local cache...")

        _processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        _model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device) # type: ignore
        _model.eval()

        print(f"Model loaded on device: {device}")

    return _processor, _model


def generate_caption(image: Image.Image) -> str:
    """
    Generate a descriptive English caption for the given image
    using the locally downloaded BLIP model.
    """
    try:
        processor, model = _load_model()

        # Ensure RGB format
        image = image.convert("RGB")

        # Preprocess
        inputs = processor(images=image, return_tensors="pt")  # type: ignore 

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate caption
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=80,
                num_beams=5,
                min_length=10,
                repetition_penalty=1.5,
                length_penalty=1.0,
            )

        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()

    except Exception as e:
        return f"Error generating caption: {str(e)}"