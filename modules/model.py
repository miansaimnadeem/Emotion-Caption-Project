from transformers import BlipProcessor, BlipForConditionalGeneration

print("=" * 50)
print("  Downloading BLIP Model (1.9 GB)")
print("  Please wait, this happens only once...")
print("=" * 50)

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)

print("\n✅ BLIP model downloaded and cached successfully!")
print("You can now run: python -m streamlit run app.py")