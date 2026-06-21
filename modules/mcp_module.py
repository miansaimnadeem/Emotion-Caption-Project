import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn
import io


app = FastAPI(
    title="Image Caption MCP Server",
    description="Local MCP Server for Emotion-Aware Image Captioning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MCP_TOOLS = [
    {"name": "generate_caption",   "description": "Generate caption from image (local BLIP model)"},
    {"name": "translate_to_urdu",  "description": "Translate English text to Urdu"},
    {"name": "detect_emotion",     "description": "Detect emotion from caption"},
    {"name": "run_full_pipeline",  "description": "Run full LangGraph agentic pipeline"},
]

@app.get("/")
async def root():
    return {"status": "running", "server": "Image Caption MCP Server", "version": "1.0.0"}

@app.get("/tools")
async def list_tools():
    return {"tools": MCP_TOOLS}

@app.post("/tools/generate_caption")
async def api_generate_caption(file: UploadFile = File(...)):
    try:
        from modules.captioner import generate_caption
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        caption = generate_caption(image)
        return {"tool": "generate_caption", "result": caption, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/translate_to_urdu")
async def api_translate(text: str):
    try:
        from modules.translator import translate_to_urdu
        urdu = translate_to_urdu(text)
        return {"tool": "translate_to_urdu", "result": urdu, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/detect_emotion")
async def api_detect_emotion(caption: str):
    try:
        from modules.emotion import detect_emotion
        result = detect_emotion(caption)
        return {"tool": "detect_emotion", "result": result, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/run_full_pipeline")
async def api_full_pipeline(file: UploadFile = File(...)):
    try:
        from modules.captioner import generate_caption
        from modules.agentic_module import run_agentic_pipeline

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        caption = generate_caption(image)
        result = run_agentic_pipeline(caption)

        return {"tool": "run_full_pipeline", "result": result, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("=" * 50)
    print("  MCP Server Starting...")
    print("  URL:      http://localhost:8000")
    print("  API Docs: http://localhost:8000/docs")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)