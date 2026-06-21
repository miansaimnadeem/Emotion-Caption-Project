import streamlit as st
from PIL import Image
import threading
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.captioner  import generate_caption
from modules.translator import translate_to_urdu
from modules.emotion    import detect_emotion
from modules.rag_module import retrieve_similar_captions


st.set_page_config(
    page_title="AI Caption Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css(filepath):
    try:
        with open(filepath) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass

load_css("assets/style.css")


st.markdown('<h1 class="main-title">🧠 AI Image Caption Generator</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Local BLIP • Multilingual • Emotion-Aware • RAG • LangGraph • MCP</p>', unsafe_allow_html=True)
st.markdown("---")


with st.sidebar:
    st.markdown("## ⚙️ Settings")
    mode = st.selectbox(
        "Select Processing Mode",
        ["Standard Mode", "RAG Mode", "Agentic AI Mode (LangGraph)"]
    )

    st.markdown("---")
    st.markdown("## ℹ️ About")
    st.markdown("""
    **Model:** Local Salesforce BLIP Large  
    **Framework:** LangChain + LangGraph  
    **RAG:** Local FAISS Vector Store  
    **MCP:** FastAPI Server  
    **Translation:** Google Translate  
    """)

    st.markdown("---")
    st.markdown("## 🖥️ MCP Server")
    if st.button("Start MCP Server"):
        def run_mcp():
            import uvicorn
            from modules.mcp_module import app as mcp_app
            uvicorn.run(mcp_app, host="0.0.0.0", port=8000, log_level="error")

        t = threading.Thread(target=run_mcp, daemon=True)
        t.start()
        st.success("MCP Server running at http://localhost:8000")
        st.info("API Docs: http://localhost:8000/docs")


st.markdown("### 📤 Upload an Image")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    st.markdown("---")

    if st.button("✨ Generate Caption & Analyze"):

        if mode == "Standard Mode":
            with st.spinner("Generating caption using local BLIP model..."):
                caption = generate_caption(image)
            with st.spinner("Translating to Urdu..."):
                urdu = translate_to_urdu(caption)
            with st.spinner("Detecting emotion..."):
                emotion_result = detect_emotion(caption)

            st.markdown("## 📊 Results — Standard Mode")
            st.markdown(f"""<div class="result-card"><div class="result-label">   English Caption</div><div class="result-text">{caption}</div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="result-card"><div class="result-label">✨ Enhanced Caption</div><div class="result-text">{emotion_result['enhanced_caption']}</div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="result-card"><div class="result-label">   Urdu Caption</div><div class="urdu-text">{urdu}</div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="result-card"><div class="result-label">🎭 Emotion</div><span class="emotion-badge">{emotion_result['emoji']} {emotion_result['emotion']}</span></div>""", unsafe_allow_html=True)

            result_text = f"English Caption: {caption}\nUrdu Caption: {urdu}\nEmotion: {emotion_result['emotion']}"
            st.download_button("📥 Download Results", result_text, "results.txt", "text/plain")


        elif mode == "RAG Mode":
            with st.spinner("Generating caption..."):
                caption = generate_caption(image)
            with st.spinner("Retrieving similar captions (local FAISS)..."):
                similar = retrieve_similar_captions(caption, k=3)
            with st.spinner("Translating..."):
                urdu = translate_to_urdu(caption)
            with st.spinner("Detecting emotion..."):
                emotion_result = detect_emotion(caption)

            st.markdown("## 📊 Results — RAG Mode")
            st.markdown(f"""<div class="result-card"><div class="result-label">   English Caption</div><div class="result-text">{caption}</div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="result-card"><div class="result-label">   Urdu Caption</div><div class="urdu-text">{urdu}</div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="result-card"><div class="result-label">🎭 Emotion</div><span class="emotion-badge">{emotion_result['emoji']} {emotion_result['emotion']}</span></div>""", unsafe_allow_html=True)

            if similar:
                st.markdown("### 🔍 RAG — Similar Captions Retrieved")
                for i, sim in enumerate(similar, 1):
                    st.markdown(f"""<div class="result-card"><div class="result-label">Similar Caption {i}</div><div class="result-text">{sim}</div></div>""", unsafe_allow_html=True)

            result_text = f"English Caption: {caption}\nUrdu Caption: {urdu}\nEmotion: {emotion_result['emotion']}\nSimilar: {'; '.join(similar)}"
            st.download_button("📥 Download Results", result_text, "results.txt", "text/plain")


        elif mode == "Agentic AI Mode (LangGraph)":
            from modules.agentic_module import run_agentic_pipeline

            with st.spinner("Generating caption..."):
                caption = generate_caption(image)
            with st.spinner("🤖 Running LangGraph Agentic Pipeline..."):
                agent_result = run_agentic_pipeline(caption)

            st.markdown("## 📊 Results — Agentic AI Mode")
            st.markdown(f"""<div class="result-card"><div class="result-label">   English Caption</div><div class="result-text">{agent_result['image_caption']}</div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="result-card"><div class="result-label">✨ Enhanced Caption</div><div class="result-text">{agent_result['enhanced_caption']}</div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="result-card"><div class="result-label">   Urdu Caption</div><div class="urdu-text">{agent_result['urdu_caption']}</div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="result-card"><div class="result-label">🎭 Emotion</div><span class="emotion-badge">{agent_result['emotion']}</span></div>""", unsafe_allow_html=True)

            if agent_result.get("similar_captions"):
                st.markdown("### 🔍 RAG Retrieved Captions")
                for i, sim in enumerate(agent_result["similar_captions"], 1):
                    st.info(f"**{i}.** {sim}")

            st.markdown("### 🤖 LangGraph Agent Steps")
            for step in agent_result.get("steps_completed", []):
                st.success(f"✅ {step}")

            result_text = f"English: {agent_result['image_caption']}\nEnhanced: {agent_result['enhanced_caption']}\nUrdu: {agent_result['urdu_caption']}\nEmotion: {agent_result['emotion']}"
            st.download_button("📥 Download Results", result_text, "results.txt", "text/plain")

else:
    st.markdown("""<div class="upload-section"><h3>👆 Upload an image to get started</h3><p style="color:#a0aec0;">Supports Standard, RAG, and Agentic AI modes</p></div>""", unsafe_allow_html=True)
