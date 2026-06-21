from langgraph.graph import StateGraph, END
from typing import TypedDict, List


class AgentState(TypedDict):
    image_caption:    str
    emotion:          str
    similar_captions: List[str]
    enhanced_caption: str
    urdu_caption:     str
    final_output:     str
    steps_completed:  List[str]


def analyze_caption_node(state: AgentState) -> AgentState:
    """Node 1: Analyze caption quality locally."""
    caption = state["image_caption"]
    steps = state.get("steps_completed", [])

    word_count = len(caption.split())
    quality = "good" if word_count >= 5 else "needs improvement"
    steps.append(f"Step 1 — Caption analyzed: {word_count} words, quality: {quality}")

    state["steps_completed"] = steps
    return state


def retrieve_context_node(state: AgentState) -> AgentState:
    """Node 2: Retrieve similar captions using local RAG + FAISS."""
    from modules.rag_module import retrieve_similar_captions

    caption = state["image_caption"]
    steps = state.get("steps_completed", [])

    try:
        similar = retrieve_similar_captions(caption, k=3)
        state["similar_captions"] = similar
        steps.append(f"Step 2 — RAG retrieved {len(similar)} similar captions from local FAISS store")
    except Exception as e:
        state["similar_captions"] = []
        steps.append(f"Step 2 — RAG retrieval skipped: {str(e)[:40]}")

    state["steps_completed"] = steps
    return state


def enhance_caption_node(state: AgentState) -> AgentState:
    """Node 3: Detect emotion and enhance caption using LangChain (local)."""
    from modules.emotion import detect_emotion
    from modules.langchain_module import enhance_with_langchain

    caption = state["image_caption"]
    similar = state.get("similar_captions", [])
    steps = state.get("steps_completed", [])

    emotion_result = detect_emotion(caption)
    state["emotion"] = emotion_result["emotion"]

    try:
        enhanced = enhance_with_langchain(caption, emotion_result["emotion"], similar)
        state["enhanced_caption"] = enhanced
        steps.append(f"Step 3 — Caption enhanced via LangChain pipeline. Emotion: {emotion_result['emotion']}")
    except Exception:
        state["enhanced_caption"] = emotion_result["enhanced_caption"]
        steps.append(f"Step 3 — Caption enhanced with emotion: {emotion_result['emotion']}")

    state["steps_completed"] = steps
    return state


def translate_node(state: AgentState) -> AgentState:
    """Node 4: Translate enhanced caption to Urdu."""
    from modules.translator import translate_to_urdu

    enhanced = state["enhanced_caption"]
    steps = state.get("steps_completed", [])

    urdu = translate_to_urdu(enhanced)
    state["urdu_caption"] = urdu
    steps.append("Step 4 — Caption translated to Urdu successfully")

    state["steps_completed"] = steps
    return state


def finalize_node(state: AgentState) -> AgentState:
    """Node 5: Compile final output."""
    steps = state.get("steps_completed", [])

    final = f"""
AGENTIC AI FINAL OUTPUT

English Caption:  {state['image_caption']}
Enhanced Caption: {state['enhanced_caption']}
Urdu Caption:     {state['urdu_caption']}
Emotion:          {state['emotion']}
Similar Scenes:   {len(state.get('similar_captions', []))} found
    """.strip()

    state["final_output"] = final
    steps.append("Step 5 — Final output compiled by LangGraph agent")
    state["steps_completed"] = steps
    return state


def build_agent_graph():
    """Building and compiling the LangGraph agent pipeline."""
    graph = StateGraph(AgentState)

    graph.add_node("analyze",   analyze_caption_node)
    graph.add_node("retrieve",  retrieve_context_node)
    graph.add_node("enhance",   enhance_caption_node)
    graph.add_node("translate", translate_node)
    graph.add_node("finalize",  finalize_node)

    graph.set_entry_point("analyze")
    graph.add_edge("analyze",   "retrieve")
    graph.add_edge("retrieve",  "enhance")
    graph.add_edge("enhance",   "translate")
    graph.add_edge("translate", "finalize")
    graph.add_edge("finalize",  END)

    return graph.compile()


def run_agentic_pipeline(caption: str) -> dict:
    """Run the complete LangGraph agentic pipeline on a caption."""
    try:
        agent = build_agent_graph()

        initial_state = AgentState(
            image_caption=caption,
            emotion="",
            similar_captions=[],
            enhanced_caption="",
            urdu_caption="",
            final_output="",
            steps_completed=[]
        )

        result = agent.invoke(initial_state)
        return result

    except Exception as e:
        print(f"Agent pipeline error: {e}")
        return {
            "image_caption":    caption,
            "emotion":          "Neutral / Informational",
            "similar_captions": [],
            "enhanced_caption": caption,
            "urdu_caption":     "",
            "final_output":     caption,
            "steps_completed":  [f"Error: {str(e)}"]
        }