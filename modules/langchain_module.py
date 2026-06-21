from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def _local_enhancement_function(inputs: dict) -> str:
   
    caption = inputs.get("caption", "")
    emotion = inputs.get("emotion", "Neutral")
    similar_captions = inputs.get("similar_captions", "")

    # Build enhanced caption using local logic
    if similar_captions and similar_captions != "None":
        first_similar = similar_captions.split("\n")[0]
        enhanced = (
            f"{caption}. This scene shares similarities with: "
            f"'{first_similar}' — overall mood detected as {emotion}."
        )
    else:
        enhanced = f"{caption}. Overall mood detected as {emotion}."

    return enhanced


def build_langchain_pipeline():
    """
    Build LangChain LCEL pipeline using local logic
    (no external API calls needed).
    """
    try:
        prompt = PromptTemplate(
            input_variables=["caption", "emotion", "similar_captions"],
            template="{caption}|{emotion}|{similar_captions}"
        )

        local_chain = RunnableLambda(_local_enhancement_function)

        chain = RunnablePassthrough() | local_chain

        return chain

    except Exception as e:
        print(f"LangChain pipeline error: {e}")
        return None


def enhance_with_langchain(caption: str, emotion: str, similar_captions: list) -> str:
    """Enhance caption using LangChain local pipeline."""
    try:
        chain = build_langchain_pipeline()
        if chain is None:
            return caption

        similar_text = "\n".join(similar_captions) if similar_captions else "None"

        result = chain.invoke({
            "caption":          caption,
            "emotion":          emotion,
            "similar_captions": similar_text
        })

        return result.strip() if result else caption

    except Exception as e:
        print(f"Enhancement error: {e}")
        return caption