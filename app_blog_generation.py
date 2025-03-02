import os
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import TypedDict, Optional
from langgraph.checkpoint.memory import MemorySaver
import re
import urllib.parse


load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Youtube_Blog_Generation")
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Enable LangSmith tracing


llm = ChatGroq(model="qwen-2.5-32b", temperature=0.7)

# ---------------------- #
# Define Blog State for LangGraph
# ---------------------- #
class BlogState(TypedDict):
    transcript: Optional[str]
    blog_draft: Optional[str]
    llm_review: Optional[str]
    feedback: Optional[str]
    final_blog: Optional[str]

from urllib.parse import urlparse, parse_qs

def get_video_id(video_url):
    query = urlparse(video_url)
    if "youtube.com" in query.netloc:
        return parse_qs(query.query).get("v", [None])[0]
    elif "youtu.be" in query.netloc:
        return query.path.lstrip("/")
    return None


# ---------------------- #
# Extract Transcript from YouTube
# ---------------------- #
def extract_transcript(state: BlogState) -> BlogState:
    video_url = st.session_state.get("video_url", "")

    if not video_url:
        st.error("No YouTube URL provided!")
        return state

    video_id = get_video_id(video_url)

    if not video_id:
        st.error("Invalid YouTube URL!")
        return state

    try:
        # Fetch full transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = " ".join(entry["text"] for entry in transcript_list)

        # Split into chunks to manage API limits
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        transcript_chunks = text_splitter.split_text(full_transcript)

        # Store full transcript separately
        state["full_transcript"] = full_transcript  

        # Use limited chunks to stay within API token limits
        state["transcript"] = " ".join(transcript_chunks[:3])  # Adjust as needed

        st.success(" Transcript Extracted Successfully!")

    except Exception as e:
        state["transcript"] = None
        state["full_transcript"] = None  # Ensure data is reset if an error occurs
        st.error(f" Error Extracting Transcript: {str(e)}")

    return state


# ---------------------- #
#  Generate Blog from Transcript
# ---------------------- #
def generate_blog(state: BlogState) -> BlogState:
    transcript = state.get("transcript", "")

    if not transcript:
        st.error(" No transcript available! Ensure the video has captions.")
        return state

    prompt = f"""
    Generate a structured blog based on the following transcript:
    {transcript}

    Structure:
    1. **Title**: A compelling blog title
    2. **Introduction**: A brief introduction
    3. **Headings & Subheadings**
    4. **Conclusion**: A strong closing statement

    Keep the response concise, and do not exceed 6000 tokens.
    """

    state["blog_draft"] = llm.invoke(prompt).content
    return state

# ---------------------- #
#  LLM Review of Blog
# ---------------------- #
def llm_blog_review(state: BlogState) -> BlogState:
    draft = state.get("blog_draft", "")

    prompt = f"Review this blog and suggest improvements:\n{draft}"
    state["llm_review"] = llm.invoke(prompt).content
    return state

# ---------------------- #
#  Human Feedback on Blog
# ---------------------- #
def human_feedback(state: BlogState) -> BlogState:
    st.subheader("🔍 Blog Draft Review")
    st.text_area("Generated Blog", value=state["blog_draft"], height=300)
    st.text_area("LLM Review Feedback", value=state["llm_review"], height=200)

    feedback = st.text_area("Provide your feedback (Optional):")
    state["feedback"] = feedback if feedback else "No feedback provided."
    
    return state

# ---------------------- #
#  Refine Blog with AI & Human Feedback
# ---------------------- #
def refine_blog(state: BlogState) -> BlogState:
    review = state.get("llm_review", "No AI review provided.")
    feedback = state.get("feedback", "No human feedback provided.")
    draft = state.get("blog_draft", "")

    if not draft:
        st.error("❌ No blog draft available to refine!")
        return state

    prompt = f"Revise the blog based on the following:\n- AI Review: {review}\n- Human Feedback: {feedback}\n\nOriginal Blog:\n{draft}"
    state["final_blog"] = llm.invoke(prompt).content

    st.success(" Blog Finalized Successfully!")
    return state

# ---------------------- #
#  Define LangGraph Workflow
# ---------------------- #
builder = StateGraph(BlogState)
builder.add_node("extract_transcript", extract_transcript)
builder.add_node("generate_blog", generate_blog)
builder.add_node("llm_blog_review", llm_blog_review)
builder.add_node("human_feedback", human_feedback)
builder.add_node("refine_blog", refine_blog)

# Define Edges
builder.add_edge(START, "extract_transcript")
builder.add_edge("extract_transcript", "generate_blog")
builder.add_edge("generate_blog", "llm_blog_review")
builder.add_edge("llm_blog_review", "human_feedback")
builder.add_edge("human_feedback", "refine_blog")
builder.add_edge("refine_blog", END)

# Compile Graph
graph = builder.compile()

# ---------------------- #
#  Streamlit UI
# ---------------------- #
st.title("🎥 YouTube Video to Blog Generator")
st.write("🚀 Convert any YouTube video transcript into a structured blog post.")

video_url = st.text_input("🔗 Enter YouTube Video URL:")
if video_url:
    st.session_state["video_url"] = video_url

    if st.button("Generate Blog"):
        initial_state: BlogState = {
            "transcript": None,
            "blog_draft": None,
            "llm_review": None,
            "feedback": None,
            "final_blog": None
        }
        
        #  Run LangGraph Workflow
        final_state = graph.invoke(initial_state)

        #  Display Final Blog
        st.subheader("📌 Final Blog Output")
        st.text_area("Final Blog", value=final_state["final_blog"], height=400)

        #  Save to File
        with open("final_blog_output.txt", "w", encoding="utf-8") as f:
            f.write(final_state["final_blog"])
        st.success(" Blog saved to final_blog_output.txt!")
