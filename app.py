"""
app.py - Streamlit frontend
Root-level frontend that calls backend.Summarizing_Agent.
"""

import streamlit as st
from backend.backend import SummarizerAgent
from models.llm_call import llm_call


# ====================================================================== #
# INIT BACKEND
# ====================================================================== #

agent = SummarizerAgent(llm_call)


# ====================================================================== #
# UI
# ====================================================================== #

st.set_page_config(page_title="Summarizer", page_icon="📝", layout="centered")
st.title("📝 Text Summarizer")

uploaded_file = st.file_uploader("Upload a file (optional)", type=["txt", "pdf", "docx"])
text_input = st.text_area("Or paste text directly", height=200, placeholder="Paste your text here...")

if st.button("📝 Summarize", type="primary", use_container_width=True):
    try:
        if uploaded_file is not None:
            result = agent.process(uploaded_file.getvalue(), uploaded_file.name)
            st.info(f"Loaded from: {uploaded_file.name}")
        elif text_input.strip():
            result = agent.process(text_input.strip())
        else:
            st.warning("Please upload a file or paste some text.")
            st.stop()

        st.write(f"**Input:** {result.input_word_count} words")

        st.markdown("---")
        st.subheader("Summary")
        st.write(result.summary)

        st.caption(
            f"Output: {result.output_word_count} words | "
            f"Compression: {result.compression_ratio:.1%} | "
            f"Time: {result.processing_time}s"
        )

    except Exception as e:
        st.error(f"Error: {str(e)}")