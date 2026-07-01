import streamlit as st
from rag_app import ask_question

st.set_page_config(
    page_title="Document Intelligence System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Document Intelligence System")
st.markdown(
    "Ask questions about your PDF documents using Retrieval-Augmented Generation (RAG)."
)

question = st.text_input(
    "Ask a question about your documents"
)

if st.button("Search"):

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching documents..."):

        answer, docs = ask_question(question)

    st.subheader("Answer")
    st.write(answer)

# No sources shown
st.caption("Powered by Retrieval-Augmented Generation (RAG)")