import streamlit as st
import ollama

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

st.title("Enterprise Knowledge RAG Assistant")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="../chroma_db",
    embedding_function=embedding_model
)

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = ""

query = st.text_input("Ask your enterprise question:")

if query:
    enhanced_query = (
        st.session_state.conversation_history + "\n" + query
    )

    results = vector_db.similarity_search(enhanced_query, k=2)
    st.subheader("Retrieved Context")

    context_text = ""

    for i, result in enumerate(results):
        st.write(f"Result {i+1}:")
        st.write(result.page_content)
        st.write("-------------------")

        context_text += result.page_content + "\n"

    prompt = f"""
    You are an enterprise knowledge assistant.

    Answer the user's question using only the retrieved enterprise context below.

    User Question:
    {query}

    Retrieved Enterprise Context:
    {context_text}

    Answer:
    """

    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response["message"]["content"]

    st.subheader("Generated Answer")
    st.write(answer)
    st.session_state.conversation_history += f"""
    User: {query}
    Assistant: {answer}
    """