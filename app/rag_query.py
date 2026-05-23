from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import ollama

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="../chroma_db",
    embedding_function=embedding_model
)

conversation_history = ""

while True:
    query = input("\nAsk your question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    enhanced_query = conversation_history + "\n" + query

    print("\nEnhanced Retrieval Query:\n")
    print(enhanced_query)

    results = vector_db.similarity_search(enhanced_query, k=2)
    print("\nRetrieved Context:\n")

    context_text = ""

    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(result.page_content)
        print("\n-------------------\n")

        context_text += result.page_content + "\n"

    print("Final Context Sent To LLM:\n")
    print(context_text)

    prompt = f"""
    You are an enterprise knowledge assistant.

    Previous Conversation:
    {conversation_history}

    Answer the user's current question using:
    1. Previous conversation context
    2. Retrieved enterprise context

    Current User Question:
    {query}

    Retrieved Enterprise Context:
    {context_text}

    Answer:
    """

    print("\nGenerated Answer:\n")

    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    print(response["message"]["content"])

    conversation_history += f"""
    User: {query}
    Assistant: {response["message"]["content"]}
    """