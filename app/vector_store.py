from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import os

DATA_PATH = "../data"

documents = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".txt"):
        loader = TextLoader(os.path.join(DATA_PATH, file))
        documents.extend(loader.load())

print(f"Loaded {len(documents)} documents")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="../chroma_db"
)

print("Vector database created successfully")

query = "How does partition pruning improve performance?"

results = vector_db.similarity_search(query, k=2)

print("\nQuery Results:\n")

for i, result in enumerate(results):
    print(f"Result {i+1}:")
    print(result.page_content)
    print("\n-------------------\n")