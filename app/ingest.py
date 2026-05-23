from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

for i, chunk in enumerate(chunks[:5]):
    print("\n------------------")
    print(f"Chunk {i+1}")
    print(chunk.page_content)