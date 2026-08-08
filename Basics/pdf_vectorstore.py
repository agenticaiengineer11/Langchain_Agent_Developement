print("================ PDF VECTOR STORE ================")

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


pdf_path = Path(__file__).parent / "University_Departments_Network_Documentation.pdf"

loader = PyPDFLoader(str(pdf_path))

documents = loader.load()

print("Total Documents:", len(documents))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print("Total Chunks:", len(chunks))

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="university_network"
)

print("Vector database created successfully!")


results = vector_store.similarity_search(
    "Who was the project submitted to?",
    k=5
)

for i, chunk in enumerate(chunks, start=1):
    if "Mam Kinza" in chunk.page_content:
        print("=" * 60)
        print("FOUND IN CHUNK:", i)
        print("=" * 60)
        print(chunk.page_content)