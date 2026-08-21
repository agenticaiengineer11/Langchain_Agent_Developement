from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Documents
documents = [
    "Artificial Intelligence is a branch of computer science.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Large Language Models are trained on large amounts of text."
]


# Create vector database
vector_store = Chroma.from_texts(
    texts=documents,
    embedding=embeddings,
    collection_name="rag_collection"
)


print("Vector database created successfully!")

query = "What is machine learning?"

results = vector_store.similarity_search(
    query,
    k=2
)

print("\nSearch Results:")

for result in results:
    print(result.page_content)