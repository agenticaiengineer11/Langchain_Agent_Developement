from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 1. Embedding model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Create vector store

documents = [
    "Artificial Intelligence is a branch of computer science.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Large Language Models are trained on large amounts of text."
]


vector_store = Chroma.from_texts(
    texts=documents,
    embedding=embeddings,
    collection_name="retriever_test"
)


# 3. Create retriever

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)


# 4. User query

query = input("Ask a question: ")


# 5. Retrieve relevant documents

results = retriever.invoke(query)


# 6. Display results

print("\n" + "=" * 60)
print("RETRIEVED DOCUMENTS")
print("=" * 60)

for i, result in enumerate(results, start=1):

    print(f"\nDOCUMENT {i}")
    print("-" * 60)
    print(result.page_content)