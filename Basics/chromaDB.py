from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

texts = [
    "Python is used for artificial intelligence.",
    "Python is used for web development.",
    "Python is popular for automation."
]


vector_store = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    collection_name="Python_Notes"
)

results = vector_store.similarity_search_with_score(
    "How is Python used in AI?",
    k=3
)

for document, score in results:
    print("=" * 60)
    print("Score:", score)
    print("Content:", document.page_content)

print("Vector database created successfully")
