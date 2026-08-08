from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

texts = [
    "Python is used for artificial intelligence.",
    "Python is used for web development.",
    "Python is popular for automation."
]

vectors = embedding_model.embed_documents(texts)

print(f"Number of vectors: {len(vectors)}")
print(f"Length of each vector: {len(vectors[0])}")
print(f"Type of vectors: {type(vectors)}")

print("First 5 values of first vector:", vectors[0][:5])