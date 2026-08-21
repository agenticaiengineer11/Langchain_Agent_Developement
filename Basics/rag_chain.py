from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()



# ============================================================
# 1. Embedding Model
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 2. Documents
# ============================================================

documents = [
    "Artificial Intelligence is a branch of computer science.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Large Language Models are trained on large amounts of text."
]


# ============================================================
# 3. Vector Store
# ============================================================

vector_store = Chroma.from_texts(
    texts=documents,
    embedding=embeddings,
    collection_name="rag_chain_test"
)


# ============================================================
# 4. Retriever
# ============================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)


# ============================================================
# 5. Model
# ============================================================

model = ChatGroq(
    model="openai/gpt-oss-120b"
)


# ============================================================
# 6. Prompt
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
Answer the user's question using only the provided context.

If the answer is not present in the context,
say that you don't know.

Context:
{context}

Question:
{question}
"""
)


# ============================================================
# 7. User Question
# ============================================================

question = input("Ask a question: ")


# ============================================================
# 8. Retrieve Relevant Documents
# ============================================================

docs = retriever.invoke(question)


# ============================================================
# 9. Convert Documents Into Context
# ============================================================

context = "\n\n".join(
    doc.page_content
    for doc in docs
)


# ============================================================
# 10. Create Prompt
# ============================================================

messages = prompt.invoke(
    {
        "context": context,
        "question": question
    }
)


# ============================================================
# 11. Send To LLM
# ============================================================

response = model.invoke(messages)


# ============================================================
# 12. Final Answer
# ============================================================

print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(response.content)