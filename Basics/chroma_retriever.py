from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# ============================================================
# 1. Load Environment Variables
# ============================================================

load_dotenv()


# ============================================================
# 2. PDF Path
# ============================================================

pdf_path = (
    Path(__file__).parent
    / "University_Departments_Network_Documentation.pdf"
)


# ============================================================
# 3. Create LLM
# ============================================================

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

parser = StrOutputParser()


# ============================================================
# 4. Create Prompt
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful RAG assistant.

Answer the user's question based ONLY on the context below.

Use the information from the context to formulate a clear,
direct answer.

If the context does not contain enough information to answer
the question, say:

"I could not find the answer in the document."

Context:
--------------------
{context}
--------------------

Question:
{question}

Answer:
"""
)


# ============================================================
# 5. Load PDF
# ============================================================

loader = PyPDFLoader(str(pdf_path))

documents = loader.load()

print(f"Length of documents: {len(documents)}")


# ============================================================
# 6. Split Documents
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Length of chunks: {len(chunks)}")


# ============================================================
# 7. Create Embedding Model
# ============================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 8. Create Vector Store
# ============================================================

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="University_Networks"
)

print("Vector database created successfully!")


# ============================================================
# 9. Create Retriever
# ============================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================================
# 10. Format Documents
# ============================================================

def format_docs(documents):
    return "\n\n".join(
        document.page_content
        for document in documents
    )


# ============================================================
# 11. Create Complete RAG Chain
# ============================================================

rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough()
    )
    | prompt
    | model
    | parser
)


# ============================================================
# 12. Get User Question
# ============================================================

query = input("Enter your question: ")


# ============================================================
# 13. Run Complete RAG Chain
# ============================================================

result = rag_chain.invoke(query)


# ============================================================
# 14. Display Final Answer
# ============================================================

print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(result)