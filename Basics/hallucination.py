from pypdf import PdfReader

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# 1. Load environment
# ============================================================

load_dotenv()


# ============================================================
# 2. Load PDF
# ============================================================

PDF_FILE = "Basics/University_Departments_Network_Documentation.pdf"

reader = PdfReader(PDF_FILE)

documents = []

for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text() or ""

    if text.strip():

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": PDF_FILE,
                    "page": page_number
                }
            )
        )


if not documents:
    raise ValueError(
        "No text could be extracted from the PDF."
    )


# ============================================================
# 3. Split documents
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print("Total chunks:", len(chunks))


# ============================================================
# 4. Embeddings
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 5. Vector Store
# ============================================================

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="hallucination_test"
)


# ============================================================
# 6. Retriever
# ============================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================================
# 7. GPT-OSS 120B
# ============================================================

model = ChatGroq(
    model="openai/gpt-oss-120b"
)


# ============================================================
# 8. Strict RAG Prompt
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a strict PDF question-answering assistant.

You MUST follow these rules:

1. Use ONLY the information provided in the context.
2. Do NOT use your own knowledge.
3. Do NOT guess or assume.
4. If the answer is not clearly present in the context,
   say exactly:

"I don't know based on this PDF."

5. Keep the answer concise.

Context:
{context}

Question:
{question}

Answer:
"""
)


# ============================================================
# 9. User Question
# ============================================================

question = input(
    "\nAsk a question about the PDF: "
)


# ============================================================
# 10. Retrieve documents
# ============================================================

docs = retriever.invoke(question)


# ============================================================
# 11. Build context
# ============================================================

context = "\n\n".join(
    doc.page_content
    for doc in docs
)


# ============================================================
# 12. Send context + question to LLM
# ============================================================

messages = prompt.invoke(
    {
        "context": context,
        "question": question
    }
)

response = model.invoke(messages)


# ============================================================
# 13. Final Answer
# ============================================================

print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(response.content)


# ============================================================
# 14. Sources
# ============================================================

print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)

seen_sources = set()

for doc in docs:

    source = (
        doc.metadata.get("source"),
        doc.metadata.get("page")
    )

    if source not in seen_sources:

        print(
            f"Source: {source[0]} | "
            f"Page: {source[1]}"
        )

        seen_sources.add(source)