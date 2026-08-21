import os

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()


# ============================================================
# 2. Load PDF
# ============================================================

PDF_FILE = "Basics/University_Departments_Network_Documentation.pdf"

reader = PdfReader(PDF_FILE)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"


# ============================================================
# 3. Check extracted text
# ============================================================

print("Extracted text length:", len(text))

if not text.strip():
    raise ValueError(
        "No text could be extracted from the PDF. "
        "Make sure the PDF contains selectable text. "
        "If it is a scanned/image PDF, OCR will be required."
    )


print("\nFirst 500 characters:")
print(text[:500])


# ============================================================
# 4. Split text into chunks
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)


print("\nTotal chunks:", len(chunks))

if not chunks:
    raise ValueError("Text splitting produced zero chunks.")


# ============================================================
# 5. Create embedding model
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 6. Create Chroma vector database
# ============================================================

vector_store = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    collection_name="pdf_rag"
)


print("\nVector database created successfully.")


# ============================================================
# 7. Create retriever
# ============================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================================
# 8. Create GPT-OSS 120B
# ============================================================

model = ChatGroq(
    model="openai/gpt-oss-120b"
)


# ============================================================
# 9. Create prompt
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the information
provided in the context.

If the answer cannot be found in the context,
say:

"I don't know based on this PDF."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""
)


# ============================================================
# 10. Ask user
# ============================================================

question = input("\nAsk a question about the PDF: ")


# ============================================================
# 11. Retrieve relevant chunks
# ============================================================

docs = retriever.invoke(question)


print("\nRetrieved chunks:", len(docs))


# ============================================================
# 12. Create context
# ============================================================

context = "\n\n".join(
    doc.page_content
    for doc in docs
)


# ============================================================
# 13. Create prompt
# ============================================================

messages = prompt.invoke(
    {
        "context": context,
        "question": question
    }
)


# ============================================================
# 14. Ask GPT-OSS 120B
# ============================================================

response = model.invoke(messages)


# ============================================================
# 15. Final answer
# ============================================================

print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(response.content)