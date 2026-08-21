from pypdf import PdfReader

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.agents import create_agent


# ============================================================
# 1. Load environment variables
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
# 3. Split PDF into chunks
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print("Total chunks:", len(chunks))


# ============================================================
# 4. Create embeddings
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 5. Create vector database
# ============================================================

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="rag_agent"
)


# ============================================================
# 6. Create retriever
# ============================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================================
# 7. Create PDF Search Tool
# ============================================================

@tool
def search_pdf(query: str) -> str:
    """
    Search the PDF for information relevant to the user's question.
    Use this tool whenever the user asks about information contained
    in the PDF.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information was found in the PDF."

    results = []

    for doc in docs:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        results.append(
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {doc.page_content}"
        )

    return "\n\n".join(results)


# ============================================================
# 8. Create GPT-OSS 120B
# ============================================================

model = ChatGroq(
    model="openai/gpt-oss-120b"
)


# ============================================================
# 9. Create Agent
# ============================================================

agent = create_agent(
    model=model,
    tools=[search_pdf],
    system_prompt="""
You are a PDF research agent.

When the user asks about information that may be
contained in the PDF, use the search_pdf tool.

Use ONLY information returned by the PDF tool
for PDF-related questions.

Do not invent information.

If the PDF search does not provide enough information,
say that you could not find the answer in the PDF.

When possible, mention the source page.
"""
)


# ============================================================
# 10. User Question
# ============================================================

query = input("\nAsk a question about the PDF: ")


# ============================================================
# 11. Run Agent
# ============================================================

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    }
)


# ============================================================
# 12. Final Answer
# ============================================================

print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(result["messages"][-1].content)