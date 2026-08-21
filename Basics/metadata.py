from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# 1. Load PDF
# ============================================================

PDF_FILE = "Basics/University_Departments_Network_Documentation.pdf"

reader = PdfReader(PDF_FILE)

documents = []


# ============================================================
# 2. Extract each page separately
# ============================================================

for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text() or ""

    if text.strip():

        document = Document(
            page_content=text,
            metadata={
                "source": PDF_FILE,
                "page": page_number
            }
        )

        documents.append(document)


print("Total pages with text:", len(documents))


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
# 4. Show metadata
# ============================================================

for i, chunk in enumerate(chunks[:5], start=1):

    print("\n" + "=" * 60)
    print(f"CHUNK {i}")
    print("=" * 60)

    print("TEXT:")
    print(chunk.page_content[:300])

    print("\nMETADATA:")
    print(chunk.metadata)


# ============================================================
# 5. Embeddings
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 6. Vector Store
# ============================================================

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="metadata_test"
)


# ============================================================
# 7. Retriever
# ============================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================================
# 8. Search
# ============================================================

query = input("\nAsk a question about the PDF: ")

results = retriever.invoke(query)


# ============================================================
# 9. Display results + sources
# ============================================================

print("\n" + "=" * 60)
print("RETRIEVED RESULTS")
print("=" * 60)


for i, result in enumerate(results, start=1):

    print(f"\nRESULT {i}")
    print("-" * 60)

    print("TEXT:")
    print(result.page_content)

    print("\nSOURCE:")
    print(result.metadata["source"])

    print("PAGE:")
    print(result.metadata["page"])