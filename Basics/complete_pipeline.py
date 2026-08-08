print("================ RAG Document Pipeline ================")

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. PDF path
pdf_path = Path(__file__).parent / "University_Departments_Network_Documentation.pdf"

# 2. Load PDF
loader = PyPDFLoader(str(pdf_path))
documents = loader.load()

print("Total Documents:", len(documents))

# 3. Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# 4. Split documents into chunks
chunks = text_splitter.split_documents(documents)

print("Total Chunks:", len(chunks))

# 5. Inspect first chunk
print("\n" + "=" * 60)
print("FIRST CHUNK")
print("=" * 60)

print(chunks[0].page_content)

# 6. Inspect metadata
print("\n" + "=" * 60)
print("METADATA")
print("=" * 60)

print(chunks[0].metadata)

# 7. Inspect types
print("\n" + "=" * 60)
print("TYPES")
print("=" * 60)

print("Documents:", type(documents))
print("Chunks:", type(chunks))
print("First Chunk:", type(chunks[0]))