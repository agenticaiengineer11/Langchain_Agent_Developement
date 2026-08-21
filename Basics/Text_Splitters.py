from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load PDF
reader = PdfReader("Basics/Network Packets and Frames.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text() or ""


# Create splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# Split PDF text
chunks = splitter.split_text(text)


# Show results
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks[:5]):

    print("\n" + "=" * 60)
    print(f"CHUNK {i + 1}")
    print("=" * 60)

    print(chunk)