from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
python is very easy. Python is used very highly in AI and machine learning.

Python is best for learning because it's very easy.

Python becomes more popular all over the world.
"""

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=20
)

chunks = text_splitter.split_text(text)

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)