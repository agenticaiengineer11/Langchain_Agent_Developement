from langchain_text_splitters import TokenTextSplitter

text ="""
Python is a powerful programming language.
It is widely used in artificial intelligence,
machine learning, automation, web development,
and data science.
"""

splitter = TokenTextSplitter(
    chunk_size = 50,
    chunk_overlap =20
)
chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print("=" * 50)
    print(f"Chunk {i}")
    print("=" * 50)
    print(chunk)
print("Total chunks:", len(chunks))
print("Type:", type(chunks[0]))