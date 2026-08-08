from langchain_text_splitters import CharacterTextSplitter

text = """
python is very easy. Python is used very highly in AI and machine learning.

Python is best for learning because it's very easy.

Python becomes more popular all over the world.
"""

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=50,
    chunk_overlap=0
)

chunks = text_splitter.split_text(text)

print(f"Length of chunks: {len(chunks)}")

for i, chunk in enumerate(chunks, start=1):
    print("=" * 50)
    print(f"Chunk {i}")
    print("=" * 50)
    print(chunk)

print("\nType of chunks:", type(chunks))
print("Type of first chunk:", type(chunks[0]))