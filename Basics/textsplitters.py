from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader  = PyPDFLoader("University_Departments_Network_Documentation.pdf")

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 20

)
chunks = text_splitter.split_documents(documents)
print(f"Length of chunks: {len(chunks)}")

print("\n" + "="*60)
print("First Chunk")
print("="*60)
print(chunks[0].page_content)

print("\nMetadata")
print("="*60)
print(chunks[0].metadata)