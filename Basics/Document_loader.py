print("=================RAG started here from document loader==========")
from langchain_community.document_loaders import TextLoader

loader = TextLoader(r"C:\Users\FINE LAPTOP\Desktop\Langchain Models\Basics\python_notes.txt")

documents = loader.load()

print("Type ", type(documents))
print("Total documents: ", len(documents))

print("\nPage Content")
print("=" * 50)
print(documents[0].page_content)

print("\nMetadata")
print("=" * 50)
print(documents[0].metadata)