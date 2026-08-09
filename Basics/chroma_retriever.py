from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
pdf_path = Path(__file__).parent / "University_Departments_Network_Documentation.pdf"
load_dotenv() 

model = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful RAG assistant.

Answer the user's question based ONLY on the context below.

Use the information from the context to formulate a clear,
direct answer.

If the context does not contain enough information to answer
the question, say:
"I could not find the answer in the document."

Context:
--------------------
{context}
--------------------

Question:
{question}

Answer:
"""
)

loader = PyPDFLoader(str(pdf_path))

documents = loader.load()

print(f"Length of documents: {len(documents)}")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Length of chunks: {len(chunks)}")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="University_Networks"
)

print("Vector database created successfully!")

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

query = input("Enter your question: ")

results = retriever.invoke(query)
print("\n" + "=" * 60)
print("RETRIEVED CONTEXT")
print("=" * 60)

for i, document in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(document.page_content)
context = "\n\n".join(
    document.page_content
    for document in results
)
print("\n" + "=" * 60)
print("CONTEXT SENT TO LLM")
print("=" * 60)
print(context)
prompt_value = prompt.invoke(
    {
    "context":context,
    "question":query
    }
)
response = model.invoke(prompt_value)

answer = parser.invoke(response)

print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)
print(answer)