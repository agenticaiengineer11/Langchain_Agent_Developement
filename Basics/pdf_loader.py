from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader( r"C:\Users\FINE LAPTOP\Desktop\Langchain Models\Basics\University_Departments_Network_Documentation.pdf")

documents = loader.load()

full_pdf_text = "\n\n".join(
    document.page_content for document in documents
)

print("=" * 60)
print("Complete PDF")
print("=" * 60)
print(full_pdf_text)

print("\nType:", type(full_pdf_text))