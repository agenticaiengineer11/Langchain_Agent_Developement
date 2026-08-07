from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("employees.csv")

documents = loader.load()

full_csv = "\n\n".join(
    document.page_content for document in documents
)

print(full_csv)
print(type(full_csv))