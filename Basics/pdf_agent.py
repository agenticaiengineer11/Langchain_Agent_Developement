from pypdf import PdfReader

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent


load_dotenv()


reader = PdfReader("Basics/University_Departments_Network_Documentation.pdf")

pdf_text = ""

for page in reader.pages:
    pdf_text += page.extract_text() or ""


model = ChatGroq(
    model="openai/gpt-oss-120b"
)

@tool
def search_pdf(question: str) -> str:
    """
    Answer questions using the content extracted from the PDF.
    """

    return pdf_text

agent = create_agent(
    model=model,
    tools=[search_pdf]
)


query = input("Ask a question about the PDF: ")

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    }
)

print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(result["messages"][-1].content)