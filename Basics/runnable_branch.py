print("=============Runnable Branch================")
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()
python_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are professional python teacher"),
        ("human", "{question}")
    ]
)
python_chain = python_prompt | model | parser
general_prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are professional AI assistant"),
        ("human", "{question}")
    ]
)
general_chain = general_prompt | model | parser 
fastapi_prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are professional FastAPI expert"),
        ("human","{question}")
    ]
)
fastapi_chain = fastapi_prompt | model | parser
langchain_prompt  = ChatPromptTemplate.from_messages(
    [
        ("system","You are professional Langchain Agent developer"),
        ("human","{question}")
    ]
)
langchain_chain = langchain_prompt | model | parser
router = RunnableBranch(
    (lambda x:"python" in x["question"].lower(),
     python_chain
     ),
    (
    lambda x:"fastapi" in x["question"].lower(),
    fastapi_chain
    ),
    (
    lambda x:"langchain" in x["question"].lower(),
    langchain_chain
    ),
     
         general_chain
)
question = input("Enter you question: ")
response = router.invoke(
    {
        "question": question
    }
)
print(response)