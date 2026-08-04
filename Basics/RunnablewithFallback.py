print("=========Runnable With Fall back===========")
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()
python_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system","You are an expert Python Teacher. Answer ONLY Python-related questions. "
            "If the question is not about Python, refuse by saying "
            "'I cannot answer this question.'"
        ),
        (
            "human","{question}"
        )
    ]
)
python_chain = python_prompt | model | parser
fastapi_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system"," You are a FastAPI Expert. Answer ONLY FastAPI questions. "
            "If the question is unrelated, refuse by saying "
            "'I cannot answer this question.;"
            
        ),
        (
            "human","{question}"
        )
    ]
)
fastapi_chain = fastapi_prompt | model | parser
general_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system","You are AI assistant.Answer every question clearly."
        ),
        (
            "human","{question}"
        )
    ]
)
general_chain = general_prompt | model | parser
final_chain = python_chain.with_fallbacks(
    [
        fastapi_chain,
        general_chain
    ]
)
question = input("Enter your question")
response = final_chain.invoke(
    {
        "question": question
    }

)
print("="*50)
print("======Model Response=====")
print("="*50)
print(response)