# src/rag/rag_engine.py

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI




SYSTEM_PROMPT = """
You are a real estate assistant.
You must NOT calculate numbers.
You must NOT change financial values.
You must ONLY explain the provided property records.
"""


def generate_rag_response(context_docs: list[str], user_query: str):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    context = "\n\n---\n\n".join(context_docs)

    prompt = f"""
{SYSTEM_PROMPT}

Property Records:
{context}

User Question:
{user_query}

Answer strictly using the above records.
"""

    return llm.invoke(prompt).content
