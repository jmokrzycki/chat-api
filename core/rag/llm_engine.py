import json
from typing import AsyncGenerator, List, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.config import OLLAMA_BASE_URL, OLLAMA_API_KEY, MODEL_NAME
from core.state import get_trained_files_list
from core.rag.vector_store import vector_store

llm = ChatOpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY,
    model=MODEL_NAME,
    streaming=True
)

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def get_history_text(chat_history: List[Dict[str, str]]) -> str:
    if not chat_history:
        return ""
    return "\n".join([f"{'UŻYTKOWNIK' if msg['sender'] == 'user' else 'ASYSTENT'}: {msg['text']}" for msg in chat_history])

async def rephrase_question(user_prompt: str, chat_history: List[Dict[str, str]], custom_rephrase: str) -> str:
    if not chat_history:
        return user_prompt

    history_text = get_history_text(chat_history)
    rephrase_str = custom_rephrase.strip() + "\n\nHISTORIA ROZMOWY:\n{chat_history}\n\nNAJNOWSZE PYTANIE:\n{question}\n\nSAMODZIELNE ZAPYTANIE DO BAZY DANYCH:"

    prompt = ChatPromptTemplate.from_template(rephrase_str)
    chain = prompt | llm | StrOutputParser()

    try:
        standalone_q = await chain.ainvoke({"chat_history": history_text, "question": user_prompt})
        return standalone_q.strip().strip('"').strip("'")
    except Exception:
        return user_prompt

async def generate_chat_response(user_prompt: str, custom_template: str, custom_rephrase: str, chat_history: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    history_text = get_history_text(chat_history)
    search_query = await rephrase_question(user_prompt, chat_history, custom_rephrase)

    active_files = get_trained_files_list()

    if not active_files:
        retrieved_docs = []
    else:
        dynamic_retriever = vector_store.as_retriever(
            search_kwargs={"k": 3, "filter": {"filename": {"$in": active_files}}}
        )
        retrieved_docs = await dynamic_retriever.ainvoke(search_query)

    template_str = custom_template.strip()
    if chat_history: template_str += "\n\nHISTORIA ROZMOWY:\n{chat_history}"
    if retrieved_docs:
        context_text = format_docs(retrieved_docs)
        template_str += "\n\nKONTEKST Z DOKUMENTÓW:\n{context}"
    else:
        context_text = ""
    template_str += "\n\nUŻYTKOWNIK:\n{question}\n\nASYSTENT:"

    prompt = ChatPromptTemplate.from_template(template_str)
    chain = prompt | llm | StrOutputParser()

    try:
        async for chunk in chain.astream({"context": context_text, "question": user_prompt, "chat_history": history_text}):
            if chunk: yield json.dumps({"response": chunk}) + "\n"
    except Exception as e:
        yield json.dumps({"response": f"\n[Błąd: {str(e)}]"}) + "\n"
