from langchain_groq import ChatGroq
from dotenv import load_dotenv
from vector_db import faiss_db
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
load_dotenv()
groq_api = st.secrets["GROQ_API_KEY"]



llm_model = ChatGroq(model="qwen/qwen3-32b")

def retrive_docs(query):
    return faiss_db.similarity_search(query)

def get_contex(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know—don't try to make up an answer.
Don't provide anything out of the given context.

Question: {question}
Context: {context}
Answer:
"""

def answer_query(documents, model, query):
    context = get_contex(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    result = chain.invoke({"question" : query, "context" : context})
    return result


"""question = "If a government forbids the right to assemble peacefully, which articles are violated and why?"
retrieved_docs = retrive_docs(question)
print("Answer:", answer_query(documents=retrieved_docs, model=llm_model, query=question))"""
