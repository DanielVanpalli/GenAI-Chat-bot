import streamlit as st
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.prompts import ChatPromptTemplate,BaseChatPromptTemplate
from langchain_community.document_loaders import CSVLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory



import os
from dotenv import load_dotenv
load_dotenv()

#streamlit 
st.title = "Iportal Chat Bot"
session_store = {}
session_id = st.text_input("session_id",value="default_session")
if "stores" not in st.session_state:
    st.session_state.store = {}


groq_api_key = "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"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

loader = CSVLoader("E:\\Downloads\\it_issues_resolutions_400.csv")
document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
splits = text_splitter.split_documents(document)

vector_store = Chroma.from_documents(splits,embeddings)
retriever = vector_store.as_retriever()

contextulize_q_system_prompt = (
    "Given chat history and the latest user question "
    "which might reference context in the chat history "
    "formulate a standalone question which can be understood "
    "without the chat history. Do Not answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextulize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",contextulize_q_system_prompt),
        (MessagesPlaceholder("chat_history")),
        ("human","{input}")
    ]
)

history_aware_retriever = create_history_aware_retriever(llm,retriever,contextulize_q_prompt)

system_promt=(
    "you are an assistant for question-answering tasks. "
    "Use the following pieces of retrived context or answer"
    "the question. if you don't know the answer, say like "
    "please reach out to support team by giving iportal@support.com as email."
    "Use three sentences maximum and keep the answers"
    "concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_promt),
        (MessagesPlaceholder("chat_history")),
        ("human","{input}")
    ]
)

question_ans_chain = create_stuff_documents_chain(llm,qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever,question_ans_chain)



def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

user_input = st.input_text("your question: ")
if user_input:
    session_history = get_session_history(session_id)
    response=conversational_rag_chain.invoke(
        {"input":user_input},
        {"configurable":{"session_id":session_id}}
    )
    st.success("Assistance: ",response["answer"])


