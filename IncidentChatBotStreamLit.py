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
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
import pdb



import os
from dotenv import load_dotenv
load_dotenv()

#streamlit 
st.title("Iportal Chat Bot")

session_id = "default_session"
if "stores" not in st.session_state:
    st.session_state.store = {}

upload_file = st.file_uploader("Choose a CSV file to upload",type="csv",accept_multiple_files=True)
documents = []
if upload_file:
    for uf in upload_file:
        temcsv=f"./temp.csv"
        with open(temcsv,"wb") as file:
            file.write(uf.getvalue())
            file_name=uf.name
        loader = CSVLoader(temcsv)
        docs = loader.load()
        documents.extend(docs)


    groq_api_key = "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"
    huggingFace_api_key = "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"
    os.environ['HF_TOKEN'] = huggingFace_api_key
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
    splits = text_splitter.split_documents(documents)

    vector_store = FAISS.from_documents(splits,embeddings)
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
        "the question. if you don't know the answer, say that you"
        "don't know. Use three sentences maximum and keep the answers"
        "concise."
        "if you don't know the answer, please share the following contacts"
        "iportal@support.com to reach out to support team."
        "Use Four sentences maximum and keep the answers"
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
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    st.write("ask anything from the uploaded file. ")
    user_input = st.text_input("your question: ")
    if user_input:
        session_history = get_session_history(session_id)
        response=conversational_rag_chain.invoke(
            {"input":user_input},
            {"configurable":{"session_id":session_id}}
        )
        st.write("Assistant: ")
        st.write(response['answer'])
        st.write("chat history: ")
        st.write(get_session_history(session_id).messages)
        

