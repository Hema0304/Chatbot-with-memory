import streamlit as st 
import os 

from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()



groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

prompt=ChatPromptTemplate.from_template(
    """ 
    Answer the questions based on the provided context only.
    please provide the most accurate response based on the questions.
    <context>
    {content}
    <context>
    Question:{input}
    """
)

def ctreate_vectore_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("research_papers")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        print("Final documents:", st.session_state.final_documents)
        print("Length of documents:", len(st.session_state.final_documents))
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
        
user_prompt = st.text_input("enter your query from the document")

if st.button("Document Embedding"):
    ctreate_vectore_embedding()
    st.write("vector DAtabase is ready")
    
import time

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retriever_chain=create_retrieval_chain(retriever,document_chain)
    
    start=time.process_time()
    response=retriever_chain.invoke({'input':user_prompt})
    print(f"REspose time:{time.process_time()-start}")
    st.write(response['answer'])
    
    with st.expander("document similarity search"):
        for i,doc in enumerate(response['content']):
            st.write(doc.page_content)
            st.write('------------------------')
    
        
        
