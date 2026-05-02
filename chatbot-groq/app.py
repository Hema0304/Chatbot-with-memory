import streamlit as st
import os
import time

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

# ---------------- LLM ----------------
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# ---------------- PROMPT ----------------
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.

<context>
{context}
</context>

Question: {input}
""")

# ---------------- VECTOR DB FUNCTION ----------------
def create_vector_embedding():
    if "vectors" not in st.session_state:

        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        loader = PyPDFDirectoryLoader("research_papers")
        docs = loader.load()

        if not docs:
            st.error("No PDF files found in 'research_papers' folder.")
            return

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        final_docs = splitter.split_documents(docs)

        if not final_docs:
            st.error("Failed to split documents.")
            return

        st.session_state.vectors = FAISS.from_documents(
            final_docs,
            st.session_state.embeddings
        )

        st.success("Vector Database Created Successfully!")

# ---------------- UI ----------------
st.title("Context-Aware Conversational Agent with Memory")

# Auto create vector DB (safe approach)
if "vectors" not in st.session_state:
    if st.button("Create Vector Database"):
        create_vector_embedding()

# Query input
user_prompt = st.text_input("Ask a question from your PDF")

# ---------------- CHAT LOGIC ----------------
if user_prompt:

    if "vectors" not in st.session_state:
        st.warning("Please create vector database first!")
        st.stop()

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = st.session_state.vectors.as_retriever()

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    start = time.process_time()

    response = retrieval_chain.invoke({"input": user_prompt})

    st.write("### Answer:")
    st.write(response["answer"])

    st.write(f"⏱ Response time: {time.process_time() - start:.2f}s")

    # ---------------- SOURCE DOCS ----------------
    with st.expander(" Source Documents"):
        for doc in response["context"]:
            st.write(doc.page_content)
            st.write("---")










# import streamlit as st 
# import os 

# from langchain_groq import ChatGroq
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain

# from langchain_core.prompts import ChatPromptTemplate

# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader

# from dotenv import load_dotenv
# load_dotenv()



# groq_api_key = os.getenv("GROQ_API_KEY")

# llm = ChatGroq(
#     groq_api_key=groq_api_key,
#     model_name="llama-3.1-8b-instant"
# )

# prompt=ChatPromptTemplate.from_template(
#     """ 
#     Answer the questions based on the provided context only.
#     please provide the most accurate response based on the questions.
#     <context>
#     {content}
#     <context>
#     Question:{input}
#     """
# )

# def ctreate_vectore_embedding():
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings=OllamaEmbeddings()
#         st.session_state.loader=PyPDFDirectoryLoader("research_papers")
#         st.session_state.docs=st.session_state.loader.load()
#         st.session_state.text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
#         st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
#         print("Final documents:", st.session_state.final_documents)
#         print("Length of documents:", len(st.session_state.final_documents))
#         st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
        
# user_prompt = st.text_input("enter your query from the document")

# if st.button("Document Embedding"):
#     ctreate_vectore_embedding()
#     st.write("vector DAtabase is ready")
    
# import time

# if user_prompt:
#     document_chain=create_stuff_documents_chain(llm,prompt)
#     retriever=st.session_state.vectors.as_retriever()
#     retriever_chain=create_retrieval_chain(retriever,document_chain)
    
#     start=time.process_time()
#     response=retriever_chain.invoke({'input':user_prompt})
#     print(f"REspose time:{time.process_time()-start}")
#     st.write(response['answer'])
    
#     with st.expander("document similarity search"):
#         for i,doc in enumerate(response['content']):
#             st.write(doc.page_content)
#             st.write('------------------------')
    
        
        
