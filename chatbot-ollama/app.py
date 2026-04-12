from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st 
import os 

import os 
from dotenv import load_dotenv
load_dotenv()

#langchain tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A chatbot using OLLAMA"

#prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are a heplful assistant . please response to the questions."),
        ("user","Question:{question}")
    ]
)

def generate_response(question,engine,temperature,max_tokens):
    llm = Ollama(model=engine)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question':question})
    return answer

#select the ollama Module
engine=st.sidebar.selectbox("select ollama model",["gemma","mistral"])
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("max tokens",min_value=50,max_value=300,value=150)

#main interface
st.write("Ask any question")
user_input = st.text_input("You:")


if user_input:
    response=generate_response(user_input,engine,temperature,max_tokens)
    st.write(response)
else:
    st.write("please provide the question")
