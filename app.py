
import openai
from langchain_openai import ChatOpenAI # openai llm model
from langchain_core.prompts import ChatPromptTemplate # promptr for llm model
from langchain_core.output_parsers import StrOutputParser # to parse output in required format

import streamlit as st # webframework

# loading environment variables
import os
from dotenv import load_dotenv
load_dotenv()

# setting up envi variables
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"]="true" # to track langsmith
os.environ["LANGCHAIN_PROJECT"]="Q&A chatbot using openai"
api_key = os.getenv("OPENAI_API_KEY")

# setting up prompt for llm (standard template for llm model)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are helpful chat assistant, please respond to user queries"), 
        ("user","ask any question:{question}")
    ])

# function to generate response
def generate_response(question,api_key,llm,temperature,max_tokens):
    openai.api_key=api_key
    llm = ChatOpenAI(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question":question})
    return answer

# using streamlit
st.title("Q&A Chatbot with OpenAI")

# side bar for settings
st.sidebar.title("Settings")

# drop down to select diff llm models
llm = st.sidebar.selectbox("Select an Open AI model",["gpt-4o","gpt-4-turbo","gpt-4"])

# to adjust response parameters
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.8) # temp is a param for creativity
max_tokens=st.sidebar.slider("Max Tokens",min_value=70,max_value=300,value=99)

# main interface
st.write("Ask any question")
user_input = st.text_input("you:")

if user_input:
    response = generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)