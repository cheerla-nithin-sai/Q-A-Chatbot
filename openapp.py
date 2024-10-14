# Q n A Chatbot using OpenSource Models

from langchain.llms import Ollama # open source llm
from langchain_core.prompts import ChatPromptTemplate # to write prompt for llm
from langchain_core.output_parsers import StrOutputParser # to parse the output
import streamlit as st # web frame work

# to load envi variables
import os
from dotenv import load_dotenv
load_dotenv()

# setting langchain variables
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q/A using Open Source OLLAMA model"

# prompt for llm
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are helpful chat assistant, please respond to user queries"), 
        ("user","ask any question:{question}")
    ])

# function to generate response
def generate_response(question,llm):
    llm = Ollama(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question":question})
    return answer

# using streamlit

st.title("Q&A Chatbot with OpenAI")

# side bar for settings
st.sidebar.title("Settings")

# drop down to select diff llm models
llm = st.sidebar.selectbox("Select an Open AI model",["mistral","gemma2:2b"])

# main interface
st.write("Ask any question")
user_input = st.text_input("you:")

if user_input:
    response = generate_response(user_input,llm)
    st.write(response)

else:
    st.write("Please provide the user input")

