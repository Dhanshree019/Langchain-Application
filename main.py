import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')

client = ChatNVIDIA(
    model="meta/llama-3.1-405b-instruct",
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to user queries."),
        ("user", "Question: {question}")
    ]
)

st.title("Welcome, Ask me anything 🤖")
input_text = st.text_input("Ask Question")


if input_text:
    formatted_prompt = prompt.format_messages(question=input_text)
    response = client.invoke(formatted_prompt)
    st.write(response.content) 
