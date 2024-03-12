from multiprocessing import AuthenticationError

from langchain import requests
from pinecone import Pinecone

from Config import Config
import streamlit as st
from openai import OpenAI, OpenAIError
from Agents import Head_Agent

st.title("Mini Project 2: Streamlit Chatbot")

if Config.openai_key:
    openai_key = Config.openai_key
else:
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if not openai_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

if Config.pinecone_key:
    pinecone_key = Config.pinecone_key
else:
    pinecone_key = st.sidebar.text_input("Pinecone API Key", type="password")
    if not pinecone_key:
        st.info("Please add your Pinecone API key to continue.")
        st.stop()

try:
    client = OpenAI(api_key=openai_key)
    message = {"role": "user", "content": "ping"}
    client.chat.completions.create(model="gpt-3.5-turbo", messages=[message])
    Head_Agent(openai_key, pinecone_key, Config.pinecone_index_name)
except AuthenticationError:
    st.error("Failed to authenticate with OpenAI. Please check your API key.")
    st.stop()
except OpenAIError as e:
    st.error(f"An error occurred while trying to communicate with OpenAI: {e}")
    st.stop()

try:
    Pinecone(api_key=pinecone_key)
except requests.exceptions.HTTPError as e:
    st.error(f"Failed to authenticate with Pinecone or communicate properly: {e}")
    st.stop()

run = Head_Agent(openai_key, pinecone_key, Config.pinecone_index_name)
run.main_loop()


