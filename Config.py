import streamlit as st

class Config:
    openai_key = st.secrets["OPENAI_KEY"]
    pinecone_key = st.secrets["PINECONE_KEY"]
    pinecone_index_name = "ee596-pinecone-index"
    chatty = True
