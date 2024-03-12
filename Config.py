import os

class Config:
    openai_key = os.environ.get("OPENAI_KEY")
    pinecone_key = os.environ.get("PINECONE_KEY")
    pinecone_index_name = "ee596-pinecone-index"
    chatty = True
