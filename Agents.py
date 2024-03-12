from Config import Config
from openai import OpenAI
import streamlit as st
from textblob import TextBlob
from langchain.vectorstores import Pinecone
from pinecone import Pinecone

class Obnoxious_Agent:
    def __init__(self, client) -> None:
        self.client = client
        self.prompt = ""

    def set_prompt(self, prompt):
        self.prompt = f"Would you describe the tone of this prompt as 'rude', 'polite', or 'neutral'?: '{prompt}'"

    def extract_action(self, response) -> bool:
        out = 'rude' in response.choices[0].message.content.lower().split()
        return out

    def check_query(self, query):
        self.set_prompt(query)
        prompt = self.prompt
        message = {"role": "user", "content": prompt}
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[message]
        )
        return self.extract_action(response)

class Query_Agent:
    def __init__(self, pinecone_index, openai_client, embeddings) -> None:
        self.pinecone_index = pinecone_index
        self.openai_client = openai_client
        self.embeddings = embeddings
        self.prompt = ""

    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return self.openai_client.embeddings.create(input=[text], model=model).data[0].embedding

    def query_vector_store(self, query, k=5):
        query_embedding = self.get_embedding(query)
        response = self.embeddings.query(vector=[query_embedding], top_k=k, namespace='ns1', include_metadata=True)
        docs = self.extract_action(response, query)
        return docs

    def set_prompt(self, prompt):
        self.prompt = prompt
        return self.prompt

    def extract_action(self, response, query = None):
        relevant_docs = ""
        for match in response['matches']:
            if match['score'] > 0.75:
                relevant_docs += match['metadata']['text']
        return relevant_docs


class Answering_Agent:
    def __init__(self, openai_client) -> None:
        self.client = openai_client

    def generate_response(self, query, docs, conv_history, k=5):
        # TODO: Generate a response to the user's query
        context_prompt =\
        f"{conv_history}"\
        f"Please reference the following context to answer the question. Context: {docs}:" \
        f" \n Question: {query}"

        message = {"role": "user", "content": context_prompt}
        response = self.client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[message],
        ).choices[0].message.content
        return response

class Relevant_Documents_Agent:
    def __init__(self, openai_client) -> None:
        self.client = openai_client

    def get_relevance(self, conversation, prompt) -> str:
        context_prompt = \
            f"is the following conversation either related to machine learning or consist of pleasanties? 'Yes', 'No', or 'Somewhat' {conversation} {prompt}:"

        message = {"role": "user", "content": context_prompt}
        response = self.client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[message],
        ).choices[0].message.content
        return response

class Head_Agent:
    def __init__(self, openai_key, pinecone_key, pinecone_index_name) -> None:
        self.client = OpenAI(api_key=openai_key)
        self.pinecone_key = pinecone_key
        self.pinecone_index_name = pinecone_index_name
        self.Obnoxious_Agent = None
        self.Query_Agent = None
        self.Answering_Agent = None
        self.setup_sub_agents()
        self.conv_history = []
        self.logs = []


    def setup_sub_agents(self):
        # Initialize Obnoxious_Agent
        self.Obnoxious_Agent = Obnoxious_Agent(self.client)

        # Initialize Query_Agent
        vectorstore = Pinecone(api_key=self.pinecone_key)
        vs_index = vectorstore.Index(self.pinecone_index_name)
        self.Query_Agent = Query_Agent(vs_index, self.client, vs_index)

        # Relevant Document Agent
        self.Relevant_Documents_Agent = Relevant_Documents_Agent(self.client)

        #Answering Agent
        self.Answering_Agent = Answering_Agent(self.client)


    def main_loop(self):
        self.logs.append("Session Start")
        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-3.5-turbo"

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask me about ML!"):
            self.logs.append(f"Prompt: {prompt}")
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            self.logs.append(f"Prompt: {prompt}")
            if self.Obnoxious_Agent.check_query(prompt):
                response = "I'm sorry, but let's keep our conversation civil."
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                self.Query_Agent.set_prompt(prompt)
                docs = self.Query_Agent.query_vector_store(prompt)
                response = None

                self.logs.append(f"docs: {docs}")
                if len(docs) == 0:
                    relevance = self.Relevant_Documents_Agent.get_relevance(st.session_state.messages[-5:], prompt)
                    print(relevance)
                    if "No" == relevance:
                        response = f"Sorry, no relevant docs found for '{prompt}'."\
                        f"\nPlease ask a question about ML"
                if not Config.chatty:
                    prompt = f"Answering in two sentences or less, {prompt}"

                if not response:
                    response = self.Answering_Agent.generate_response(prompt, docs, st.session_state.messages[-5:])

                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            self.logs.append(f"response: {response}")
            print(self.logs)