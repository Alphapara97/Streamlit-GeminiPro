"""
Module Name: ChatBot using Gemini and Langchain
Author: Ansuman Sasmal
"""

__author__ = "Ansuman Sasmal"
__version__ = "2.0"



import pathlib
import textwrap
import os
import tempfile
import streamlit as st
import urllib
import warnings
from pathlib import Path as p
from pprint import pprint
import uuid
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_google_vertexai import ChatVertexAI
from streamlit_chat import message
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader


st.set_page_config(page_title="Chat with Documents", page_icon="⛓️")
st.title("Chat with Documents")
st.text("Please enter the access key provided* and upload the research papers")
st.text("*: Present in the resume or you can click the github link provided below.")
st.text("link to demo video :https://github.com/Alphapara97/Streamlit-GeminiPro ")

@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_file, google_api_key):
    # Generate a unique identifier for the file
    unique_identifier = str(uuid.uuid4())
    temp_filename = f"file_{unique_identifier}"

    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, temp_filename)

    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Load document
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()

    # Split document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_text(context)

    # Create embeddings and store in vectordb and extract vector_index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 5})

    return vector_index

def clear_chat():
    del st.session_state.past[:]
    del st.session_state.generated[:]        
        
google_api_key = st.sidebar.text_input("Gemini API Key", type="password")
if not google_api_key:
    st.info("Please add access key to continue.")
    st.stop()
    
uploaded_file = st.sidebar.file_uploader(
    label="Upload PDF file", type=["pdf"], accept_multiple_files=False
)
if not uploaded_file:
    st.info("Please upload document to continue.")
    st.stop()

retriever = configure_retriever(uploaded_file, google_api_key)
# Continue with the rest of your Streamlit app logic using 'retriever'...


# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup GeminiLLM and QA chain

generation_config = {
  "temperature": 0,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 4096,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  },
]


model_safety_none = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=google_api_key,
                             temperature=0.0,convert_system_message_to_human=True, safety_settings=safety_settings)

#model_safety_none = ChatVertexAI(model="gemini-pro",google_api_key=google_api_key,
#                             temperature=0.0,convert_system_message_to_human=True, safety_settings=safety_settings)


qa_chain = RetrievalQA.from_chain_type(
    model_safety_none,
    retriever=retriever,
    return_source_documents=False,
    memory=memory, verbose=True
)


def conversational_chat(query):

    result = qa_chain(query)
    st.session_state['history'].append((query, result["result"]))

    return result["result"]

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

#container for the chat history
response_container = st.container()
#container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):

        user_input = st.text_input("Query:", placeholder="Ask your questions regarding your document", key='input')
        submit_button = st.form_submit_button(label='Send')
        

    if submit_button and user_input:
        output = conversational_chat(user_input)

        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            #avatar_style check for options
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="adventurer",seed="midnight")
            message(st.session_state["generated"][i], key=str(i), avatar_style="bottts",seed="bob")
    
    st.button("Clear message", on_click=clear_chat)
