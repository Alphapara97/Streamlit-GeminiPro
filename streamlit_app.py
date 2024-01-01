import google.generativeai as genai
from IPython.display import display, Markdown
import pathlib
import textwrap
import os
import tempfile
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings, GoogleGenerativeAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain, RetrievalQA, load_qa_chain
from langchain.vectorstores import DocArrayInMemorySearch, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import urllib
import warnings
from pathlib import Path as p
from pprint import pprint

import pandas as pd
from langchain.prompts import PromptTemplate



st.set_page_config(page_title="Chat with Documents", page_icon="⛓️")
st.title("Chat with Documents")


@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())
        

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
      

    # Create embeddings and store in vectordb and extract vector_index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=google_api_key)
    vector_index = Chroma.from_texts(splits, embeddings).as_retriever(search_kwargs={"k":5})


    return vector_index

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)
        

class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")
        
        
google_api_key = st.sidebar.text_input("Gemini API Key", type="password")
if not openai_api_key:
    st.info("Please add your Gemini API key to continue.")
    st.stop()
    
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF file", type=["pdf"], accept_multiple_files=False
)
if not uploaded_files:
    st.info("Please upload PDF document to continue.")
    st.stop()
    
retriever = configure_retriever(uploaded_files)

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
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
]


model_safety_none = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=google_api_key,
                             temperature=0.0,convert_system_message_to_human=True, safety_settings=safety_settings)



qa_chain = RetrievalQA.from_chain_type(
    model_safety_none,
    retriever=retriever,
    return_source_documents=True,
    memory=memory, verbose=True
)


if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
