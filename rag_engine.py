import os
import streamlit as st
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


def get_vector_store():
    loader = TextLoader("galaxy.txt")
    os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["AZURE_OPENAI_API_KEY"]
    os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets["AZURE_OPENAI_ENDPOINT"]
    os.environ["AZURE_API_VERSION"] = st.secrets["AZURE_API_VERSION"]
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(docs)
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-small",
        azure_endpoint="https://gl-testing-azure.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15",
        api_version=os.getenv("AZURE_API_VERSION"),
    )
    return FAISS.from_documents(texts, embeddings)
