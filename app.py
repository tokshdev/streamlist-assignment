import streamlit as st
import os
from langchain_openai import AzureChatOpenAI
from rag_engine import get_vector_store
from router import classify_query
from prompts import get_prompt_template

# 1. Page Config
st.set_page_config(page_title="Smart Support Copilot", layout="wide")
st.title("Smart Support Copilot")

# 2. Initialize Persistent State
if "vector_store" not in st.session_state:
    with st.spinner("Initializing Knowledge Base..."):
        st.session_state.vector_store = get_vector_store()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Setup Azure LLM
os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["AZURE_OPENAI_API_KEY"]
os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets["AZURE_OPENAI_ENDPOINT"]
os.environ["AZURE_API_VERSION"] = st.secrets["AZURE_API_VERSION"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]


# Ensure "azure_deployment" exactly matches the deployed name in your Azure portal
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini", # Changed to match your endpoint's target, or use "gpt-4o" if that is your actual deployment name
    api_version=os.getenv("AZURE_API_VERSION"), 
    temperature=0
)

# 4. Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Handle User Input
if query := st.chat_input("How can I help you with your device today?"):
    # Append User Query
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Classification & Retrieval
    q_type = classify_query(query, llm)
    docs = st.session_state.vector_store.similarity_search(query, k=2)
    context = "\n".join([d.page_content for d in docs])
    
    # Build Prompt with Memory (Last 3 messages)
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-3:]])
    prompt = f"History: {history_text}\n\n{get_prompt_template(q_type)}\n\nContext: {context}\n\nQuery: {query}"
    
    # Generate Response
    response = llm.invoke(prompt)
    
    # Display & Store Assistant Response
    st.session_state.messages.append({"role": "assistant", "content": response.content})
    with st.chat_message("assistant"):
        st.caption(f"Intent Classified As: **{q_type}**")
        st.markdown(response.content)
        
        # 6. Source Awareness (Rubric Requirement)
        with st.expander("View Retrieved Sources"):
            for i, doc in enumerate(docs):
                st.write(f"Source {i+1}: {doc.page_content[:200]}...")        
