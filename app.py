import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
DB_CONNECTION = "postgresql+psycopg://myuser:mypassword@localhost:5432/vector_db"  #protocol://username:password@host:port/database_name
COLLECTION_NAME = "diet_guide"

# --- UI SETUP ---
st.set_page_config(page_title="JagaMakan", page_icon="🥗")
st.title("🥗 JagaMakan: Malaysian Diet Guide Assistant")
st.caption("Powered by Llama 3.2, AWS S3, and Local Vectors")

# --- BACKEND SETUP (Cached) ---
# We use @st.cache_resource so it doesn't reconnect to DB on every click
@st.cache_resource
def get_vector_store():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DB_CONNECTION,
        use_jsonb=True,
    )

try:
    vector_store = get_vector_store()
    llm = ChatOllama(model="llama3.2")
except Exception as e:
    st.error(f"Error connecting to Database: {e}")
    st.stop()

# --- CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT LOGIC ---
if prompt := st.chat_input("Ask a question about the Malaysian Diet Guide..."):
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Searching the Knowledge Base..."):
            try:
                # Search DB
                docs = vector_store.similarity_search(prompt, k=5)
                context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
                
                # Generate Answer
                prompt_template = ChatPromptTemplate.from_template("""
                You are a helpful nutrition assistant. Answer based ONLY on the context provided.
                If the answer is not in the context, say "I don't know."
                
                CONTEXT:
                {context}
                
                QUESTION: 
                {question}
                """)
                
                chain = prompt_template | llm
                response = chain.invoke({"context": context_text, "question": prompt})
                
                # Display Answer
                st.markdown(response.content)
                
                # Save to History
                st.session_state.messages.append({"role": "assistant", "content": response.content})
                
                # Optional: Show sources in an expander
                with st.expander("View Source Chunks"):
                    st.markdown(context_text)
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")