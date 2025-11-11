import streamlit as st
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from vector_db import faiss_db
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import shutil
import tempfile

# Load environment variables
load_dotenv()
groq_api = st.secrets["GROQ_API_KEY"]



# Set page configuration
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# Initialize models
@st.cache_resource
def load_llm():
    return ChatGroq(model="qwen/qwen3-32b")

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm_model = load_llm()
embedding_model = load_embedding_model()

# PDF processing functions
def upload_pdf(file):
    """Save uploaded PDF to temporary directory"""
    pdf_directory = "temp_pdfs/"
    os.makedirs(pdf_directory, exist_ok=True)
    
    filename = file.name
    destination_path = os.path.join(pdf_directory, filename)
    
    with open(destination_path, "wb") as f:
        f.write(file.getbuffer())
    
    return destination_path

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

def create_vector_db(file_path):
    """Create FAISS vector database from PDF"""
    documents = load_pdf(file_path)
    text_chunks = create_chunks(documents)
    
    vector_db = FAISS.from_documents(text_chunks, embedding_model)
    return vector_db

# RAG functions
def retrieve_docs(query, vector_db):
    return vector_db.similarity_search(query)

def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know—don't try to make up an answer.
Don't provide anything out of the given context.

Question: {question}
Context: {context}
Answer:
"""

def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    result = chain.invoke({"question": query, "context": context})
    return result.content

# Sidebar for PDF upload
with st.sidebar:
    st.title("📁 Document Management")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Upload a PDF file to chat with its content"
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            # Save uploaded file
            file_path = upload_pdf(uploaded_file)
            
            # Create vector database
            st.session_state.vector_db = create_vector_db(file_path)
            st.session_state.pdf_processed = True
            
            st.success(f"✅ PDF processed successfully!")
            st.info(f"Document: {uploaded_file.name}")
            
            # Clean up temporary file
            try:
                os.remove(file_path)
                os.rmdir("temp_pdfs/")
            except:
                pass

    st.markdown("---")
    st.markdown("### ℹ️ How to use:")
    st.markdown("""
    1. Upload a PDF document
    2. Ask questions about the content
    3. Get AI-powered answers based on the document
    """)
    
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("🤖 RAG Chat Assistant")
st.markdown("Chat with your documents using AI!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        if not st.session_state.pdf_processed:
            st.error("⚠️ Please upload a PDF document first to start chatting!")
        else:
            with st.spinner("Thinking..."):
                try:
                    # Retrieve relevant documents
                    documents = retrieve_docs(prompt, st.session_state.vector_db)
                    
                    # Generate answer
                    response = answer_query(documents, llm_model, prompt)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Show source documents (optional)
                    with st.expander("📚 View source documents"):
                        for i, doc in enumerate(documents[:3]):  # Show top 3 sources
                            st.markdown(f"**Source {i+1}:**")
                            st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                            st.markdown("---")
                            
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "Powered by Groq • LangChain • FAISS • Streamlit",
    help="Built with modern AI technologies for document understanding"
)