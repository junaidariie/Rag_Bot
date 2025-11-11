from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os, shutil

pdf_directory = "pdfs/"
FAISS_DB_PATH = "vectorstore/db_faiss"
os.makedirs(pdf_directory, exist_ok=True)

def upload_pdf(file_input):
    """
    Saves the uploaded PDF locally and returns its path
    """
    if isinstance(file_input, str):
        filename = os.path.basename(file_input)
        dest = os.path.join(pdf_directory, filename)
        shutil.copy2(file_input, dest)
    else:  # uploaded file
        filename = file_input.name
        dest = os.path.join(pdf_directory, filename)
        with open(dest, "wb") as f:
            shutil.copyfileobj(file_input, f)

    return dest

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def create_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return splitter.split_documents(docs)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=model_name)

def build_vector_db_from_pdf(uploaded_file):
    pdf_path = upload_pdf(uploaded_file)        
    docs = load_pdf(pdf_path)                    
    chunks = create_chunks(docs)                 
    db = FAISS.from_documents(chunks, get_embedding_model())  
    db.save_local(FAISS_DB_PATH)                

def load_vector_db():
    if os.path.exists(FAISS_DB_PATH):
        return FAISS.load_local(FAISS_DB_PATH, get_embedding_model(), allow_dangerous_deserialization=True)
    return None
