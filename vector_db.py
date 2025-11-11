from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os, shutil

embedding_model = HuggingFaceEmbeddings(
    
)

pdf_directory = "pdfs/"
os.makedirs(pdf_directory, exist_ok=True)

def upload_pdf(file_input):
    """
    Accepts either a file path (string) or a file object
    """
    if isinstance(file_input, str):
        # It's a file path
        filename = os.path.basename(file_input)
        destination_path = os.path.join(pdf_directory, filename)
        shutil.copy2(file_input, destination_path)
    else:
        # It's a file object
        filename = file_input.name
        destination_path = os.path.join(pdf_directory, filename)
        with open(destination_path, "wb") as f:
            shutil.copyfileobj(file_input, f)
    
    return f"File {filename} uploaded successfully to {destination_path}"


def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

def create_chunk(documents):
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )
    text_chunk = text_spliter.split_documents(documents)
    return text_chunk


file_path = "United Nations Universal Declaration of Human Rights 1948.pdf"
document = load_pdf(file_path)
text_chunks = create_chunk(document)


model="sentence-transformers/all-MiniLM-L6-v2"
def get_embedding_model(model):
    embeddings = HuggingFaceEmbeddings(model_name=model)
    return embeddings

FAISS_DB_PATH = "vectorstore/db_faiss"
faiss_db = FAISS.from_documents(text_chunks, get_embedding_model(model))
faiss_db.save_local(FAISS_DB_PATH)




"""file_path = "United Nations Universal Declaration of Human Rights 1948.pdf"
document = load_pdf(file_path)

text_chunks = create_chunk(document)
print("chunk_count", len(text_chunks))"""