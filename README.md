# 🤖 RAG Bot – Chat With Your PDF Documents

RAG Bot is an AI-powered assistant that lets you upload a PDF and ask questions about its content.  
It uses **Retrieval-Augmented Generation (RAG)** to search your document and answer based only on the provided context — **no hallucination, no made-up answers.**

✅ Powered by **Groq + Qwen 32B**  
✅ Uses **FAISS** vector database  
✅ Built with **Streamlit + LangChain**  
✅ Runs fully on CPU (no GPU required)

---

## 🚀 Live Demo
Try the app here:  
👉 **https://ragbot-tkedmmzozbxni6zgor7erm.streamlit.app/**

---

## 📁 Repository
GitHub Source Code:  
👉 **https://github.com/junaidariie/Rag_Bot**

---

## ✅ Features

- 📄 Upload any PDF document  
- 🔍 Extracts and chunks text automatically  
- 🧠 Builds a FAISS vector database for search  
- 💬 Ask natural language questions
- ✅ Answers only from the document context (no hallucination)
- 🧾 Shows source text used to answer
- 🔥 Powered by **Groq Qwen 3-32B** for fast reasoning

---

## 🧠 Tech Stack

| Component | Technology |
|----------|------------|
| LLM | `ChatGroq(model="qwen/qwen3-32b")` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | FAISS |
| PDF Loader | PDFPlumber |
| Frontend | Streamlit |
| Framework | LangChain |

---

## 🧩 How It Works

```
1. User uploads PDF
2. Text extracted and split into chunks
3. Chunks embedded using MiniLM
4. Stored in FAISS vector DB
5. User asks a question
6. FAISS retrieves the most relevant chunks
7. Qwen-32B answers using only the retrieved context
```

✅ Prevents hallucinations  
✅ Ensures answers come from the actual document

---

## 📦 Installation & Setup

### 1️⃣ Clone repository

```bash
git clone https://github.com/junaidariie/Rag_Bot
cd Rag_Bot
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Add your API keys

Create a `.streamlit/secrets.toml` file:

```toml
GROQ_API_KEY="your_groq_key"
```

### 4️⃣ Run the app

```bash
streamlit run main.py
```

---

## 📚 Usage

1. Upload a PDF file  
2. Wait for “✅ PDF processed successfully”  
3. Ask any question in the chat box  
4. The bot responds with answers extracted from the document  
5. Expand **📚 View source documents** to verify the answer

---

## 🧠 RAG Prompt Logic

```text
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know—don't try to make up an answer.
Don't provide anything out of the given context.
```

✅ No hallucination  
✅ Trustworthy responses  
✅ Fully explainable

---

## ✅ Future Enhancements

- Support for multiple PDFs
- Download conversation as a text file
- Vector DB caching for faster startup
- Chat history memory
- Web URLs & TXT imports

---

## 🤝 Contributing

Pull requests and improvements are welcome!

---

## ⭐ If you like the project
Give the repo a ⭐ on GitHub and share the app!

---

Made with ❤️ using Streamlit, LangChain, Groq, and FAISS.
