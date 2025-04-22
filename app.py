from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings


import os
import tempfile

app = FastAPI()
templates = Jinja2Templates(directory="templates")
#app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory vector store
vector_store = None

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    global vector_store

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    with open(temp_file.name, "wb") as f:
        content = await file.read()
        f.write(content)

    loader = PyPDFLoader(temp_file.name)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    #embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Or "llama3" if supported
    #embeddings = OllamaEmbeddings(model="llama3")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)

    return {"message": "Document uploaded and indexed successfully."}


@app.post("/chat/")
async def chat_with_doc(query: str = Form(...)):
    global vector_store
    if vector_store is None:
        return {"error": "Please upload a document first."}

    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        #llm=Ollama(model="llama3"),
        llm = Ollama(model="mistral"),
        retriever=retriever,
        return_source_documents=False,
    )
    result = qa_chain.run(query)
    return {"response": result}
