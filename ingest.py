import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def ingest_docs(docs_folder="./docs"):
    all_chunks = []

    # Get API key from environment or Streamlit secrets
    api_key = os.getenv("OPENAI_API_KEY")

    # Try Streamlit secrets if env var not found
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets["OPENAI_API_KEY"]
            os.environ["OPENAI_API_KEY"] = api_key
        except Exception:
            raise ValueError("OPENAI_API_KEY not found in environment or Streamlit secrets")

    if not os.path.exists(docs_folder):
        raise FileNotFoundError(f"docs folder not found: {docs_folder}")

    pdf_files = [f for f in os.listdir(docs_folder) if f.endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {docs_folder}")

    print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(docs_folder, pdf_file)
        print(f"Loading: {pdf_file}")

        loader    = PyPDFLoader(pdf_path)
        documents = loader.load()

        for doc in documents:
            doc.metadata["source_file"] = pdf_file

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)
        all_chunks.extend(chunks)
        print(f"   {len(documents)} pages -> {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Creating vector database...")

    embeddings  = OpenAIEmbeddings(api_key=api_key)
    vectorstore = FAISS.from_documents(
        documents=all_chunks,
        embedding=embeddings
    )
    vectorstore.save_local("./db")
    print(f"Done! {len(all_chunks)} chunks stored in ./db")

if __name__ == "__main__":
    ingest_docs()