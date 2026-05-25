import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def ingest_docs():
    docs_folder = "./docs"
    all_chunks = []

    pdf_files = [f for f in os.listdir(docs_folder) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in /docs folder")
        return

    print(f"Found {len(pdf_files)} PDF(s): {pdf_files}\n")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(docs_folder, pdf_file)
        print(f"Loading: {pdf_file}")

        loader = PyPDFLoader(pdf_path)
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

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(
        documents=all_chunks,
        embedding=embeddings
    )
    vectorstore.save_local("./db")
    print(f"\nDone! {len(all_chunks)} chunks from {len(pdf_files)} PDF(s) stored.")
    print("Vector database saved to ./db folder")

if __name__ == "__main__":
    ingest_docs()