import argparse
from dotenv import load_dotenv
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from utils.get_embedding_function import get_embedding_function
from langchain_chroma import Chroma

load_dotenv()

CHROMA_PATH = "knowledge_base_chroma"
KNOWLEDGE_PATH = "conhecimento"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database")
    args = parser.parse_args()

    if args.reset:
        print("Rebuilding database...")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents():
    documents = []
    for file in os.listdir(KNOWLEDGE_PATH):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(KNOWLEDGE_PATH, file)
            try:
                print(f"Processing: {file}")
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file
                    doc.metadata["page"] = doc.metadata.get("page", 0)
                documents.extend(docs)
                print(f"Loaded {len(docs)} pages from {file}")
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
    return documents

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       
        chunk_overlap=80,     
        length_function=len,  
        is_separator_regex=False,
        separators=["\n\n", "\n", "(?<=\. )", "(?<=\! )", "(?<=\? )", ", "]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def add_to_chroma(chunks: list[Document]):
    os.makedirs(CHROMA_PATH, exist_ok=True)
    
    embedding_function = get_embedding_function()
    
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )
    
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    print(f"Adding {len(chunks_with_ids)} chunks to database")
    db.add_documents(chunks_with_ids)
    print(f"Database created at: {os.path.abspath(CHROMA_PATH)}")

def calculate_chunk_ids(chunks):
    """
    Calculate unique IDs for each chunk based on source and page
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page", 0)
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    """Remove existing database"""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Previous database removed")

if __name__ == "__main__":
    main()