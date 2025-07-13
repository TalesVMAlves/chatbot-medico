import argparse
from dotenv import load_dotenv
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from utils.get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
import wandb

load_dotenv()

CHROMA_PATH = "knowledge_base_chroma"
KNOWLEDGE_PATH = "conhecimento"
WANDB_API = os.getenv("WANDB_API")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database")
    parser.add_argument("--wandb_project", type=str, default="health-chroma-db", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default="process_health_docs", help="WandB run name")
    args = parser.parse_args()

    # Initialize WandB
    if not WANDB_API:
        print("Warning: WANDB_API environment variable not set. WandB features disabled.")
    else:
        wandb.login(key=WANDB_API)
        wandb_run = wandb.init(
            project=args.wandb_project, 
            name=args.wandb_run_name,
            config={"reset_db": args.reset}
        )

    if args.reset:
        print("Rebuilding database...")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    
    # Create WandB artifact
    if WANDB_API:
        artifact = wandb.Artifact(
            name="health_knowledge_base",
            type="vector-database",
            description="Chroma vector store with health knowledge documents"
        )
        artifact.add_dir(CHROMA_PATH)
        wandb_run.log_artifact(artifact)
        
        # Wait for artifact to finish uploading
        artifact.wait()
        print(f"✅ Database saved as WandB artifact: {artifact.name}:{artifact.version}")
        wandb_run.finish()

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
                
                # Track with WandB
                if WANDB_API:
                    wandb.log({"file_processed": file, "pages_loaded": len(docs)})
                    
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                if WANDB_API:
                    wandb.log({"file_error": file, "error_message": str(e)})
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
    
    if WANDB_API:
        wandb.log({"total_chunks": len(chunks)})
        
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
    
    # Track Chroma stats
    if WANDB_API:
        collection = db.get()
        wandb.log({
            "chroma_documents": len(collection["ids"]),
            "chroma_embeddings": len(collection["embeddings"]) if collection["embeddings"] else 0
        })

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