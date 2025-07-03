import os
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
import wandb

def download_artifact(artifact_name: str, project_path: str) -> str:
    api = wandb.Api()
    artifact = api.artifact(f"{project_path}/{artifact_name}:latest")
    return artifact.download()

def query_health_rag(query_text: str, chroma_path: str) -> str:
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_path, 
                embedding_function=embedding_function)
    
    augmented_query = f"sintomas: {query_text} diretrizes saúde brasil"
    
    results = db.similarity_search_with_score(augmented_query, k=3)
    
    context_text = "\n\n---\n\n".join(
        [f"Fonte: {doc.metadata['source']}\n{doc.page_content}" 
         for doc, _score in results]
    )
    return context_text