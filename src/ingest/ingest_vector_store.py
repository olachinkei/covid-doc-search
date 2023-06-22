import argparse
import json
import logging
import os
import pathlib
from typing import List, Tuple
import tiktoken

import langchain
import wandb
from langchain.cache import SQLiteCache
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings 

langchain.llm_cache = SQLiteCache(database_path="langchain.db")

logger = logging.getLogger(__name__)

def load_documents(data_dir: str):
    """Load documents from pdf files

    Args:
        data_dir (str): The directory containing pdf files

    Returns:
        List[Document]: A list of documents
    """
    loader = PyPDFDirectoryLoader(data_dir)
    # use this when encountering token size limit
    #text_splitter = RecursiveCharacterTextSplitter(
    #            chunk_size = 2000,
    #            chunk_overlap  = 20,
    #            length_function = len,
    #            add_start_index = True,
    #        )
    #documents = loader.load_and_split(text_splitter=text_splitter) 
    documents = loader.load_and_split()
    return documents

def create_vector_store(
                        documents,
                        vector_store_path: str = "./vector_store",
                        ) -> Chroma:
    """Create a ChromaDB vector store from a list of documents

    Args:
        documents (_type_): A list of documents to add to the vector store
        vector_store_path (str, optional): The path to the vector store. Defaults to "./vector_store".

    Returns:
        Chroma: A ChromaDB vector store containing the documents.
    """
    api_key = os.environ.get("OPENAI_API_KEY", None)
    embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=vector_store_path,
    )
    vector_store.persist()
    return vector_store


def log_dataset(documents: List[Document], run: "wandb.run"):
    """Log a dataset to wandb

    Args:
        documents (List[Document]): A list of documents to log to a wandb artifact
        run (wandb.run): The wandb run to log the artifact to.
    """
    document_artifact = wandb.Artifact(name="documentation_dataset", type="dataset")
    with document_artifact.new_file("documents.json") as f:
        for document in documents:
            f.write(document.json() + "\n")
    run.log_artifact(document_artifact)

def log_index(vector_store_dir: str, run: "wandb.run"):
    """Log a vector store to wandb

    Args:
        vector_store_dir (str): The directory containing the vector store to log
        run (wandb.run): The wandb run to log the artifact to.
    """
    index_artifact = wandb.Artifact(name="vector_store", type="search_index")
    index_artifact.add_dir(vector_store_dir)
    run.log_artifact(index_artifact)


def ingest_document_vectorstore(
                docs_dir: str,
                vector_store_path: str,
                ) -> Tuple[List[Document], Chroma]:
    """Ingest a directory of markdown files into a vector store

    Args:
        docs_dir (str):
        vector_store_path (str):
    """
    # load the documents
    documents = load_documents(docs_dir)
    # create document embeddings and store them in a vector store
    vector_store = create_vector_store(documents, vector_store_path)
    return documents, vector_store

def ingest_prompt(prompt_template,run):
    with open(prompt_template, "r") as file:
        prompt = file.read()
    prompt = json.dumps(prompt)
    
    with open("prompt_data/prompt_for_production.json", "w") as outfile:
        outfile.write(prompt)
    artifact =wandb.Artifact("prompt_for_production", type="prompt")
    artifact.add_file("prompt_data/prompt_for_production.json")
    run.log_artifact(artifact)
    


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs_dir",
        type=str,
        default="originaldoc",
        help="The directory containing the wandb documentation",
    )
    parser.add_argument(
        "--vector_store",
        type=str,
        default="./vector_store",
        help="The directory to save or load the Chroma db to/from",
    )
    parser.add_argument(
        "--wandb_project",
        default="covid-doc-search",
        type=str,
        help="The wandb project to use for storing artifacts",
    )
    parser.add_argument(
        "--wandb_entity",
        default="keisukekamata",
        type=str,
        help="The wandb's entity",
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)
        
    documents, vector_store = ingest_document_vectorstore(
        docs_dir=args.docs_dir,
        vector_store_path=args.vector_store,
        )
    log_dataset(documents, run)
    log_index(args.vector_store, run)

    run.finish()


if __name__ == "__main__":
    main()

# docker exec -it covid-doc-search python ingest_vectorstore.py