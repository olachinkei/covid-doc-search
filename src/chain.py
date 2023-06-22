import logging
import wandb
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from prompts import load_chat_prompt
from langchain.chains.qa_with_sources import stuff_prompt

logger = logging.getLogger(__name__)

def load_vector_store(wandb_run: wandb.run, openai_api_key: str) -> Chroma:
    """
    Load a vector store from a Weights & Biases artifact
    Args:
        run (wandb.run): An active Weights & Biases run
        openai_api_key (str): The OpenAI API key to use for embedding
    Returns:
        Chroma: A chroma vector store object
    """
    # load vector store artifact
    vector_store_artifact_dir = wandb_run.use_artifact(
        wandb_run.config.vector_store_artifact, type="search_index"
    ).download()
    embedding_fn = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # load vector store
    vector_store = Chroma(
        embedding_function=embedding_fn, persist_directory=vector_store_artifact_dir
    )

    return vector_store


def load_chain(wandb_run: wandb.run, vector_store: Chroma, openai_api_key: str):
    """Load a ConversationalQA chain from a config and a vector store
    Args:
        wandb_run (wandb.run): An active Weights & Biases run
        vector_store (Chroma): A Chroma vector store object
        openai_api_key (str): The OpenAI API key to use for embedding
    Returns:
        ConversationalRetrievalChain: A ConversationalRetrievalChain object
    """
    
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name=wandb_run.config.model_name,
        temperature=wandb_run.config.chat_temperature,
        max_retries=wandb_run.config.max_fallback_retries,
        streaming = True,
    )
    chat_prompt_dir = wandb_run.use_artifact(
        wandb_run.config.chat_prompt_artifact, type="prompt"
    ).download()

    with open(f"{chat_prompt_dir}/question_template.txt", "r") as file:
        qa_template = file.read()
        
    QA_PROMPT = PromptTemplate(
        template=qa_template,
        input_variables=["summaries", "question"]) 

    chain_type_kwargs = {"prompt":QA_PROMPT,
                         "document_prompt":stuff_prompt.EXAMPLE_PROMPT,
                         "document_variable_name": "summaries"}
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True,
        reduce_k_below_max_tokens=True,
        max_tokens_limit=3500,
    )
    return qa_chain


def get_answer(
    chain: RetrievalQAWithSourcesChain,
    question: str,
):
    """Get an answer from a ConversationalRetrievalChain
    Args:
        chain (ConversationalRetrievalChain): A ConversationalRetrievalChain object
        question (str): The question to ask
        chat_history (list[tuple[str, str]]): A list of tuples of (question, answer)
    Returns:
        str: The answer to the question
    """
    result = chain(
        inputs={"question": question},
        return_only_outputs=True,
    )
    answer = result["answer"]
    source = result["source_documents"][0].metadata["source"].replace("originaldoc/","")
    page = result["source_documents"][0].metadata["page"] + 1
    response = f"Answer:\t{answer}\nSource:\t{source}\npage:\t{page}"
    return response


