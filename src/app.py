import os
from types import SimpleNamespace
import logging
import gradio as gr
import wandb
from chain import get_answer, load_chain, load_vector_store
from config import default_config
import pandas as pd

logger = logging.getLogger(__name__)


class Chat:
    """A chatbot interface that persists the vectorstore and chain between calls."""
    def __init__(
        self,
        config: SimpleNamespace,
    ):
        """Initialize the chat
        Args:
            config (SimpleNamespace): The configuration.
        """
        self.config = config
        self.wandb_run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            job_type=self.config.job_type,
            config=self.config,
        )
        self.vector_store = None
        self.chain = None

    def __call__(
        self,
        question: str,
        openai_api_key: str = None,
    ):
        """Answer a question about COVID-19 using the LangChain QA chain and vector store retriever.
        Args:
            question (str): The question to answer.
            openai_api_key (str, optional): The OpenAI API key. Defaults to None.
        Returns:
            list[tuple[str, str]], list[tuple[str, str]]: The chat history before and after the question is answered.
        """
        if openai_api_key is not None:
            openai_key = openai_api_key
        #elif os.environ["OPENAI_API_KEY"]:
        #    openai_key = os.environ["OPENAI_API_KEY"]
        else:
            raise ValueError(
                "Please provide your OpenAI API key as an argument or set the OPENAI_API_KEY environment variable"
            )

        if self.vector_store is None:
            self.vector_store = load_vector_store(
                wandb_run=self.wandb_run, openai_api_key=openai_key
            )
        if self.chain is None:
            self.chain = load_chain(
                self.wandb_run, self.vector_store, openai_api_key=openai_key
            )

        #history = history or [] # not use history in this chat
        #history=[]
        question = question.lower()
        response = get_answer(
            chain=self.chain,
            question=question,
            #chat_history=history,
        )
        # history.append((question, response))
        return response
    
def csv_to_markdown(csv_file):
    with open(csv_file, "r") as file:
        df = pd.read_csv(csv_file)
    markdown_table = df.to_markdown(index=False)
    return markdown_table


csv_file_path = "path/to/your/csv/file.csv"
markdown_table = csv_to_markdown(csv_file_path)
print(markdown_table)

with gr.Blocks() as demo:
    with gr.Row():
            with gr.Column():
                gr.HTML(
                    """<b><center>QUICK SEARCH FROM PAPERS REGARDING COVID-19</center></b>
                    <p>Papers regareding COVID-19 are stored behind this chatbot. This chatbot is not intended for a clinical advice tool, but for just a search assistant.</p>
                    <p>Please make sure to read the original document by tracking the source before conclusion.</p>""")

            openai_api_key = gr.Textbox(placeholder="Paste your own OpenAI API key (sk-...)",
                                                show_label=False, lines=1, type='password')
    with gr.Row():
        question = gr.Textbox(
            label="Type in your questions about COVID19 here",
            placeholder="What are the main adverse events in patients with COVID-19 treated with molnupiravir?",
            scale = 5
        )
        clear_question = gr.ClearButton(
            value="clear",
            components=[question],
            variant="secondary",
            scale = 1,
        )
        

    with gr.Row():
        btn = gr.Button(
            value="Submit"
        )
#    state = gr.State()
    output = gr.Textbox(
        label="Output"
        )

    btn.click(
        Chat(
            config=default_config,
        ),
        inputs=[question,openai_api_key],
        outputs=output
        )

    
    
    gr.Markdown(
    """
    # List of papers stored in data base
    
    """)
    gr.Markdown(
    csv_to_markdown("doc_list.csv")
    )
    

if __name__ == "__main__":
    demo.launch(
        show_error=True,debug=True
        #share=True, server_name="0.0.0.0", server_port=8884, show_error=True,debug=True
    )
    demo.integrate(wandb=wandb)