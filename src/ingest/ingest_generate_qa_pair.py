import argparse
from typing import List, Tuple
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential, # for exponential backoff
)
import wandb
import openai
import pandas as pd
from ingest_vector_store import load_documents



@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def generate_pairs_qa(documents,
                      system_template_for_qa_generation: str,
                      prompt_template_for_qa_generation: str,
                      n_generations=1):
    """generate pairs of question and answer 
    Args:
        system_template_for_qa_generation (str): File name of system template
        prompt_template_for_qa_generation (str): File name of prompt template
        n_generations (int)): number of questions to create per chunk

    Returns:
        dataframe: a list of pairs of questions and answersb with context and source
    """
    with open(system_template_for_qa_generation, "r") as file:
        system_prompt_for_qa_generation = file.read()

    generations = []
    for document in documents:
        user_prompt = generate_context_prompt(prompt_template_for_qa_generation = prompt_template_for_qa_generation,
                                              CONTEXT = document.page_content
                                              )
        messages=[
            {"role": "system", "content": system_prompt_for_qa_generation},
            {"role": "user", "content": user_prompt},
        ]
        response = completion_with_backoff(
            model="gpt-3.5-turbo",
            temperature=0.1,
            messages=messages,
            n = n_generations,
            )
        context = document.page_content
        source = document.metadata["source"]
        page = document.metadata["page"]+1
        generations.extend([f"CONTEXT:{document.page_content} \n"+ response.choices[i].message.content+f"\n SOURCE:{source}\t page({page})"
                            for i in range(n_generations)])

    generated_examples = []
    for generation in generations:

        context, question, answer, source = parse_generation(generation)
        generated_examples.append({"context": context, "question": question, "answer": answer,"source":source})
    return generated_examples

def generate_context_prompt(prompt_template_for_qa_generation,CONTEXT):
    
    with open(prompt_template_for_qa_generation, "r") as file:
        prompt_template_for_qa_generation = file.read()

    user_prompt = prompt_template_for_qa_generation.format(CONTEXT=CONTEXT)
    return user_prompt

def parse_generation(generation):
    lines = generation.split("\n")
    context = []
    question = []
    answer = []
    source = []
    flag = None
    
    for line in lines:
        if "CONTEXT:" in line:
            flag = "context"
            line = line.replace("CONTEXT:", "").strip()
        elif "QUESTION:" in line:
            flag = "question"
            line = line.replace("QUESTION:", "").strip()
        elif "ANSWER:" in line:
            flag = "answer"
            line = line.replace("ANSWER:", "").strip()
        elif "SOURCE:" in line:
            flag = "source"
            line = line.replace("SOURCE:", "").strip()

        if flag == "context":
            context.append(line)
        elif flag == "question":
            question.append(line)
        elif flag == "answer":
            answer.append(line)
        elif flag == "source":
            source.append(line)


    context = "\n".join(context)
    question = "\n".join(question)
    answer = "\n".join(answer)
    source = "\n".join(source)
    return context, question, answer, source

def log_generated_examples(generated_examples, 
                           run: "wandb.run",
                           system_template_for_qa_generation,
                           prompt_template_for_qa_generation
                           ):
    
    df = pd.DataFrame(generated_examples)
    df.to_csv('eval/generated_examples.csv', index=False)
    run.log({"generated_examples": wandb.Table(dataframe=df)})
    # log csv file as an artifact to W&B for later use
    artifact =wandb.Artifact("generated_examples", type="dataset")
    artifact.add_file("eval/generated_examples.csv")
    run.log_artifact(artifact)
    artifact =wandb.Artifact("prompt_template_for_qa_generation", type="prompt")
    artifact.add_file(prompt_template_for_qa_generation)
    run.log_artifact(artifact)
    artifact =wandb.Artifact("system_template_for_qa_generation", type="prompt")
    artifact.add_file(system_template_for_qa_generation)
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
        "--document_artifact",
        default='keisukekamata/covid-doc-search/documentation_dataset:v0',
        type=str,
        help="artifact of document in wandb",
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
    parser.add_argument(
        "--system_template_for_qa_generation",
        default="prompt_data/system_template_for_qa_generation.txt",
        type=str,
    )
    parser.add_argument(
        "--prompt_template_for_qa_generation",
        default="prompt_data/prompt_template_for_qa_generation.txt",
        type=str,
    )


    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)
    documents = load_documents(args.docs_dir)
    generated_examples = generate_pairs_qa(documents,
                                           system_template_for_qa_generation=args.system_template_for_qa_generation,
                                           prompt_template_for_qa_generation=args.prompt_template_for_qa_generation,
                                           n_generations=1)
    log_generated_examples(generated_examples,
                            run,
                            system_template_for_qa_generation=args.system_template_for_qa_generation,
                            prompt_template_for_qa_generation=args.prompt_template_for_qa_generation,
                            )
    run.finish()

if __name__ == "__main__":
    main()

#docker exec -it covid-doc-search python ingest/ingest_generate_qa_pair.py