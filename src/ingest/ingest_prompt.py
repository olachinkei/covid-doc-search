# for creating a prompt
import argparse
import os
import wandb
import json
import pandas as pd
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate



def log_prompt(prompt_template, run):
    #with open(prompt_template, "r") as file:
    #    template = file.read()
    #prompt = json.dumps(template)
    #with open("prompt_data/prompt_for_production.json", "w") as outfile:
    #    outfile.write(prompt)
    artifact =wandb.Artifact("question_template", type="prompt")
    artifact.add_file("prompt_data/question_template.txt")
    run.log_artifact(artifact)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="prompt_data/prompt_template.txt",
        help="The template file of prompt",
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
    log_prompt(args.prompt_template, run)
    run.finish()

if __name__ == "__main__":
    main()
