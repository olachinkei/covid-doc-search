"""Configuration for the LLM Apps Course"""
from types import SimpleNamespace

TEAM = "keisukekamata"
PROJECT = "covid-doc-search"
JOB_TYPE = "stating"

default_config = SimpleNamespace(
    project=PROJECT,
    entity=TEAM,
    job_type=JOB_TYPE,
    vector_store_artifact='keisukekamata/covid-doc-search/vector_store:v0',
    chat_prompt_artifact='keisukekamata/covid-doc-search/question_template:v0',
    chat_temperature=0.1,
    max_fallback_retries=5,
    model_name="gpt-3.5-turbo",
    eval_model="gpt-3.5-turbo",
    eval_artifact='keisukekamata/covid-doc-search/generated_examples:v0',
)