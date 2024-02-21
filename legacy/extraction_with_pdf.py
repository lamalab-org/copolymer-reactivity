import openai
from openai import OpenAI
import os
import yaml
import json
import copolextractor.prompter as prompter
import langchain
from dotenv import load_dotenv
from langchain.cache import SQLiteCache
from langchain.llms import OpenAI as LangChainOpenAI



load_dotenv()
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
llm = OpenAI()

input_folder = "./../pdfs"
output_folder = "model_output_assistant"
input_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".pdf")])
max_section_length = 16385
model = "gpt-4-1106-preview"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


for i, filename in enumerate(input_files):
    file_path = os.path.join(input_folder, filename)
    print(file_path)
    print(filename)
    file = client.files.create(
        file=open(file_path, "rb"),
        purpose="assistants"
    )
    assistant = client.beta.assistants.create(
        instructions="You are a scientific assistant, extracting important information about polymerization conditions out of pdfs.",
        model="gpt-4-turbo-preview",
        tools=[{"type": "retrieval"}],
        name="Extractor",
        file_ids=[file.id]
    )
    prompt_template = prompter.get_prompt_template()
    output = prompter.call_openai_agent(assistant, file, prompt_template)
    print('Output: ', output)

    parts = output.split("```")

    if len(parts) >= 3:
        output_part = parts[1]
        print(output_part)
    else:
        output_part = ""
        print("Output in json format is empty.")
    output_name_json = os.path.join(output_folder, f"output_data_assistant{i+1}.json")
    output_name_yaml = os.path.join(output_folder, f"output_data_assistant{i+1}.yaml")

    if output_part.startswith("json\n"):
        output_cleaned = output_part.split("json\n", 1)[1]
    else:
        output_cleaned = output_part
    print(output_cleaned)
    try:
        json_data = json.loads(output_cleaned)

        with open(output_name_json, "w", encoding="utf-8") as json_file:
            json.dump(json_data, json_file, ensure_ascii=False, indent=4)
        print("output saved as JSON-file.")
        with open(output_name_yaml, "w") as yaml_file:
            yaml.dump(json_data, yaml_file, allow_unicode=True)
    except json.JSONDecodeError as e:
        print("error at parsing the output to JSON-file:", e)
