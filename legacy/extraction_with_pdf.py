import openai
from openai import OpenAI
import os
import yaml
import copolextractor.prompter as prompter
import langchain
from dotenv import load_dotenv
from langchain.cache import SQLiteCache
from langchain.llms import OpenAI as LangChainOpenAI


def get_prompt_template():
    prompt = """From the PDF,extract the polymerization information from each polymerization and report it in json format. 
                Extract the following information:

                   reactions: 
                    monomers: [name of pair of involved monomers] 
                    -combinations: 
                     -polymerization_type: polymerization reaction type (free radical polymerization, anionic polymerization, cationic polymerizatio,...)
                       solvent: used solvent
                       method: used polymerization method (solvent, bulk,...)
                       temperature: used polymerization temperature
                       temperature_unit: unit of temperature (°C, °F, ...)
                       reaction_constants: polymerization reaction constants r1 and r2
                        -constant_1:
                        -constant_2:
                       reaction_constant_conf: confidence interval of polymerization reaction constant r1 and r2
                        -constant_conf_1:
                        -constant_conf_2:
                       determination_method: method for determination of the r-values (Kelen-Tudor, ...)
                   source: doi url or source  


                   If the information is not provided put NA. If there are multiple polymerization's with different 
                   parameters report as a separate reaction and combinations."""
    return prompt


load_dotenv()
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
llm = OpenAI()

input_folder = "./../pdfs"
output_folder = "model_output"
max_section_length = 16385
model = "gpt-4-1106-preview"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


for i, filename in enumerate(os.listdir(input_folder)):
    if filename.endswith(".pdf"):
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
        prompt_template = get_prompt_template()
        output = prompter.call_openai_agent(assistant, file, prompt_template)
        print('Output: ', output)
        output_name = os.path.join(output_folder, f"output_data_assistant{i + 1}.yaml")
        with open(output_name, "w") as yaml_file:
            yaml.dump(output, yaml_file, allow_unicode=True)
