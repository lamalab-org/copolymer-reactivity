from openai import OpenAI
import os
import yaml
import json
#import langchain
#I had an issue with dotev, I tried to fix it, but decided after some time to do something else first
#from dotenv import load_dotenv
#from langchain.cache import SQLiteCache
#from langchain.llms import OpenAI


def split_document(document, max_length):
    return [document[j:j+max_length] for j in range(0, len(document), max_length)]


input_folder = 'markdown_output'
output_folder = 'model_output'
max_section_length = 4096
json_data = ""
collected_data = {}

client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY')
)

prompt = ""
#load_dotenv()
    #langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
    #llm = OpenAI()

for i, filename in enumerate(os.listdir(input_folder)):
    if filename.endswith(".mmd"):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
        sections = split_document(file_content, max_section_length)
        for section in sections:
            prompt = (f"""Here is the content of a section of the file: {section}
                   Extract the polymerization information from each polymerization and report it in json format. 
                   Extract the following information:

                   source: source as doi url
                   polymerization type: polymerization reaction type
                   Monomers: name of pair of involved monomers 
                   solvent: used solvent
                   method: used polymerization method
                   temperature: used polymerization temperature
                   reaction constants: polymerization reaction constants r1 and r2
                   reaction constant conf: confidence interval of polymerization reaction constant r1 and r2

                   If the information is not provided put NA. If there are multiple polymerization's with different 
                   parameters report as a separate polymerization.""")
            full_prompt = prompt + json_data
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system",
                     "content": "You are a scientific assistant, extracting important information about polymerization "
                                "conditions out of text."},
                    {"role": "user", "content": full_prompt}
                ]
            )
            message_content = completion.choices[0].message.content
            new_data = json.loads(message_content)
            collected_data.update(new_data)
            json_string = json.dumps(collected_data, indent=4)
            json_data = (f"Here are the previously collected data: {json_string}. Please add more information based on"
                         f" the following text:")
            print(message_content)

        output_name = os.path.join(output_folder, f'output_data{i + 1}.yaml')
        with open(output_name, 'w') as yaml_file:
            yaml.dump(collected_data, yaml_file, allow_unicode=True)
