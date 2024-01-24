from openai import OpenAI
import os
import yaml
import copolextractor.prompter as prompter


input_folder = "markdown_output"
output_folder = "model_output"
max_section_length = 4096

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

prompt_template = """Here is the content of a section of the file: {}
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
                   parameters report as a separate polymerization."""
# load_dotenv()
# langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
# llm = OpenAI()

for i, filename in enumerate(os.listdir(input_folder)):
    if filename.endswith(".mmd"):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()
        output = prompter.repeated_call_model(file_content, prompt_template, max_section_length, prompter.call_openai())
        print('Output: ', output)
        output_name = os.path.join(output_folder, f"output_data{i + 1}.yaml")
        with open(output_name, "w") as yaml_file:
            yaml.dump(output, yaml_file, allow_unicode=True)
