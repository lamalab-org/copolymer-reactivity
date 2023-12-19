from openai import OpenAI
import os
import json

input_folder = 'markdown_output'
output_folder = 'model_output'

client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY')
)

prompt = ""

for i, filename in enumerate(os.listdir(input_folder)):
    if filename.endswith(".mmd"):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
        prompt = (f"""Here is the content of the file {filename}: {file_content}
        Extract the polymerization information from each polymerization and report it in json format. 
        Extract the following information:

        source
        polymerization reaction
        involved monomers
        used solvent
        used polymerization method
        temperature
        polymerization reaction constants r1 and r2
        confidence interval of polymerization reaction constant r1 and r2
         
        If the information is not provided put NA. If there are multiple polymerization's with different parameters 
        report as a separate polymerization.""")

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system",
                 "content": "You are a scientific assistant, extracting important information about polymerization "
                            "conditions out of text."},
                {"role": "user", "content": prompt}
            ]
        )
        message_content = completion.choices[0].message.content
        polymerization_data = json.loads(message_content)
        print(message_content)
        output_name = os.path.join(output_folder, f'output_data{i+1}.json')
        with open(output_name, 'w') as file:
            json.dump(polymerization_data, file, indent=4)
