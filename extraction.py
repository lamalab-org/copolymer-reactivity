import openai
import os

openai.api_key = "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"
input_folder = 'test/markdown_output'
output_folder = 'test/model_output'

prompt = ""

for filename in os.listdir(input_folder):
    if filename.endswith(".mmd"):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
        prompt = (f"""Here is the content of the file {filename}: {file_content}
        Extract the polymerization information from each polymerization and report it as a table with the following 
        columns and information
        
        source
        polymerization reaction
        involved monomers
        used solvent
        used polymerization method
        temperature
        polymerization reaction constants r1 and r2
        confidence interval of polymerization reaction constant r1 and r2
         
        If the information is not provided put NA. If there are multiple polymerization's with different parameters 
        report as a separate line.""")
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            temperature=0.5
        )
        print(response)
