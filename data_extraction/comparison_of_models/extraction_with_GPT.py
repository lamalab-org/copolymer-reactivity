import os
import yaml
import copolextractor.prompter as prompter
import copolextractor.analyzer as az
import time
from openai import OpenAI


start = time.time()

input_folder = "output_marker"
output_folder = "model_output"
max_section_length = 16385
number_of_model_calls = 2
total_input_tokens = 0
total_output_tokens = 0
number_of_calls = 0
input_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".md")])


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

prompt_template = """Here is the content of a section of the file: {text}. The content of the Markdown is a scientific paper about copolymerization of monomers. We only consider
copolymerizations with 2 different monomers. If you find a polymerization with just one or more than 2
monomers ignore them. Its possible, that there is also the beginning of a new paper about polymers in
the PDF. Ignore these. In each paper there could be multiple different reaction with different pairs of monomers and same reactions with different
reaction conditions. The reaction constants for the copolymerization with the monomer pair is the most
important information. Be careful with numbers and do not miss the decimal points.
If there are polymerizations without these constants, ignore these.
From the PDF, extract the polymerization information from each polymerization and report it in valid json format. 
Don't use any abbreviations, always use the whole word.
Try to keep the string short. Exclude comments out of the json output. Return one json object. Stick to the
given output datatype (string, or float).

Extract the following information:

    reactions: [
    {{
    "monomers": ["Monomer 1", "Monomer 2"] as STRING (only the whole Monomer name without abbreviation)
    "reaction_conditions": [
    {{
        "polymerization_type": "polymerization reaction type (free radical, anionic, cationic, ...)" as STRING,
        "solvent": "used solvent for the polymerization reaction as STRING (whole name without
                abbreviation, just name no further details); if the solvent is water put just 'water';",
        "method": "used polymerization method (solvent, bulk, emulsion...)" as STRING,
        "temperature": "used polymerization temperature" as FLOAT,
        "temperature_unit": "unit of temperature (°C, °F, ...)" as STRING,
        "reaction_constants": {{ "polymerization reaction constants r1 and r2" as FLOAT,
        "constant_1": "",
        "constant_2": "" }},
        "reaction_constant_conf": {{ "confidence interval of polymerization reaction constant r1 and r2" as FLOAT
        "constant_conf_1": "",
        "constant_conf_2": "" }},
        "determination_method": "method for determination of the r-values (Kelen-Tudor, ...)" as STRING
    }},
    {{
        "polymerization_type": "",
        "solvent": "",
        ...
        }}
    ]
    }},
    {{
    "monomers": "",
    "reaction_condition": [
    {{ ... }}
    ]
    }},
    "source": "doi url or source" as STRING (just one source),
    "PDF_name": "name of the pdf document"
    ]
    }}

    If the information is not provided put null. If there are multiple polymerizations with different
    parameters report as a separate reaction (for different pairs of monomers) and reaction_conditions(for
    different reaction conditions of the same monomers).
"""

for i, filename in enumerate(input_files):
    if filename.endswith(".md"):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()
        output, input_token, output_token, number_of_call = (
            prompter.repeated_call_model(
                file_content,
                prompt_template,
                max_section_length,
                prompter.call_openai_chucked,
            )
        )
        total_input_tokens += input_token
        total_output_tokens += output_token
        number_of_calls += number_of_call
        print("Output: ", output)
        output_name = os.path.join(output_folder, f"output_data{i + 1}.yaml")
        with open(output_name, "w") as yaml_file:
            yaml.dump(output, yaml_file, allow_unicode=True)

        for a in range(number_of_model_calls):
            na_count = az.count_na_values(output)
            total_entry_count = az.count_total_entries(output)
            rate = az.calculate_rate(na_count, total_entry_count)
            print("NA-rate: ", rate)
            if rate > 0.3 or output is None:
                print(f"model call number {a + 2} of {filename}")
                updated_prompt = prompter.update_prompt_chucked(prompt_template, output)
                output, input_token, output_token, number_of_call = (
                    prompter.repeated_call_model(
                        file_content,
                        prompt_template,
                        max_section_length,
                        prompter.call_openai_chucked,
                    )
                )
                print("Output: ", output)
                output_name = os.path.join(output_folder, f"output_data{i + 1}.yaml")
                with open(output_name, "w") as yaml_file:
                    yaml.dump(output, yaml_file, allow_unicode=True)
                total_input_tokens += input_token
                total_output_tokens += output_token
                number_of_calls += number_of_call
            else:
                print("NA-rate under 30%")
            print("input tokens used: ", total_input_tokens)
            print("output tokens used: ", total_output_tokens)
            print("total number of model call: ", number_of_calls)

end = time.time()
execution_time = start - end

print("execution time: ", execution_time)
