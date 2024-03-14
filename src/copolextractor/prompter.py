import time
import os
import yaml
from openai import OpenAI
import json
from typing import List
import anthropic


class RunTimeExpired(Exception):
    "Raised when the run status of the open-ai call is expired"
    pass


def get_prompt_template():
    prompt = """The content of the PDF is a scientific paper about Copolymerization of Monomers. There are only 
                Copolymerization with 2 different monomers. If you find a polymerisation with just one or more than 2 
                monomers ignore them. Its possible, that there is also the beginning of a new paper about polymers on 
                the PDF. Ignore these. In each paper there 
                could be multiple different reaction with different pairs of monomers and same reactions with different 
                reaction conditions. The reaction constants for the copolymerization with the monomer pair is the most 
                important information. Be careful with numbers and do not miss the decimal points.
                If there are polymerization's without these constants, ignore these. From the PDF,extract the 
                polymerisation information from each polymerisation and report it in valid json format. Try to keep the 
                str short. Exclude comments out of the json output. Give the Output as one json object. Stick to the 
                given output datatype (String or Integer).
                
                Extract the following information:

                   reactions: [
                   {
                    "monomers": ["Monomer 1", "Monomer 2"] as STRING (only the whole Monomer name without abbreviation)
                    "reaction_conditions": [
                    {
                       "polymerization_type": polymerization reaction type (free radical, anionic, cationic, ...) as STRING,
                       "solvent": used solvent for the polymerization reaction as STRING (whole name without 
                                abbreviation, just name no further details) if the polymerization method is bulk put 
                                solvent: None; if the solvent is water put just "water"; ,
                       "method": used polymerisation method (solvent, bulk, emulsion...) as STRING,
                       "temperature": used polymerization temperature as INTEGER ,
                       "temperature_unit": unit of temperature (°C, °F, ...) as STRING,
                       "reaction_constants": { polymerization reaction constants r1 and r2 as INTEGER, 
                        "constant_1":
                        "constant_2": },
                       "reaction_constant_conf": { confidence interval of polymerization reaction constant r1 and r2 as INTEGER
                        "constant_conf_1":
                        "constant_conf_2": },
                       "determination_method": method for determination of the r-values (Kelen-Tudor, ...) as STRING
                    },
                    {
                        "polymerization_type": 
                        "solvent":
                        ...
                     }   
                    ]
                    },
                    {
                    "monomers":
                    "reaction_condition": [
                    { ... }
                    ]
                    }
                   "source": doi url or source as STRING (just one source)
                   "PDF_name": name of the pdf document
                   ]


                   If the information is not provided put 'NA'. If there are multiple polymerization's with different 
                   parameters report as a separate reaction (for different pairs of monomers) and reaction_conditions(for 
                   different reaction conditions of the same monomers)."""
    return prompt


def get_prompt_addition():
    prompt_addition = """Here is the previously collected data from the same PDF: {}. Try to fill up the 
        entries with NA and correct entries if they are wrong. Pay particular attention on numbers and at the decimal point. Combine different reaction if they belong to the same 
        polymerization with the same reaction conditions. Report every different polymerization and every different 
        reaction condition separately. Do this based on this prompt:"""
    return prompt_addition


def get_prompt_addition_with_data(new_data):
    prompt_addition_base = get_prompt_addition()
    prompt_addition_with_data = prompt_addition_base.format(new_data)
    return prompt_addition_with_data


def split_document(document: str, max_length: int) -> List[str]:
    return [document[j : j + max_length] for j in range(0, len(document), max_length)]


def format_prompt(template, data):
    return template.format(data)


def call_openai(prompt, model="gpt-3.5-turbo-1106", temperature: float = 0, **kwargs):
    """Call chat openai model

    Args:
        prompt (str): Prompt to send to model
        model (str, optional): Name of the API. Defaults to "gpt-3.5-turbo-1106".
        temperature (float, optional): inference temperature. Defaults to 0.

    Returns:
        dict: new data
    """
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a scientific assistant, extracting important information about polymerization "
                "conditions out of text.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        **kwargs,
    )
    message_content = completion.choices[0].message.content
    new_data = json.loads(message_content)
    return new_data


def call_openai_agent(assistant, file, prompt, **kwargs):
    print("openai call has started")
    client = OpenAI()
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )

    while run.status != "completed":
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        print(f"Run Satus: {run.status}")
        time.sleep(5)
        if run.status == "expired":
            raise RunTimeExpired
    else:
        print("Run completed!")

    message_response = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    messages = message_response.data
    latest_message = messages[0]
    output = latest_message.content[0].text.value
    return output


def update_data(new_data: dict) -> str:
    old_data_template = get_prompt_addition_with_data(new_data)
    return old_data_template


def update_prompt(prompt, data):
    new_prompt = update_data(data) + prompt
    return new_prompt


def repeated_call_model(text, prompt_template, max_length: int, model: str, model_call_fn) -> dict:
    chunks = split_document(text, max_length=max_length)
    output = {}
    extracted = ""
    for chunk in chunks:
        prompt = format_prompt(prompt_template, {"text": chunk})
        prompt += extracted

        new_data = model_call_fn(prompt, model)
        extracted = update_data(new_data)
        output.update(new_data)
    return output


def format_output_as_json_and_yaml(i, output, output_folder, ):

    parts = output.split("```")

    if len(parts) >= 3:
        output_part = parts[1]
        print(output_part)
    else:
        output_part = ""
        print("Output in json format is empty.")
    output_name_json = os.path.join(output_folder, f"output_data_assistant{i + 1}.json")
    output_name_yaml = os.path.join(output_folder, f"output_data_assistant{i + 1}.yaml")

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
        return json_data
    except json.JSONDecodeError as e:
        print("error at parsing the output to JSON-file:", e)


def format_output_claude_as_json_and_yaml(i, content_blocks, output_folder):
    for content_block in content_blocks:
        if hasattr(content_block, 'text'):
            json_str = content_block.text
        elif hasattr(content_block, 'get'):
            json_str = content_block.get('text')
        else:
            print("ContentBlock hat kein zugängliches 'text'-Attribut oder eine 'get'-Methode.")
            return
        output_name_json = os.path.join(output_folder, f"output_data_claude{i + 1}.json")
        output_name_yaml = os.path.join(output_folder, f"output_data_claude{i + 1}.yaml")

        try:
            json_data = json.loads(json_str)

            with open(output_name_json, "w", encoding="utf-8") as json_file:
                json.dump(json_data, json_file, ensure_ascii=False, indent=4)
            print(f"Output saved as JSON-file at {output_name_json}.")

            with open(output_name_yaml, "w", encoding="utf-8") as yaml_file:
                yaml.dump(json_data, yaml_file, allow_unicode=True)
            print(f"Output saved as YAML-file at {output_name_yaml}.")
            return json_data
        except Exception as e:
            print(f"Error parsing the output: {e}")


def call_claude3(prompt):
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        system="You are a scientific assistant, extracting important information about polymerization conditions "
               "out of pdfs in valid json format. If the info is not found put 'NA'.",
        temperature=0.0,
        messages=prompt
    )
    print(message.content)
    return message.content
