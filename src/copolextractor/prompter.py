import time
import os
import yaml
from openai import OpenAI
import json
from typing import List, Tuple
import anthropic
import base64


class RunTimeExpired(Exception):
    "Raised when the run status of the open-ai call is expired"

    pass


def get_prompt_rxn_number():
    prompt = """The content of the pictures is a scientific paper about copolymerization of monomers.
The main focus here is to find the copolymerizations which have r-values for a pair of two Monomers.
We only consider copolymerizations with 2 different monomers. If you find a polymerization with just one or more than 2 monomers ignore them. 
Its possible, that there is also the beginning of a new paper about polymers in the PDF. 
Ignore these. In each paper there could be multiple different reaction with different pairs of monomers and same reactions with different reaction conditions. 
Count each different reaction with an r-value as one. Ignore copolymerization with reference to previse work.
If there are also reactions from previous works included set the 'additional sources' variable to true.
Just count copolymerizations with r-values which are carried out in the article.  Return one json object. 
Stick to the given output datatype (string, or float). json:
{
    "number of reactions": number of the different relevant copolymerizations with r-values in the paper (FLOAT),
    "additional sources": if there are additional reaction from previous works included = true (BOOLEAN),
    "language": language of the main text (STRING),
    "paper name": name of the pdf document (STRING
}
"""
    return prompt


def get_prompt_pdf_quality():
    prompt = """The content of the pictures is a scientific paper about copolymerization of monomers.
    The main focus here is to find the copolymerizations which have r-values for a pair of two Monomers.
    Its possible, that there is also the beginning of a new paper about polymers in the PDF. 
    Ignore these. Rate the quality of the provided paper form 0 (hard to extract) to 10 (easy to extract) in terms of readability and easiness of data extraction.
    Return one json object. 
    Stick to the given output datatype (string, integer or float). json:
    {
        "pdf_quality": the quality of the provided PDF document in terms of e.g. resolution and easiness of data extraction form 0 (hard to extract) to 10 (easy to extract) (FLOAT),
        "table_quality": the quality and structuredness of the tables in the PDF document in terms of e.g. easiness of data extraction and clearity form 0 (hard to extract) to 10 (easy to extract) (FLOAT),
        "quality_of_numbers": the readability of the numbers in the PDF document form 0 (hard to extract) to 10 (easy to extract) (FLOAT),
        "year": the year of the publication (INTEGER)
    }
    """
    return prompt


def get_prompt_template():
    prompt = """The content of the pictures is a scientific paper about copolymerization of monomers. 
We only consider copolymerizations with 2 different monomers. If you find a polymerization with just one or more than 2 monomers ignore them. 
Its possible, that there is also the beginning of a new paper about polymers in the PDF. 
Ignore these. In each paper there could be multiple different reaction with different pairs of monomers and same reactions with different reaction conditions. 
The reaction constants for the copolymerization with the monomer pair is the most important information. Be careful with numbers and do not miss the decimal points.
If there are polymerization's without these constants, ignore these.
From the PDF, extract the polymerization information from each polymerization and report it in valid json format. 
Also pay attention to the caption of figures.
Don't use any abbreviations, always use the whole word.
Try to keep the string short. Exclude comments out of the json output. Return one json object. 
Stick to the given output datatype (string, or float).

Extract the following information:

reactions: [
    {
        "monomers": ["Monomer 1", "Monomer 2"] as STRING (only the whole Monomer name without abbreviation)
        "reaction_conditions": [
            {
                "polymerization_type": polymerization reaction type (free radical, anionic, cationic, ...) as STRING,
                "solvent": used solvent for the polymerization reaction as STRING (whole name without
                        abbreviation, just name no further details like 'sulfur or water free'); if the solvent is water put just "water"; ,
                "method": used polymerization method (solvent(polymerization takes place in a solvent), bulk (polymerization takes place without any solvent, only reactants like monomers built the reaction mixture), emulsion...) as STRING,
                "temperature": used polymerization temperature as FLOAT ,
                "temperature_unit": unit of temperature (°C, °F, ...) as STRING,
                "reaction_constants": { polymerization reaction constants r1 and r2 as FLOAT (be careful and just take the individual values, not the product of these two),
                "constant_1":
                "constant_2": },
                "reaction_constant_conf": { confidence interval of polymerization reaction constant r1 and r2 as FLOAT
                "constant_conf_1":
                "constant_conf_2": },
                "determination_method": method for determination of the r-values (Kelen-Tudor, EVM Program...) as STRING
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


If the information is not provided put null.
If there are multiple polymerization's with different parameters report as a separate reaction (for different pairs of monomers) and reaction_conditions (for different reaction conditions of the same monomers)."""
    return prompt


def get_prompt_addition() -> str:
    prompt_addition = """Here is the previously collected data from the same Markdowns: {}. 
Try to fill up the entries with NA and correct entries if they are wrong. Pay particular attention on numbers and at the decimal point. 
Combine different reaction if they belong to the same polymerization with the same reaction conditions. 
Report every different polymerization and every different reaction condition separately. Do this based on this prompt:"""
    return prompt_addition


def get_prompt_addition_with_data(new_data) -> str:
    prompt_addition_base = get_prompt_addition()
    prompt_addition_with_data = prompt_addition_base.format(new_data)
    return prompt_addition_with_data


def split_document(document: str, max_length: int) -> List[str]:
    return [document[j : j + max_length] for j in range(0, len(document), max_length)]


def format_prompt(template: str, data: dict) -> str:
    return template.format(**data)


def call_openai(
    prompt, model="gpt-4o", temperature: float = 0.0, **kwargs
):
    """Call chat openai model

    Args:
        prompt (str): Prompt to send to model
        model (str, optional): Name of the API. Defaults to ""gpt-4-vision-preview".
        temperature (float, optional): inference temperature. Defaults to 0.

    Returns:
        dict: new data
    """
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a scientific assistant, extracting important information about polymerization conditions"
                "out of pdfs in valid json format. Extract just data which you are 100% confident about the "
                "accuracy. Keep the entries short without details. Be careful with numbers.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
        seed=12345,
        **kwargs,
    )
    input_tokens = completion.usage.prompt_tokens
    output_token = completion.usage.completion_tokens
    message_content = completion.choices[0].message.content
    return message_content, input_tokens, output_token


def call_openai_chucked(
    prompt, model="gpt-3.5-turbo-1106", temperature: float = 0.0, **kwargs
):
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
                "content": "You are a scientific assistant, extracting important information about polymerization conditions"
                "out of pdfs in valid json format. Extract just data which you are 100% confident about the "
                "accuracy. Keep the entries short without details. Be careful with numbers.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        **kwargs,
    )
    message_content = completion.choices[0].message.content
    input_tokens = completion.usage.prompt_tokens
    output_token = completion.usage.completion_tokens
    new_data = json.loads(message_content)
    return new_data, input_tokens, output_token


def call_openai_agent(assistant, file, prompt, **kwargs):
    print("openai call has started")
    client = OpenAI()
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=prompt
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id, assistant_id=assistant.id
    )
    run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    while run.status != "completed":
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        print(f"Run Satus: {run.status}")
        time.sleep(5)
        if run.status == "expired" or run.status == "failed":
            print(run.last_error)
            raise RunTimeExpired
    else:
        print("Run completed!")

    message_response = client.beta.threads.messages.list(thread_id=thread.id)
    messages = message_response.data
    latest_message = messages[0]
    output = latest_message.content[0].text.value
    input_token = run.usage.prompt_tokens
    output_token = run.usage.completion_tokens
    return output, input_token, output_token


def update_data(new_data: dict) -> str:
    old_data_template = get_prompt_addition_with_data(new_data)
    return old_data_template


def update_prompt(prompt, data):
    new_prompt = update_data(data) + prompt
    return new_prompt


def update_prompt_chucked(prompt, data):
    new_prompt = update_data_chucked(data) + prompt
    return new_prompt


def update_data_chucked(new_data: dict) -> str:
    old_data_template = f"""Here are the previously collected data: {new_data}. Please add more information based on"""
    return old_data_template


def repeated_call_model(
    text, prompt_template, max_length: int, model_call_fn
) -> Tuple[dict, int, int, int]:
    chunks = split_document(text, max_length=max_length)
    output = {}
    extracted = ""
    input_tokens = 0
    output_tokens = 0
    number_of_model_calls = 0
    for chunk in chunks:
        prompt = format_prompt(prompt_template, {"text": chunk})
        prompt += extracted

        new_data, input_token, output_token = model_call_fn(prompt)
        number_of_model_calls += 1
        input_tokens += input_token
        output_tokens += output_token
        extracted = update_data(new_data)
        output.update(new_data)
    return output, input_token, output_token, number_of_model_calls


def format_output_as_json_and_yaml(
    i,
    output,
    output_folder,
    pdf_name
):
    parts = output.split("```")

    if len(parts) >= 3:
        output_part = parts[1]
    else:
        output_part = ""
        print("Output in json format is empty.")
    output_name_json = os.path.join(output_folder, f"output_data{i + 1}.json")
    output_name_yaml = os.path.join(output_folder, f"output_data{i + 1}.yaml")

    if output_part.startswith("json\n"):
        output_cleaned = output_part.split("json\n", 1)[1]
    else:
        output_cleaned = output_part
    print(output_cleaned)
    try:
        json_data = json.loads(output_cleaned)
        json_data['source_pdf'] = pdf_name

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
        if hasattr(content_block, "text"):
            json_str = content_block.text
        elif hasattr(content_block, "get"):
            json_str = content_block.get("text")
        else:
            return
        output_name_json = os.path.join(output_folder, f"output_data_claude{i}.json")
        output_name_yaml = os.path.join(output_folder, f"output_data_claude{i}.yaml")

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
        "out of images in valid json format. If the info is not found put 'NA'. Always be truthful and do not "
        "extract anything false or made up.",
        temperature=0.0,
        messages=prompt,
    )
    input_token = message.usage.input_tokens
    output_token = message.usage.output_tokens
    print(message.content)
    return message.content, input_token, output_token


def update_prompt_with_text_and_images(original_prompt, data, prompt):
    new_text = update_prompt(prompt, data)
    original_prompt[-1]["text"] = new_text
    updated_prompt_str = json.dumps(original_prompt)
    print(f"updated_prompt_str after update: {updated_prompt_str}")
    return updated_prompt_str


def create_image_content(image, detail="high"):
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": detail},
    }


def get_prompt_vision_model(images_base64, prompt_text):
    content = []
    for data in images_base64:
        content.append(create_image_content(data))

    content.append({"type": "text", "text": prompt_text})
    return content


def encode_image_to_base64(filepath):
    with open(filepath, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_prompt_claude_vision(output_folder_images, filename, pdf_images, prompt_text):
    name_without_ext, _ = os.path.splitext(filename)
    images = [
        (
            "image/png",
            encode_image_to_base64(
                os.path.join(
                    output_folder_images,
                    f"corrected_{name_without_ext}_page{idx + 1}.png",
                )
            ),
        )
        for idx in range(len(pdf_images))
    ]
    prompt = [{"role": "user", "content": []}]

    for index, (media_type, data) in enumerate(images, start=1):
        prompt[0]["content"].append({"type": "text", "text": f"Image {index}:"})

        prompt[0]["content"].append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
            }
        )

    prompt[0]["content"].append({"type": "text", "text": prompt_text})
    return prompt
