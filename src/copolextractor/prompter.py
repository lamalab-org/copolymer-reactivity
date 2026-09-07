import base64
import json
import os
import time
from typing import List, Tuple, Union

import anthropic
import yaml
from openai import OpenAI


class RunTimeExpired(Exception):
    "Raised when the run status of the open-ai call is expired"

    pass


def get_prompt_pdf_quality() -> str:
    """Return the LLM prompt used to rate a paper's PDF quality for data extraction.

    Returns:
        The prompt text, asking the model to score PDF/table/number quality and
        report year, reaction count and language as a JSON object.
    """
    prompt = """Question: The content of the pictures is a scientific paper about copolymerization of monomers.
    The main focus here is to find the copolymerizations which have r-values for a pair of two Monomers.
    Its possible, that there is also the beginning of a new paper about polymers in the PDF.
    Ignore these. Rate the quality of the provided paper form 0 (hard to extract data) to 10 (easy to extract data) in terms of readability and easiness of data extraction.
    In each paper there could be multiple different reaction with different pairs of monomers and same reactions with different reaction conditions.
    Count each different reaction with an r-value as one. Ignore copolymerization with reference to previse work.
    Just count copolymerizations with r-values which are carried out in the article.
    Stick to the given output_2 datatype (string, integer or float). json:
    {
        "pdf_quality": the quality of the provided PDF document in terms of e.g. resolution and easiness of data extraction form 0 (hard to extract) to 10 (easy to extract) (FLOAT),
        "table_quality": the quality and structuredness of the tables in the PDF document in terms of e.g. easiness of data extraction and clearity form 0 (hard to extract) to 10 (easy to extract) (FLOAT),
        "quality_of_numbers": the readability of the numbers in the PDF document form 0 (hard to extract) to 10 (easy to extract) (FLOAT),
        "year": the year of the publication (INTEGER),
        "number_of_reactions": number of the different relevant copolymerizations with r-values in the paper (FLOAT),
        "language": language of the main text (STRING),
    }
    Possible Answer (including comments in the brackets):
    {
        "pdf_quality": 4 (resolution is poor but still readable),
        "table_quality": 5 (basic structure provided; no clear lines in between; line slip possible),
        "quality_of_number": 7 (numbers are readable),
        "year": 1957,
        "number_of_reactions": 20 (20 different copolymerization reaction with r-value in this paper),
        "language": english,
    }
    Question: Please reason concisely about the quality of the given categorys in terms of how likely you would make mistakes in extracting data for this article.
    Then provide your answer in the given json format. Please be really realistic about your score and try to give a calibrated estimate.
    """
    return prompt


def get_prompt_template() -> str:
    """Return the main LLM extraction prompt describing the reaction JSON schema.

    Returns:
        The prompt text instructing the model to extract copolymerization reactions,
        conditions and reaction constants from a PDF into the documented JSON schema.
    """
    prompt = """The content of the pictures is a scientific paper about copolymerization of monomers.
    We only consider copolymerizations with 2 different monomers. If you find a polymerization with just one or more than 2 monomers ignore them.
    Its possible, that there is also the beginning of a new paper about polymers in the PDF.
    Ignore these. In each paper there could be multiple different reaction with different pairs of monomers and same reactions with different reaction conditions.
    The reaction constants for the copolymerization with the monomer pair is the most important information. Be careful with numbers and do not miss the decimal points.
    If there are polymerization's without these constants, ignore these.
    From the PDF, extract the polymerization information from each polymerization and report it in valid json format.
    Also pay attention to the caption of figures.
    Don't use any abbreviations, always use the whole word.
    Be careful with the sequenz of the monomers and reaction constants. The monomer 1 should belong to r-value 1. t6zt
    Try to keep the string short. Exclude comments out of the json output_2. Return one json object.
    Stick to the given output_2 datatype (string, or float).

    Extract the following information:

    reactions: [
        {
            "monomers": [
                "monomer1" (monomer assigned to reaction constant 1),
                "monomer2" (monomer assigned to reaction constant 2)
                ] as STRING (only the whole Monomer name without abbreviation)
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
                    Q-value": {another reaction value provided in some articles as FLOAT
                        "constant_1":
                        "constant_2": },
                    "e-Value": {another reaction value provided in some articles as FLOAT
                        "constant_1":
                        "constant_2": },
                    "r_product": the product of r1 and r2 sometimes provided in articles, do not calculate this, if its provided extracted it, if not put null
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


    If the information is not provided put 'na'.
    If there are multiple polymerization's with different parameters report as a separate reaction (for different pairs of monomers) and reaction_conditions (for different reaction conditions of the same monomers)."""
    return prompt


def get_prompt_addition() -> str:
    """Return the prompt fragment used to ask the model to refine previously extracted data.

    Returns:
        A template string with a single `{}` placeholder for the prior extraction data.
    """
    prompt_addition = """Here is the previously collected data from the same Markdowns: {}.
Try to fill up the entries with NA and correct entries if they are wrong. Pay particular attention on numbers and at the decimal point.
Combine different reaction if they belong to the same polymerization with the same reaction conditions.
Report every different polymerization and every different reaction condition separately. Do this based on this prompt:"""
    return prompt_addition


def get_prompt_addition_with_data(new_data: dict) -> str:
    """Fill the refinement prompt template (`get_prompt_addition`) with prior extraction data.

    Args:
        new_data: Previously extracted data to embed in the prompt.

    Returns:
        The refinement prompt with `new_data` inserted.
    """
    prompt_addition_base = get_prompt_addition()
    prompt_addition_with_data = prompt_addition_base.format(new_data)
    return prompt_addition_with_data


def split_document(document: str, max_length: int) -> List[str]:
    """Split a document into fixed-size, non-overlapping character chunks.

    Args:
        document: Full text to split.
        max_length: Maximum number of characters per chunk.

    Returns:
        The list of chunks, in order.
    """
    return [document[j : j + max_length] for j in range(0, len(document), max_length)]


def format_prompt(template: str, data: dict) -> str:
    """Fill a prompt template's named placeholders with values from `data`.

    Args:
        template: Prompt template using `str.format`-style `{key}` placeholders.
        data: Mapping of placeholder names to values.

    Returns:
        The formatted prompt string.
    """
    return template.format(**data)


def call_openai(
    prompt: Union[str, list], model: str = "chatgpt-4o-latest", temperature: float = 0.0, **kwargs
) -> Tuple[str, int, int]:
    """Call chat openai model

    Args:
        prompt (Union[str, list]): Prompt to send to model. Either plain text, or a
            list of vision message content blocks (see `get_prompt_vision_model`).
        model (str, optional): Name of the API. Defaults to ""gpt-4-vision-preview".
        temperature (float, optional): inference temperature. Defaults to 0.

    Returns:
        A (message_content, input_tokens, output_tokens) tuple, where
        `message_content` is the raw JSON string returned by the model.
    """
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a scientific assistant, extracting important information about polymerization conditions"
                "out of PDFs in valid json format. Extract just data which you are 100% confident about the "
                "accuracy. Keep the entries short without details. Be careful with numbers.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        seed=12345,
        response_format={"type": "json_object"},
        **kwargs,
    )
    input_tokens = completion.usage.prompt_tokens
    output_token = completion.usage.completion_tokens
    message_content = completion.choices[0].message.content
    return message_content, input_tokens, output_token


def call_openai_chucked(
    prompt: str, model: str = "gpt-3.5-turbo-1106", temperature: float = 0.0, **kwargs
) -> Tuple[dict, int, int]:
    """Call chat openai model

    Args:
        prompt (str): Prompt to send to model
        model (str, optional): Name of the API. Defaults to "gpt-3.5-turbo-1106".
        temperature (float, optional): inference temperature. Defaults to 0.

    Returns:
        A (new_data, input_tokens, output_tokens) tuple, where `new_data` is the
        model's response parsed from JSON.
    """
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a scientific assistant, extracting important information about polymerization conditions"
                "out of PDFs in valid json format. Extract just data which you are 100% confident about the "
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


def call_openai_agent(assistant, file, prompt: str, **kwargs) -> Tuple[str, int, int]:
    """Run a prompt through an OpenAI Assistants-API thread and poll until completion.

    Args:
        assistant: OpenAI assistant object (as returned by the Assistants API) to run.
        file: Currently unused by this function; kept for call-site compatibility.
        prompt: User message content to send to the assistant.

    Raises:
        RunTimeExpired: If the run status becomes "expired" or "failed".

    Returns:
        A (output_text, input_tokens, output_tokens) tuple with the assistant's
        latest reply and token usage for the run.
    """
    print("openai call has started")
    client = OpenAI()
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(thread_id=thread.id, role="user", content=prompt)
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)
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
    """Build the refinement prompt fragment for a round of previously extracted data.

    Args:
        new_data: Previously extracted data to embed in the prompt.

    Returns:
        The refinement prompt fragment with `new_data` inserted.
    """
    old_data_template = get_prompt_addition_with_data(new_data)
    return old_data_template


def update_prompt(prompt: str, data: dict) -> str:
    """Prefix a base prompt with a refinement fragment built from prior extraction data.

    Args:
        prompt: Base prompt to append after the refinement fragment.
        data: Previously extracted data to embed in the refinement fragment.

    Returns:
        The combined prompt (refinement fragment followed by `prompt`).
    """
    new_prompt = update_data(data) + prompt
    return new_prompt


def update_prompt_chucked(prompt: str, data: dict) -> str:
    """Prefix a base prompt with the chunked-mode refinement fragment.

    Args:
        prompt: Base prompt to append after the refinement fragment.
        data: Previously extracted data to embed in the refinement fragment.

    Returns:
        The combined prompt (refinement fragment followed by `prompt`).
    """
    new_prompt = update_data_chucked(data) + prompt
    return new_prompt


def update_data_chucked(new_data: dict) -> str:
    """Build the chunked-mode refinement prompt fragment for prior extraction data.

    Args:
        new_data: Previously extracted data to embed in the prompt.

    Returns:
        The refinement prompt fragment with `new_data` inserted.
    """
    old_data_template = f"""Here are the previously collected data: {new_data}. Please add more information based on"""
    return old_data_template


def repeated_call_model(
    text: str, prompt_template: str, max_length: int, model_call_fn
) -> Tuple[dict, int, int, int]:
    """Extract data from a long document by prompting the model chunk-by-chunk.

    Each chunk's result is merged into a running output dict and fed back into the
    prompt for the next chunk via `update_data`, so later chunks can refine earlier
    extractions.

    Args:
        text: Full document text to process.
        prompt_template: Prompt template with a `{text}` placeholder for the chunk.
        max_length: Maximum number of characters per chunk (see `split_document`).
        model_call_fn: Callable taking a prompt string and returning
            (new_data, input_tokens, output_tokens).

    Returns:
        A (output, input_tokens, output_tokens, number_of_model_calls) tuple, where
        `output` is the merged extraction result and the token counts are only from
        the last chunk call (accumulators `input_tokens`/`output_tokens` are tracked
        but not returned).
    """
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
    i: int, output: str, output_folder: str, pdf_name: str
) -> Union[dict, None]:
    """Parse a fenced-code-block model response and persist it as JSON and YAML.

    Args:
        i: Zero-based index used to derive the output filenames.
        output: Raw model response, expected to contain the JSON payload inside a
            ``` fenced code block (optionally prefixed with "json\\n").
        output_folder: Directory the JSON/YAML files are written to.
        pdf_name: Source PDF filename, stored under the "source_pdf" key.

    Returns:
        The parsed data dict (with "source_pdf" added), or None if the response
        could not be parsed as JSON.
    """
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
        json_data["source_pdf"] = pdf_name

        with open(output_name_json, "w", encoding="utf-8") as json_file:
            json.dump(json_data, json_file, ensure_ascii=False, indent=4)
        print("output_2 saved as JSON-file.")
        with open(output_name_yaml, "w") as yaml_file:
            yaml.dump(json_data, yaml_file, allow_unicode=True)
        return json_data
    except json.JSONDecodeError as e:
        print("error at parsing the output_2 to JSON-file:", e)


def format_output_claude_as_json_and_yaml(
    i: int, content_blocks: list, output_folder: str
) -> Union[dict, None]:
    """Parse the first Claude content block containing JSON text and persist it.

    Args:
        i: Zero-based index used to derive the output filenames.
        content_blocks: Anthropic message content blocks; the first block exposing
            a "text" attribute or key is used.
        output_folder: Directory the JSON/YAML files are written to.

    Returns:
        The parsed data dict, or None if no usable text block was found or parsing
        failed.
    """
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
            print(f"Error parsing the output_2: {e}")


def call_claude3(prompt: list) -> Tuple[list, int, int]:
    """Call the Claude 3 Opus chat model with a pre-built message list.

    Args:
        prompt: List of Anthropic message dicts (as accepted by `messages.create`).

    Returns:
        A (content_blocks, input_tokens, output_tokens) tuple with the model's
        response content blocks and token usage.
    """
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        system="You are a scientific assistant, extracting important information about polymerization conditions"
        "out of PDFs in valid json format. Extract just data which you are 100% confident about the "
        "accuracy. Keep the entries short without details. Be careful with numbers.",
        temperature=0.0,
        messages=prompt,
    )
    input_token = message.usage.input_tokens
    output_token = message.usage.output_tokens
    print(message.content)
    return message.content, input_token, output_token


def update_prompt_with_text_and_images(original_prompt: list, data: dict, prompt: str) -> str:
    """Replace the trailing text block of a Claude vision prompt with a refined prompt.

    Args:
        original_prompt: Claude message list (as built by `get_prompt_claude_vision`)
            whose last content block's "text" value is replaced in place.
        data: Previously extracted data to embed via `update_prompt`.
        prompt: Base prompt text to combine with the refinement fragment.

    Returns:
        The updated `original_prompt` serialized as a JSON string.
    """
    new_text = update_prompt(prompt, data)
    original_prompt[-1]["text"] = new_text
    updated_prompt_str = json.dumps(original_prompt)
    print(f"updated_prompt_str after update: {updated_prompt_str}")
    return updated_prompt_str


def create_image_content(image: str, detail: str = "high") -> dict:
    """Build an OpenAI vision message content block for a base64-encoded image.

    Args:
        image: Base64-encoded JPEG image data.
        detail: OpenAI image detail level ("low", "high" or "auto").

    Returns:
        An "image_url" content block as expected by the OpenAI chat completions API.
    """
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": detail},
    }


def get_prompt_vision_model(images_base64: List[str], prompt_text: str) -> list:
    """Build the OpenAI vision message content list from images and prompt text.

    Args:
        images_base64: Base64-encoded JPEG images to include, in order.
        prompt_text: Text prompt appended after the images.

    Returns:
        A list of content blocks (images followed by the text block) as expected by
        the OpenAI chat completions API.
    """
    content = []
    for data in images_base64:
        content.append(create_image_content(data))

    content.append({"type": "text", "text": prompt_text})
    return content


def encode_image_to_base64(filepath: str) -> str:
    """Read an image file and return its base64-encoded content.

    Args:
        filepath: Path to the image file.

    Returns:
        The base64-encoded image data as a UTF-8 string.
    """
    with open(filepath, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_prompt_claude_vision(
    output_folder_images: str, filename: str, pdf_images: list, prompt_text: str
) -> list:
    """Build a Claude vision message list from previously saved, deskewed page images.

    Args:
        output_folder_images: Directory containing the deskewed page PNGs produced by
            `image_processer.correct_text_orientation`.
        filename: Original PDF filename, used to derive the page image filenames.
        pdf_images: List of page images; only its length is used to determine how
            many page images to load.
        prompt_text: Text prompt appended after the images.

    Returns:
        A single-element list containing one Anthropic user message dict, whose
        content alternates "Image N:" labels with base64-encoded image blocks,
        followed by the text prompt.
    """
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
