import time

from openai import OpenAI
import json
from typing import List


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
        print("Run Satus: {run.status}")
        time.sleep(1)
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
    old_data_template = f"""Here are the previously collected data: {new_data}. Please add more information based on"""
    return old_data_template


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
