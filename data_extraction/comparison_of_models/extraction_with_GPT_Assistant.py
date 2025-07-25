from openai import OpenAI
import os
import json
import copolextractor.prompter as prompter
import copolextractor.analyzer as az
import langchain
from dotenv import load_dotenv
from langchain.cache import SQLiteCache
import random
import time


start = time.time()

load_dotenv()
langchain.llm_cache = SQLiteCache(database_path="../.langchain.db")
llm = OpenAI()

input_folder = "../pdf_testset"
output_folder = "model_output_assistant"
input_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".pdf")])
# random_pdf_selection = random.sample(input_files, 20) enumerate(random_pdf_selection)
# print("random selection of PDFs: ", random_pdf_selection)

max_section_length = 16385
model = "gpt-4-1106-preview"
number_of_model_calls = 2
parsing_error = 0
run_time_expired_error = 0
total_input_tokens = 0
total_output_token = 0
number_of_calls = 0

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


for i, filename in enumerate(input_files):
    file_path = os.path.join(input_folder, filename)
    print(file_path)
    print(filename)
    file = client.files.create(file=open(file_path, "rb"), purpose="assistants")
    assistant = client.beta.assistants.create(
        instructions="You are a scientific assistant, extracting important information about polymerization conditions"
        "out of pdf_testset in valid json format. Extract just data which you are 100% confident about the "
        "accuracy. Keep the entries short without details. Be careful with numbers.",
        model="gpt-4-turbo-preview",
        tools=[{"type": "retrieval"}],
        name="Extractor",
        file_ids=[file.id],
    )
    prompt_template = prompter.get_prompt_template()
    try:
        output, input_token, output_token = prompter.call_openai_agent(
            assistant, file, prompt_template
        )
        total_input_tokens += input_token
        total_output_token += output_token
        number_of_calls += 1
    except prompter.RunTimeExpired:
        run_time_expired_error += 1
        continue

    # format output_2 and convert into json and yaml file
    try:
        output_model = prompter.format_output_as_json_and_yaml(i, output, output_folder)
    except json.JSONDecodeError as e:
        parsing_error += 1
        print(f"json format of output_2 of {filename} is not valid")
        continue

    # if there are more than 30 % "NA" entries it calls again the model (max 2 times)
    for a in range(number_of_model_calls):
        na_count = az.count_na_values(output_model)
        total_entry_count = az.count_total_entries(output_model)
        rate = az.calculate_rate(na_count, total_entry_count)
        if rate > 0.3:
            print(f"model call number {a+2} of {filename}")
            try:
                prompt = prompter.update_prompt(prompt_template, output_model)
            except prompter.RunTimeExpired:
                run_time_expired_error += 1
                continue

            output, input_tokens, output_token = prompter.call_openai_agent(
                assistant, file, prompt
            )
            total_input_tokens += input_token
            total_output_token += output_token
            number_of_calls += 1
            output_model = prompter.format_output_as_json_and_yaml(
                i, output, output_folder
            )
    print("input tokens used: ", total_input_tokens)
    print("output_2 tokens used: ", total_output_token)
    print("total number of model call: ", number_of_calls)
print("parsing error:", parsing_error)

end = time.time()
execution_time = start - end

print("execution time: ", execution_time)
