from pdf2image import convert_from_path
import os
from openai import OpenAI
import copolextractor.prompter as prompter
import copolextractor.analyzer as az
import copolextractor.image_processer as ip
import time
import random

start = time.time()

input_folder = "../pdfs"
output_folder_images = "./images"
output_folder = "model_output"
number_of_model_calls = 2
input_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".pdf")])
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
total_input_tokens = 0
total_output_token = 0
number_of_calls = 0

#random_pdf_selection = random.sample(input_files, 20)
#print("random selection of PDFs: ", random_pdf_selection)

prompt_text = prompter.get_prompt_template()


for i, filename in enumerate(input_files):
    file_path = os.path.join(input_folder, filename)
    print("processing ", filename)
    pdf_images = convert_from_path(file_path)

    images_base64 = [ip.process_image(image, 2048, output_folder_images, file_path, j)[0] for j, image in enumerate(pdf_images)]

    content = prompter.get_prompt_vision_model(images_base64, prompt_text)

    print("model call starts")
    output, input_token, output_token = prompter.call_openai(prompt=content)
    total_input_tokens += input_token
    total_output_token += output_token
    number_of_calls += 1
    output_model = prompter.format_output_as_json_and_yaml(i, output, output_folder, filename)
    print("output_model: ", output_model)

    for a in range(number_of_model_calls):
        na_count = az.count_na_values(output_model)
        total_entry_count = az.count_total_entries(output_model)
        rate = az.calculate_rate(na_count, total_entry_count)
        print("NA-rate: ", rate)
        if rate > 0.3 or output_model is None:
            print(f"model call number {a+2} of {filename}")
            updated_prompt = prompter.update_prompt(prompt_text, output_model)
            content = prompter.get_prompt_vision_model(images_base64, updated_prompt)
            output, input_token, output_token = prompter.call_openai(content)
            total_input_tokens += input_token
            total_output_token += output_token
            number_of_calls += 1
            output_model = prompter.format_output_as_json_and_yaml(i, output, output_folder, filename)
        else:
            print("NA-rate under 30%")

    print("input tokens used: ", total_input_tokens)
    print("output tokens used: ", total_output_token)
    print("total number of model call: ", number_of_calls)

end = time.time()
execution_time = start-end

print("execution time: ", execution_time)
