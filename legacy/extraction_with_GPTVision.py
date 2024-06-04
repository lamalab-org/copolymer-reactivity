from pdf2image import convert_from_path
import os
from openai import OpenAI
import copolextractor.prompter as prompter
import copolextractor.analyzer as az
import copolextractor.image_processer as ip
import time
import json
import yaml


start = time.time()

input_folder = "./PDF_from_scidownl/pdfs"
output_folder_images = "./images"
output_folder = "./PDF_from_scidownl/model_output_extraction"

#selected_entries_path = './PDF_from_scidownl/selected_entries.json'
#with open(selected_entries_path, 'r', encoding='utf-8') as file:
    #selected_entries = json.load(file)

#selected_pdfs = [entry['out'] for entry in selected_entries if os.path.exists(os.path.join(input_folder, os.path.basename(entry['out'])))]
input_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".pdf")])

number_of_model_calls = 2
input_files = sorted([f for f in input_files if f.endswith(".pdf")])
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
print(os.environ.get("OPENAI_API_KEY"))
total_input_tokens = 0
total_output_token = 0
number_of_calls = 0

prompt_text = prompter.get_prompt_template()

for i, filename in enumerate(input_files):

    #file_path = os.path.join('./PDF_from_scidownl', 'PDF', os.path.basename(filename))
    file_path = os.path.join('./PDF_from_scidownl/pdfs', filename)
    print("processing ", filename)
    pdf_images = convert_from_path(file_path)

    images_base64 = [ip.process_image(image, 2048, output_folder_images, file_path, j)[0] for j, image in enumerate(pdf_images)]

    content = prompter.get_prompt_vision_model(images_base64, prompt_text)

    print("model call starts")
    output, input_token, output_token = prompter.call_openai(prompt=content)
    total_input_tokens += input_token
    total_output_token += output_token
    number_of_calls += 1
    output_filename = os.path.basename(filename)
    output_name_json = os.path.join(output_folder, output_filename.replace('.pdf', '.json'))
    output_name_yaml = os.path.join(output_folder, output_filename.replace('.pdf', '.yaml'))

    with open(output_name_json, "w", encoding="utf-8") as json_file:
        json.dump(output, json_file, ensure_ascii=True, indent=4)
    print("output saved as JSON-file.")
    with open(output_name_yaml, "w") as yaml_file:
        yaml.dump(output, yaml_file, allow_unicode=True)
    #output_model = prompter.format_output_as_json_and_yaml(i, output, output_folder, filename)
    print("output_model: ", output)

    for a in range(number_of_model_calls):
        na_count = az.count_na_values(output)
        total_entry_count = az.count_total_entries(output)
        rate = az.calculate_rate(na_count, total_entry_count)
        print("NA-rate: ", rate)
        if rate > 0.3 or output is None:
            print(f"model call number {a+2} of {filename}")
            updated_prompt = prompter.update_prompt(prompt_text, output)
            content = prompter.get_prompt_vision_model(images_base64, updated_prompt)
            output, input_token, output_token = prompter.call_openai(content)
            total_input_tokens += input_token
            total_output_token += output_token
            number_of_calls += 1
            #output_model = prompter.format_output_as_json_and_yaml(i, output, output_folder, filename)
            with open(output_name_json, "w", encoding="utf-8") as json_file:
                json.dump(output, json_file, ensure_ascii=False, indent=4)
            print("output saved as JSON-file.")
            with open(output_name_yaml, "w") as yaml_file:
                yaml.dump(output, yaml_file, allow_unicode=True)
            print("output_model: ", output)
        else:
            print("NA-rate under 30%")

    print("input tokens used: ", total_input_tokens)
    print("output tokens used: ", total_output_token)
    print("total number of model call: ", number_of_calls)

    #for item in selected_entries:
        ##if item['out'].endswith(filename):
            #item['extracted'] = True
    #with open(selected_entries_path, 'w') as file:
       # json.dump(selected_entries, file, indent=4)

end = time.time()
execution_time = start-end

print("execution time: ", execution_time)
