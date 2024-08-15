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

input_folder = "../pdf_testset"
output_folder_images = "./processed_images"
output_folder = "./model_output_GPT4-o"

# Load the selected entries from the unique list
#selected_entries_path = './data_extraction_GPT-4o/enhanced_doi_list_unique.json'
#with open(selected_entries_path, 'r', encoding='utf-8') as file:
    #selected_entries = json.load(file)

# Get the list of PDFs to process
#selected_pdfs = [entry['out'] for entry in selected_entries if os.path.exists(os.path.join(input_folder, os.path.basename(entry['out'])))]

#input_files = sorted([os.path.basename(f) for f in selected_pdfs if f.endswith(".pdf")])
input_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".pdf")])
print(input_files)
print('number of input files:', len(input_files))

number_of_model_calls = 2
input_files = sorted([f for f in input_files if f.endswith(".pdf")])
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
total_input_tokens = 0
total_output_token = 0
number_of_calls = 0

prompt_text = prompter.get_prompt_template()

for i, filename in enumerate(input_files):

    # Skip the file if already extracted or if precision_score is not 1
    #entry = next((item for item in selected_entries if os.path.basename(item['out']) == filename), None)
    #if entry:
        #if entry.get('extracted') or entry.get('precision_score') != 1:
            #print(f"Skipping {filename} as it is already extracted or precision_score is not 1")
            #continue

    #file_path = os.path.join('./data_extraction_GPT-4o', 'PDF', os.path.basename(filename))
    #file_path = os.path.join('./data_extraction_GPT-4o/PDF', filename)
    file_path = os.path.join('../model_evaluation/pdf_testset', filename)
    print("processing ", filename)
    pdf_images = convert_from_path(file_path)

    images_base64 = [ip.process_image(image, 2048, output_folder_images, file_path, j)[0] for j, image in enumerate(pdf_images)]

    content = prompter.get_prompt_vision_model(images_base64, prompt_text)

    print("model call starts")
    output, input_token, output_token = prompter.call_openai(prompt=content)
    print(output)
    total_input_tokens += input_token
    total_output_token += output_token
    number_of_calls += 1
    output_filename = os.path.basename(filename)
    output_name_json = os.path.join(output_folder, output_filename.replace('.pdf', '.json'))
    output_name_yaml = os.path.join(output_folder, output_filename.replace('.pdf', '.yaml'))

    #with open(output_name_json, "w", encoding="utf-8") as json_file:
       # json.dump(output, json_file, ensure_ascii=False, indent=4)
    #print("Output saved as JSON-file.")

    #with open(output_name_yaml, "w", encoding="utf-8") as yaml_file:
        #yaml.dump(output, yaml_file, allow_unicode=True, default_flow_style=False)
    #print("Output saved as YAML-file.")
    output_model = prompter.format_output_as_json_and_yaml(i, output, output_folder, filename)

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
                json.dump(output, json_file, ensure_ascii=True, indent=4)
            print("output saved as JSON-file.")
            with open(output_name_yaml, "w") as yaml_file:
                yaml.dump(output, yaml_file, allow_unicode=True, default_flow_style=False)
            print("output_model: ", output)
        else:
            print("NA-rate under 30%")

    print("input tokens used: ", total_input_tokens)
    print("output tokens used: ", total_output_token)
    print("total number of model call: ", number_of_calls)

    # Mark the entry as extracted
    #entry['extracted'] = True
    #with open(selected_entries_path, 'w', encoding='utf-8') as file:
        #json.dump(selected_entries, file, indent=4)

# Save the updated selected entries back to the file
#with open(selected_entries_path, 'w', encoding='utf-8') as file:
    #json.dump(selected_entries, file, indent=4)

end = time.time()
execution_time = end - start

print("execution time: ", execution_time)
