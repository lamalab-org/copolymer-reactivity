import json
import os
from pdf2image import convert_from_path
import time
from openai import OpenAI
import copolextractor.prompter as prompter
import copolextractor.image_processer as ip


input_folder = "./pdfs"
output_folder_images = "./images"
output_folder = "model_output_rxn_count"
number_of_model_calls = 2

selected_entries_path = 'selected_entries.json'
with open(selected_entries_path, 'r', encoding='utf-8') as file:
    selected_entries = json.load(file)

#selected_pdfs = [entry['out'] for entry in selected_entries if os.path.exists(os.path.join(input_folder, os.path.basename(entry['out'])))]

input_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".pdf")])

#input_files = sorted([os.path.basename(f) for f in selected_pdfs if f.endswith(".pdf")])
print(input_files)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
total_input_tokens = 0
total_output_token = 0
number_of_calls = 0

output_file_path = os.path.join(os.getcwd(), "enhanced_doi_list.json")
enhanced_doi_list = json.load(open(output_file_path))
prompt_text = prompter.get_prompt_pdf_quality()

start = time.time()


for i, filename in enumerate(input_files):

    #item = next((item for item in enhanced_doi_list if item['out'].endswith(filename)), None)
    item = next((item for item in enhanced_doi_list if 'pdf_name' in item and item['pdf_name'] == filename), None)

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
    api_response = json.loads(output)
    print("API Response Parsed: ", api_response)

    output_json_path = os.path.join(output_folder, filename.replace('.pdf', '.json'))
    with open(output_json_path, 'w') as json_file:
        json.dump(api_response, json_file, indent=4)

    paper_name = filename
    for item in enhanced_doi_list:
        if item['out'].endswith(paper_name):
            item['pdf_quality'] = api_response['pdf_quality']
            item['table_quality'] = api_response['table_quality']
            item['quality_of_number'] = api_response['quality_of_numbers']
            item['year'] = api_response['year']
            item['processed_quality'] = True
            break

    print("input tokens used: ", total_input_tokens)
    print("output tokens used: ", total_output_token)
    print("total number of model call: ", number_of_calls)
    with open(output_file_path, 'w') as file:
        json.dump(enhanced_doi_list, file, indent=4)

with open(output_file_path, 'w') as file:
    json.dump(enhanced_doi_list, file, indent=4)

end = time.time()
execution_time = end - start

print("execution time: ", execution_time)