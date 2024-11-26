from pdf2image import convert_from_path
import os
from openai import OpenAI
import copolextractor.prompter as prompter
import copolextractor.image_processer as ip
import time
import json

start = time.time()

input_folder = "./PDF"
output_folder_images = "./processed_images"
output_folder = "model_output_rxn_count"
number_of_model_calls = 2
input_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".pdf")])
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
total_input_tokens = 1232390
total_output_token = 6850
number_of_calls = 120

output_file_path = os.path.join(os.getcwd(), "enhanced_doi_list1.json")
enhanced_doi_list = json.load(open(output_file_path))
prompt_text = prompter.get_prompt_rxn_number()

for i, filename in enumerate(input_files):

    item = next((item for item in enhanced_doi_list if item['out'].endswith(filename)), None)

    if item is None:
        print("No match found in enhanced_doi_list for ", filename)
        continue

    if 'processed' not in item:
        item['processed'] = False

    if item['processed']:
        print("Skipping already processed file: ", filename)
        continue

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
            item['name'] = api_response['paper name']
            item['language'] = api_response['language']
            item['rxn_number'] = api_response['number of reactions']
            item['additional_sources'] = api_response['additional sources']
            item['processed'] = True
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
