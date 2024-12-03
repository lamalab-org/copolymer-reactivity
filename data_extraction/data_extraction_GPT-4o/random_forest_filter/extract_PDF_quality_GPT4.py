import json
import os
import time
from pdf2image import convert_from_path
from openai import OpenAI
import copolextractor.prompter as prompter
import copolextractor.image_processer as ip


def process_pdfs(input_folder, output_folder_images, output_folder, selected_entries_path):
    # Load the selected entries from the unique list
    with open(selected_entries_path, 'r', encoding='utf-8') as file:
        selected_entries = json.load(file)

    # Get the list of PDFs to process
    selected_pdfs = [
        entry['out'] for entry in selected_entries
        if os.path.exists(os.path.join(input_folder, os.path.basename(entry['out'])))
    ]
    input_files = sorted([os.path.basename(f) for f in selected_pdfs if f.endswith(".pdf")])
    print("Input files:", input_files)
    print('Number of input files:', len(input_files))

    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    total_input_tokens = 1011486
    total_output_tokens = 4518
    number_of_calls = 107

    # Load the enhanced DOI list
    with open(selected_entries_path, 'r', encoding='utf-8') as file:
        enhanced_doi_list = json.load(file)

    # Generate the base prompt text
    prompt_text = prompter.get_prompt_pdf_quality()

    start_time = time.time()

    for i, filename in enumerate(input_files):
        item = next((item for item in enhanced_doi_list if 'out' in item and item['out'].endswith(filename)), None)

        if item is None or item.get('processed_quality'):
            print(f"Skipping {filename} as it is already processed.")
            continue

        file_path = os.path.join(input_folder, filename)
        print("Processing:", filename)

        pdf_images = convert_from_path(file_path)
        images_base64 = [
            ip.process_image(image, 2048, output_folder_images, file_path, j)[0]
            for j, image in enumerate(pdf_images)
        ]

        content = prompter.get_prompt_vision_model(images_base64, prompt_text)

        print("Model call starts")
        output, input_token, output_token = prompter.call_openai(prompt=content)
        total_input_tokens += input_token
        total_output_tokens += output_token
        number_of_calls += 1

        api_response = json.loads(output)
        print("API Response Parsed:", api_response)

        output_json_path = os.path.join(output_folder, filename.replace('.pdf', '.json'))
        with open(output_json_path, 'w') as json_file:
            json.dump(api_response, json_file, indent=4)

        # Update the enhanced DOI list
        for item in enhanced_doi_list:
            if item['out'].endswith(filename):
                item['pdf_quality'] = api_response.get('pdf_quality')
                item['table_quality'] = api_response.get('table_quality')
                item['quality_of_number'] = api_response.get('quality_of_numbers')
                item['year'] = api_response.get('year')
                item['processed_quality'] = True
                item['language'] = api_response.get('language')
                item['rxn_count'] = api_response.get('number_of_reactions')
                break

        print("Input tokens used:", total_input_tokens)
        print("Output tokens used:", total_output_tokens)
        print("Total number of model calls:", number_of_calls)

        # Save the updated enhanced DOI list
        with open(selected_entries_path, 'w') as file:
            json.dump(enhanced_doi_list, file, indent=4)

    end_time = time.time()
    print("Execution time:", end_time - start_time)


def main(input_folder, output_folder_images, output_folder, selected_entries_path):

    # Ensure output directories exist
    os.makedirs(output_folder_images, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Process the PDFs
    process_pdfs(input_folder, output_folder_images, output_folder, selected_entries_path)


if __name__ == "__main__":

    # Define input and output folders
    input_folder = "./PDF"
    output_folder_images = "./processed_images"
    output_folder = "./model_output_score"
    selected_entries_path = "../collected_data/enhanced_doi_list_unique.json"

    main(input_folder, output_folder_images, output_folder, selected_entries_path)
