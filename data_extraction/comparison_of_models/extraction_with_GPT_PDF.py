import os
import time
import json
import yaml
from pdf2image import convert_from_path
import copolextractor.prompter as prompter
import copolextractor.analyzer as az
import copolextractor.image_processer as ip


def process_pdf_files(paper_list_path, output_folder_images, output_folder, number_of_model_calls):
    """
    Process PDF files based on entries from paper_list.json and update the JSON file with extraction results.
    """
    start = time.time()

    # Load paper list JSON
    with open(paper_list_path, 'r', encoding='utf-8') as file:
        paper_list = json.load(file)

    # Filter for entries with "precision_score": 1
    selected_papers = [
        entry for entry in paper_list if entry.get("precision_score") == 1 and not entry.get("extracted")
    ]
    print(f"Number of PDFs to process: {len(selected_papers)}")

    # Token and model call counters
    total_input_tokens = 0
    total_output_tokens = 0
    number_of_calls = 0

    # Load prompt template
    prompt_text = prompter.get_prompt_template()

    for i, paper in enumerate(selected_papers):
        filename = paper['pdf_name']
        file_path = os.path.join("pdf_testset", filename)

        if not os.path.exists(file_path):
            print(f"Skipping {filename}: File not found.")
            continue

        print(f"Processing {filename}")

        # Convert PDF to images
        pdf_images = convert_from_path(file_path)
        images_base64 = [ip.process_image(image, 2048, output_folder_images, file_path, j)[0] for j, image in enumerate(pdf_images)]

        # Generate initial prompt
        content = prompter.get_prompt_vision_model(images_base64, prompt_text)

        # Call the model and process output
        print("Model call starts")
        output, input_token, output_token = prompter.call_openai(prompt=content)
        total_input_tokens += input_token
        total_output_tokens += output_token
        number_of_calls += 1

        # Save output as JSON and YAML
        output_name_json = os.path.join(output_folder, filename.replace('.pdf', '.json'))
        output_name_yaml = os.path.join(output_folder, filename.replace('.pdf', '.yaml'))

        for attempt in range(number_of_model_calls):
            na_count = az.count_na_values(output)
            total_entry_count = az.count_total_entries(output)
            na_rate = az.calculate_rate(na_count, total_entry_count)
            print(f"NA-rate: {na_rate}")

            if na_rate > 0.3 or output is None:
                print(f"Retrying model call {attempt + 2} for {filename}")
                updated_prompt = prompter.update_prompt(prompt_text, output)
                content = prompter.get_prompt_vision_model(images_base64, updated_prompt)
                output, input_token, output_token = prompter.call_openai(content)
                total_input_tokens += input_token
                total_output_tokens += output_token
                number_of_calls += 1

                # Save updated output
                with open(output_name_json, "w", encoding="utf-8") as json_file:
                    json.dump(output, json_file, ensure_ascii=True, indent=4)
                with open(output_name_yaml, "w", encoding="utf-8") as yaml_file:
                    yaml.dump(output, yaml_file, allow_unicode=True, default_flow_style=False)
            else:
                print("NA-rate below 30%, no further retries needed.")
                break

        # Update the JSON entry
        paper.update({
            "extracted": True,
            "extracted_data": output
        })

    # Save the updated JSON file
    extracted_json_path = "data_extracted.json"
    with open(extracted_json_path, 'w', encoding='utf-8') as file:
        json.dump(paper_list, file, indent=4)

    print("Total input tokens used:", total_input_tokens)
    print("Total output tokens used:", total_output_tokens)
    print("Total number of model calls:", number_of_calls)

    end = time.time()
    print("Execution time:", end - start)


def main(input_folder_images, output_folder, paper_list_path):
    """
    Main function to process PDFs and update JSON entries.
    """
    # Number of retries for model calls
    number_of_model_calls = 2

    # Ensure output folders exist
    os.makedirs(input_folder_images, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Process PDF files
    process_pdf_files(paper_list_path, input_folder_images, output_folder, number_of_model_calls)


if __name__ == "__main__":
    # Input and output folders
    input_folder_images = "./processed_images"
    output_folder = "./model_output_GPT4-o"
    paper_list_path = "../data_extraction_GPT-4o/output/paper_list.json"

    main(input_folder_images, output_folder, paper_list_path)
