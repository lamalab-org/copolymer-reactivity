import json
import os
import time
from pdf2image import convert_from_path
import copolextractor.prompter as prompter
import copolextractor.image_processer as ip
import traceback


def is_valid_pdf(file_path):
    """
    Check if the file is a valid PDF.
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(4)
            return header == b"%PDF"
    except Exception:
        return False


def process_pdfs(
    input_folder,
    output_folder_images,
    output_folder,
    selected_entries_path,
    log_file_path,
    output_file,
):
    """
    Process PDFs and update the JSON entries with extracted information.
    """
    # Load the selected entries from the JSON file
    with open(selected_entries_path, "r", encoding="utf-8") as file:
        selected_entries = json.load(file)

    total_input_tokens = 0
    total_output_tokens = 0
    number_of_calls = 0

    # Generate the base prompt text
    prompt_text = prompter.get_prompt_pdf_quality()

    start_time = time.time()

    for entry in selected_entries:
        filename = entry.get("pdf_name")
        if not filename:
            print(f"Skipping entry {entry} because filename {filename} was not found.")
            continue

        file_path = os.path.join(input_folder, filename)
        output_json_path = os.path.join(
            output_folder, filename.replace(".pdf", ".json")
        )

        # Check if the result already exists in the output folder
        if os.path.exists(output_json_path):
            print(f"Loading existing result for {filename} from {output_json_path}.")
            with open(output_json_path, "r", encoding="utf-8") as json_file:
                api_response = json.load(json_file)

            # Update the current entry with the already processed data
            entry.update(
                {
                    "pdf_quality": api_response.get("pdf_quality"),
                    "table_quality": api_response.get("table_quality"),
                    "quality_of_number": api_response.get("quality_of_numbers"),
                    "year": api_response.get("year"),
                    "processed_quality": True,
                    "language": api_response.get("language"),
                    "rxn_count": api_response.get("number_of_reactions"),
                }
            )
            print(entry)
            continue  # Skip further processing since the result already exists

        # Skip if the PDF file doesn't exist
        if not os.path.exists(file_path):
            print(f"Skipping {filename}: File not found.")
            entry["pdf_exists"] = False
            continue
        else:
            entry["pdf_exists"] = True

        # Validate if the file is a valid PDF
        if not is_valid_pdf(file_path):
            print(f"Skipping {filename}: Invalid or corrupted PDF file.")
            entry["processed_quality"] = False
            continue

        # Process the PDF
        print(f"Processing {filename}")
        try:
            # Convert PDF to images
            pdf_images = convert_from_path(file_path)
            images_base64 = []
            for i, image in enumerate(pdf_images):
                base64_image, _ = ip.process_image(
                    image, 2048, output_folder_images, file_path, i
                )
                images_base64.append(base64_image)
        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")
            with open(log_file_path, "a") as log_file:
                log_file.write(f"Error processing {filename}:\n")
                log_file.write(str(e) + "\n")
                traceback.print_exc(file=log_file)
            entry["processed_quality"] = False
            continue

        # Prepare and send the prompt to the model
        content = prompter.get_prompt_vision_model(images_base64, prompt_text)

        print("Model call starts")
        output, input_token, output_token = prompter.call_openai(prompt=content)
        total_input_tokens += input_token
        total_output_tokens += output_token
        number_of_calls += 1

        api_response = json.loads(output)
        print("API Response Parsed:", api_response)

        # Save the API response to a JSON file
        with open(output_json_path, "w", encoding="utf-8") as json_file:
            json.dump(api_response, json_file, indent=4)
        print(f"Saved result for {filename} to {output_json_path}.")

        # Update the current entry with the new data
        entry.update(
            {
                "pdf_quality": api_response.get("pdf_quality"),
                "table_quality": api_response.get("table_quality"),
                "quality_of_number": api_response.get("quality_of_numbers"),
                "year": api_response.get("year"),
                "processed_quality": True,
                "language": api_response.get("language"),
                "rxn_count": api_response.get("number_of_reactions"),
            }
        )
        print(entry)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(selected_entries, file, indent=4)

    end_time = time.time()
    print("Execution time:", end_time - start_time)
    print("Total input tokens:", total_input_tokens)
    print("Total output tokens:", total_output_tokens)
    print("Total number of model calls:", number_of_calls)


def main(
    input_folder,
    output_folder_images,
    output_folder,
    selected_entries_path,
    output_file,
):
    """
    Main function to process PDFs and update JSON entries.
    """
    # Define log file path
    log_file_path = "./error_log.txt"

    import os
    print("Working directory: ", os.getcwd())

    # Ensure output directories exist
    os.makedirs(output_folder_images, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Process the PDFs
    process_pdfs(
        input_folder,
        output_folder_images,
        output_folder,
        selected_entries_path,
        log_file_path,
        output_file,
    )
