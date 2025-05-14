import json
import os
import time
import datetime
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


def load_or_create_token_stats(stats_file_path):
    """
    Load existing token statistics or create a new file if it doesn't exist.
    """
    if os.path.exists(stats_file_path):
        try:
            with open(stats_file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError:
            # File exists but is corrupted or empty
            return {"runs": []}
    else:
        # Create a new stats file
        return {"runs": []}


def save_token_stats(stats_file_path, stats_data):
    """
    Save token statistics to the JSON file.
    """
    with open(stats_file_path, "w", encoding="utf-8") as file:
        json.dump(stats_data, file, indent=4)


def process_pdfs(
        input_folder,
        output_folder_images,
        output_folder,
        selected_entries_path,
        log_file_path,
        output_file,
        stats_file_path,
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
    processed_papers = 0

    # Load or create token statistics
    token_stats = load_or_create_token_stats(stats_file_path)

    # Create a new run entry with timestamp
    current_run = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "processed_papers": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "execution_time": 0,
        "calls": []
    }

    # Generate the base prompt text
    prompt_text = prompter.get_prompt_pdf_quality()

    start_time = time.time()

    for entry in selected_entries:
        filename = entry.get("filename")
        if not filename:
            print(f"Skipping entry {entry} because filename {filename} was not found.")
            continue

        file_path = os.path.join(input_folder, filename.replace(".json", ".pdf"))
        output_json_path = os.path.join(
            output_folder, filename.replace(".pdf", ".json")
        )

        # Check if the result already exists in the output_2 folder
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
            processed_papers += 1
            continue  # Skip further processing since the result already exists

        # Skip if the PDF file doesn't exist
        if not os.path.exists(file_path):
            print(f"Skipping {filename}: File not found in {file_path}.")
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
            total_size = 0
            initial_resolution = 2048  # Initial resolution
            current_resolution = initial_resolution

            # First attempt at processing images
            for i, image in enumerate(pdf_images):
                base64_image, img_size = ip.process_image(
                    image, current_resolution, output_folder_images, file_path, i
                )
                images_base64.append(base64_image)
                total_size += len(base64_image)

            # Check if total size exceeds 50 MB (50 * 1024 * 1024 bytes)
            max_size = 50 * 1024 * 1024

            # If images are too large, rescale them with progressively lower resolution
            scaling_factor = 0.75  # Reduce resolution by 25% each iteration
            max_attempts = 3  # Prevent infinite loops
            attempt = 0

            while total_size > max_size and attempt < max_attempts:
                attempt += 1
                current_resolution = int(current_resolution * scaling_factor)
                print(
                    f"Total size of images ({total_size / (1024 * 1024):.2f} MB) exceeds 50 MB. Downscaling to {current_resolution}px...")

                # Reset and reprocess
                images_base64 = []
                total_size = 0

                for i, image in enumerate(pdf_images):
                    base64_image, img_size = ip.process_image(
                        image, current_resolution, output_folder_images, file_path, i
                    )
                    images_base64.append(base64_image)
                    total_size += len(base64_image)

                print(f"After downscaling: Total size is now {total_size / (1024 * 1024):.2f} MB")

            if total_size > max_size:
                print(
                    f"Warning: Even after {max_attempts} downscaling attempts, images for {filename} are still {total_size / (1024 * 1024):.2f} MB (exceeding 50 MB)")

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

        print("LLM call starts")
        output, input_token, output_token = prompter.call_openai(prompt=content)

        # Track tokens for this call
        total_input_tokens += input_token
        total_output_tokens += output_token
        number_of_calls += 1
        processed_papers += 1

        # Log tokens after each call
        print(f"Call {number_of_calls} for {filename}: Input tokens: {input_token}, Output tokens: {output_token}")

        # Add call info to current run
        current_run["calls"].append({
            "filename": filename,
            "input_tokens": input_token,
            "output_tokens": output_token,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        print("API response:", output)

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
    execution_time = end_time - start_time

    # Update the current run stats
    current_run["processed_papers"] = processed_papers
    current_run["total_input_tokens"] = total_input_tokens
    current_run["total_output_tokens"] = total_output_tokens
    current_run["execution_time"] = execution_time

    # Add the current run to the token stats
    token_stats["runs"].append(current_run)

    # Save the updated token stats
    save_token_stats(stats_file_path, token_stats)

    print(f"LLM scores saved to {output_file}")
    print(f"Token statistics saved to {stats_file_path}")

    print("Execution time:", execution_time)
    print("Total input tokens:", total_input_tokens)
    print("Total output tokens:", total_output_tokens)
    print("Total number of model calls:", number_of_calls)
    print("Number of processed papers:", processed_papers)


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

    # Define token stats file path
    stats_file_path = "./token_stats.json"

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
        stats_file_path,
    )


if __name__ == "__main__":
    # Example usage
    main(
        input_folder="./pdfs",
        output_folder_images="./output_images",
        output_folder="./output_json",
        selected_entries_path="./selected_entries.json",
        output_file="./updated_entries.json",
    )