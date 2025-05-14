import time
import datetime
import os
import json
from pdf2image import convert_from_path
import copolextractor.prompter as prompter
import copolextractor.analyzer as az
import copolextractor.image_processer as ip
import copolextractor.utils as utils


failed_smiles_list = []


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


def process_pdf_files(
        paper_list_path,
        output_folder_images,
        output_folder,
        number_of_model_calls,
        pdf_folder,
        stats_file_path,
):
    """
    Process PDF files based on entries from paper_list.json and update the JSON file with extraction results.
    """
    start = time.time()

    # Load paper list JSON
    with open(paper_list_path, "r", encoding="utf-8") as file:
        paper_list = json.load(file)

    # Filter for entries with "precision_score": 1
    selected_papers = [
        entry
        for entry in paper_list
        if entry.get("precision_score") == 1 and not entry.get("extracted" and entry.get("rxn_number", 0) > 0)
    ]
    print(f"Number of PDFs to process: {len(selected_papers)}")

    # Token and model call counters
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

    # Load prompt template
    prompt_text = prompter.get_prompt_template()

    for i, paper in enumerate(selected_papers):
        filename = paper["filename"]
        file_path = os.path.join(pdf_folder, filename.replace('.json', '.pdf'))

        json_file_path = os.path.join(output_folder, filename.replace(".pdf", ".json"))
        if os.path.exists(json_file_path):
            print(f"Skipping {filename}: JSON file already exists.")
            continue

        if not os.path.exists(file_path):
            print(f"Skipping {filename}: File not found.")
            continue

        print(f"Processing {filename}")
        processed_papers += 1

        # Convert PDF to images with size check and downscaling if needed
        try:
            pdf_images = convert_from_path(file_path)
            images_base64 = []
            total_size = 0
            initial_resolution = 2048  # Initial resolution
            current_resolution = initial_resolution

            # First attempt at processing images
            for j, image in enumerate(pdf_images):
                base64_image, img_size = ip.process_image(
                    image, current_resolution, output_folder_images, file_path, j
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

                for j, image in enumerate(pdf_images):
                    base64_image, img_size = ip.process_image(
                        image, current_resolution, output_folder_images, file_path, j
                    )
                    images_base64.append(base64_image)
                    total_size += len(base64_image)

                print(f"After downscaling: Total size is now {total_size / (1024 * 1024):.2f} MB")

            if total_size > max_size:
                print(
                    f"Warning: Even after {max_attempts} downscaling attempts, images for {filename} are still {total_size / (1024 * 1024):.2f} MB (exceeding 50 MB)")

        except Exception as e:
            print(f"An error occurred while processing images for {filename}: {e}")
            continue

        # Generate initial prompt
        content = prompter.get_prompt_vision_model(images_base64, prompt_text)

        # Call the model and process output
        print("Model call starts")
        output, input_token, output_token = prompter.call_openai(prompt=content)

        # Track tokens for this call
        total_input_tokens += input_token
        total_output_tokens += output_token
        number_of_calls += 1

        # Log tokens after each call
        print(f"Call {number_of_calls} for {filename}: Input tokens: {input_token}, Output tokens: {output_token}")

        # Add call info to current run
        current_run["calls"].append({
            "filename": filename,
            "call_type": "initial",
            "input_tokens": input_token,
            "output_tokens": output_token,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # Update and save token statistics after each call
        current_run["total_input_tokens"] = total_input_tokens
        current_run["total_output_tokens"] = total_output_tokens
        current_run["processed_papers"] = processed_papers
        current_run["execution_time"] = time.time() - start

        token_stats["runs"][-1] = current_run  # Update the latest run
        save_token_stats(stats_file_path, token_stats)

        # Save output as JSON
        output_name_json = os.path.join(
            output_folder, filename.replace(".pdf", ".json")
        )

        for retry_attempt in range(number_of_model_calls):
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    output = None
                    continue

            if output is None:
                print(f"Output is None for {filename}, skipping quality check")
                continue

            na_count = az.count_na_values(output, null_value="na")
            total_entry_count = az.count_total_entries(output)
            na_rate = az.calculate_rate(na_count, total_entry_count)
            print(f"NA-rate: {na_rate}")

            if na_rate > 0.4:
                print(f"Retrying model call {retry_attempt + 2} for {filename}")
                updated_prompt = prompter.update_prompt(prompt_text, output)
                content = prompter.get_prompt_vision_model(
                    images_base64, updated_prompt
                )
                output, input_token, output_token = prompter.call_openai(content)

                # Track tokens for retry call
                total_input_tokens += input_token
                total_output_tokens += output_token
                number_of_calls += 1

                # Log tokens after each retry
                print(
                    f"Retry {retry_attempt + 1} for {filename}: Input tokens: {input_token}, Output tokens: {output_token}")

                # Add retry call info to current run
                current_run["calls"].append({
                    "filename": filename,
                    "call_type": f"retry_{retry_attempt + 1}",
                    "input_tokens": input_token,
                    "output_tokens": output_token,
                    "na_rate": na_rate,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

                # Update and save token statistics after each retry
                current_run["total_input_tokens"] = total_input_tokens
                current_run["total_output_tokens"] = total_output_tokens
                token_stats["runs"][-1] = current_run
                save_token_stats(stats_file_path, token_stats)

                # Try to parse and save the output
                try:
                    if isinstance(output, str):
                        parsed_output = json.loads(output)
                    else:
                        parsed_output = output

                    # Save updated output
                    with open(output_name_json, "w", encoding="utf-8") as json_file:
                        json.dump(parsed_output, json_file, indent=4)
                    print(f"Output successfully saved to {output_name_json}")

                except (json.JSONDecodeError, TypeError) as e:
                    print(f"Error processing output: {e}")
                    continue
            else:
                print("NA-rate below threshold, no further retries needed.")

                # Save the output
                try:
                    if isinstance(output, str):
                        output = json.loads(output)

                    with open(output_name_json, "w", encoding="utf-8") as json_file:
                        json.dump(output, json_file, indent=4)
                    print(f"Output successfully saved to {output_name_json}")
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"Error saving output: {e}")

                break

        # Update the JSON entry
        paper.update({"extracted": True, "extracted_data": output})

    # Final update to token statistics
    current_run["execution_time"] = time.time() - start
    token_stats["runs"][-1] = current_run
    save_token_stats(stats_file_path, token_stats)

    # Save the updated JSON file
    extracted_json_path = "./comparison_of_models/data_extracted.json"
    with open(extracted_json_path, "w", encoding="utf-8") as file:
        json.dump(paper_list, file, indent=4)

    print("Total input tokens used:", total_input_tokens)
    print("Total output tokens used:", total_output_tokens)
    print("Total number of model calls:", number_of_calls)
    print("Number of processed papers:", processed_papers)
    print(f"Token statistics saved to {stats_file_path}")

    end = time.time()
    print("Execution time:", end - start)


def decode_nested_json(data):
    """
    Recursively decode string-encoded JSON fields in a dictionary or list.
    Args:
        data: The data structure to decode (dict, list, or str).
    Returns:
        Decoded data structure with all nested JSON strings parsed.
    """
    if isinstance(data, str):
        try:
            return decode_nested_json(
                json.loads(data)
            )  # Parse the string and decode further
        except (json.JSONDecodeError, TypeError):
            return data  # Return as-is if not JSON
    elif isinstance(data, dict):
        return {key: decode_nested_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [decode_nested_json(item) for item in data]
    return data


def process_files(input_folder, output_file):
    """
    Process JSON files in the input folder and generate extracted results.
    """
    results = []
    file_count = 0
    reaction_count = 0

    input_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".json")])

    for filename in input_files:
        file_count += 1
        file_path = os.path.join(input_folder, filename)
        print(f"Processing: {file_path}")
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
                if isinstance(data, str):
                    print("Detected double-encoded JSON. Attempting to parse again...")
                    data = json.loads(data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file: {filename} - {e}")
            continue

        try:
            reactions = data.get("reactions", [])
            if not isinstance(reactions, list):
                print(f"'reactions' in {filename} is not a list. Skipping file.")
                continue

            for reaction in reactions:
                if not isinstance(reaction, dict):
                    print(f"Skipping non-dict reaction in {filename}: {reaction}")
                    continue

                monomers = reaction.get("monomers", [])
                if len(monomers) != 2:
                    print(f"Unexpected number of monomers in {filename}: {monomers}")
                    continue

                monomer1, monomer2 = monomers

                reaction_conditions = reaction.get("reaction_conditions", [])
                for condition in reaction_conditions:
                    temp = condition.get("temperature")
                    unit = condition.get("temperature_unit")
                    temperature = az.convert_unit(temp, unit)
                    solvent = condition.get("solvent")
                    solvent_smiles = utils.name_to_smiles(solvent)
                    logP = utils.calculate_logP(solvent_smiles) if solvent_smiles else None
                    r_values = condition.get("reaction_constants", {})
                    conf_intervals = condition.get("reaction_constant_conf", {})
                    method = condition.get("method")
                    product = condition.get("r_product")
                    polymerization_type = condition.get("polymerization_type")
                    determination_method = condition.get("determination_method")

                    result = {
                        "file": filename,
                        "monomer1_s": utils.name_to_smiles(monomer1),
                        "monomer2_s": utils.name_to_smiles(monomer2),
                        "monomer1": monomer1,
                        "monomer2": monomer2,
                        "r_values": r_values,
                        "conf_intervals": conf_intervals,
                        "temperature": temperature,
                        "temperature_unit": "°C",
                        "solvent": solvent,
                        "solvent_smiles": solvent_smiles,
                        "logP": logP,
                        "method": method,
                        "r-product": product,
                        "source": data.get("source"),
                        "polymerization_type": polymerization_type,
                        "determination_method": determination_method,
                    }

                    results.append(result)
                    reaction_count += 1
        except (AttributeError, json.JSONDecodeError, TypeError) as e:
            print(f"Error processing file {filename}: {e}")
            continue

    print(f"Total files processed: {file_count}")
    print(f"Total reactions extracted: {reaction_count}")

    with open(output_file, "w") as file:
        json.dump(results, file, indent=4)
    print(f"Results saved to {output_file}")

    # Print the failed SMILES summary
    if failed_smiles_list:
        unique_failed_smiles = set(failed_smiles_list)
        print(f"Failed SMILES for logP calculation: {len(unique_failed_smiles)}")
        print(unique_failed_smiles)
    else:
        print("All SMILES were successfully processed.")


def main(
        input_folder_images, output_folder, paper_list_path, pdf_folder, extracted_data_file
):
    """
    Main function to process PDFs and extracted JSON files.
    """
    # Define token stats file path
    stats_file_path = "./token_stats.json"

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

    # Add the current run to the token stats
    token_stats["runs"].append(current_run)

    # Save the initial token stats
    save_token_stats(stats_file_path, token_stats)

    # Ensure necessary folders exist
    os.makedirs(input_folder_images, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Process PDF files
    process_pdf_files(
        paper_list_path,
        input_folder_images,
        output_folder,
        2,
        pdf_folder,
        stats_file_path
    )

    # Process extracted JSON files
    process_files(output_folder, extracted_data_file)


if __name__ == "__main__":
    # Input and output folders
    input_folder_images = "./processed_images"
    output_folder = "./model_output_GPT4-o"
    paper_list_path = "../../data_extraction/data_extraction_GPT-4o/output_2/paper_list.json"
    pdf_folder = "../obtain_data/output_2/PDF"
    extracted_data_file = "../../data_extraction/comparison_of_models/extracted_data.json"

    main(
        input_folder_images,
        output_folder,
        paper_list_path,
        pdf_folder,
        extracted_data_file,
    )