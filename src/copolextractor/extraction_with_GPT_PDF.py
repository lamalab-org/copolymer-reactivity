import time
from pdf2image import convert_from_path
import copolextractor.prompter as prompter
import copolextractor.analyzer as az
import copolextractor.image_processer as ip
import copolextractor.utils as utils
import os
import json
from copolextractor.mongodb_storage import CoPolymerDB


failed_smiles_list = []
db = CoPolymerDB()


def process_pdf_files(
    paper_list_path,
    output_folder_images,
    output_folder,
    number_of_model_calls,
    pdf_folder,
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
        if entry.get("precision_score") == 1 and not entry.get("extracted")
    ]
    print(f"Number of PDFs to process: {len(selected_papers)}")

    # Token and model call counters
    total_input_tokens = 0
    total_output_tokens = 0
    number_of_calls = 0

    # Load prompt template
    prompt_text = prompter.get_prompt_template()

    for i, paper in enumerate(selected_papers):
        filename = paper["pdf_name"]
        file_path = os.path.join(pdf_folder, filename)

        json_file_path = os.path.join(output_folder, filename.replace(".pdf", ".json"))
        if os.path.exists(json_file_path):
            print(f"Skipping {filename}: JSON file already exists.")
            continue

        if not os.path.exists(file_path):
            print(f"Skipping {filename}: File not found.")
            continue

        print(f"Processing {filename}")

        # Convert PDF to images
        pdf_images = convert_from_path(file_path)
        images_base64 = [
            ip.process_image(image, 2048, output_folder_images, file_path, j)[0]
            for j, image in enumerate(pdf_images)
        ]

        # Generate initial prompt
        content = prompter.get_prompt_vision_model(images_base64, prompt_text)

        # Call the model and process output
        print("Model call starts")
        output, input_token, output_token = prompter.call_openai(prompt=content)
        total_input_tokens += input_token
        total_output_tokens += output_token
        number_of_calls += 1

        # Save output as JSON and YAML
        output_name_json = os.path.join(
            output_folder, filename.replace(".pdf", ".json")
        )

        for attempt in range(number_of_model_calls):
            if isinstance(output, str):
                output = json.loads(output)

            na_count = az.count_na_values(output, null_value="na")
            total_entry_count = az.count_total_entries(output)
            na_rate = az.calculate_rate(na_count, total_entry_count)
            print(f"NA-rate: {na_rate}")

            if na_rate > 0.4 or output is None:
                print(f"Retrying model call {attempt + 2} for {filename}")
                updated_prompt = prompter.update_prompt(prompt_text, output)
                content = prompter.get_prompt_vision_model(
                    images_base64, updated_prompt
                )
                output, input_token, output_token = prompter.call_openai(content)
                total_input_tokens += input_token
                total_output_tokens += output_token
                number_of_calls += 1

                if output is not None:
                    # Save updated output
                    try:
                        # Parse the output string into a Python dictionary
                        parsed_output = json.loads(output)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        parsed_output = None

                    if parsed_output is not None:
                        # Save updated output
                        with open(output_name_json, "w", encoding="utf-8") as json_file:
                            json.dump(parsed_output, json_file, indent=4)
                        print(f"Output successfully saved to {output_name_json}")
                    else:
                        print("Failed to save output due to JSON parsing error.")
            else:
                print("NA-rate below 30%, no further retries needed.")

                if output is not None:
                    # Save updated output
                    with open(output_name_json, "w", encoding="utf-8") as json_file:
                        json.dump(output, json_file, indent=4)
                    print(f"Output successfully saved to {output_name_json}")
                else:
                    print("Failed to save output due to JSON parsing error.")

                break

        # Update the JSON entry
        paper.update({"extracted": True, "extracted_data": output})

    # Save the updated JSON file
    extracted_json_path = (
        "../../data_extraction/comparison_of_models/data_extracted.json"
    )
    with open(extracted_json_path, "w", encoding="utf-8") as file:
        json.dump(paper_list, file, indent=4)

    print("Total input tokens used:", total_input_tokens)
    print("Total output tokens used:", total_output_tokens)
    print("Total number of model calls:", number_of_calls)

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

                    database = db.save_data(data)
                    print(database)

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

    # Ensure necessary folders exist
    os.makedirs(input_folder_images, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Process PDF files
    process_pdf_files(
        paper_list_path, input_folder_images, output_folder, 2, pdf_folder
    )

    # Process extracted JSON files
    process_files(output_folder, extracted_data_file)


if __name__ == "__main__":
    # Input and output folders
    input_folder_images = "./processed_images"
    output_folder = "./model_output_GPT4-o"
    paper_list_path = (
        "../../data_extraction/data_extraction_GPT-4o/output/paper_list.json"
    )
    pdf_folder = "../obtain_data/output/PDF"
    extracted_data_file = (
        "../../data_extraction/comparison_of_models/extracted_data.json"
    )

    main(
        input_folder_images,
        output_folder,
        paper_list_path,
        pdf_folder,
        extracted_data_file,
    )
