import json
import os
from rdkit import Chem
from rdkit.Chem import Descriptors

failed_smiles_list = []
count = []


# Function to calculate logP from SMILES
def calculate_logP(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            logP = Descriptors.MolLogP(mol)
            return logP
        else:
            print(f"Conversion failed for SMILES: {smiles}")
            failed_smiles_list.append(smiles)
            return None
    except Exception as e:
        print(f"Error processing SMILES: {smiles} with error {e}")
        failed_smiles_list.append(smiles)
        return None


# Function to load additional monomer data from the corresponding JSON files
def load_monomer_data(smiles, monomer_data_path):
    filename = os.path.join(monomer_data_path, f"{smiles}.json")
    if os.path.exists(filename):
        with open(filename, "r") as file:
            monomer_data = json.load(file)
            return monomer_data
    else:
        print(f"File not found for SMILES: {smiles} at {filename}")
        return None


# Main function
def main():
    # Path to the monomer features files
    monomer_data_path = "./molecule_properties"

    # Load the main data JSON
    data_file = "../data_extraction/data_extraction_GPT-4o/output/copol_database/copol_extracted_data_without_fp.json"
    with open(data_file, "r") as file:
        data = json.load(file)

    # Track the number of times each paper name has been used
    paper_count = {}
    all_entries = []

    # Process each entry in the data
    for idx, entry in enumerate(data):
        # Calculate logP for the solvent
        solvent_smiles = entry.get("solvent")
        if solvent_smiles:
            entry["logP"] = calculate_logP(solvent_smiles)

        # Load and add monomer1 data
        monomer1_smiles = entry.get("monomer1_s")
        monomer1_data = load_monomer_data(monomer1_smiles, monomer_data_path)
        if monomer1_data:
            entry["monomer1_data"] = monomer1_data
        else:
            count.append(monomer1_smiles)

        # Load and add monomer2 data
        monomer2_smiles = entry.get("monomer2_s")
        monomer2_data = load_monomer_data(monomer2_smiles, monomer_data_path)
        if monomer2_data:
            entry["monomer2_data"] = monomer2_data
        else:
            count.append(monomer2_smiles)

        # Determine a unique filename for each reaction
        base_filename = entry.get("file", "unknown").replace(".json", "")
        if base_filename in paper_count:
            paper_count[base_filename] += 1
        else:
            paper_count[base_filename] = 1
        unique_filename = f"{base_filename}_reaction_{paper_count[base_filename]}.json"

        output_path = os.path.join("./processed_reactions", unique_filename)

        with open(output_path, "w") as output_file:
            json.dump(entry, output_file, indent=4)

        all_entries.append(entry)

    # Save all entries in a single file
    all_reactions_output_path = os.path.join(
        "./processed_reactions", "all_reactions.json"
    )
    with open(all_reactions_output_path, "w") as all_reactions_file:
        json.dump(all_entries, all_reactions_file, indent=4)
    print("Processed files saved in ./processed_reactions/")


if __name__ == "__main__":
    # Create directory for processed files if it doesn't exist
    if not os.path.exists("./processed_reactions"):
        os.makedirs("./processed_reactions")

    main()

    print("failed smiles: ", len(set(count)))
    print(set(count))

    # Print the list of SMILES that failed logP calculation
    if failed_smiles_list:
        print("Failed SMILES for logP calculation:")
        for smi in failed_smiles_list:
            print(smi)
    else:
        print("All SMILES were successfully processed.")
