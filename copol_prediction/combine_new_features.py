import json
import os
from rdkit import Chem
from rdkit.Chem import Descriptors

failed_smiles_list = []

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
        with open(filename, 'r') as file:
            monomer_data = json.load(file)
            return monomer_data
    else:
        print(f"File not found for SMILES: {smiles} at {filename}")
        return None

# Main function
def main():
    # Path to the monomer features files
    monomer_data_path = './molecule_properties'

    # Load the main data JSON
    data_file = '../data_extraction/data_extraction_GPT-4o/collected_data/extracted_data_collected_without_fp.json'
    with open(data_file, 'r') as file:
        data = json.load(file)

    # Process each entry in the data
    for entry in data:
        # Calculate logP for the solvent
        solvent_smiles = entry.get('solvent')
        if solvent_smiles:
            entry['logP'] = calculate_logP(solvent_smiles)

        # Load and add monomer1 data
        monomer1_smiles = entry.get('monomer1_s')
        monomer1_data = load_monomer_data(monomer1_smiles, monomer_data_path)
        if monomer1_data:
            entry['monomer1_data'] = monomer1_data

        # Load and add monomer2 data
        monomer2_smiles = entry.get('monomer2_s')
        monomer2_data = load_monomer_data(monomer2_smiles, monomer_data_path)
        if monomer2_data:
            entry['monomer2_data'] = monomer2_data

    # Save the updated data back to a JSON file
    output_file = 'extracted_data_collected_with_logP_and_features.json'
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Data successfully updated and saved to {output_file}")


if __name__ == "__main__":
    main()
    print(failed_smiles_list)
