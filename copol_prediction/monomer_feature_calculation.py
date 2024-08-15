import os
from morfeus.conformer import ConformerEnsemble
from morfeus import XTB
import json
import time
from tqdm import tqdm  # Import tqdm


# Function to optimize and calculate properties of a molecule
def calculate_properties(smiles):
    # Create a conformer ensemble from the SMILES string
    ce = ConformerEnsemble.from_rdkit(smiles, optimize="MMFF94")

    # Prune conformers based on RMSD and sort by energy
    ce.prune_rmsd()
    ce.sort()

    # Optimize with GFN-FF and perform single-point calculations with GFN2-xTB
    model_opt = {"method": "GFN-FF"}
    ce.optimize_qc_engine(program="xtb", model=model_opt, procedure="geometric")

    # Run optimization with GFN2-xTB
    model_opt = {"method": "GFN2-xTB"}
    ce.optimize_qc_engine(program="xtb", model=model_opt, procedure="geometric")

    # Run single-point calculations with GFN2-xTB
    model_sp = {"method": "GFN2-xTB"}
    ce.sp_qc_engine(program="xtb", model=model_sp)

    # Calculate Boltzmann weights
    weights = ce.boltzmann_weights()

    # Calculate properties for the lowest energy conformer
    best_conformer = ce.conformers[0]
    elements, coordinates = best_conformer.elements, best_conformer.coordinates
    xtb = XTB(elements, coordinates)

    properties = {
        'smiles': smiles,
        'ip': xtb.get_ip(),
        'ip_corrected': xtb.get_ip(corrected=True),
        'ea': xtb.get_ea(),
        'homo': xtb.get_homo(),
        'lumo': xtb.get_lumo(),
        'charges': xtb.get_charges(),
        'dipole': xtb.get_dipole().tolist(),
        'global_electrophilicity': xtb.get_global_descriptor("electrophilicity", corrected=True),
        'global_nucleophilicity': xtb.get_global_descriptor("nucleophilicity", corrected=True),
        'fukui_electrophilicity': xtb.get_fukui("electrophilicity"),
        'fukui_nucleophilicity': xtb.get_fukui("nucleophilicity"),
        'boltzmann_weights': weights.tolist()
    }

    return properties


# Save the properties to a file
def save_properties(properties, filename):
    with open(filename, 'w') as f:
        # Save to json
        json.dump(properties, f)


# Function to extract unique SMILES strings
def extract_unique_smiles(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    unique_smiles = set()
    for entry in data:
        if entry['monomer1_s'] is not None and entry['monomer2_s'] is not None:
            unique_smiles.add(entry["monomer1_s"])
            unique_smiles.add(entry["monomer2_s"])

    return unique_smiles


# Main function
def main(json_file, output_folder, smiles_error):
    # Extract unique SMILES strings
    unique_smiles = extract_unique_smiles(json_file)

    # Print the unique SMILES strings and their count
    print(f"Unique SMILES: {len(unique_smiles)}")
    time.sleep(10)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Calculate properties for each unique SMILES string with a progress bar
    for smiles in tqdm(sorted(unique_smiles), desc="Processing SMILES"):
        filename = os.path.join(output_folder, f"{smiles.replace('/', '_')}.json")
        if os.path.exists(filename):
            print(f"File for {smiles} already exists. Skipping calculation.")
            continue
        elif smiles in smiles_error:
            print(f"SMILES {smiles} is in the error list. Skipping calculation.")
            continue

        try:
            properties = calculate_properties(smiles)
            save_properties(properties, filename)
            print(f"Properties for {smiles} saved to {filename}")
        except Exception as e:
            print(f'For {smiles} the error {e} occurred.')
            error_entry = {
                'smiles': smiles,
                'error': str(e)
            }
            smiles_error.append(error_entry)
            with open("smiles_error.json", 'w') as f:
                json.dump(smiles_error, f, indent=4)
            continue


if __name__ == "__main__":
    json_file = "extracted_data_collected_with_logP_and_features.json"
    output_folder = "molecule_properties"
    smiles_error_path = "smiles_error.json"

    try:
        with open(smiles_error_path, 'r') as f:
            smiles_error = json.load(f)
    except FileNotFoundError:
        smiles_error = []

    try:
        main(json_file, output_folder, smiles_error)
    except Exception as e:
        print(e)
        pass
