import os
from morfeus.conformer import ConformerEnsemble
from morfeus import XTB
import json
import time
from tqdm import tqdm  # Import tqdm
import numpy as np


# Function to calculate only missing properties of a molecule
def calculate_missing_properties(smiles, existing_properties):
    missing_properties = {}

    if 'best_conformer_elements' in existing_properties and 'best_conformer_coordinates' in existing_properties:
        elements = existing_properties['best_conformer_elements']
        coordinates = existing_properties['best_conformer_coordinates']
    else:
        print(f"Best conformer data is missing for {smiles}. Cannot calculate missing properties.")
        elements, coordinates, energy = conformer_opt(smiles)
        missing_properties['best_conformer_elements'] = elements
        missing_properties['best_conformer_coordinates'] = coordinates
        missing_properties['best_conformer_energy'] = energy

    xtb = XTB(elements, coordinates)

    # Check for missing properties and calculate them
    if 'ip' not in existing_properties:
        missing_properties['ip'] = xtb.get_ip()
    if 'ip_corrected' not in existing_properties:
        missing_properties['ip_corrected'] = xtb.get_ip(corrected=True)
    if 'ea' not in existing_properties:
        missing_properties['ea'] = xtb.get_ea()
    if 'homo' not in existing_properties:
        missing_properties['homo'] = xtb.get_homo()
    if 'lumo' not in existing_properties:
        missing_properties['lumo'] = xtb.get_lumo()
    if 'charges' not in existing_properties:
        missing_properties['charges'] = xtb.get_charges()
    if 'dipole' not in existing_properties:
        missing_properties['dipole'] = xtb.get_dipole().tolist()
    if 'global_electrophilicity' not in existing_properties:
        missing_properties['global_electrophilicity'] = xtb.get_global_descriptor("electrophilicity", corrected=True)
    if 'global_nucleophilicity' not in existing_properties:
        missing_properties['global_nucleophilicity'] = xtb.get_global_descriptor("nucleophilicity", corrected=True)
    if 'fukui_electrophilicity' not in existing_properties:
        missing_properties['fukui_electrophilicity'] = xtb.get_fukui("electrophilicity")
    if 'fukui_nucleophilicity' not in existing_properties:
        missing_properties['fukui_nucleophilicity'] = xtb.get_fukui("nucleophilicity")
    if 'fukui_radical' not in existing_properties:
        missing_properties['fukui_radical'] = xtb.get_fukui("radical")

    return missing_properties


def conformer_opt(smiles):
    # Create a conformer ensemble from the SMILES string
    ce = ConformerEnsemble.from_rdkit(smiles, optimize="MMFF94")

    # Prune conformers based on RMSD and sort by energy
    ce.prune_rmsd()
    ce.sort()

    try:
        # Attempt optimization with GFN-FF
        model_opt = {"method": "GFN-FF"}
        ce.optimize_qc_engine(program="xtb", model=model_opt, procedure="geometric")
    except Exception as e:
        print(f"GFN-FF optimization failed for {smiles}: {e}")
        # Continue with the next steps regardless of the error

    try:
        # Run optimization with GFN2-xTB
        model_opt = {"method": "GFN2-xTB"}
        ce.optimize_qc_engine(program="xtb", model=model_opt, procedure="geometric")
    except Exception as e:
        print(f"GFN-FF optimization failed for {smiles}: {e}")
        # Continue with the next steps regardless of the error

    # Run single-point calculations with GFN2-xTB
    model_sp = {"method": "GFN2-xTB"}
    ce.sp_qc_engine(program="xtb", model=model_sp)

    # Calculate Boltzmann weights
    weights = ce.boltzmann_weights()

    # Select the best conformer
    best_conformer = ce.conformers[0]
    elements, coordinates = best_conformer.elements, best_conformer.coordinates
    energy = best_conformer.energy

    # Convert numpy arrays to lists for JSON serialization
    elements = elements.tolist() if isinstance(elements, np.ndarray) else elements
    coordinates = coordinates.tolist() if isinstance(coordinates, np.ndarray) else coordinates

    return elements, coordinates, energy


def calculate_property(smiles):
        elements, coordinates, energy = conformer_opt(smiles)

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
            'fukui_radical': xtb.get_fukui("radical"),
            'best_conformer_coordinates': coordinates,
            'best_conformer_elements': elements,
            'best_conformer_energy': energy
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
        print('SMILES: ', smiles)
        filename = os.path.join(output_folder, f"{smiles.replace('/', '_')}.json")
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                existing_properties = json.load(f)
            required_properties = [
                'ip', 'ip_corrected', 'ea', 'homo', 'lumo', 'charges', 'dipole',
                'global_electrophilicity', 'global_nucleophilicity',
                'fukui_electrophilicity', 'fukui_nucleophilicity', 'fukui_radical'
            ]
            if all(prop in existing_properties for prop in required_properties):
                print(f"All properties for {smiles} already exist. Skipping calculation.")
                continue
            else:
                print(f"Some properties for {smiles} are missing. Calculating missing properties.")
                missing_properties = calculate_missing_properties(smiles, existing_properties)
                existing_properties.update(missing_properties)
                save_properties(existing_properties, filename)
                print(f"Updated properties for {smiles} saved to {filename}")
                continue
        elif smiles in smiles_error:
            print(f"SMILES {smiles} is in the error list. Skipping calculation.")
            continue

        try:
            properties = calculate_property(smiles)
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
