import os
import json
from tqdm import tqdm
from morfeus.conformer import ConformerEnsemble
from morfeus import XTB


# Function to calculate only missing properties of a molecule
def calculate_missing_properties(smiles, existing_properties):
    missing_properties = {}

    if 'best_conformer_elements' in existing_properties and 'best_conformer_coordinates' in existing_properties:
        elements = existing_properties['best_conformer_elements']
        coordinates = existing_properties['best_conformer_coordinates']
    else:
        print(f"Best conformer data is missing for {smiles}. Calculating it.")
        elements, coordinates, energy = conformer_opt(smiles)
        missing_properties.update({
            'best_conformer_elements': elements,
            'best_conformer_coordinates': coordinates,
            'best_conformer_energy': energy
        })

    xtb = XTB(elements, coordinates)

    # Calculate and store missing properties
    xtb_properties = [
        ('ip', xtb.get_ip),
        ('ip_corrected', lambda: xtb.get_ip(corrected=True)),
        ('ea', xtb.get_ea),
        ('homo', xtb.get_homo),
        ('lumo', xtb.get_lumo),
        ('charges', xtb.get_charges),
        ('dipole', lambda: xtb.get_dipole().tolist()),
        ('global_electrophilicity', lambda: xtb.get_global_descriptor("electrophilicity", corrected=True)),
        ('global_nucleophilicity', lambda: xtb.get_global_descriptor("nucleophilicity", corrected=True)),
        ('fukui_electrophilicity', lambda: xtb.get_fukui("electrophilicity")),
        ('fukui_nucleophilicity', lambda: xtb.get_fukui("nucleophilicity")),
        ('fukui_radical', lambda: xtb.get_fukui("radical")),
    ]

    for prop, func in xtb_properties:
        if prop not in existing_properties:
            try:
                missing_properties[prop] = func()
            except Exception as e:
                print(f"Error calculating {prop} for {smiles}: {e}")

    return missing_properties


# Conformer optimization function
def conformer_opt(smiles):
    ce = ConformerEnsemble.from_rdkit(smiles, optimize="MMFF94")
    ce.prune_rmsd()
    ce.sort()

    try:
        print('Started GFN-FF optimization.')
        ce.optimize_qc_engine(program="xtb", model={"method": "GFN-FF"}, procedure="geometric")
    except Exception as e:
        print(f"GFN-FF optimization failed for {smiles}: {e}")
        ce = ConformerEnsemble.from_rdkit(smiles, optimize="MMFF94")
        ce.prune_rmsd()
        ce.sort()

    try:
        print('Started GFN2-xTB optimization.')
        ce.optimize_qc_engine(program="xtb", model={"method": "GFN2-xTB"}, procedure="geometric")
    except Exception as e:
        print(f"GFN2-xTB optimization failed for {smiles}: {e}")

    ce.sp_qc_engine(program="xtb", model={"method": "GFN2-xTB"})
    best_conformer = ce.conformers[0]

    elements = best_conformer.elements.tolist()
    coordinates = best_conformer.coordinates.tolist()
    energy = best_conformer.energy

    return elements, coordinates, energy


# Function to calculate all properties for a given SMILES
def calculate_property(smiles):
    elements, coordinates, energy = conformer_opt(smiles)
    xtb = XTB(elements, coordinates)

    properties = {
        'smiles': smiles,
        'best_conformer_coordinates': coordinates,
        'best_conformer_elements': elements,
        'best_conformer_energy': energy,
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
    }

    return properties


# Function to update extracted_data.json with new properties
# Function to update extracted_data.json with molecular properties for both monomers
def update_extracted_data(json_file, output_file, updated_data):
    # Load the original JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    for entry in data:
        # Get SMILES for monomer1 and monomer2
        monomer1_smiles = entry.get('monomer1_s')
        monomer2_smiles = entry.get('monomer2_s')

        # Retrieve properties for each monomer from the updated data
        monomer1_data = updated_data.get(monomer1_smiles, {})
        monomer2_data = updated_data.get(monomer2_smiles, {})

        # Add the properties as lists in the entry
        entry['monomer1_data'] = monomer1_data
        entry['monomer2_data'] = monomer2_data

    # Save the updated data back to the output file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Updated {json_file} with new properties.")


# Main function
# Main function to process SMILES, calculate molecular properties, and update extracted data
def main(json_file, output_folder, smiles_error_path, output_file):
    # Load the error log or initialize it if the file does not exist
    try:
        with open(smiles_error_path, 'r') as f:
            smiles_error = json.load(f)
    except FileNotFoundError:
        smiles_error = []

    # Load the extracted data and extract unique SMILES strings
    with open(json_file, 'r') as f:
        extracted_data = json.load(f)
    unique_smiles = set(entry['monomer1_s'] for entry in extracted_data if entry['monomer1_s']) | \
                    set(entry['monomer2_s'] for entry in extracted_data if entry['monomer2_s'])

    print(f"Unique SMILES: {len(unique_smiles)}")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    updated_data = {}

    # Iterate through each unique SMILES string
    for smiles in tqdm(sorted(unique_smiles), desc="Processing SMILES"):
        print(f"Processing {smiles}")
        filename = os.path.join(output_folder, f"{smiles.replace('/', '_')}.json")

        if os.path.exists(filename):
            # If the file exists, load existing properties and calculate missing ones
            with open(filename, 'r') as f:
                existing_properties = json.load(f)
            missing_properties = calculate_missing_properties(smiles, existing_properties)
            existing_properties.update(missing_properties)
            save_properties(existing_properties, filename)
            updated_data[smiles] = existing_properties
        elif smiles not in [error['smiles'] for error in smiles_error]:
            try:
                # If the file doesn't exist, calculate all properties
                properties = calculate_property(smiles)
                save_properties(properties, filename)
                updated_data[smiles] = properties
            except Exception as e:
                # Log any errors encountered
                print(f"Error processing {smiles}: {e}")
                smiles_error.append({'smiles': smiles, 'error': str(e)})
                with open(smiles_error_path, 'w') as f:
                    json.dump(smiles_error, f, indent=4)

    # Update the extracted data JSON file with molecular properties
    update_extracted_data(json_file, output_file, updated_data)


# Save properties to a file
def save_properties(properties, filename):
    with open(filename, 'w') as f:
        json.dump(properties, f, indent=4)


if __name__ == "__main__":
    json_file = "../data_extraction/comparison_of_models/extracted_data.json"
    output_folder = "output/molecule_properties"
    smiles_error_path = "output/smiles_error.json"
    output_file = "output/extracted_data_w_features.json"
    main(json_file, output_folder, smiles_error_path, output_file)
