import os
import json
import time
from tqdm import tqdm
from morfeus.conformer import ConformerEnsemble
from morfeus import XTB
import numpy as np

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
        ce.optimize_qc_engine(program="xtb", model={"method": "GFN-FF"}, procedure="geometric")
    except Exception as e:
        print(f"GFN-FF optimization failed for {smiles}: {e}")

    try:
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
def update_extracted_data(json_file, output_file, updated_data):
    with open(json_file, 'r') as f:
        data = json.load(f)

    for entry in data:
        smiles_set = {entry['monomer1_s'], entry['monomer2_s']}
        for smiles, properties in updated_data.items():
            if smiles in smiles_set:
                entry.update(properties)

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Updated {json_file} with new properties.")


# Main function
def main(json_file, output_folder, smiles_error_path, output_file):
    # Load errors or initialize
    try:
        with open(smiles_error_path, 'r') as f:
            smiles_error = json.load(f)
    except FileNotFoundError:
        smiles_error = []

    # Extract unique SMILES strings
    with open(json_file, 'r') as f:
        extracted_data = json.load(f)
    unique_smiles = set(entry['monomer1_s'] for entry in extracted_data if entry['monomer1_s']) | \
                    set(entry['monomer2_s'] for entry in extracted_data if entry['monomer2_s'])

    print(f"Unique SMILES: {len(unique_smiles)}")

    # Create output folder if necessary
    os.makedirs(output_folder, exist_ok=True)

    updated_data = {}

    for smiles in tqdm(sorted(unique_smiles), desc="Processing SMILES"):
        print(f"Processing {smiles}")
        filename = os.path.join(output_folder, f"{smiles.replace('/', '_')}.json")

        if os.path.exists(filename):
            with open(filename, 'r') as f:
                existing_properties = json.load(f)
            missing_properties = calculate_missing_properties(smiles, existing_properties)
            existing_properties.update(missing_properties)
            save_properties(existing_properties, filename)
            updated_data[smiles] = existing_properties
        elif smiles not in [error['smiles'] for error in smiles_error]:
            try:
                properties = calculate_property(smiles)
                save_properties(properties, filename)
                updated_data[smiles] = properties
            except Exception as e:
                print(f"Error processing {smiles}: {e}")
                smiles_error.append({'smiles': smiles, 'error': str(e)})
                with open(smiles_error_path, 'w') as f:
                    json.dump(smiles_error, f, indent=4)

    update_extracted_data(json_file, output_file, updated_data)


# Save properties to a file
def save_properties(properties, filename):
    with open(filename, 'w') as f:
        json.dump(properties, f, indent=4)


if __name__ == "__main__":
    json_file = "extracted_data.json"
    output_folder = "molecule_properties"
    smiles_error_path = "output/smiles_error.json"
    output_file = "extracted_data_w_features.json"
    main(json_file, output_folder, smiles_error_path, output_file)
