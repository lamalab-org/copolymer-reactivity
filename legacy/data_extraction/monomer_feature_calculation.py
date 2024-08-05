import os
from morfeus.conformer import ConformerEnsemble
#from morfeus import XTB
#import qcengine as qcng
#import geometric

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
        for prop, value in properties.items():
            f.write(f"{prop}: {value}\n")

# Main function
def main(smiles_list, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for smiles in smiles_list:
        properties = calculate_properties(smiles)
        filename = os.path.join(output_folder, f"{smiles.replace('/', '_')}.txt")
        save_properties(properties, filename)
        print(f"Properties for {smiles} saved to {filename}")

if __name__ == "__main__":
    # Example SMILES list
    smiles_list = ["CCO", "CCCC", "CC(N)C(O)=O"]
    output_folder = "molecule_properties"
    main(smiles_list, output_folder)
