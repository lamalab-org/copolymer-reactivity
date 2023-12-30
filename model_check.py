import json
import os
import pubchempy as pcp


def get_smiles_from_name(monomer):
    try:
        compound = pcp.get_compounds(monomer, 'name')
        if compound:
            return compound[0].canonical_smiles
        else:
            return "SMILES not found for this chemical name"
    except pcp.PubChemHTTPError as e:
        return f"PubChem Error: {str(e)}"


monomer_error = 0
reaction_const_error = 0
sum_error = 0
not_found_error = 0

print(monomer_error)

model_folder = 'model_output'
test_folder = 'test_data'

model_files = sorted(os.listdir(model_folder))
test_files = sorted(os.listdir(test_folder))

for model_file, test_file in zip(model_files, test_files):

    model_path = os.path.join(model_folder, model_file)
    test_path = os.path.join(test_folder, test_file)
    with open(model_path, 'r') as file:
        model_data = json.load(file)
    with open(test_path, 'r') as file:
        test_data = json.load(file)

    for model_reaction, test_reaction in zip(model_data['polymerizations'], test_data['reaction']):

        model_monomer = model_reaction['involved monomers']
        test_monomer = test_reaction['monomers']
        smiles_model_monomer = get_smiles_from_name(model_monomer)
        smiles_test_monomer = get_smiles_from_name(test_monomer)

        if set(smiles_test_monomer) == set(smiles_model_monomer):
            print("Monomers match:", test_monomer)
        else:
            monomer_error += 1
            print("Monomers do not match:", test_monomer, "vs", model_monomer)

        for model_combination, test_combination in zip(model_data['combinations'], test_data['combinations']):

            test_constants = test_combination['reaction_constants']
            model_constants = model_combination['polymerization_reaction_constants']

            for key, value in test_constants.items():
                if key in model_constants:
                    if abs(value - model_constants[key]) == 0:
                        print(f"Reaction constant {key} matches:", value)
                    else:
                        reaction_const_error += 1
                        print(f"Reaction constant {key} does not match:", value, "vs", model_constants[key])
                else:
                    not_found_error += 1
                    print(f"Reaction constant {key} not found in extracted data")

sum_error = monomer_error + reaction_const_error + not_found_error
print("monomer names do not match:" + str(monomer_error))
print("reaction constant do not match:" + str(reaction_const_error))
print("data was not found in the document:" + str(not_found_error))
print("sum of errors:" + str(sum_error))
