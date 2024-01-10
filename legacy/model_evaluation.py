import os
import src.copolextractor.analyzer as az


def load_yaml_combinations(file_path):
    data = az.load_yaml(file_path)
    combinations_count = az.get_total_number_of_combinations(data)
    return combinations_count


def load_yaml_reactions(file_path):
    data = az.load_yaml(file_path)
    reaction_count = az.get_number_of_reactions(data)
    return reaction_count


def get_file_path(path, file):
    file_path = os.path.join(path, file)
    return file_path


reaction_number_error = 0
combination_number_error = 0
matching_monomer_error = 0


test_path = "./../test_data"
model_path = "./../test_data2"
test_files = sorted([f for f in os.listdir(test_path) if f.endswith(".yaml")])
model_files = sorted([f for f in os.listdir(model_path) if f.endswith(".yaml")])

for test_file, model_file in zip(test_files, model_files):
    test_file_path = get_file_path(test_path, test_file)
    model_file_path = get_file_path(model_path, model_file)

    len_combinations_test = load_yaml_combinations(test_file_path)
    len_combinations_model = load_yaml_combinations(model_file_path)

    len_reactions_test = load_yaml_reactions(test_file_path)
    len_reactions_model = load_yaml_reactions(model_file_path)

    print(f"Comparing {test_file} and {model_file}")
    print(f"Number of different reactions in test data: {len_reactions_test}")
    print(f"Number of different reactions in model data: {len_reactions_model}")
    print(f"Number of different combinations in test data: {len_combinations_test}")
    print(f"Number of different combinations in model data: {len_combinations_model}")

    if len_reactions_test == len_reactions_model:
        print("Number of reactions is equal.")
    else:
        print("Number of reactions is different.")
        reaction_number_error += 1
    if len_combinations_test == len_combinations_model:
        print("Number of combinations is equal.")
    else:
        print("Number of combinations is different.")
        combination_number_error += 1

print(f"reaction-number-error is {reaction_number_error}")
print(f"combination-number-error is {combination_number_error}")

for test_file, model_file in zip(test_files, model_files):
    test_file_path = get_file_path(test_path, test_file)
    model_file_path = get_file_path(model_path, model_file)
    test_data = az.load_yaml(test_file_path)
    model_data = az.load_yaml(model_file_path)
    matching_monomer_error += az.find_matching_reaction(model_data, test_data)
    #entweder vorher alle in eine liste oder hier wieder beide sachen von model und test eingeben
    #gibt die anzahl der gefundenen matching combinations und den matching score aus als tupel
    #matching_combinations_count = az.find_matching_combination(combination: List[dict], polymerization_type: str, solvent: str, temperature: Union[str, float, int], method: str)

print("matching-monomer-error: ", matching_monomer_error)

