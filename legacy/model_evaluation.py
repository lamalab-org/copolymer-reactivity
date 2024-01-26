import os
import copolextractor.analyzer as az
from sklearn.metrics import mean_squared_error


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
reaction_const_conf_error = 0
reaction_const_error = 0
test_reaction_constants = []
model_reaction_constants = []
combined_score = []
combined_mse_const = []
combined_mse_conf = []


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


for test_file, model_file in zip(test_files, model_files):
    print(test_file)
    print(model_file)
    test_file_path = get_file_path(test_path, test_file)
    model_file_path = get_file_path(model_path, model_file)
    test_data = az.load_yaml(test_file_path)
    model_data = az.load_yaml(model_file_path)
    for i, reaction in enumerate(test_data['reaction']):
        print('iteration: ', i)
        test_monomers = reaction['monomers']
        model_monomer_index = az.find_matching_reaction(model_data, test_monomers)
        test_monomer_index = i
        if model_monomer_index is not None:
            for j, combination in enumerate(reaction['combinations']):

                print('matching_monomer_error: ', matching_monomer_error)
                specific_reaction = model_data['reaction'][model_monomer_index]
                specific_combination = specific_reaction['combinations']
                model_combinations = az.extract_combinations(specific_combination)
                temperature, polym_method, polym_type, solvent, reaction_constants, reaction_constant_confidence, determination_method = az.get_metadata_polymerization(combination)
                index, score = az.find_matching_combination(model_combinations, polym_type,
                                                            solvent, temperature, polym_method, determination_method)
                print("combination match: ", score)
                combined_score.append(score)

                test_reaction_constants, test_reaction_const_conf = az.get_reaction_constant(reaction_constants, reaction_constant_confidence)
                model_reaction_constants, model_reaction_const_conf = az.get_reaction_constant(combination['reaction_constants'], combination['reaction_constant_conf'])
                if model_reaction_constants[0] is None or model_reaction_constants[1] is None:
                    reaction_const_error += 1
                else:
                    mse_const = mean_squared_error(test_reaction_constants, model_reaction_constants)

                if test_reaction_const_conf == model_reaction_const_conf == [None, None]:
                    continue
                elif model_reaction_const_conf[0] is None or model_reaction_const_conf[1] is None:
                    reaction_const_conf_error += 1
                else:
                    mse_conf = mean_squared_error(test_reaction_const_conf, model_reaction_const_conf)
                combined_mse_const.append(mse_const)
                combined_mse_conf.append(mse_conf)
        else:
            matching_monomer_error += 1

average_mse_const = sum(combined_mse_const) / len(combined_mse_const)
average_mse_conf = sum(combined_mse_conf) / len(combined_mse_conf)
average_score = sum(combined_score) / len(combined_score)
print("matching-monomer-error: ", matching_monomer_error)
print(f"reaction-number-error is {reaction_number_error}")
print(f"combination-number-error is {combination_number_error}")
print(f"reaction constant error is {reaction_const_error}")
print(f"reaction constant confidence error is {reaction_const_conf_error}")
print(f"average score of fuzzy matching is {average_score}")
print(f"average mse of reaction constants is {average_mse_const}")
print(f"average mse of reaction constants confidence is {average_mse_conf}")
