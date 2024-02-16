import os
import copolextractor.analyzer as az
from sklearn.metrics import mean_squared_error
import wandb


# from extraction import get_prompt_template


def count_combinations(file_path):
    data = az.load_yaml(file_path)
    combinations_count = az.get_total_number_of_combinations(data)
    return combinations_count


def count_reactions(file_path):
    data = az.load_yaml(file_path)
    reaction_count = az.get_number_of_reactions(data)
    return reaction_count


def get_file_path(path, file):
    file_path = os.path.join(path, file)
    return file_path


def calculate_mse(x1, x2):
    return (x1 - x2) ** 2


reaction_number_error = 0
combination_number_error = 0
matching_monomer_error = 0
reaction_const_conf_error = None
reaction_const_error = None
test_reaction_constants = []
model_reaction_constants = []
combined_score = []
combined_mse_const = []
combined_mse_conf = []
mse_temp = []
total_monomer_count = 0
combined_count_reactions_test = 0
combined_count_reactions_model = 0
combined_count_combinations_test = 0
combined_count_combinations_model = 0
solvent_error = 0
# prompt = get_prompt_template()


test_path = "./../test_data"
model_path = "./model_output"
test_files = sorted([f for f in os.listdir(test_path) if f.endswith(".yaml")])
model_files = sorted([f for f in os.listdir(model_path) if f.endswith(".yaml")])

wandb.init(
    project="Copolymer_reactivity",

    config={
        "model": "GPT3.5 turbo-1106",
        "temperature": 0.0,
        "paper number": 10,
        "token length": 16385,
    }
)

# comparison of numer of reactions and combinations
for test_file, model_file in zip(test_files, model_files):
    test_file_path = get_file_path(test_path, test_file)
    model_file_path = get_file_path(model_path, model_file)

    combinations_test_count = count_combinations(test_file_path)
    combinations_model_count = count_combinations(model_file_path)

    reactions_test_count = count_reactions(test_file_path)
    reactions_model_count = count_reactions(model_file_path)

    print(f"Comparing {test_file} and {model_file}")
    print(f"Number of different reactions in test data: {reactions_test_count}")
    print(f"Number of different reactions in model data: {reactions_model_count}")
    print(f"Number of different combinations in test data: {combinations_test_count}")
    print(f"Number of different combinations in model data: {combinations_model_count}")
    combined_count_reactions_test = combined_count_reactions_test + reactions_test_count
    combined_count_reactions_model = combined_count_reactions_model + reactions_model_count
    combined_count_combinations_test = combined_count_combinations_test + combinations_test_count
    combined_count_combinations_model = combined_count_combinations_model + combinations_model_count

    if reactions_test_count != reactions_model_count:
        reaction_number_error += 1
    if combinations_test_count != combinations_model_count:
        combination_number_error += 1

# comparison of the data of each unique reaction and combination
for test_file, model_file in zip(test_files, model_files):
    print(test_file)
    print(model_file)
    test_file_path = get_file_path(test_path, test_file)
    model_file_path = get_file_path(model_path, model_file)
    test_data = az.load_yaml(test_file_path)
    model_data = az.load_yaml(model_file_path)
    for i, reaction in enumerate(test_data['reactions']):
        print('iteration: ', i + 1)
        total_monomer_count += 1
        test_monomers = reaction['monomers']
        print(test_monomers)
        # try to find a matching monomer pair
        model_monomer_index = az.find_matching_reaction(model_data, test_monomers)
        test_monomer_index = i
        # if a monomer match is found: compare reaction conditions of this specific match
        if model_monomer_index is not None:
            sequence_change = az.get_sequence_of_monomers(model_data['reactions'][model_monomer_index]['monomers'], test_monomers)
            for j, combination in enumerate(reaction['combinations']):
                print('matching_monomer_error: ', matching_monomer_error)
                specific_reaction = model_data['reactions'][model_monomer_index]
                specific_combination = specific_reaction['combinations']
                model_combinations = az.extract_combinations(specific_combination)
                temperature, temp_unit, polym_method, polym_type, solvent, reaction_constants, reaction_constant_confidence, determination_method = az.get_metadata_polymerization(
                    combination)
                print(model_combinations)
                # try to find mathing reaction conditions or find best matching reaction conditions
                index, score = az.find_matching_combination(model_combinations, solvent, temperature, temp_unit, polym_type, polym_method, determination_method)
                print("combination match: ", score)
                combined_score.append(score)

                # comparison of temperature
                temperature_model, temp_unit_model = az.get_temp(combination, index)
                temperature_model, temperature = az.convert_unit(temperature_model, temperature, temp_unit_model,
                                                                 temp_unit)
                mse_temp_individual = calculate_mse(temperature_model, temperature)
                mse_temp.append(mse_temp_individual)
                print(temperature_model, temperature)

                # comparison of solvents
                solvent_model, smiles_solvent_model = az.get_solvent(combination, index)
                smiles_solvent_test = az.name_to_smiles(solvent)
                solvent_error += az.compare_smiles(smiles_solvent_test, smiles_solvent_model)

                # comparison of reactivity constant and confidence of reactivity constant
                test_reaction_constants, test_reaction_const_conf = az.get_reaction_constant(reaction_constants, reaction_constant_confidence)
                model_reaction_constants, model_reaction_const_conf = az.get_reaction_constant(
                    combination['reaction_constants'], combination['reaction_constant_conf'])
                if sequence_change == 1:
                    test_reaction_constants, model_reaction_constants = az.change_sequence(test_reaction_constants,
                                                                                           model_reaction_constants)
                    test_reaction_const_conf, model_reaction_const_conf = az.change_sequence(test_reaction_const_conf,
                                                                                             model_reaction_const_conf)

                if model_reaction_constants[0] is None or model_reaction_constants[1] is None:
                    if reaction_const_conf_error is None:
                        reaction_const_error = 1
                    else:
                        reaction_const_error += 1
                else:
                    for test_val, model_val in zip(test_reaction_constants, model_reaction_constants):
                        mse_const_individual = mean_squared_error([test_val], [model_val])
                        combined_mse_const.append(mse_const_individual)

                if test_reaction_const_conf == model_reaction_const_conf == [None, None]:
                    continue
                elif model_reaction_const_conf[0] is None or model_reaction_const_conf[1] is None:
                    if reaction_const_conf_error is None:
                        reaction_const_conf_error = 1
                    else:
                        reaction_const_conf_error += 1

                else:
                    for test_val, model_val in zip(test_reaction_const_conf, model_reaction_const_conf):
                        mse_conf_individual = mean_squared_error([test_val], [model_val])
                        combined_mse_conf.append(mse_conf_individual)
        else:
            matching_monomer_error += 1

# comparative metrics for each model run
monomer_error_rate = (matching_monomer_error / total_monomer_count)
average_mse_const = az.average(combined_mse_const)
average_mse_conf = az.average(combined_mse_conf)
average_mse_temp = az.average(mse_temp)
average_score = az.average(combined_score)
combination_error_rate = (abs(combined_count_combinations_model / combined_count_combinations_test))
reaction_error_rate = (abs(combined_count_reactions_model / combined_count_reactions_test))
print("matching-monomer-error: ", matching_monomer_error)
print(f"{matching_monomer_error} of {total_monomer_count} Monomer pairs are not found. Error rate: {round(((matching_monomer_error / total_monomer_count) * 100), 1)} %")
print(f"reaction-number-error is {reaction_number_error}. The reaction error rate is {round(reaction_error_rate * 100, 1)} %.")
print(f"combination-number-error is {combination_number_error}. The combination error rate is {round(combination_error_rate * 100, 1)} %")
print(f"reaction constant error is {reaction_const_error}")
print(f"reaction constant confidence error is {reaction_const_conf_error}")
print(f"average score of fuzzy matching is {average_score}")
print(f"average mse of reaction constants is {average_mse_const}")
print(f"average mse of reaction constants confidence is {average_mse_conf}")
print(f"average mse of temperature is {average_mse_temp}")

# metrics saved on weights and biases
wandb.log({"monomer-error": matching_monomer_error, "monomer-error-rate": monomer_error_rate,
           "reaction-number-error": reaction_number_error, "reaction-error-rate": reaction_error_rate,
           "combination-error": combination_number_error, "combination-error-rate": combination_error_rate,
           "reaction-constant-error": reaction_const_error, "reaction-const-conf-error": reaction_const_conf_error,
           "fuzzy matching score": average_score, "mse reaction const": average_mse_const,
           "mse const conf": average_mse_conf, "mse temperature": average_mse_temp})
