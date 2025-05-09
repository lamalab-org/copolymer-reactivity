import os
import src.copolextractor.analyzer as az
from sklearn.metrics import mean_squared_error, mean_absolute_error
import wandb
import src.copolextractor.prompter as prompter
import src.copolextractor.utils as utils
import traceback
import json
import matplotlib.pyplot as plt


def count_reaction_conditions(file_path):
    data = utils.load_yaml(file_path)
    reaction_conditions_count = az.get_total_number_of_reaction_conditions(data)
    return reaction_conditions_count


def count_reactions(file_path):
    data = utils.load_yaml(file_path)
    reaction_count = az.get_number_of_reactions(data)
    return reaction_count


def get_file_path(path, file):
    file_path = os.path.join(path, file)
    return file_path


def calculate_mse(x1, x2):
    return (x1 - x2) ** 2


def calculate_rate(x1, x2):
    if x2 != 0:
        rate = abs(x1 / x2)
        return rate
    else:
        return None


def calculate_deviation(mae, y_true):
    if y_true != 0:
        deviation = mae / y_true
    else:
        deviation = mae
    return deviation


def load_existing_data(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {"test_data": [], "model_data": []}


def append_new_data(existing_data, new_test_data, new_model_data):
    existing_data["test_data"].append(new_test_data)
    existing_data["model_data"].append(new_model_data)
    return existing_data


def save_data(file_path, data):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


reaction_number_error = 0
reaction_conditions_number_error = 0
matching_monomer_error = 0
reaction_const_conf_NA_count = 0
reaction_const_NA_count = 0
test_reaction_constants = []
model_reaction_constants = []
combined_score = []
combined_mse_const = []
combined_mse_conf = []
mse_temp = []
total_monomer_count = 0
combined_count_reactions_test = 0
combined_count_reactions_model = 0
combined_count_reaction_conditions_test = 0
combined_count_reaction_conditions_model = 0
solvent_error = 0
parsing_error = 0
file_count = 0
prompt = prompter.get_prompt_template()
prompt_addition = prompter.get_prompt_addition()
reaction_condition_count = 0
total_entries_count = 0
na_count = 0
correct_reaction_count = 0
mae_const_current_rxn = 0
mae_conf_current_rxn = 0
deviation_conf = None
deviation_const = None
deviation_temp = None
all_test_data = []
all_model_data = []
all_test_data_temp = []
all_model_data_temp = []

test_path = "test_data"
model_path = "../model_output_GPT4-o"
test_files = sorted([f for f in os.listdir(test_path) if f.endswith(".yaml")])
model_files = sorted([f for f in os.listdir(model_path) if f.endswith(".yaml")])

wandb.init(
    project="Copolymer_extraction",
    config={
        "model": "gpt4-o",
        "paper number": 10,
        "token length": "",
        "input": "processed_images",
        "number of model calls max": 3,
        "number of model calls min": 1,
        "temperature": 0.0,
        "seed": 12345,
        "max resolution": "high",
        "deviation of correct rxn": 0.01,
        "input tokens used": 100085,
        "output_2 tokens used": 7813,
        "total model calls": 10,
        "time used": 383.4063169956207,
    },
)

# comparison of number of reactions and reactions with different reaction_conditions
for test_file, model_file in zip(test_files, model_files):
    test_file_path = get_file_path(test_path, test_file)
    model_file_path = get_file_path(model_path, model_file)

    file_count += 1
    try:
        reaction_conditions_test_count = count_reaction_conditions(test_file_path)
        reaction_conditions_model_count = count_reaction_conditions(model_file_path)

        reactions_test_count = count_reactions(test_file_path)
        reactions_model_count = count_reactions(model_file_path)

        print(f"Comparing {test_file} and {model_file}")
        print(f"Number of different reactions in test data: {reactions_test_count}")
        print(f"Number of different reactions in model data: {reactions_model_count}")
        print(
            f"Number of different reaction_conditions in test data: {reaction_conditions_test_count}"
        )
        print(
            f"Number of different reaction_conditions in model data: {reaction_conditions_model_count}"
        )
        combined_count_reactions_test += reactions_test_count
        combined_count_reactions_model += reactions_model_count
        combined_count_reaction_conditions_test += reaction_conditions_test_count
        combined_count_reaction_conditions_model += reaction_conditions_model_count

        if reactions_test_count != reactions_model_count:
            reaction_number_error += 1
        if reaction_conditions_test_count != reaction_conditions_model_count:
            reaction_conditions_number_error += 1

        # comparing reactions and reaction conditions
        test_data = utils.load_yaml(test_file_path)
        model_data = utils.load_yaml(model_file_path)

        # counting empty entries
        total_entries_count += az.count_total_entries(model_data)
        na_count += az.count_na_values(model_data)
        # iteration over each monomer pair of test data
        for i, reaction in enumerate(test_data["reactions"]):
            print("iteration: ", i + 1)
            total_monomer_count += 1
            test_monomers = reaction["monomers"]

            # try to find a matching monomer pair
            model_monomer_index = az.find_matching_reaction(model_data, test_monomers)
            test_monomer_index = i
            # if a monomer match is found: compare reaction conditions of this specific match
            if model_monomer_index is not None:
                sequence_change = az.get_sequence_of_monomers(
                    model_data["reactions"][model_monomer_index]["monomers"],
                    test_monomers,
                )
                # iteration over each reaction condition of the test monomer pair with found index
                for j, reaction_conditions in enumerate(
                    reaction["reaction_conditions"]
                ):
                    reaction_condition_count += 1
                    print("matching_monomer_error: ", matching_monomer_error)
                    # model data: specific reaction condition
                    specific_reaction = model_data["reactions"][model_monomer_index]
                    specific_reaction_conditions = specific_reaction[
                        "reaction_conditions"
                    ]
                    model_reaction_conditions = az.extract_reaction_conditions(
                        specific_reaction_conditions
                    )
                    # test data: specific reaction condition for this iteration
                    (
                        temperature,
                        temp_unit,
                        polym_method,
                        polym_type,
                        solvent,
                        reaction_constants,
                        reaction_constant_confidence,
                        determination_method,
                    ) = az.get_metadata_polymerization(reaction_conditions)

                    # try to find matching or best matching reaction conditions of model data to specific test conditions
                    index, score = az.find_matching_reaction_conditions(
                        model_reaction_conditions,
                        solvent,
                        temperature,
                        temp_unit,
                        polym_type,
                        polym_method,
                        determination_method,
                    )
                    print("reaction_conditions match: ", score)
                    combined_score.append(score)

                    # comparison of temperature
                    temperature_model, temp_unit_model = az.get_temp(
                        model_reaction_conditions, index
                    )
                    if temperature_model is not None and temp_unit_model is not None:
                        temperature = az.convert_unit(temperature, temp_unit)
                        temperature_model = az.convert_unit(
                            temperature_model, temp_unit_model
                        )
                    if temperature_model is not None:
                        temperature_model_list = [temperature_model]
                        temperature_list = [temperature]

                        all_test_data_temp.append({"temperature": temperature})
                        all_model_data_temp.append({"temperature": temperature_model})

                        mse_temp_individual = calculate_mse(
                            temperature_model, temperature
                        )
                        mse_temp.append(mse_temp_individual)
                        mae_temp_current_rxn = mean_absolute_error(
                            temperature_list, temperature_model_list
                        )
                        deviation_temp = mae_temp_current_rxn / temperature
                        print(
                            "mae temp: ",
                            mae_temp_current_rxn,
                            "deviation: ",
                            deviation_temp,
                        )

                    # comparison of solvents
                    solvent_model, smiles_solvent_model = az.get_solvent(
                        model_reaction_conditions, index
                    )
                    print(
                        "solvent model: ", solvent_model, " vs. solvent test: ", solvent
                    )
                    smiles_solvent_test = utils.name_to_smiles(solvent)
                    solvent_missmatch = False
                    solvent_error += az.compare_smiles(
                        smiles_solvent_test, smiles_solvent_model
                    )
                    if (
                        az.compare_smiles(smiles_solvent_test, smiles_solvent_model)
                        == 1
                    ):
                        solvent_error += 1
                        solvent_missmatch = True

                    # comparison of reactivity constant and confidence of reactivity constant
                    test_reaction_constants, test_reaction_constant_confidence = (
                        az.get_reaction_const_list(
                            reaction_constants, reaction_constant_confidence
                        )
                    )
                    print(
                        f"test reaction_const: {reaction_constants}, test reaction_const_confidence:"
                        f" {test_reaction_constant_confidence}"
                    )
                    model_reaction_constants, model_reaction_const_conf = (
                        az.get_reaction_constant(model_reaction_conditions, index)
                    )
                    print(
                        "model-reaction-const: ",
                        model_reaction_constants,
                        ", model-reaction-conf: ",
                        model_reaction_const_conf,
                    )

                    if sequence_change == 1:
                        test_reaction_constants, model_reaction_constants = (
                            az.change_sequence(
                                test_reaction_constants, model_reaction_constants
                            )
                        )
                        test_reaction_constant_confidence, model_reaction_const_conf = (
                            az.change_sequence(
                                test_reaction_constant_confidence,
                                model_reaction_const_conf,
                            )
                        )

                    all_test_data.append(
                        {
                            "constants": test_reaction_constants,
                            "confidence": test_reaction_constant_confidence,
                        }
                    )
                    all_model_data.append(
                        {
                            "constants": model_reaction_constants,
                            "confidence": model_reaction_const_conf,
                        }
                    )

                    if (
                        model_reaction_constants[0] is None
                        or model_reaction_constants[1] is None
                    ):
                        if (
                            test_reaction_constants[0] is not None
                            or test_reaction_constants[1] is not None
                        ):
                            reaction_const_NA_count += 1
                            mse_const_individual = 1
                            mae_const_current_rxn = 1
                            deviation_const = 1
                        else:
                            mse_const_individual = 0
                            mae_const_current_rxn = 0
                            deviation_const = 0
                    else:
                        for test_val, model_val in zip(
                            test_reaction_constants, model_reaction_constants
                        ):
                            if model_val is not None and test_val is not None:
                                mse_const_individual = mean_squared_error(
                                    [test_val], [model_val]
                                )
                                mae_const_current_rxn = mean_absolute_error(
                                    [test_val], [model_val]
                                )
                                combined_mse_const.append(mse_const_individual)
                                if deviation_conf is not None:
                                    deviation_const = (
                                        deviation_conf
                                        + (
                                            calculate_deviation(
                                                mae_const_current_rxn, test_val
                                            )
                                        )
                                    ) / 2
                                else:
                                    deviation_const = calculate_deviation(
                                        mae_const_current_rxn, test_val
                                    )

                    if (
                        model_reaction_const_conf[0] is None
                        or model_reaction_const_conf[1] is None
                    ):
                        if (
                            test_reaction_constant_confidence[0] is not None
                            or test_reaction_constant_confidence[1] is not None
                        ):
                            reaction_const_conf_NA_count += 1
                            mse_conf_individual = 1
                            mae_conf_current_rxn = 1
                            deviation_conf = 1
                        else:
                            mse_conf_individual = 0
                            mae_conf_current_rxn = 0
                            deviation_conf = 0
                    else:
                        for test_val, model_val in zip(
                            test_reaction_constant_confidence, model_reaction_const_conf
                        ):
                            if (
                                test_reaction_constant_confidence[0] is None
                                or test_reaction_constant_confidence[1] is None
                            ):
                                mse_const_individual = 1
                                mae_conf_current_rxn = 1
                            elif model_val is not None and test_val is not None:
                                mse_conf_individual = mean_squared_error(
                                    [test_val], [model_val]
                                )
                                combined_mse_conf.append(mse_conf_individual)
                                mae_conf_current_rxn = mean_absolute_error(
                                    [test_val], [model_val]
                                )
                                if deviation_conf is not None:
                                    deviation_conf = (
                                        deviation_conf
                                        + (
                                            calculate_deviation(
                                                mae_conf_current_rxn, test_val
                                            )
                                        )
                                    ) / 2
                                else:
                                    deviation_conf = calculate_deviation(
                                        mae_conf_current_rxn, test_val
                                    )
                    try:
                        print(
                            f"deviation temp: {deviation_temp}, deviation const: {deviation_const}, deviation conf: {deviation_conf}"
                        )
                        if (
                            deviation_temp < 0.01
                            and deviation_conf < 0.01
                            and deviation_const < 0.01
                        ):
                            correct_reaction_count += 1
                            print("reaction is completely correct")
                        else:
                            print("rxn is not completely correct")
                    except (TypeError, KeyError, ValueError) as e:
                        print(ValueError, e)
                        print(TypeError, e)
                        print(KeyError, e)
            else:
                matching_monomer_error += 1
    except (TypeError, KeyError, ValueError) as e:
        print(ValueError, e)
        print(TypeError, e)
        print(KeyError, e)
        error_details = traceback.format_exc()
        print("error details: " + error_details)
        print(f"file {test_file} not parsable")
        parsing_error += 1
    mae_const_current_rxn = 1
    mae_conf_current_rxn = 1
    deviation_conf = None
    deviation_const = None

correct_reaction_rate = (
    calculate_rate(correct_reaction_count, combined_count_reaction_conditions_test)
    * 100
)

# existing_data_const = load_existing_data(const_json_file_path)
# existing_data_temp = load_existing_data(temp_json_file_path)
# updated_data_const = append_new_data(existing_data_const, all_test_data, all_model_data)
# updated_data_temp = append_new_data(existing_data_temp, all_test_data_temp, all_model_data_temp)
# save_data(const_json_file_path, updated_data_const)
# save_data(temp_json_file_path, updated_data_temp)

print("reaction count", reaction_condition_count)
print("correct rxn count", correct_reaction_count)

print(
    f"out of {reaction_condition_count} matching reactions, {correct_reaction_count} are completely correct. ({calculate_rate(correct_reaction_count, reaction_condition_count) * 100} %)"
)
print(
    f"out of {combined_count_reaction_conditions_test} reactions, {correct_reaction_count} are completely correct. ({calculate_rate(correct_reaction_count, combined_count_reaction_conditions_test) * 100} %)"
)

# comparative metrics for each model run
monomer_error_rate = calculate_rate(matching_monomer_error, total_monomer_count) * 100
average_mse_const = az.average(combined_mse_const)
average_mse_conf = az.average(combined_mse_conf)
average_mse_temp = az.average(mse_temp)
average_score = az.average(combined_score)
reaction_conditions_number_error_rate = (
    calculate_rate(
        combined_count_reaction_conditions_model,
        combined_count_reaction_conditions_test,
    )
    * 100
)
reaction_number_error_rate = (
    calculate_rate(combined_count_reactions_model, combined_count_reactions_test) * 100
)
parsing_error_rate = calculate_rate(parsing_error, file_count) * 100
reaction_constant_NA_rate = (
    calculate_rate(reaction_const_NA_count, reaction_condition_count) * 100
)
reaction_constant_conf_NA_rate = (
    calculate_rate(reaction_const_conf_NA_count, reaction_condition_count) * 100
)
na_entry_rate = calculate_rate(na_count, total_entries_count) * 100
solvent_error_rate = (
    calculate_rate(solvent_error, combined_count_reaction_conditions_test) * 100
)

print(
    f"number of empty entries: {na_count}. rate of empty entries is: {na_entry_rate} %."
)
print(
    f"parsing error is {parsing_error}. The parsing error rate is {parsing_error_rate}."
)
print("matching-monomer-error: ", matching_monomer_error)
print(
    f"{matching_monomer_error} of {total_monomer_count} Monomer pairs are not found. Error rate: {correct_reaction_rate} %."
)
print(
    f"reaction-number-error is {reaction_number_error}. The reaction error rate is {reaction_number_error_rate} %."
)
print(
    f"reaction_conditions-number-error is {reaction_conditions_number_error}. The reaction_conditions error rate is {reaction_conditions_number_error_rate} %."
)
print(
    f"reaction constant error is {reaction_const_NA_count}. The reaction constant error rate is {reaction_constant_NA_rate} %."
)
print(
    f"reaction constant confidence error is {reaction_const_conf_NA_count}. The reaction constant confidence error rate is {reaction_constant_conf_NA_rate} %."
)
print(f"average score of fuzzy matching is {average_score}")
print(f"average mse of reaction constants is {average_mse_const}")
print(f"average mse of reaction constants confidence is {average_mse_conf}")
print(f"average mse of temperature is {average_mse_temp}")

print("reaction cond. test count: ", combined_count_reaction_conditions_test)

# metrics saved on weights and biases
wandb.log(
    {
        "prompt": prompt,
        "prompt_addition": prompt_addition,
        "parsing-error": parsing_error,
        "parsing-error-rate": parsing_error_rate,
        "number of empty entries": na_count,
        "rate of empty entries": na_entry_rate,
        "correct reaction count": correct_reaction_count,
        "correct-reaction-rate": correct_reaction_rate,
        "matching-monomer-error": matching_monomer_error,
        "matching-monomer-error-rate": monomer_error_rate,
        "reaction-number-error": reaction_number_error,
        "reaction-number-error-rate": reaction_number_error_rate,
        "reaction_conditions-error": reaction_conditions_number_error,
        "reaction_conditions-number-error-rate": reaction_conditions_number_error_rate,
        "reaction-constant-NA-count": reaction_const_NA_count,
        "reaction-const-NA-rate": reaction_constant_NA_rate,
        "reaction-const-conf-NA-rate": reaction_constant_conf_NA_rate,
        "reaction-const-conf-NA-count": reaction_const_conf_NA_count,
        "fuzzy matching score": average_score,
        "mse reaction const": average_mse_const,
        "mse reaction const conf": average_mse_conf,
        "mse temperature": average_mse_temp,
        "solvent error": solvent_error,
        "solvent error rate": solvent_error_rate,
    }
)
