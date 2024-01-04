import yaml
import os


def load_yaml_combinations(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
        combinations_count = 0
        for reaction in data["reaction"]:
            combinations_count += len(reaction["combinations"])
        return combinations_count


def load_yaml_reactions(file_path):
    with open(file_path, "r") as file:
        yaml.safe_load(file)
        reaction_count = 0
        reaction_count += len("reaction")
        return reaction_count


reaction_number_error = 0
combination_number_error = 0

test_path = "test_data"
model_path = "test_data2"
test_files = sorted([f for f in os.listdir(test_path) if f.endswith(".yaml")])
model_files = sorted([f for f in os.listdir(model_path) if f.endswith(".yaml")])

for test_file, model_file in zip(test_files, model_files):
    test_file_path = os.path.join(test_path, test_file)
    model_file_path = os.path.join(model_path, model_file)

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
