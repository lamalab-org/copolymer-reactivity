import json
import yaml
import os

json_dir_path = "test_data"
yaml_dir_path = "test_data"

for i, filename in enumerate(os.listdir(json_dir_path)):
    json_file_path = os.path.join(json_dir_path, filename)
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)

    output_name = os.path.join(yaml_dir_path, f"paper{i + 1}.yaml")
    with open(output_name, "w") as yaml_file:
        yaml.dump(data, yaml_file, allow_unicode=True)
