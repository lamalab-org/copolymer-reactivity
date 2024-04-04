from pdf2image import convert_from_path
import os
from openai import OpenAI
import copolextractor.prompter as prompter
import copolextractor.analyzer as az
import copolextractor.image_processer as ip


input_folder = "./../pdfs"
output_folder_images = "./images"
output_folder = "model_output"
number_of_model_calls = 2
input_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".pdf")])
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

prompt_text = prompter.get_prompt_template()


for i, filename in enumerate(input_files):
    file_path = os.path.join(input_folder, filename)
    print("processing ", filename)
    pdf_images = convert_from_path(file_path)

    images_base64 = [ip.process_image(image, 2048, output_folder_images, file_path, j)[0] for j, image in enumerate(pdf_images)]

    content = prompter.get_prompt_vision_model(images_base64, prompt_text)

    print("model call starts")
    output = prompter.call_openai(prompt=content)
    output_model = prompter.format_output_as_json_and_yaml(i, output, output_folder)
    print("output_model: ", output_model)
    #print(f"model call number 2 of {filename}")
    #updated_prompt = prompter.update_prompt(prompt_text, output_model)
    #content = prompter.get_prompt_vision_model(images_base64, updated_prompt)
    #output = prompter.call_openai(content)
    #output_model = prompter.format_output_as_json_and_yaml(i, output, output_folder)

    for a in range(number_of_model_calls):
        na_count = az.count_na_values(output_model)
        total_entry_count = az.count_total_entries(output_model)
        rate = az.calculate_rate(na_count, total_entry_count)
        print("NA-rate: ", rate)
        if rate > 0.3 or output_model is None:
            print(f"model call number {a+3} of {filename}")
            updated_prompt = prompter.update_prompt(prompt_text, output_model)
            content = prompter.get_prompt_vision_model(images_base64, updated_prompt)
            output = prompter.call_openai(content)
            output_model = prompter.format_output_as_json_and_yaml(i, output, output_folder)
        else:
            print("NA-rate under 30%")
