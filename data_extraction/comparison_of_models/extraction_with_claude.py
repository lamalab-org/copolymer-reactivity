from pdf2image import convert_from_path
import os
import anthropic
import time
import copolextractor.prompter as prompter
import copolextractor.analyzer as az
import copolextractor.image_processer as ip


start = time.time()

input_folder = "./../pdf_testset"
output_folder_images = "./processed_images"
output_folder = "model_output_claude"
number_of_model_calls = 2
parsing_error = 0
total_input_token = 0
total_output_token = 0
number_of_calls = 0
input_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".pdf")])
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
prompt_text = prompter.get_prompt_template()


for i, filename in enumerate(input_files):
    i += 1
    file_path = os.path.join(input_folder, filename)
    print(file_path)
    print(filename)
    pdf_images = convert_from_path(file_path)

    for idx in range(len(pdf_images)):
        image_path = os.path.join(
            output_folder_images, f"{filename}_page_{idx + 1}.png"
        )
        pdf_images[idx].save(image_path, "PNG")
    print("Successfully converted PDF to processed_images")
    for j, image in enumerate(pdf_images):
        resized_image = ip.resize_image(image, 1024)
        rotate_image = ip.correct_text_orientation(
            resized_image, output_folder_images, file_path, j
        )

    prompt = prompter.get_prompt_claude_vision(
        output_folder_images, filename, pdf_images, prompt_text
    )

    print("model call starts")
    output, input_token, output_token = prompter.call_claude3(prompt)
    total_input_token += input_token
    total_output_token += output_token
    number_of_calls += 1
    output_model = prompter.format_output_claude_as_json_and_yaml(
        i, output, output_folder
    )
    print("output_model: ", output_model)

    for a in range(number_of_model_calls):
        na_count = az.count_na_values(output_model)
        total_entry_count = az.count_total_entries(output_model)
        rate = az.calculate_rate(na_count, total_entry_count)
        print("NA-rate: ", rate)
        if rate > 0.3 or output_model is None:
            print(f"model call number {a+2} of {filename}")
            updated_prompt_text = prompter.update_prompt(prompt_text, output_model)
            prompt = prompter.get_prompt_claude_vision(
                output_folder_images, filename, pdf_images, updated_prompt_text
            )
            output, input_token, output_token = prompter.call_claude3(prompt)
            total_input_token += input_token
            total_output_token += output_token
            number_of_calls += 1
            output_model = prompter.format_output_claude_as_json_and_yaml(
                i, output, output_folder
            )
        else:
            print("NA-rate under 30%")
        print("tokens input used: ", input_token)
        print("tokens output_2 used: ", output_token)
        print("total number of model call: ", number_of_calls)
    if output_model is None:
        parsing_error += 1

print(f"out of {i} papers, {parsing_error} papers are extracted in invalid json format")


end = time.time()
execution_time = start - end

print("execution time: ", execution_time)
