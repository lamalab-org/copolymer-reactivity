from pdf2image import convert_from_path
import os
import anthropic
import copolextractor.prompter as prompter
import base64
import copolextractor.analyzer as az


input_folder = "./../pdfs"
output_folder_images = "./images"
output_folder = "model_output_claude"
number_of_model_calls = 2
parsing_error = 0
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
        image_path = os.path.join(output_folder_images, f"{filename}_page_{idx + 1}.png")
        pdf_images[idx].save(image_path, 'PNG')
    print("Successfully converted PDF to images")
    prompt = prompter.get_prompt_claude_vision(output_folder_images, filename, pdf_images, prompt_text)

    print("model call starts")
    output = prompter.call_claude3(prompt)
    output_model = prompter.format_output_claude_as_json_and_yaml(i, output, output_folder)
    print("output_model: ", output_model)

    for a in range(number_of_model_calls):
        na_count = az.count_na_values(output_model)
        total_entry_count = az.count_total_entries(output_model)
        rate = az.calculate_rate(na_count, total_entry_count)
        print("NA-rate: ", rate)
        if rate > 0.3 or output_model is None:
            print(f"model call number {a+2} of {filename}")
            updated_prompt_text = prompter.update_prompt(prompt_text, output_model)
            prompt = prompter.get_prompt_claude_vision(output_folder_images, filename, pdf_images, updated_prompt_text)
            output = prompter.call_claude3(prompt)
            output_model = prompter.format_output_claude_as_json_and_yaml(i, output, output_folder)
        else:
            print("NA-rate under 30%")
    if output_model is None:
        parsing_error += 1

print(f"out of {i} papers, {parsing_error} papers are extracted in invalid json format")
