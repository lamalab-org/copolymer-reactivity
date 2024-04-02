from pdf2image import convert_from_path
import os
import copolextractor.prompter as prompter
import base64
import copolextractor.analyzer as az
from openai import OpenAI
from PIL import Image
import time
import io


def resize_image(image, max_dimension):
    width, height = image.size

    # Check if the image has a palette and convert it to true color mode
    if image.mode == "P":
        if "transparency" in image.info:
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")
    # convert to black and white
    # image = image.convert("L")

    if width > max_dimension or height > max_dimension:
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        image = image.resize((new_width, new_height), Image.LANCZOS)

        timestamp = time.time()

    return image


def convert_to_jpeg(image):
    with io.BytesIO() as output:
        image.save(output, format="jpeg")
        return output.getvalue()


def process_image(image, max_size):
    width, height = image.size
    # resized_image = resize_image(image, max_size)
    jpeg_image = convert_to_jpeg(image)
    return (
        base64.b64encode(jpeg_image).decode("utf-8"),
        max(width, height),  # same tuple metadata
    )


input_folder = "./../pdfs"
output_folder_images = "./images"
output_folder = "model_output"
number_of_model_calls = 2
input_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".pdf")])
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

prompt_text = prompter.get_prompt_template()


for i, filename in enumerate(input_files):
    file_path = os.path.join(input_folder, filename)
    print(file_path)
    print(filename)
    pdf_images = convert_from_path(file_path)

    images_base64 = [process_image(image, 1024)[0] for image in pdf_images]

    content = prompter.get_prompt_vision_model(images_base64, prompt_text)

    print("model call starts")
    output = prompter.call_openai(prompt=content)
    output_model = prompter.format_output_as_json_and_yaml(i, output, output_folder)
    print("output_model: ", output_model)
    print(f"model call number 2 of {filename}")
    updated_prompt = prompter.update_prompt(prompt_text, output_model)
    content = prompter.get_prompt_vision_model(images_base64, updated_prompt)
    print(updated_prompt)
    output = prompter.call_openai(content)

    output_model = prompter.format_output_as_json_and_yaml(i, output, output_folder)

    for a in range(number_of_model_calls):
        na_count = az.count_na_values(output_model)
        total_entry_count = az.count_total_entries(output_model)
        rate = az.calculate_rate(na_count, total_entry_count)
        print("NA-rate: ", rate)
        if rate > 0.3 or output_model is None:
            print(f"model call number {a+3} of {filename}")
            updated_prompt = prompter.update_prompt(prompt_text, output_model)
            content = prompter.get_prompt_vision_model(images_base64, updated_prompt)
            print(updated_prompt)
            output = prompter.call_openai(content)
            output_model = prompter.format_output_as_json_and_yaml(i, output, output_folder)
        else:
            print("NA-rate under 30%")
