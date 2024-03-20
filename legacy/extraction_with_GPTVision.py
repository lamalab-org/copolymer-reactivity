from pdf2image import convert_from_path
import os
import copolextractor.prompter as prompter
import base64
import copolextractor.analyzer as az
from openai import OpenAI
import json
from PIL import Image


def encode_image_to_base64(filepath):
    with open(filepath, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def compress_and_save_image(image, output_path, quality=20, format='JPEG'):
    image.save(output_path, format=format, quality=quality)


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

    for idx, img in enumerate(pdf_images):
        image_path = os.path.join(output_folder_images,
                                  f"{filename}_page_{idx + 1}.jpg")
        compress_and_save_image(img, image_path, quality=20)
    print("Successfully converted PDF to images")
    images_base64 = [encode_image_to_base64(os.path.join(output_folder_images, f"{filename}_page_{idx + 1}.jpg")) for
                     idx in range(len(pdf_images))]

    prompt = [
        {
            "role": "user",
            "content": []
        }
    ]

    media_type = "image/jpeg"
    for index, data in enumerate(images_base64, start=1):
        prompt[0]["content"].append(
            {
                "type": "text",
                "text": f"Image {index}:"
            }
        )

        prompt[0]["content"].append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                    "detail": "low"
                },
            }
        )

    prompt[0]["content"].append(
        {
            "type": "text",
            "text": prompt_text
        }
    )
    prompt_str = json.dumps(prompt)

    print("model call starts")
    output = prompter.call_openai(prompt_str)
    output_model = prompter.format_output_as_json_and_yaml(i, output, output_folder)
    print("output_model: ", output_model)

    for a in range(number_of_model_calls):
        na_count = az.count_na_values(output_model)
        total_entry_count = az.count_total_entries(output_model)
        rate = az.calculate_rate(na_count, total_entry_count)
        print("NA-rate: ", rate)
        if rate > 0.3:
            print(f"model call number {a+2} of {filename}")
            prompt = prompter.update_prompt_with_text_and_images(prompt_text, output_model)
            print(prompt)
            #output = prompter.call_openai(prompt)
            #output_model = prompter.format_output_as_json_and_yaml(i, output, output_folder)
        else:
            print("NA-rate under 30%")
