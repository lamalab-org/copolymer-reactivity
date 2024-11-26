import os
import requests
import time
import logging

# Set up logging
logging.basicConfig(filename='../files_pygetpaper/XML_cleaning/processing_errors.log', level=logging.ERROR)

# Function to process a single XML file
def process_xml_file(xml_file, output_file, timeout=60):
    files = {
        "input": open(xml_file, "rb"),
        "segmentSentences": (
            None,
            "1",
        ),  # Optional, set to '1' for sentence segmentation
        "grobidRefine": (None, "1"),  # Optional, set to '1' for refining with Grobid
    }
    for attempt in range(5):  # Retry up to 5 times
        try:
            response = requests.post(server_url, files=files, timeout=timeout)
            if response.status_code == 200:
                with open(output_file, "wb") as f:
                    f.write(response.content)
                print(f"Processed {xml_file} successfully.")
                return output_file
            else:
                print(f"Failed to process {xml_file}. Status code: {response.status_code}")
            break
        except requests.exceptions.Timeout:
            print(f"Timeout error for {xml_file}. Moving to the next file.")
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error: {e}. Retrying in 10 seconds...")
            time.sleep(10)
        except requests.exceptions.RequestException as e:
            logging.error(f"Error processing {xml_file}: {e}")
            print(f"An error occurred: {e}. Moving to the next file.")
            return None
        except Exception as e:
            logging.error(f"Unexpected error processing {xml_file}: {e}")
            print(f"An unexpected error occurred: {e}. Moving to the next file.")
            return None


# Main function to process all XML files in subdirectories
def process_all_xml_files(input_dir, output_dir):
    # Iterate over all subdirectories in the input directory
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)

        # Ensure it's a directory before proceeding
        if os.path.isdir(subdir_path):
            # Iterate over all files in the subdirectory
            for filename in os.listdir(subdir_path):
                if filename.endswith(".xml"):
                    print(f'processing {filename} in {subdir_path}.')
                    input_file = os.path.join(subdir_path, filename)
                    output_file = os.path.join(output_dir, subdir, filename.replace(".xml", ".tei.xml"))

                    # Ensure the output directory for this subfolder exists
                    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

                    # Check if the output file already exists
                    if os.path.exists(output_file):
                        print(f"Output file {output_file} already exists. Skipping.")
                        continue

                    # Process the XML file with a timeout and error handling
                    process_xml_file(input_file, output_file)

if __name__ == "__main__":
    input_dir = "../files_pygetpaper/"  # Path to your input directory
    print(input_dir)
    output_dir = "../files_pygetpaper/XML_cleaning/clean_XML_files"  # Path to your output directory

    # Define the Pub2TEI server URL
    server_url = "http://localhost:8060/service/processXML"  # Use port 8080

    process_all_xml_files(input_dir, output_dir)
