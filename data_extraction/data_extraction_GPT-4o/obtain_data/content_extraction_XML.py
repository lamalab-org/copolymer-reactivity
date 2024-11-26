import os
import re
import json
from bs4 import BeautifulSoup as bs


# Function to remove newlines and extra spaces
def remove_newlines(text):
    return text.replace('\n', ' ').strip()


# Function to extract child elements
def extract_children(element):
    children = []
    for child in element.children:
        if child.name is not None or (child.string and child.string.strip() != ''):
            children.append(child)
    if len(children) < 1:
        raise ValueError(f'{element.name} should not be empty')
    else:
        return children


# Function to check the name of an element
def element_name(element, name):
    if ':' in name:
        prefix, node_name = name.split(':')
        return (element.prefix == prefix and element.name == node_name)
    else:
        return element.name == name


# Function to extract text from paragraph elements
def extract_para_elements(element):
    return re.sub(' +', ' ', remove_newlines(element.text))


# Function to extract and preserve original table formatting
def extract_table_elements(element):
    # Get the entire table element as a string to preserve formatting
    return str(element)


# Function to recursively extract sections and their content
def extract_section_elements(element):
    paragraphs = []
    title = ""
    for child in extract_children(element):
        if child.name == 'head':  # Section title
            title = remove_newlines(child.text)
        elif child.name == 'p':  # Paragraphs
            paragraphs.append(extract_para_elements(child))
        elif child.name == 'div':  # Nested sections
            paragraphs.append(extract_section_elements(child))
        elif child.name == 'figure':  # Skip figures
            continue  # Skip figure elements or handle them if needed
        elif child.name == 'table':  # Handle tables
            table_content = extract_table_elements(child)
            paragraphs.append({"table": table_content})
        else:
            print(f"Unexpected element: {child.name}")
            continue  # Skip unexpected elements

    return {title: paragraphs}


# Function to extract all sections and their content from the document
def extract_sections_text(soup):
    body = soup.find('body')
    sections_content = []
    if body is not None:
        for child in extract_children(body):
            if child.name == 'div':
                sections_content.append(extract_section_elements(child))
            elif child.name == 'p':  # Handle paragraphs directly within the body
                paragraphs = extract_para_elements(child)
                sections_content.append({"Unnamed Section": [paragraphs]})
            elif child.name == 'figure':  # Skip or handle figures directly in the body
                continue  # Skip figure elements, or you could extract figure captions if needed
            elif child.name == 'table':  # Handle tables directly in the body
                table_content = extract_table_elements(child)
                sections_content.append({"table": table_content})
            else:
                raise ValueError(
                    f'Unexpected section name. Obtained: "{child.name}". Expecting: "div", "p", "table", or "figure".')
    return sections_content


# Function to extract the DOI or another identifier from the XML
def extract_identifier(soup):
    # Assuming the DOI is in an element like <idno type="doi"> or similar
    doi_tag = soup.find('idno', {'type': 'doi'})
    if doi_tag and doi_tag.text:
        return doi_tag.text.strip()

    # Fallback: Return None if no DOI is found
    return None


# Function to process each XML file to extract content and save it
def process_xml_file(input_file, output_file, i):
    with open(input_file, 'r') as f:
        file = f.read()

    soup = bs(file, 'xml')
    sections_content = extract_sections_text(soup)

    # Extract DOI or another identifier
    identifier = extract_identifier(soup)
    if not identifier:
        identifier = f"Unknown Source {i}"  # Fallback if no DOI is found

    # Save the extracted content to a JSON file
    output_data = {
        "doi": identifier,
        "content": sections_content
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)


def process_all_xml_files(input_dir, output_dir):
    i = 0  # Initialize counter for `i`

    # Iterate over all subdirectories in the input directory
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)

        # Ensure it's a directory before proceeding
        if os.path.isdir(subdir_path):
            # Iterate over all files in the subdirectory
            for filename in os.listdir(subdir_path):
                if filename.endswith(".xml"):
                    i += 1  # Increment `i` for each file
                    input_file = os.path.join(subdir_path, filename)
                    output_file = os.path.join(output_dir, f"content_{i}.json")

                    # Process the XML file
                    process_xml_file(input_file, output_file, i)


if __name__ == "__main__":
    input_dir = "../files_pygetpaper/XML_cleaning/clean_XML_files"  # Path to your input directory
    output_dir = "../files_pygetpaper/XML_cleaning/clean_JSON_files"  # Path to your collected output directory

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process all XML files
    process_all_xml_files(input_dir, output_dir)
