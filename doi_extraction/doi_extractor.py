import json
import re
import math

f = open('output.json')
data = json.load(f)

doi_list = []
no_doi_counter = 0
total_rows = len(data)

for i, row in enumerate(data):
    string = str(row)

    # Define a regex pattern to match the entire <a> tag with specific class names and extract the URL
    pattern = r'<a\s+class=\"(\S+\s+)*badge\s+badge-pill\s+badge-primary(\s+\S+)*\"\s+href=\"(https?://\S+?)\"\s+.*?>'

    # Use re.search to find the <a> tag with specified classes and extract the URL
    match = re.search(pattern, string)

    if match:
        extracted_url = match.group(3)  # Extracted URL within the tag
        doi_list.append(extracted_url)
    else:
        no_doi_counter +=1

    if i%10==0:
        print(round(i / total_rows * 100, 2))


print(len(doi_list))
print(no_doi_counter)


# Code to save doi_list to a JSON file
output_data = {"doi_list": doi_list}

with open('doi_output.json', 'w') as output_file:
    json.dump(output_data, output_file)

print("DOI list saved to doi_output.json")
