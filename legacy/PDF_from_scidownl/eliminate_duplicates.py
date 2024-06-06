import json


with open('enhanced_doi_list1.json', 'r') as file:
    data = json.load(file)


unique_papers = {}
for entry in data:
    paper = entry['paper']
    if paper not in unique_papers:
        unique_papers[paper] = entry

unique_data = list(unique_papers.values())

with open('enhanced_doi_list_unique.json', 'w') as file:
    json.dump(unique_data, file, indent=4)


print(f"Original number of entries: {len(data)}")
print(f"Number of entries after removing duplicates: {len(unique_data)}")
