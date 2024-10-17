import matextract  # noqa: F401
from crossref.restful import Works
import json

works = Works(timeout=60)

# Performing the search for sources on the topic of buchwald-hartwig coupling for 10 papers
query_result = (
    works.query(bibliographic="'copolymerization' AND 'reactivity ratio")
    .select("DOI", "title", "author", "type", "publisher", "issued")
)

results = [item for item in query_result]

# Save 100 results including their metadata in a json file
with open("copol_crossref.json", "w") as file:
    json.dump(results, file)

print(results)