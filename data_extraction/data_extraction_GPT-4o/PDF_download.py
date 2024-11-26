from scidownl import scihub_download
import json
from scidownl.core.task import ScihubTask
import copolextractor.utils as utils
import time


paper_count = 0
failed_download_count = 0
downloaded_paper_count = 0

file_path = "../doi_extraction/doi_output.json"
data = utils.load_json(file_path)
doi_list = data['doi_list']

print("doi list: ", doi_list)

#enhanced_doi_list = [
    #{
        #"paper": url,
        #"downloaded": False,
        #"paper_type": "doi" if "doi.org" in url else "other",
        #"index": index,
        #"out": f"./data_extraction_GPT-4o/paper_{index}.pdf"
   # }
   # for index, url in enumerate(doi_list)
#]

#print("enhanced doi list: ", enhanced_doi_list)
output_file_path = "data_extraction_GPT-4o/enhanced_doi_list1.json"
enhanced_doi_list = json.load(open(output_file_path, 'r'))
with open(output_file_path, 'w') as file:
    json.dump(enhanced_doi_list, file, indent=4)

out = "./data_extraction_GPT-4o/paper/"

help(scihub_download)

for index, paper_dict in enumerate(enhanced_doi_list):
    paper_count += 1
    paper = paper_dict["paper"]
    paper_type = paper_dict["paper_type"]
    out = paper_dict["out"]
    task = ScihubTask(source_keyword=paper, out=out)
    task.run()

    if task.context.get('status') in ['downloading_failed', 'extracting_failed']:
        print(f"Download failed: {task.context.get('error')}")
        failed_download_count += 1
    else:
        print(f"Download successful: PDF saved in {task.context.get('out')}")
        paper_dict["downloaded"] = True
        downloaded_paper_count += 1

    with open(output_file_path, 'w') as file:
        json.dump(enhanced_doi_list, file, indent=4)


print(f"out of {paper_count} papers {downloaded_paper_count} got successfully downloaded, {failed_download_count} "
      f"downloads failed")
