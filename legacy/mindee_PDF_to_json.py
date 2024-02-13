import os
import json

os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/lib'
os.environ['LD_LIBRARY_PATH'] = '/opt/homebrew/lib'


from doctr.models import ocr_predictor
from doctr.io import DocumentFile

model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True, export_as_straight_boxes=True)

doc = DocumentFile.from_pdf("./../pdfs/paper1.pdf")
result = model(doc)

json_output = result.export()
output_file_path = './output_json_mindee/paper1.json'


with open(output_file_path, 'w') as json_file:
    json.dump(json_output, json_file, indent=4)