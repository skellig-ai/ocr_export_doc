from google.cloud import documentai_v1 as documentai
from google.cloud import storage
import numpy as np
import argparse
import imutils
import cv2
import pandas as pd
from config import *
from utility import *


ap = argparse.ArgumentParser()
ap.add_argument('-f', '--file', required=False, 
                help="file to be digitised,  valid formats: PDF, TIFF, GIF, JPEG, PNG, BMP, WEBP",
                default='EUR-MED_C1300_Skywalker.pdf')
ap.add_argument('-t', '--template', required=False, 
                help="file to be digitised,  valid formats: PDF, TIFF, GIF, JPEG, PNG, BMP, WEBP",
                default='EUR-MED_C1300-1.png')
args = vars(ap.parse_args())

# TODO(developer): Uncomment these variables before running the sample.
project_id= 'edd-gcp-01'
location = 'eu' # Format is 'us' or 'eu'
processor_id = '10c5d397b0870d86' # Create processor in Cloud Console
file_path = args["file"]
template_path = args['template']
# file_path = 'EUR-MED_C1300_Skywalker.pdf'
# file_path = 'Scanned_form.pdf'

def process_document_sample(
    project_id: str, location: str, processor_id: str, file_path: str, template_path: str
):

    # You must set the api_endpoint if you use a location other than 'us', e.g.:
    opts = {}
    if location == "eu":
        opts = {"api_endpoint": "eu-documentai.googleapis.com"}

    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    # The full resource name of the processor, e.g.:
    # projects/project-id/locations/location/processor/processor-id
    # You must create new processors in the Cloud Console first
    name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

    with open(file_path, "rb") as image:
        image_content = image.read()
    template = cv2.imread(template_path)
    
    # Read the file into memory
    document = {"content": image_content, "mime_type": "application/pdf"}

    # Configure the process request
    request = {"name": name, "raw_document": document}

    # Recognizes text entities in the PDF document
    result = client.process_document(request=request)

    document = result.document

    print("Document processing complete.")

    # For a full list of Document object attributes, please reference this page: https://googleapis.dev/python/documentai/latest/_modules/google/cloud/documentai_v1beta3/types/document.html#Document

    document_pages = document.pages

    #initalising accuracy arrays
    doc_acc = []
    field_acc = np.zeros( (len(FIELD_NAMES),1) )
    doc_text = []
    confidence = []

    # Read the text recognition output from the processor
    print("The document contains the following paragraphs:")
    for page in document_pages:
        # # field_text = ''
        # for idx, paragraph in enumerate(page.paragraphs):
        #   field_text = get_text(paragraph.layout, document)
        #   print(f'Paragraph: {idx} \n{field_text} \n\n')

       for idx, field in enumerate(FIELD_NAMES):
         field_text = ''
         text_loc = []
         paragraph_confidence = []
         ground_field = ground_truth[field[0]]
         for paragraph_num in field.paragraph:
           field_text += get_text(page.paragraphs[paragraph_num].layout, document)
           paragraph_confidence.append(page.paragraphs[paragraph_num].layout.confidence)
           text_loc.append(page.paragraphs[paragraph_num].layout)
         
            
         field_text = remove_from_text(field_text, OCR_LOCATIONS[idx].filter_keywords)
         
         doc_text.append(field_text.replace('\n', ', '))
         field_acc[idx], doc_acc, field_text = ocr_acc(ground_field, field_text.split('\n'), doc_acc)
         # print(f'Paragraph {idx}: (Field Accuracy = {field_acc[idx]}) \nParagraph Confidence: {sum(paragraph_confidence)/len(paragraph_confidence)} \n{field_text}\n\n')
         confidence.append(sum(paragraph_confidence)/len(paragraph_confidence))
         
    doc2csv(doc_text,confidence)
    print( f'Document Accuracy = {doc_acc[idx]} \n Confidence: {confidence}')
    return document_pages, document, template

def get_text(doc_element: dict, document: dict):
    """
    Document AI identifies form fields by their offsets
    in document text. This function converts offsets
    to text snippets.
    """
    response = ""
    # If a text segment spans several lines, it will
    # be stored in different text segments.
    for segment in doc_element.text_anchor.text_segments:
        start_index = (
            int(segment.start_index)
            if segment in doc_element.text_anchor.text_segments
            else 0
        )
        end_index = int(segment.end_index)
        response += document.text[start_index:end_index]
    return response

def remove_from_text(field_text, removal_list):
    for word in removal_list:
        field_text = field_text.replace(word, '')
    return field_text.strip(' \n')

def put_OCR(field_text, layout, image):
  box_shape = get_bbox(layout, image.shape)
  #cv2.rectangle(image, box_shape[0], box_shape[1], (0, 255, 0), 2)
  put_text(image, field_text, box_shape)
  
def get_bbox(layout,img_shape):
  x1 = round(img_shape[1]*layout.bounding_poly.normalized_vertices[0].x)
  y1 = round(img_shape[0]*layout.bounding_poly.normalized_vertices[0].y)
  x2 = round(img_shape[1]*layout.bounding_poly.normalized_vertices[2].x)
  y2 = round(img_shape[0]*layout.bounding_poly.normalized_vertices[2].y)
  return (x1,y1),(x2,y2)

def put_text(image, field_text, box_shape):
  for j, line in enumerate(field_text.split('\n')):
    b_shape = (box_shape[0][0], box_shape[0][1] + ((j+2) * 25) + 10)
    cv2.putText(image, line, b_shape, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2);

def doc2csv(text,confidence):
    sec = [section.replace(' \n',', ') for section in sections]
    data = {'Form Field': sec,
             'Input': text,
             'Confidence': confidence}
    df = pd.DataFrame(data).to_csv('Skywalker.csv', index=False)
    
_, _, template = process_document_sample(project_id, location, processor_id, file_path,template_path)
cv2.imshow("Input", imutils.resize(template, width=500))
cv2.waitKey(0)