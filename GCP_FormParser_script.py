from google.cloud import documentai_v1 as documentai
from google.cloud import storage
from config import *

# TODO(developer): Uncomment these variables before running the sample.
project_id= 'edd-gcp-01'
location = 'eu' # Format is 'us' or 'eu'
processor_id = '10c5d397b0870d86' # Create processor in Cloud Console
#file_path = '\invent_lab_projects\offline-sandbox\EUR-MED_C1300_Skywalker.pdf'
file_path = 'EUR-MED_C1300_Skywalker.pdf'

def process_document_sample(
    project_id: str, location: str, processor_id: str, file_path: str
):
    from google.cloud import documentai_v1 as documentai

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

    # Read the text recognition output from the processor
    print("The document contains the following paragraphs:")
    for page in document_pages:

      for field in FIELD_NAMES:
        paragraph_text = ''
        for idx in field.paragraph:
          paragraph_text += get_text(page.paragraphs[idx].layout, document)
        print(paragraph_text)

    return document_pages, document

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
process_document_sample(project_id, location, processor_id, file_path)

