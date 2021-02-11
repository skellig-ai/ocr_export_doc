# USAGE
# python ocr_form.py --image scans/scan_01.jpg --template form_w4.png

# import the necessary packages
from pyimagesearch.alignment import align_images
from collections import namedtuple
import pytesseract
import argparse
import imutils
import cv2
import numpy as np

def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

def acc(ground_value, ocr_value):
    bigger = max(len(ground_value), len(ocr_value))
    lev = damerau_levenshtein_distance(ground_value, ocr_value)
    return (bigger - lev)/bigger

def ocr_acc(field_truth, field_ocr):
    k = 0
    max_line_acc = np.zeros( (len(field_truth),) )
    for i, fline in enumerate(field_truth):
        line_acc = np.zeros( (len(field_ocr[k:]),) )
        for j, oline in enumerate(field_ocr[k:]):
            line_acc[j] = acc(fline, oline)
        
        if len(line_acc) == 0:
            max_line_acc[i] = 0
        elif max(line_acc) > 0.5:
            max_line_acc[i] = max(line_acc)
            k += np.argmax(line_acc) + 1
        
    return max_line_acc.mean()

##def opening(image):
##    kernel = np.ones((5,5),np.uint8)
##    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image that we'll align to template")
ap.add_argument("-t", "--template", required=True,
	help="path to input template image")
ap.add_argument("-v", "--threshold", required=True,
	help="path to input template image")
args = vars(ap.parse_args())
threshold = float(args['threshold'])

# create a named tuple which we can use to create locations of the
# input document which we wish to OCR
OCRLocation = namedtuple("OCRLocation", ["id", "bbox",
	"filter_keywords"])

# define the locations of each area of the document we wish to OCR

OCR_LOCATIONS = [
    OCRLocation("Exporter", (100, 71, 734, 201),
        ['Exporter', '(Name', "full address", "country)"]),
    OCRLocation("preferential_trade_between", (866, 216, 689, 186),
        ["and", "."]),
    OCRLocation("Consignee", (125, 273, 734, 334),
        ["Consignee", "(Name", "full address", "country)", ",", "(Optional)", "3"]),
    OCRLocation("transport_details", (125, 607, 734, 266),
        ["Transport", "details", "(Optional)", "6"]),
    OCRLocation("item_number", (126, 876, 377, 990),
        ["8", "Item", "number:", "marks", "and", "numbers"]),
    OCRLocation("description_of_goods", (533, 876, 691, 990),
        ["Number", "and", "kind", "of", "packages", "(1):", "description", "goods"]),
    OCRLocation("gross_weight", (1235, 877, 166, 990), 
        ["Gross", "weight", "(kg)", "or", "other", "measure", "(litres,", "cu.", "m.,", "etc)"]),
    OCRLocation("customs_office", (124, 1873, 900, 208),
        ["Customs", "11.", "office", 'Declaration', 'certified', 'Export', 'document', '(2):', 'From', 'No.', 		'Endorsement', 'stamp', 'Issuing', 'country', 'or', 'territory', 'UNITED', 'KINGDOM']),
    OCRLocation("customs_date_signature", (124, 2082, 900, 190),
        ["Date", "(Signature)", 'Issuing', 'country', 'or', 'territory', 'UNITED', 'KINGDOM']),
    OCRLocation("exporter_date_signature", (1024, 1871, 554, 401),
        ['12.', 'Declaration', 'by', 'Exporter', 'I,', 'the', 'undersigned,', 'declare', 'that', 'the', 	'goods', 'described', 'above', 'meet', 'conditions', 'required', 'for', 'issue', 'of', 'this', 		'certificate.',"(Place", "and", "date", "Signature", "."]),
]

headings = ['1. Exporter \n(Name, full address, country)',
            '2. Certificate used in \npreferential trade between',
            '3. Consignee \n(Name, full address, country)(Optional)',
            '6. Transport details (Optional)',
            '8.1. Item number: \nmarks and numbers',
            '8.2. Number and kind of packages (1): \ndescription of goods',
            '9. Gross \nweight (kg) \nof other \nmeasure \n(liters, cu. m., etc)',
            '11.1. Customs Endorsement',
            '11.2. Customs Signature',
            '12. Declaration by the Exporter']

# load the input image and template from disk
print("[INFO] loading images...")
image = cv2.imread(args["image"])
template = cv2.imread(args["template"])

# align the images
print("[INFO] aligning images...")
aligned = align_images(image, template, debug=False)

# initialize a results list to store the document OCR parsing results
print("[INFO] OCR'ing document...")
parsingResults = []

# loop over the locations of the document we are going to OCR
for loc in OCR_LOCATIONS:
    # extract the OCR ROI from the aligned image
    (x, y, w, h) = loc.bbox
    roi = aligned[y:y + h, x:x + w]

    # OCR the ROI using Tesseract
    binary = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY)[1]
    img = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
##  binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
##  opening = opening(image)
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(binary)

    # break the text into lines and loop over them
    for line in text.split("\n"):
        # if the line is empty, ignore it
        if len(line) == 0:
            continue

        # convert the line to lowercase and then check to see if the
        # line contains any of the filter keywords (these keywords
        # are part of the *form itself* and should be ignored)
        lower = line.lower()
        count = sum([lower.count(x) for x in loc.filter_keywords])

        # if the count is zero than we know we are *not* examining a
        # text field that is part of the document itself (ex., info,
        # on the field, an example, help text, etc.)
        if count == 0:
            # update our parsing results dictionary with the OCR'd
            # text if the line is *not* empty
            parsingResults.append([loc, line])

# initialize a dictionary to store our final OCR results
results = {}

# loop over the results of parsing the document
for (loc, line) in parsingResults:
    # grab any existing OCR result for the current ID of the document
    r = results.get(loc.id, None)

    # if the result is None, initialize it using the text and location
    # namedtuple (converting it to a dictionary as namedtuples are not
    # hashable)
    if r is None:
        results[loc.id] = (line, loc._asdict())

    # otherwise, there exists a OCR result for the current area of the
    # document, so we should append our existing line
    else:
        # unpack the existing OCR result and append the line to the
        # existing text
        (existingText, loc) = r
        text = "{}\n{}".format(existingText, line)

        # update our results dictionary
        results[loc["id"]] = (text, loc)


ocred = np.ones(aligned.shape)

# loop over the results
for (idx, result) in enumerate(results.values()):
    # unpack the result tuple
    (text, loc) = result
    # display the OCR result to our terminal
    print(loc["id"])
    print("=" * len(loc["id"]))
    print("{}\n\n".format(text))

    # extract the bounding box coordinates of the OCR location and
    # then strip out non-ASCII text so we can draw the text on the
    # output image using OpenCV
    (x, y, w, h) = loc["bbox"]
    clean = cleanup_text(text)

    # draw a bounding box around the text
    cv2.rectangle(ocred, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.rectangle(aligned, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    startY = y + 35
    for (i, heading) in enumerate(headings[idx].split("\n")):
        cv2.putText(ocred, heading, (x, startY), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1)
        cv2.putText(aligned, heading, (x, startY), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1)
        startY = startY + 30
    
    # loop over all lines in the text
    for (i, line) in enumerate(text[:-1].split("\n")):
        # draw the line on the output image
        cv2.putText(ocred, line, (x, startY),
            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
        cv2.putText(aligned, line, (x, startY),
            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
        startY = startY + 30

# show the input and output images, resizing it such that they fit
# on our screen


cv2.imshow("Input", imutils.resize(image, width=700))
cv2.imshow("Output", imutils.resize(aligned, width=700))
cv2.imshow('OCR\'ed', imutils.resize(ocred, width=700));
cv2.waitKey(0)
