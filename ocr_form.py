# USAGE
# python ocr_form.py --image scans/scan_01.jpg --template form_w4.png

# import the necessary packages
from pyimagesearch.alignment import align_images
from collections import namedtuple
from utility import *
from config import *
import pytesseract
import argparse
import imutils
import cv2
import numpy as np


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

# load the input image and template from disk
print("[INFO] loading images...")
image = cv2.imread(args["image"])
template = cv2.imread(args["template"])

# align the images
print("[INFO] aligning images...")
aligned = align_images(cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1], template, debug=False)
y, x, _ = aligned.shape
aligned = cv2.resize(aligned, (2*x, 2*y))

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
    text = pytesseract.image_to_string(rgb)

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

#initalising accuracy arrays
doc_acc = []
field_acc = np.zeros( (len(results.values()),1) )

# loop over the results
for (idx, result) in enumerate(results.values()):
    # unpack the result tuple
    (text, loc) = result
    ground_field = ground_truth[result[1]['id']]
    
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
    
    startY = y + 65
    cv2.putText(ocred, headings[idx], (x, startY), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(aligned, headings[idx], (x, startY), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    
    # calculate accuracy
    field_acc[idx], doc_acc, ocr_text = ocr_acc(ground_field, text.split("\n"), doc_acc)
    print(field_acc[idx])
    
    # loop over all lines in the text
    for (i, line) in enumerate(ocr_text):
        # draw the line on the output image
        startY = y + ((i+2) * 25) + 40
        cv2.putText(ocred, line, (x, startY),
            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(aligned, line, (x, startY),
            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

print(f'Accuracy: {field_acc.mean()}')
cv2.putText(ocred, f'Document Accuracy: {np.round(100*np.asarray(doc_acc).mean())}%', (800, 100),
            cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 255), 3)

# show the input and output images, resizing it such that they fit
# on our screen


cv2.imshow("Input", imutils.resize(image, width=700))
cv2.imshow("Output", imutils.resize(aligned, width=700))
cv2.imshow('OCR\'ed', imutils.resize(ocred, width=700));
cv2.waitKey(0)
