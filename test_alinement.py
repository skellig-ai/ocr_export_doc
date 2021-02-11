# import the necessary packages
from pyimagesearch.alignment import align_images
from collections import namedtuple
import pytesseract
import argparse
import imutils
import cv2
import numpy as np

image = 'Documents/ocr-document/scans/File_000.jpg'
template = 'Documents/ocr-document/form_w4.png'

image = cv2.imread(image)
template = cv2.imread(template)

aligned = align_images(image, template, debug=True)

