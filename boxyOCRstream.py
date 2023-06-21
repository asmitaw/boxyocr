"""
# My first app
Here's our first attempt at using data to create a table:
"""

import pandas as pd
import os
import torch
import random
import easyocr
import cv2
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import streamlit as st
from PIL import Image
 
def resizeAndPad(img, size, padColor=255):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA

    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = float(w)/h 
    saspect = float(sw)/sh

    if (saspect > aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0

    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img



def deskew_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to obtain a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Convert the binary image to np.uint8 data type
    binary = np.uint8(binary)

    # Apply morphological operations to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Detect lines in the image using Hough transform
    lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    # Calculate the angle of rotation based on the dominant line
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)

    # Find the most common angle (mode) in the list of angles
    mode_angle = np.mean(np.array(angles))

    # Rotate the image to deskew it
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, mode_angle, 1.0)
    deskewed_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC)

    return deskewed_image

def openCVEasyOCR(img):
    
    reader = easyocr.Reader(['en'], gpu=True)
    result = reader.readtext(img,detail=1)
    #result

    font = cv2.FONT_HERSHEY_SIMPLEX

    fontScale = 1
    fontThickness = 1
    fontFace=cv2.FONT_HERSHEY_DUPLEX

    # make sure font thickness is an integer, if not, the OpenCV functions that use this may crash
    fontThickness = int(fontThickness)

    imageHeight = np.size(img, 0)
    imageWidth = np.size(img, 1)

    fontScale = (imageWidth * imageHeight) / (300 * 300) # Would work best for almost square images

    # write the text on the image
    #cv2.putText(img, text, top_left, fontFace, fontScale, (255,255,255),
      #          fontThickness)
    print_text = ""

    for detection in result:
        top_left = tuple(int(val) for val in detection[0][0])
        bottom_right = tuple(int(val) for val in detection[0][2])
        text = detection[1]
        print_text = print_text + "\n" + text
        font = cv2.FONT_HERSHEY_SIMPLEX
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        img = cv2.rectangle(img, top_left,bottom_right, (red,blue,green),5)
        img = cv2.putText(img,text,top_left,font,0.5,(255,255,255),2,cv2.LINE_AA)
 
    print("<td>" + print_text + "</td>")
    return img , print_text

st.title("BoxyLink Cloud - OCR demo")
st.divider()
col1, col2 = st.columns([2, 2])
image = np.zeros((300,300,3), np.uint8)
result_text = ""

with col1:
    st.subheader("Upload Files here")
    uploaded_file = st.file_uploader("", accept_multiple_files=False)
    #  for uploaded_file in uploaded_files:
 
    #uploaded_file = st.file_uploader("Choose a image file", type="jpg")
    #de_skew = st.checkbox('Skew images before detection',value=False )

    if uploaded_file is not None:
        de_skew = st.checkbox('Skew images before detection',value=False )

    # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = resizeAndPad(image, (1200,1200), 255)

    # Now do something with the image! For example, let's display it:
#       bytes_data = uploaded_file.read()
        st.image(image, caption='Uploaded Image', use_column_width=True, channels="BGR")
        d_img = image
    
       #compressing image before processing 

        if de_skew:
            deskew_image(image)
            d_img = deskew_image (image)
            image, result_text = openCVEasyOCR(d_img)
        else:
            image, result_text = openCVEasyOCR(image)

with col2:
   st.header("Results")
   st.image(image, caption='Uploaded Image', use_column_width=True, channels="BGR")
   st.write(result_text)

# SHOW PROJECTED IMAGE
#   st.image(image, caption='Image with OCR results', use_column_width=True)
        
