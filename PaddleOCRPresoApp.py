"""
# My first app
Here's our first attempt at using data to create a table:
"""
from paddleocr import PaddleOCR,draw_ocr
import pandas as pd
import os
#import torch
import random
#import easyocr
import cv2
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image
import streamlit as st
import sys

st.set_page_config(layout="wide")

css = '''
<style>
    [data-testid="stSidebar"]{
        min-width: 400px;
        max-width: 800px;
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True)    


# sidebar
mapping = {"PP-OCR": "BoxyOCR - PP (xx MS)", "PP-OCRv3": "BoxyOCR - PPV3 (XX MS)"}
with st.sidebar:

    st.header("Boxylink OCR")               
    st.write("Custom models for industrial OCR")

    st.divider()

    PPModel = st.radio(
        "Boxylink GPU Based Models",
        ("PP-OCR", "PP-OCRv3"), format_func=lambda x: mapping[x],key="ChosenModel"
    )

    useGPU = st.checkbox(
        "Use GPU    ", True
    )

    st.write(mapping)

# Custom stream wrapper class
class StreamWrapper:
    def __init__(self, write_func):
        self.write_func = write_func

    def write(self, text):
        self.write_func(text)


def createtraining_data(file_path, result_text, boxes):
    training_data_line_format = file_path + "\t"
    print("blah")
    for i in range(len(result_text)):
          # image_path\t[{“points”:[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], “transcription”:text_annotation}, {“points”:…..}]
        training_data_line_format = training_data_line_format + "[{\"points\":" + str(boxes[i] ) + ", \"transcription\": \"" + str(result_text[i]) + "\"},"
    #training_data_line_format =  "[{\“points\”:[[" x1,y1 "],["+  x2,y2 "],[" + x3,y3 "],[" x4,y4 "]], \"transcription\":" + text_annotation "},"
    training_data_line_format = training_data_line_format + "\n"                 

    print(training_data_line_format)  
    with open('./train_data/rec/rec_gt_train.txt', 'a') as f:
        f.write(training_data_line_format+ "\n")

                

# Redirect console output to Streamlit
def redirect_output():
    stream_wrapper = StreamWrapper(st.write)
    sys.stdout = stream_wrapper
    sys.stderr  = stream_wrapper
 

def paddle_ocr(image,file_name):
    ocr = PaddleOCR(use_angle_cls=True, ocr_version =  PPModel , lang='en' ) # need to run only once to download and load model into memory
    result = ocr.ocr(image, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)

    # draw result
    result = result[0]
   # image = Image.open(image).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    print(txts)
    print(scores)
    im_show = draw_ocr(image, boxes, None, None, font_path='./usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-Regular.ttf')
    im_show = Image.fromarray(im_show)
    createtraining_data(file_name,txts, boxes)
    im_show.save( "results/Image_1_result" + path_in)
    return im_show, txts, scores
   

col1, col2 = st.columns([2, 2])
image = np.zeros((300,300,3), np.uint8)
result_text = ""
redirect_output()
 

with col1:
    st.header("Upload Files here")
    
    uploaded_file = st.file_uploader("", accept_multiple_files=False)
    #  for uploaded_file in uploaded_files:
 
    #uploaded_file = st.file_uploader("Choose a image file", type="jpg")
    #de_skew = st.checkbox('Skew images before detection',value=False )

    if uploaded_file is not None:
        path_in = uploaded_file.name
        print(path_in)
    # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        uploaded_image = cv2.imdecode(file_bytes,1)
        paddle_image = cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)

    # Now do something with the image! For example, let's display it:
        st.image(uploaded_image, caption='Uploaded Image', width=400, channels="BGR")
        image, result_text, scores = paddle_ocr(paddle_image, path_in)
 
with col2:

   months = ['January', 'january', 'january', 'February', 'february', 'february', 'March', 'march', 'march',
          'April', 'april',  'May',   'may', 'June', 'june' , 'July', 'july' ,
          'August', 'august', 'august', 'September', 'september', 'september', 'October', 'october', 'october',
          'November', 'november', 'november', 'December', 'december', 'december',
          'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUNE', 'JULY', 'AUG', 'SEP',  'SEPT', 'OCT', 'NOV', 'DEC']
 
   st.subheader("Results")
   for i in range(len(result_text)):
    if result_text[i].upper().startswith("RS"):
            #something
            st.write("Price:")
            st.write (result_text[i])
    elif any( result_text[i].startswith(month) for month in months):
            monthvalue = result_text[i]
            st.write("Month:")
            st.write (result_text[i])

   st.image(image, caption='Uploaded Image', width=400, channels="BGR")

   for i in range(len(result_text)):
    st.write(result_text[i], scores[i])


# SHOW PROJECTED IMAGE
#   st.image(image, caption='Image with OCR results', use_column_width=True)
        
