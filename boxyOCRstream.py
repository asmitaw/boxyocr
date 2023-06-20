"""
# My first app
Here's our first attempt at using data to create a table:
"""

import pandas as pd
import streamlit as st

from PIL import Image
 
#Add Streamlit widgets
#Streamlit also makes it easy to add interactivity to your app with widgets like sliders, dropdown menus, and checkboxes. For example, you can add a slider to your app that allows users to control the value of a parameter in your model like this:
 
st.title("Boxylink Cloud - OCR demo for TTK")
st.divider()


col1, col2 = st.columns([2, 2])


with col1:
   st.subheader("Upload Files here")
   uploaded_files = st.file_uploader("", accept_multiple_files=True)
   for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

with col2:
   st.subheader("Results")
   st.image("https://static.streamlit.io/examples/dog.jpg")






