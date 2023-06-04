# Import Library
import streamlit as st
from pymongo import MongoClient
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep
import time
import requests
from io import BytesIO
from PIL import Image
import easyocr  # Ganti dari pytesseract ke easyocr
import pandas as pd
from transformers import AutoTokenizer, BertForSequenceClassification
import torch
from pandas import DataFrame
import os

# Connect to MongoDB

MONGODB_URL = os.getenv('mongodb+srv://ricardo8bdg:simarmataas123@jobads.94mucvv.mongodb.net/')
client = MongoClient(MONGODB_URL)
db = client['projek_ml']
collection = db['scraping']

# Set your Instagram credentials
my_user = "scrapetesting"
my_pwd = 'do.ricard0'

# Load the model
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=2)
model.load_state_dict(torch.load('IndoBERT_classifier.pt'))
model.eval()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')

@st.cache_resource
def get_driver():
    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--headless')
    return webdriver.Chrome(executable_path=ChromeDriverManager().install(), options=options)


@st.cache_data
def classify_text(text):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors='pt')

    # Make prediction
    outputs = model(**inputs)

    # Get the predicted class
    _, predicted = torch.max(outputs.logits, 1)

    # Convert the prediction to 'bias' or 'Non-bias'
    if predicted.item() == 1:
        return 'bias'
    else:
        return 'Non-bias'

@st.cache_data
def perform_image_scraping_ocr():
    driver = get_driver()
    driver.get("https://www.instagram.com/accounts/login")
    driver.maximize_window()
    sleep(3)

    user_name = driver.find_element(By.XPATH, "//input[@name='username']")
    user_name.send_keys(my_user)
    sleep(1)

    password = driver.find_element(By.XPATH, "//input[@name='password']")
    password.send_keys(my_pwd)
    password.send_keys(Keys.RETURN)
    sleep(3)

    # Keyword to search
    keyword = "lowongan"
    driver.get("https://www.instagram.com/explore/tags/" + keyword + "/")
    time.sleep(8)
    my_images = set()

    # Get all images on the page
    images = driver.find_elements(By.XPATH, "//img[@class='x5yr21d xu96u03 x10l6tqk x13vifvy x87ps6o xh8yej3']")
    while len(my_images) < 5:
        for image in images:
            source = image.get_attribute('src')
            if collection.count_documents({'image_url': source}, limit=1) == 0:
                my_images.add(source)
                collection.insert_one({'image_url': source})
            if len(my_images) >= 5:
                break
        if len(my_images) < 5:
            driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
            sleep(3)
            images = driver.find_elements(By.XPATH, "//img[@class='x5yr21d xu96u03 x10l6tqk x13vifvy x87ps6o xh8yej3']")

    driver.quit()

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en', 'id'])  # Supports English and Indonesian

    # Perform OCR on each image and save to MongoDB
    for image_url in collection.find({'ocr_result': {'$exists': False}}):
        url = image_url['image_url']
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        result = reader.readtext(image)
        text = ' '.join([item[1] for item in result])
        prediction = classify_text(text)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        collection.update_one({'image_url': url}, {'$set': {'ocr_result': text, 'timestamp': timestamp, 'prediction': prediction}})


def get_data_from_db():
    # Connect to MongoDB
    client = MongoClient(MONGODB_URL)
    db = client['projek_ml']
    collection = db['scraping']

    # Get data from MongoDB
    data = collection.find()

    # Create a DataFrame
    df = DataFrame(list(data))

    # Sort by timestamp
    df.sort_values(by='timestamp', ascending=False, inplace=True)

    return df

def main():
    MONGODB_URL = os.getenv('mongodb+srv://ricardo8bdg:simarmataas123@jobads.94mucvv.mongodb.net/')
    client = MongoClient(MONGODB_URL)
    db = client['projek_ml']
    collection = db['scraping']
    st.title('Bias and Discrimination Recognition')

    # Load initial data
    df = get_data_from_db()

    # Search feature
    search_term = st.text_input('Enter search term')

    # Button to perform Image Scraping and OCR
    if st.button("Perform Image Scraping and OCR"):
        st.text("Performing Image Scraping and OCR...")
        perform_image_scraping_ocr()
        st.text("Image Scraping and OCR completed!")

        # Rerun the app after scraping
        df = get_data_from_db()
        st.experimental_rerun()

    search_results = df[df['ocr_result'].str.contains(search_term, na=False)]
    page_size = 5
    page_number = st.number_input(
        label="Page Number", min_value=1, max_value=len(search_results)//page_size+1, step=1)

    current_start = (page_number-1)*page_size
    current_end = page_number*page_size

    # Show OCR results, predictions, and image URLs in a single table
    for index, row in search_results.iloc[current_start:current_end].iterrows():
        # Determine row color based on prediction
        if row['prediction'] == 'bias':
            row_color = '#ffc0cb'
        else:
            row_color = '#fff'

        st.markdown(
        f"""
        <table style='background-color: #283e4a; color: black;'> <!-- Ubah color: #fff menjadi color: black -->
            <tr style='background-color: {row_color};'>
                <td style='width: 400px; text-align: center; vertical-align: middle;'><b>Job Description</b></td>
                <td style='text-align: center; vertical-align: middle;'><b>Identification Job</b></td>
                <td style='text-align: center; vertical-align: middle;'><b>Image URL</b></td>
            </tr>
            <tr style='background-color: {row_color};'>
                <td style='width: 400px; text-align: center; vertical-align: middle;'>{row['ocr_result']}</td>
                <td style='text-align: center; vertical-align: middle;'>{row['prediction']}</td>
                <td style='text-align: center; vertical-align: middle;'><a href="{row['image_url']}">link</a></td>
            </tr>
        </table>
        """,
        unsafe_allow_html=True)

    st.header("Text Classification Job Description to Bias or Non-Bias")
    text = st.text_area('Text to predict', 'Enter text here...')
    if st.button('Predict'):
        prediction = classify_text(text)
        st.write(prediction)

if __name__ == "__main__":
    main()
