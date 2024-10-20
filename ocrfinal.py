from PIL import Image
import pytesseract
import cv2
import os
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet

# Set up Tesseract path (adjust based on your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Ensure these downloads only once
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Function to extract bill details from text
def extract_bill_details(text, img_path):
    date = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)  # Find dates in DD/MM/YYYY or similar formats
    amount = re.findall(r'[$£€₹](\d+(?:\.\d{1,2})?)', text)  # Find amounts with currency symbols
    title = text.splitlines()[0]  # The first line as the title of the organization
    mrp = re.findall(r'MRP[^\d]*(\d+(?:\.\d{1,2})?)', text, re.IGNORECASE)  # MRP details
    expiry = re.findall(r'EXP[^\d]*(\b(?:[A-Za-z]{3}-\d{4}|\d{2}-\d{2}-\d{4})\b)', text, re.IGNORECASE)  # Expiry date
    ean = re.findall(r'\b\d{12,13}\b', text)  # EAN number (usually 12-13 digits)
    net_quantity = re.findall(r'Net\s*Quantity[^\d](\d+\s(?:kg|g|ml|l|pcs|units|packs)?)', text, re.IGNORECASE)  # Net quantity
    
    return {
        'image_path': img_path,
        'date': date[0] if date else None,
        'title': title,
        'amount': max(map(float, amount)) if amount else None,
        'mrp': max(map(float, mrp)) if mrp else None,
        'expiry': expiry[0] if expiry else None,
        'ean': ean[0] if ean else None,
        'net_quantity': net_quantity[0] if net_quantity else None
    }

# Example of usage with an image
img_path =r"C:\Users\Tamilarasan S\OneDrive\Pictures\Screenshots\Screenshot (187).png"
image = cv2.imread(img_path, 0)  # Read the image in grayscale
text = pytesseract.image_to_string(image).lower()  # Perform OCR on the image

# Extract details from the text and include the image path in the output
bill_details = extract_bill_details(text, img_path)
# Output each key-value pair on a new line
for key, value in bill_details.items():
    print(f"{key}: {value}")
