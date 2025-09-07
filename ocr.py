import requests

# API details
api_key = "K89675956188957"
url = "https://api.ocr.space/parse/image"

# File to upload (replace 'your_image.jpg' with your image file path)
file_path = "image.png"

# Prepare the file for upload
with open(file_path, 'rb') as image_file:
    files = {'file': image_file}
    data = {'apikey': api_key, 'language': 'eng', 'isOverlayRequired': 'false'}

    # Send POST request
    response = requests.post(url, files=files, data=data)

    # Check response
    if response.status_code == 200:
        result = response.json()
        if result.get("IsErroredOnProcessing", False):
            print("Error:", result.get("ErrorMessage", "Unknown error"))
        else:
            print("OCR Result:", result.get("ParsedResults")[0].get("ParsedText", "No text extracted"))
    else:
        print("Failed to connect to API. Status code:", response.status_code)