from pdf2image import convert_from_path
import pytesseract
from typing import List
import os

# --- Document ---
DIR = 'tools/dococr/documents/'
FILE = 'jotform.pdf'
PATH_FILE = os.path.join(DIR, FILE)

def extract_text_from_pdf(PATH_FILE: str) -> List[str]:
    full_text_list = []

    # 1. Convert PDF pages to a list of PIL Image objects
    # Note: pdf2image needs the Poppler library (installed with 'pacman -S poppler')
    try:
        # Setting a higher DPI (e.g., 300) often improves OCR accuracy significantly
        print(f"Converting pages from {PATH_FILE}...")
        images = convert_from_path(PATH_FILE, dpi=300)
    except Exception as e:
        print("Error: Could not convert PDF. Ensure Poppler is installed.")
        print(f"Underlying error: {e}")
        return []

    # 2. Process each image/page
    for i, image in enumerate(images):
        print(f"Processing Page {i+1}...")

        # Use Tesseract to run OCR on the PIL Image object
        # You can specify languages if needed: lang='eng+spa'
        page_text = pytesseract.image_to_string(image)

        # Split the text into tokens (words) and add them to the main list
        full_text_list.extend(page_text.split())

    print("\nOCR Complete.")
    return full_text_list


if os.path.exists(PATH_FILE):
    ocr_result_tokens = extract_text_from_pdf(PATH_FILE)

    if ocr_result_tokens:
        print(f"Total tokens extracted: {len(ocr_result_tokens)}")
        print("First 20 tokens:")
        print(ocr_result_tokens[:20])

        # This output format (List[str]) is now ready for your robust data extraction function!
        # final_data = extract_final_receipt_values(ocr_result_tokens, ['SUB', 'TOTAL'])

else:
    print(f"\nError: Please ensure a PDF file named '{FILE}' exists in the current directory.")
