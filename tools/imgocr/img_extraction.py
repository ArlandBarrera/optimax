import easyocr
import os

# this needs to run only once to load the model into memory
reader = easyocr.Reader(['es','en'])

DIR = 'tools/imgocr/images/'
FILE = 'shoes.jpg'
PATH_IMG = os.path.join(DIR, FILE)

result = reader.readtext(PATH_IMG, detail = 0)

print(result)
