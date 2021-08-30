import PIL
import os
import os.path
from PIL import Image

f = r'C:\Users\Ishan\Downloads\Arrow Images\New folder' #change folder path
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = img.resize((640,480))
    img.save(f_img)