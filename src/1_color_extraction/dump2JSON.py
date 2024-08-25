import numpy as np
from PIL import Image
import json
import os

COMPRESS_RATE = 0.2

def dump(input_path, filename, output_path):
    img = Image.open(os.path.join(input_path, filename))
    img.thumbnail((int(img.width * COMPRESS_RATE),
                   int(img.height * COMPRESS_RATE)))
    img_data = np.array(img).reshape(-1, 3)
    print(img_data.shape)

    with open(os.path.join(output_path, os.path.splitext(filename)[0]+'.json'), "w") as fw:
        fw.writelines(json.dumps(img_data.tolist()))

if __name__ == "__main__":
    print("begin dump")

    current_dir = os.path.dirname(os.path.abspath(__file__))

    input_path = os.path.join(current_dir, '..', '..', 'data', '1_color_extraction_input')

    output_path = os.path.join(current_dir, '..', '..', 'data', '1_color_extraction_output', 'json')
    os.makedirs(output_path, exist_ok=True)
    
    for file in os.listdir(input_path):
        if file.endswith((".png",".jpg","jpeg")):
            dump(input_path, file, output_path)

    print("finish dump")