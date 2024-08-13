import numpy as np
from PIL import Image
import json
import argparse
import os

compress_rate = 0.2

print("begin dump")

def dump(filename):
    img = Image.open("../imgs/test/{}".format(filename))
    img.thumbnail((int(img.width*compress_rate),
                   int(img.height * compress_rate)))
    img_data = np.array(img).reshape(-1, 3)
    print(img_data.shape)

    with open("../imgs/test/{}.json".format(os.path.splitext(filename)[0]), "w") as fw:
        fw.writelines(json.dumps(img_data.tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='image to json')
    
    parser.add_argument('--base_dir_in', type=str, default='/home/gliamanti/myApps/FSCS/color-extraction/imgs/test/', help='Base input directory')
    
    args = parser.parse_args()
    
    base_dir_in = args.base_dir_in
    
    for file in os.listdir(base_dir_in):
        if file.endswith((".png",".jpg","jpeg")):
            dump(file)

    print("finish dump")