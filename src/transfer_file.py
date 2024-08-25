import os
import shutil

current_dir = current_dir = os.path.dirname(os.path.abspath(__file__))

def fscs_to_depth_segment():
    from_path = os.path.join(current_dir, '..', 'data', '2_fscs_output')
    to_path = os.path.join(current_dir, '..', 'data', '3_depth_segment_input')
    for img_name_with_png in os.listdir(from_path):
        png_dir = os.path.join(from_path, img_name_with_png)
        img_name = img_name_with_png.split('.')[0]
        if os.path.isdir(os.path.join(png_dir)):
            for png_file in os.listdir(png_dir):
                
                os.makedirs(os.path.join(to_path, 'alpha', img_name), exist_ok=True)
                os.makedirs(os.path.join(to_path, 'layer', img_name), exist_ok=True)
                #alpha
                if 'proc-alpha' in png_file:
                    shutil.copy2(os.path.join(png_dir, png_file), os.path.join(to_path, 'alpha', img_name))

                # layer
                if 'img-00_layer' in png_file:
                    shutil.copy2(os.path.join(png_dir, png_file), os.path.join(to_path, 'layer', img_name))

if __name__ == '__main__':
    
    fscs_to_depth_segment()