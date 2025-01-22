import os
import shutil

current_dir = current_dir = os.path.dirname(os.path.abspath(__file__))

def color_extraction_to_fscs():
    from_path = os.path.join(current_dir, '..', 'data', '1_color_extraction_input')
    to_path = os.path.join(current_dir, '..', 'data', '2_fscs_input', 'mask')
    os.makedirs(to_path, exist_ok=True)
    for img_name in os.listdir(from_path):
        img_path = os.path.join(from_path, img_name)
        shutil.copy2(img_path, os.path.join(to_path, img_name))

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

def prepare_blender_input():
    # for alpha
    from_path = os.path.join(current_dir, '..', 'data', '3_depth_segment_output', 'patch_alpha')
    for img_name in os.listdir(from_path):
        to_path = os.path.join(current_dir, '..', 'data', '4_blender_input', img_name, 'alpha')
        os.makedirs(to_path, exist_ok=True)
        for alpha_name in os.listdir(os.path.join(from_path, img_name)):
            shutil.copy2(os.path.join(from_path, img_name, alpha_name), os.path.join(to_path, alpha_name))

    # for layer
    from_path = os.path.join(current_dir, '..', 'data', '3_depth_segment_output', 'patch')
    for img_name in os.listdir(from_path):
        to_path = os.path.join(current_dir, '..', 'data', '4_blender_input', img_name, 'layer')
        os.makedirs(to_path, exist_ok=True)
        for layer_name in os.listdir(os.path.join(from_path, img_name)):
            shutil.copy2(os.path.join(from_path, img_name, layer_name), os.path.join(to_path, layer_name))

        # background
        os.makedirs(os.path.join(current_dir, '..', 'data', '4_blender_input', img_name, 'background'), exist_ok=True)

if __name__ == '__main__':
    try:
        color_extraction_to_fscs()
        fscs_to_depth_segment()
        prepare_blender_input()
    except Exception as e:
        pass
