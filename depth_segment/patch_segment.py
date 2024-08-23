import cv2
import os
import numpy as np
import shutil
import argparse


class myLayer:
    def __init__(self, layer_path='', alpha_path=''):
        self.__layer_path = layer_path
        self.__alpha_path = alpha_path

    def get_layer_path(self):
        return self.__layer_path
    def get_alpha_path(self):
        return self.__alpha_path
    def set_layer_path(self, layer_path):
        self.__layer_path = layer_path
    def set_alpha_path(self, alpha_path):
        self.__alpha_path = alpha_path
    
def get_patch(layer_path, alpha_path, mask):
    layer = cv2.imread(layer_path, cv2.IMREAD_UNCHANGED)
    alpha = cv2.imread(alpha_path, cv2.IMREAD_ANYDEPTH)

    patch = np.empty(layer.shape, dtype=np.uint8)
    patch_alpha = np.empty(alpha.shape, dtype=np.uint8)
    
    # Apply the mask to the layer image
    patch[:, :, :3] = layer[:, :, :3]
    patch[:, :, 3] = np.where(mask, layer[:, :, 3], 0)
    
    # Apply the mask to the alpha image
    patch_alpha = np.where(mask, alpha, 0)
    
    # Apply a Gaussian blur to the image
    patch_alpha = cv2.GaussianBlur(patch_alpha, (65, 65), 0)
    
    return patch, patch_alpha

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Layer and alpha images segmentation')
    
    # parser.add_argument('--layer_dir', type=str, default='./myInput/layer/', required=True, help='Layer directory')
    # parser.add_argument('--mask_dir', type=str, default='./myInput/mask/', required=True, help='Mask directory')
    # parser.add_argument('--alpha_dir', type=str, default='./myInput/alpha/', required=True, help='Alpha directory')
    # parser.add_argument('--base_dir_out', type=str, default='./myOutput/', required=True, help='Base output directory')
    
    # parser.add_argument('--input_name', type=str, default='human', required=True, help='Input name')
    
    # args = parser.parse_args()
    
    # dir_name = args.input_name
    # layer_dir = args.layer_dir + dir_name + '/'
    # mask_dir = args.mask_dir + dir_name + '/'
    # alpha_dir = args.alpha_dir + dir_name + '/'
    # base_dir_out = args.base_dir_out
    
    # for debugging
    dir_name = 'human'
    layer_dir = os.path.join('myInput', 'layer', dir_name)
    mask_dir = os.path.join('myInput', 'mask', dir_name)
    alpha_dir = os.path.join('myInput', 'alpha', dir_name)
    base_dir_out = 'myOutput'
    
    
    patch_output_dir = os.path.join(base_dir_out, 'patch', dir_name)
    patch_alpha_output_dir = os.path.join(base_dir_out, 'patch_alpha', dir_name)
    
    myLayers = []
    
    # Create the destination directory if it doesn't exist
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    if not os.path.exists(patch_output_dir):
        os.makedirs(patch_output_dir)
    if not os.path.exists(patch_alpha_output_dir):
        os.makedirs(patch_alpha_output_dir)
        
        
    # Rename the layer image
    # for i, file in enumerate(sorted(os.listdir(layer_dir))):
    #     new_name = f"layer_{i}{os.path.splitext(file)[1]}"
    #     new_fullname = os.path.join(layer_dir, new_name)
    #     os.rename(os.path.join(layer_dir, file), new_fullname)

    # Delete all files in patch directory
    for file in os.listdir(patch_output_dir):
        file_path = os.path.join(patch_output_dir, file)
        os.unlink(file_path)
            
    # Delete all files in patch_alpha directory
    for file in os.listdir(patch_alpha_output_dir):
        file_path = os.path.join(patch_alpha_output_dir, file)
        os.unlink(file_path)
    
    # Delete all files in mask directory
    for file in os.listdir(mask_dir):
        file_path = os.path.join(mask_dir, file)
        os.unlink(file_path)

    
    # Copy mask files from myOutput to myInput
    mask_src_dir = './myOutput/mask/' + dir_name + '/'
    for file in os.listdir(mask_src_dir):
        src_file = os.path.join(mask_src_dir, file)
        shutil.copy2(src_file, mask_dir)
    
    layers = sorted(os.listdir(layer_dir), key=lambda x: int(x.split('_')[1].split('.')[0].lstrip('layer-')))
    alphas = sorted(os.listdir(alpha_dir), key=lambda x: int(x.split('_')[1].split('.')[0].lstrip('layer-')))
    # print(layers)
    # print(alphas)
    for layer, alpha in zip(layers, alphas):
        full_layer_path = os.path.join(layer_dir, layer)
        full_alpha_path = os.path.join(alpha_dir, alpha)
        # print(full_layer_path)
        # print(full_alpha_path)
        layer = myLayer(layer_path=full_layer_path, alpha_path=full_alpha_path)
        myLayers.append(layer)
    
    masks = sorted(os.listdir(mask_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))
    # print(len(masks))
    # print(masks)
    for masks_index, mask in enumerate(masks):  ####### mask 從 1 開始編號
        full_mask_path = os.path.join(mask_dir, mask)
        
        mask = cv2.imread(full_mask_path, cv2.IMREAD_ANYDEPTH)
        # print(alpha.shape)
        # print(depth.shape)
        # depth = np.resize(depth, (layer.shape[0], layer.shape[1]))
        
        for mylayer_index, mylayer in enumerate(myLayers):  ####### layer 與 alpha 從 1 開始編號
            full_layer_path = mylayer.get_layer_path()
            full_alpha_path = mylayer.get_alpha_path()
            
            patch, patch_alpha = get_patch(full_layer_path, full_alpha_path, mask)
            cv2.imwrite(os.path.join(patch_output_dir, 'layer_' + str(masks_index+1) + '-' + str(mylayer_index+1) + '.png'), patch)
            cv2.imwrite(os.path.join(patch_alpha_output_dir, 'alpha_' + str(masks_index+1) + '-' + str(mylayer_index+1) + '.png'), patch_alpha)