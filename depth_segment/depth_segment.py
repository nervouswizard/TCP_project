import numpy as np
import cv2
from PIL import Image
import sys
import os
import argparse

np.set_printoptions(threshold=sys.maxsize)

def label_segment(depth_map, num_groups, threshold):
    background_thresh = np.min(depth_map) + threshold
    
    # Calculate the histogram of the depth map
    hist, bins = np.histogram(depth_map, bins=np.max(depth_map))
    print('depth map max: ', np.max(depth_map))
    print('depth min max: ', np.min(depth_map))
    
    # Calculate the cumulative distribution function (CDF) of the histogram
    cdf = np.cumsum(hist) / np.sum(hist)
    
   # Initialize the group sizes array using a for loop
    cdf_area = np.zeros(num_groups)
    for i in range(num_groups):
        cdf_area[i] = (i + 1) / num_groups
    print('cdf_area', cdf_area)
    
    group_bounds = np.interp(cdf_area, cdf, bins[1:])
    print('group_bounds', group_bounds)
    
    # Create a 2D array to store the segmentation result
    segmentation = np.zeros((depth_map.shape[0], depth_map.shape[1]), dtype=np.uint8)

    # Iterate over the depth map and assign each pixel to a segment based on its depth value
    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            depth_value = depth_map[i, j]
            if depth_value > background_thresh:  # foreground pixel
                label = np.searchsorted(group_bounds, depth_value)
                segmentation[i, j] = label  # +1 because label 0 is background
                
    # print(segmentation)
    return segmentation

def get_label_mask(segmentation, output_path):
    for i in range(np.max(segmentation)+1):
        mask = np.where(segmentation==i, 255, 0).astype('uint8')
        image = Image.fromarray(mask)
        
        image.save(output_path + 'mask_' + str(i) +'.png')
                

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Depth map segmentation')
    
    # parser.add_argument('--base_dir_in', type=str, default='./myInput/depth/', required=True, help='Base input directory')
    # parser.add_argument('--base_dir_out', type=str, default='./myOutput/mask/', required=True, help='Base output directory')
    # parser.add_argument('--input_name', type=str, default='human', required=True, help='Input name')
    # parser.add_argument('--num_groups', type=int, default=10, help='Number of groups')
    # parser.add_argument('--threshold', type=int, default=10000, help='Background threshold')
    
    # args = parser.parse_args()
    
    # base_dir_in = args.base_dir_in
    # base_dir_out = args.base_dir_out
    # dir_name = args.input_name
    # num_groups = args.num_groups
    # threshold = args.threshold
    # for debugging
    base_dir_in = os.path.join('myInput', 'depth')
    base_dir_out = os.path.join('myOutput', 'mask')
    dir_name = 'squirrel'
    num_groups = 10
    threshold = 10000
    
    
    
    input_path = os.path.join(base_dir_in, dir_name+'.png')
    output_path = os.path.join(base_dir_out, dir_name)
    
    # 使用 try 建立目錄
    try:
        os.makedirs(output_path)
    # 檔案已存在的例外處理
    except FileExistsError:
        print('Clean the directory')
        # Delete all files in the directory
        for file in os.listdir(output_path):
            file_path = os.path.join(output_path, file)
            os.unlink(file_path)
            
    depth_map = cv2.imread(input_path, cv2.IMREAD_ANYDEPTH)
    print(depth_map.shape)
    
    segmentation = label_segment(depth_map, num_groups, threshold)
    get_label_mask(segmentation, output_path)