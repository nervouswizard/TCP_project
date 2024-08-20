# region import
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from net import MaskGenerator, ResiduePredictor
from mydataset import MyDataset
import cv2
import os
import time

from guided_filter_pytorch.guided_filter import GuidedFilter
# endregion

def proc_guidedfilter(alpha_layers, guide_img):
    # guide_imgは， 1chのモノクロに変換
    # target_imgを使う． bn, 3, h, w
    guide_img = (guide_img[:, 0, :, :]*0.299 + guide_img[:, 1, :, :]*0.587 + guide_img[:, 2, :, :]*0.114).unsqueeze(1)
        
    # lnのそれぞれに対してguideg filterを実行
    for i in range(alpha_layers.size(1)): # bn, ln, 1, h, w
        # layerは，bn, 1, h, w
        layer = alpha_layers[:, i, :, :, :]
        
        processed_layer = GuidedFilter(3, 1*1e-6)(guide_img, layer)
        
        # processed_layer = torch.where(processed_layer > 0.05, processed_layer, torch.tensor(0))
        # print('processed_layer', processed_layer)
        
        # レイヤーごとの結果をまとめてlayersの形に戻す (bn, ln, 1, h, w)
        if i == 0: 
            processed_alpha_layers = processed_layer.unsqueeze(1)
            # print(i, processed_alpha_layers.shape)
        else:
            processed_alpha_layers = torch.cat((processed_alpha_layers, processed_layer.unsqueeze(1)), dim=1)
            # print(i, processed_alpha_layers.shape)
    
    return processed_alpha_layers

# 必要な関数を定義する
def replace_color(primary_color_layers, manual_colors):
    temp_primary_color_layers = primary_color_layers.clone()
    for layer in range(len(manual_colors)):
        for color in range(3):
                temp_primary_color_layers[:,layer,color,:,:].fill_(manual_colors[layer][color])
    return temp_primary_color_layers

def cut_edge(target_img):
    #print(target_img.size())
    target_img = F.interpolate(target_img, scale_factor=resize_scale_factor, mode='area')
    #print(target_img.size())
    h = target_img.size(2)
    w = target_img.size(3)
    h = h - (h % 8)
    w = w - (w % 8)
    target_img = target_img[:,:,:h,:w]
    #print(target_img.size())
    return target_img

def alpha_normalize(alpha_layers):
    # constraint (sum = 1)
    # layersの状態で受け取り，その形で返す. bn, ln, 1, h, w
    return alpha_layers / (alpha_layers.sum(dim=1, keepdim=True) + 1e-8)

def read_backimage():
    img = cv2.imread('../dataset/backimage.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2,0,1))
    img = img/255
    img = torch.from_numpy(img.astype(np.float32))

    return img.view(1,3,256,256).to(device)

## Define functions for mask operation.
# マスクを受け取る関数
# target_layer_numberが冗長なレイヤーの番号（２つ）のリスト．これらのレイヤーに操作を加える
def load_mask(mask_path):
    mask = cv2.imread(mask_path, 0) #白黒で読み込み
    mask[mask<128] = 0.
    mask[mask >= 128] = 1.
    # tensorに変換する
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
    
    return mask
        

def mask_operate(alpha_layers, target_layer_number, mask_path):
    layer_A = alpha_layers[:, target_layer_number[0], :, :, :]
    layer_B = alpha_layers[:, target_layer_number[1], :, :, :]
    
    layer_AB = layer_A + layer_B    
    
    mask = load_mask(mask_path)
    
    mask = cut_edge(mask)
    
    layer_A = layer_AB * mask
    layer_B = layer_AB * (1. - mask)
    
    return_alpha_layers = alpha_layers.clone()
    return_alpha_layers[:, target_layer_number[0], :, :, :] = layer_A
    return_alpha_layers[:, target_layer_number[1], :, :, :] = layer_B
    
    return return_alpha_layers


def my_mask_operate(alpha_layers, mask_path):
    return_alpha_layers = alpha_layers.clone()
    
    for i in range(num_primary_color):
        layer = alpha_layers[:, i, :, :, :]
        
        mask = load_mask(mask_path)
        
        mask = cut_edge(mask)

        layer = layer * mask
        
        return_alpha_layers[:, i, :, :, :] = layer
        
    return return_alpha_layers

# region User inputs
run_name = 'sample'
num_primary_color = 7 
# num_primary_color = 6 
csv_path = 'my_test.csv' # なんでも良い．後方でパスを置き換えるから
resize_scale_factor = 1

boost_scale = 1
bound_flag = True

# image name and palette color values

### k-means
# img_name = 'bluebird.png'; manual_color_0 = [254, 254, 254]; manual_color_1 = [119, 135, 160]; manual_color_2 = [189, 203, 219]; manual_color_3 = [71, 68, 70]; manual_color_4 = [220, 129, 91]; manual_color_5 = [149, 165, 188]; manual_color_6 = [246, 203, 154];
# img_name = 'human.png'; manual_color_0 = [64, 59, 54]; manual_color_1 = [214, 197, 177]; manual_color_2 = [138, 131, 125]; manual_color_3 = [247, 237, 229]; manual_color_4 = [100, 94, 88]; manual_color_5 = [32, 28, 25]; manual_color_6 = [165, 158, 151];
# img_name = 'lotus.png'; manual_color_0 = [16, 15, 14]; manual_color_1 = [179, 179, 178]; manual_color_2 = [65, 62, 56]; manual_color_3 = [207, 207, 206]; manual_color_4 = [126, 113, 96]; manual_color_5 = [132, 28, 43]; manual_color_6 = [154, 153, 151];
# img_name = 'rooster.png'; manual_color_0 = [250, 240, 234]; manual_color_1 = [34, 31, 30]; manual_color_2 = [112, 114, 114]; manual_color_3 = [72, 71, 70]; manual_color_4 = [157, 32, 55]; manual_color_5 = [180, 118, 98]; manual_color_6 = [196, 166, 153];
# img_name = 'rosemaling.png'; manual_color_0 = [50, 64, 61]; manual_color_1 = [253, 253, 246]; manual_color_2 = [222, 144, 59]; manual_color_3 = [131, 151, 144]; manual_color_4 = [90, 108, 95]; manual_color_5 = [27, 22, 16]; manual_color_6 = [205, 204, 185];
# img_name = 'buffalo.png'; manual_color_0 = [252, 254, 254]; manual_color_1 = [61, 72, 91]; manual_color_2 = [212, 218, 220]; manual_color_3 = [110, 116, 133]; manual_color_4 = [17, 28, 53]; manual_color_5 = [234, 240, 243]; manual_color_6 = [153, 162, 175];
# img_name = 'crab.png'; manual_color_0 = [53, 49, 77]; manual_color_1 = [253, 253, 253]; manual_color_2 = [181, 183, 188]; manual_color_3 = [86, 87, 102]; manual_color_4 = [208, 210, 214]; manual_color_5 = [121, 121, 131]; manual_color_6 = [154, 154, 161];
# img_name = 'lotus2.png'; manual_color_0 = [253, 254, 254]; manual_color_1 = [111, 141, 102]; manual_color_2 = [242, 132, 139]; manual_color_3 = [17, 20, 17]; manual_color_4 = [152, 177, 138]; manual_color_5 = [215, 53, 73]; manual_color_6 = [230, 204, 195];
# img_name = 'monkey.png'; manual_color_0 = [254, 234, 198]; manual_color_1 = [44, 32, 23]; manual_color_2 = [202, 161, 128]; manual_color_3 = [236, 207, 170]; manual_color_4 = [94, 72, 54]; manual_color_5 = [232, 91, 70]; manual_color_6 = [147, 119, 92];
# img_name = 'owl.png'; manual_color_0 = [254, 254, 254]; manual_color_1 = [163, 158, 143]; manual_color_2 = [188, 185, 171]; manual_color_3 = [94, 92, 84]; manual_color_4 = [211, 209, 198]; manual_color_5 = [133, 128, 115]; manual_color_6 = [230, 230, 227];
# img_name = 'pandas.png'; manual_color_0 = [38, 41, 31]; manual_color_1 = [254, 254, 254]; manual_color_2 = [133, 134, 126]; manual_color_3 = [229, 231, 230]; manual_color_4 = [5, 8, 2]; manual_color_5 = [189, 189, 185]; manual_color_6 = [84, 86, 77];
# img_name = 'sparrow.png'; manual_color_0 = [254, 254, 254]; manual_color_1 = [110, 106, 125]; manual_color_2 = [233, 110, 220]; manual_color_3 = [161, 170, 179]; manual_color_4 = [52, 54, 73]; manual_color_5 = [218, 221, 226]; manual_color_6 = [206, 58, 138];
# img_name = 'squirrel.png'; manual_color_0 = [254, 254, 251]; manual_color_1 = [137, 143, 137]; manual_color_2 = [188, 181, 169]; manual_color_3 = [94, 106, 106]; manual_color_4 = [207, 98, 109]; manual_color_5 = [47, 61, 67]; manual_color_6 = [220, 215, 202];

# img_name = 'Kim_Jisoo.jpg'; manual_color_0 = [175, 202, 211]; manual_color_1 = [28, 24, 23]; manual_color_2 = [211, 166, 151]; manual_color_3 = [115, 145, 145]; manual_color_4 = [229, 214, 205]; manual_color_5 = [174, 117, 90]; manual_color_6 = [89, 53, 43]; 

### octree
img_name = 'rooster.png'; manual_color_0 = [249, 239, 233]; manual_color_1 = [61, 59, 59]; manual_color_2 = [164, 74, 77]; manual_color_3 = [155, 155, 152]; manual_color_4 = [180, 139, 112]; manual_color_5 = [219, 165, 151]; manual_color_6 = [100, 119, 132];
# img_name = 'squirrel.png'; manual_color_0 = [239, 238, 234]; manual_color_1 = [97, 109, 108]; manual_color_2 = [197, 87, 101]; manual_color_3 = [180, 139, 116]; manual_color_4 = [122, 135, 135]; manual_color_5 = [55, 74, 82]; manual_color_6 = [214, 109, 139];
# img_name = 'bluebird.png'; manual_color_0 = [254, 254, 254]; manual_color_1 = [148, 157, 173]; manual_color_2 = [157, 176, 204]; manual_color_3 = [87, 85, 88]; manual_color_4 = [246, 210, 167]; manual_color_5 = [110, 138, 174]; manual_color_6 = [181, 199, 221];
# img_name = 'human.png'; manual_color_0 = [245, 236, 227]; manual_color_1 = [162, 155, 148]; manual_color_2 = [38, 34, 31]; manual_color_3 = [99, 93, 87]; manual_color_4 = [215, 197, 175]; manual_color_5 = [138, 131, 124]; manual_color_6 = [203, 187, 170];
# img_name = 'lotus.png'; manual_color_0 = [206, 207, 206]; manual_color_1 = [169, 169, 168]; manual_color_2 = [57, 51, 48]; manual_color_3 = [152, 65, 74]; manual_color_4 = [156, 143, 91]; manual_color_5 = [190, 192, 190]; manual_color_6 = [190, 192, 192];
# img_name = 'rooster.png'; manual_color_0 = [249, 239, 233]; manual_color_1 = [61, 59, 59]; manual_color_2 = [164, 74, 77]; manual_color_3 = [155, 155, 152]; manual_color_4 = [180, 139, 112]; manual_color_5 = [219, 165, 151]; manual_color_6 = [100, 119, 132];
# img_name = 'rosemaling.png'; manual_color_0 = [251, 251, 244]; manual_color_1 = [69, 80, 72]; manual_color_2 = [218, 152, 69]; manual_color_3 = [156, 166, 160]; manual_color_4 = [173, 107, 49]; manual_color_5 = [114, 139, 138]; manual_color_6 = [232, 209, 167];
# img_name = 'buffalo.png'; manual_color_0 = [243, 246, 247]; manual_color_1 = [84, 91, 107]; manual_color_2 = [13, 23, 49]; manual_color_3 = [39, 52, 73]; manual_color_4 = [149, 156, 169]; manual_color_5 = [112, 119, 136]; manual_color_6 = [57, 69, 88];
# img_name = 'crab.png'; manual_color_0 = [252, 252, 253]; manual_color_1 = [159, 159, 166]; manual_color_2 = [205, 206, 210]; manual_color_3 = [91, 91, 105]; manual_color_4 = [50, 43, 75]; manual_color_5 = [187, 188, 194]; manual_color_6 = [120, 121, 132];
# img_name = 'lotus2.png'; manual_color_0 = [251, 250, 250]; manual_color_1 = [157, 169, 154]; manual_color_2 = [62, 69, 61]; manual_color_3 = [244, 159, 161]; manual_color_4 = [114, 150, 103]; manual_color_5 = [222, 75, 90]; manual_color_6 = [142, 171, 114];
# img_name = 'monkey.png'; manual_color_0 = [237, 210, 174]; manual_color_1 = [254, 236, 200]; manual_color_2 = [161, 108, 83]; manual_color_3 = [213, 178, 144]; manual_color_4 = [44, 33, 25]; manual_color_5 = [169, 142, 112]; manual_color_6 = [92, 73, 55];
# img_name = 'owl.png'; manual_color_0 = [252, 252, 252]; manual_color_1 = [247, 171, 167]; manual_color_2 = [169, 158, 255]; manual_color_3 = [170, 168, 156]; manual_color_4 = [163, 252, 170]; manual_color_5 = [221, 221, 178]; manual_color_6 = [168, 255, 214];
# img_name = 'pandas.png'; manual_color_0 = [253, 253, 253]; manual_color_1 = [34, 27, 255]; manual_color_2 = [255, 31, 33]; manual_color_3 = [30, 33, 26]; manual_color_4 = [25, 253, 27]; manual_color_5 = [254, 160, 161]; manual_color_6 = [164, 159, 255];
# img_name = 'sparrow.png'; manual_color_0 = [250, 250, 251]; manual_color_1 = [154, 160, 169]; manual_color_2 = [73, 73, 89]; manual_color_3 = [213, 87, 191]; manual_color_4 = [230, 159, 223]; manual_color_5 = [182, 202, 202]; manual_color_6 = [111, 114, 138];
# img_name = 'squirrel.png'; manual_color_0 = [239, 238, 234]; manual_color_1 = [97, 109, 108]; manual_color_2 = [197, 87, 101]; manual_color_3 = [180, 139, 116]; manual_color_4 = [122, 135, 135]; manual_color_5 = [55, 74, 82]; manual_color_6 = [214, 109, 139];


target_layer_number = [0, 1] # マスクで操作するレイヤーの番号
mask_path = '../myInput/mask/'+ os.path.splitext(img_name)[0] + '_mask.png'
print(mask_path)

#  endregion

# region load weight
img_path = '../myInput/' + img_name
# img_path = '../dataset/test/' + img_name

path_mask_generator = 'results/' + run_name + '/mask_generator.pth'
path_residue_predictor = 'results/' + run_name + '/residue_predictor.pth'

if num_primary_color == 7:
    manual_colors = np.array([manual_color_0, manual_color_1, manual_color_2, manual_color_3,\
                                               manual_color_4, manual_color_5, manual_color_6]) /255
elif num_primary_color == 6:
    manual_colors = np.array([manual_color_0, manual_color_1, manual_color_2, manual_color_3,\
                                               manual_color_4, manual_color_5]) /255
    
# endregion  

# region mkdir
try:
    os.makedirs('results/%s/%s' % (run_name, img_name))
except OSError:
    pass
# endregion

# region model
test_dataset = MyDataset(csv_path, num_primary_color, mode='test')
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    )



device = 'cpu'

# define model
mask_generator = MaskGenerator(num_primary_color).to(device)
residue_predictor = ResiduePredictor(num_primary_color).to(device)


# load params
mask_generator.load_state_dict(torch.load(path_mask_generator, map_location=torch.device('cpu')))
residue_predictor.load_state_dict(torch.load(path_residue_predictor, map_location=torch.device('cpu')))


# eval mode
mask_generator.eval()
residue_predictor.eval()

backimage = read_backimage()

# datasetにある画像のパスを置き換えてしまう
test_dataset.imgs_path[0] = img_path

# endregion

# region main process
print('Start!')
img_number = 0


mean_estimation_time = 0
with torch.no_grad():
    for batch_idx, (target_img, primary_color_layers) in enumerate(test_loader):
        print(batch_idx, img_number)
        if batch_idx != img_number:
            print('Skip ', batch_idx)
            continue
        print('img #', batch_idx)
        target_img = cut_edge(target_img)   # resize the image and make sure the H & W can be divided by 8  
        target_img = target_img.to(device) # bn, 3ch, h, w
        primary_color_layers = primary_color_layers.to(device) # primary_color_layers #out: (bn, ln, 3, h, w)
        #primary_color_layers = color_regresser(target_img)
        ##
        ##
        primary_color_layers = replace_color(primary_color_layers, manual_colors) #ここ
        ##
        #print(primary_color_layers.mean())
        #print(primary_color_layers.size())
        start_time = time.time()
        primary_color_pack = primary_color_layers.view(primary_color_layers.size(0), -1 , primary_color_layers.size(3), primary_color_layers.size(4))

        import torch
        import matplotlib.pyplot as plt
        from PIL import Image
        
        # image1_tensor= primary_color_layers[0][0]
        # print(image1_tensor.shape)
        # out = image1_tensor.permute(2, 1, 0).detach().cpu().numpy()
        # arr_ = np.squeeze(out)
        # plt.imshow(arr_)
        # plt.show()
        
        primary_color_pack = cut_edge(primary_color_pack)
        primary_color_layers = primary_color_pack.view(primary_color_pack.size(0),-1,3,primary_color_pack.size(2), primary_color_pack.size(3))
        pred_alpha_layers_pack = mask_generator(target_img, primary_color_pack)
        pred_alpha_layers = pred_alpha_layers_pack.view(target_img.size(0), -1, 1, target_img.size(2), target_img.size(3))
        
        # print('primary_color_layers', primary_color_layers.shape)
        # print('primary_color_layers', primary_color_layers)
        # print('pred_alpha_layers', pred_alpha_layers.shape)
        
        ## Alpha Layer Proccessing
        processed_alpha_layers = alpha_normalize(pred_alpha_layers) 
        # processed_alpha_layers = mask_operate(processed_alpha_layers, target_layer_number, mask_path) # Option
        
        processed_alpha_layers = my_mask_operate(processed_alpha_layers, mask_path)
        
        processed_alpha_layers = proc_guidedfilter(processed_alpha_layers, target_img) # Option
        # processed_alpha_layers = alpha_normalize(processed_alpha_layers)  # Option
        
        # print('processed_alpha_layers', processed_alpha_layers.shape)
        # print('processed_alpha_layers', processed_alpha_layers)
        
        ### my add
        processed_alpha_layers *= boost_scale
        if bound_flag:
            processed_alpha_layers = torch.where(processed_alpha_layers > 1, torch.tensor(1), processed_alpha_layers)
        
            
        # print('processed_alpha_layers_2', processed_alpha_layers.shape)
        # print('processed_alpha_layers_2', processed_alpha_layers)
        
        ##
        mono_color_layers = torch.cat((primary_color_layers, processed_alpha_layers), 2) #shape: bn, ln, 4, h, w
        mono_color_layers_pack = mono_color_layers.view(target_img.size(0), -1 , target_img.size(2), target_img.size(3))
        residue_pack  = residue_predictor(target_img, mono_color_layers_pack)
        residue = residue_pack.view(target_img.size(0), -1, 3, target_img.size(2), target_img.size(3))
        
        # print('residue', residue.shape)
        # print('residue', residue)
        
        pred_unmixed_rgb_layers = torch.clamp((primary_color_layers + residue), min=0., max=1.0)
        reconst_img = (pred_unmixed_rgb_layers * processed_alpha_layers).sum(dim=1)
        # print('reconst_img', reconst_img[0][0][0][0])
        
        ### my add
        no_residue_reconst_img = (primary_color_layers * processed_alpha_layers).sum(dim=1)
        
        # print('pred_unmixed_rgb_layers', pred_unmixed_rgb_layers.shape)
        # print('pred_unmixed_rgb_layers', pred_unmixed_rgb_layers)
        # print('reconst_img', reconst_img.shape)
        
        end_time = time.time()
        estimation_time = end_time - start_time
        print(estimation_time)
        mean_estimation_time += estimation_time
        
        if True:
            # batchsizeは１で計算されているはず．それぞれ保存する．
            save_layer_number = 0
            save_image(primary_color_layers[save_layer_number,:,:,:,:],
                   'results/%s/%s/test' % (run_name, img_name) + '_img-%02d_primary_color_layers.png' % batch_idx)
            save_image(reconst_img[save_layer_number,:,:,:].unsqueeze(0),
                   'results/%s/%s/test' % (run_name, img_name)  + '_img-%02d_reconst_img.png' % batch_idx)
            save_image(target_img[save_layer_number,:,:,:].unsqueeze(0),
                   'results/%s/%s/test' % (run_name, img_name)  + '_img-%02d_target_img.png' % batch_idx)
            
            ### my add
            save_image(no_residue_reconst_img[save_layer_number,:,:,:].unsqueeze(0),
                   'results/%s/%s/test' % (run_name, img_name)  + '_img-%02d_no_risidue_reconst_img.png' % batch_idx)

            # RGBAの４chのpngとして保存する
            RGBA_layers = torch.cat((pred_unmixed_rgb_layers, processed_alpha_layers), dim=2) # out: bn, ln, 4, h, w
            # test ではバッチサイズが１なので，bn部分をなくす
            RGBA_layers = RGBA_layers[0] # ln, 4. h, w
            
            # print('RGBA_layers', RGBA_layers[0][0][0][0])
            
            # ln ごとに結果を保存する
            # for i in range(len(RGBA_layers)):
            #     save_image(RGBA_layers[i, :, :, :], 'results/%s/%s/img-%02d_layer-%02d.png' % (run_name, img_name, batch_idx, i) )
            # print('Saved to results/%s/%s/...' % (run_name, img_name))

            for i in range(len(RGBA_layers)):
                save_image(RGBA_layers[i, :, :, :], 'results/%s/%s/img-%02d_layer-%02d.png' % (run_name, img_name, batch_idx, i) )
                
                # layer = cv2.imread('results/%s/%s/img-%02d_layer-%02d.png' % (run_name, img_name, batch_idx, i), cv2.IMREAD_UNCHANGED)
                # im1 = Image.open('results/%s/%s/img-%02d_layer-%02d.png' % (run_name, img_name, batch_idx, i))
                # r, g, b ,a = im1.split()
                # output = cv2.add(layer, layer)
                # cv2.imwrite('output.png', output)
                
            print('Saved to results/%s/%s/...' % (run_name, img_name))
            
            
        if False:
            ### mono_colorの分も保存する ###
            # RGBAの４chのpngとして保存する
            mono_RGBA_layers = torch.cat((primary_color_layers, processed_alpha_layers), dim=2) # out: bn, ln, 4, h, w
            # test ではバッチサイズが１なので，bn部分をなくす
            mono_RGBA_layers = mono_RGBA_layers[0] # ln, 4. h, w
            # ln ごとに結果を保存する
            for i in range(len(mono_RGBA_layers)):
                save_image(mono_RGBA_layers[i, :, :, :], 'results/%s/%s/mono_img-%02d_layer-%02d.png' % (run_name, img_name, batch_idx, i) )

            save_image((primary_color_layers * processed_alpha_layers).sum(dim=1)[save_layer_number,:,:,:].unsqueeze(0),
                   'results/%s/%s/test' % (run_name, img_name)  + '_mono_img-%02d_reconst_img.png' % batch_idx)   
        
        
        if batch_idx == 0:
            break # debug用

# endregion

# 処理まえのアルファを保存
for i in range(len(pred_alpha_layers[0])):
            save_image(pred_alpha_layers[0,i, :, :, :], 'results/%s/%s/pred-alpha-00_layer-%02d.png' % (run_name, img_name, i) )
            
# 処理後のアルファの保存 processed_alpha_layers
for i in range(len(processed_alpha_layers[0])):
            save_image(processed_alpha_layers[0,i, :, :, :], 'results/%s/%s/proc-alpha-00_layer-%02d.png' % (run_name, img_name, i) )
            
            # layer = cv2.imread('results/%s/%s/proc-alpha-00_layer-%02d.png' % (run_name, img_name, i))
            # blur = cv2.GaussianBlur(layer, (35, 35), 0)
            # cv2.imwrite('results/%s/%s/proc-alpha-00_layer-%02d.png' % (run_name, img_name, i), blur)

# 処理後のRGBの保存
for i in range(len(pred_unmixed_rgb_layers[0])):
    save_image(pred_unmixed_rgb_layers[0,i, :, :, :], 'results/%s/%s/rgb-00_layer-%02d.png' % (run_name, img_name, i) )