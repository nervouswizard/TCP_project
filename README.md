# 基於擴散模型方法生成 2.5 維國畫視覺藝術

From [PJ_TCP_project](https://github.com/GliAmanti/TCP_project)  
Notion link: [Notion](https://dynamic-guitar-7bc.notion.site/2-5D-TCP-Visual-Art-For-Others-8b1005e21a524fdab62847301f6d604b)

## Environment

### local 

python 3.10.4  
or  
python 3.8 with conda

deno

### server

unknown

## TCP Select

Select appropriate TCPs for running our code. 水墨畫 is suggested.

<img src="https://github.com/user-attachments/assets/be3b3ff5-5dde-4145-9f39-51d0ff195520" width = "49%">
<img src="https://github.com/user-attachments/assets/66840d06-b96d-4461-8719-29473ac8742e" width = "49%">

💡 **Tips of selection**
- 畫面背景盡量乾淨
- 前景主體不用太多
- 前景主體不能過於抽象

## Preprocessing

### Image Matting

- Any matting method which can get the **mask of object** is fine. I use [Matte Anything](https://github.com/hustvl/Matte-Anything) in my thesis. And there is a virtual environment of it in `lab server 83`**.**
    
    ```powershell
    ssh cgvsl@140.116.247.73
    (password: 12345)
    conda activate pj_matteAnything
    ```
    

- Run the following command to activate the GUI.
    
    ```powershell
    cd /home/cgvsl/p76111351/Matte-Anything
    python matte_anything.py
    ```
    

- Remember to fill in the image name at tab `Save Config`. And you will find the matting result in folder `your_demos`. Be sure to download the matting mask in gradio interface.

    下載Alpha Matte
<img src="https://github.com/user-attachments/assets/986cbab3-0113-4a81-a0be-b0cdf01bfd5b" width = "100%">

### Image Inpainting

Any inpainting method which can remove the signature and stamp is fine. I use [iopaint-lama](https://huggingface.co/spaces/Sanster/iopaint-lama) in my thesis.

<img src="https://github.com/user-attachments/assets/ddb91e62-665e-42bf-b34a-059e915fe7d9" width = "100%">

## Layer Decomposition

- Clone the project, which includes the code of **Layer Decomposition**, **Layer Ordering** and **2.5D Model Generation.**
    
    ```powershell
    git clone git@github.com:GliAmanti/TCP_project.git
    ```
    
- Create a virtual environment and install packages for project `fscs`.
    
    ```powershell
    conda create --name TCP_project python=3.8
    
    cd fscs
    conda env create -f environment.yml
    ```
    <img src="https://github.com/user-attachments/assets/aa23d93a-1280-4b3a-8abe-c2132699e5f0">

### [Color Extraction](https://github.com/GliAmanti/OctreeColorExtraction)

- [Install deno package](https://nugine.github.io/deno-manual-cn/getting_started/installation.html)

- Put TCP image into folder `imgs/test`,  and run the following command to get the **primary colors** of the image in `txt` format.
    
    ```powershell
    cd color-extraction/src
    
    python dump2JSON.py # output the json file first
    deno run -A octree.ts
    ```
    octree.ts輸出在 color-extraction/output/圖片名稱.txt

    💡 The content of output file will be like this:
    
    ```
    img_name = 'squirrel.png'; manual_color_0 = [239, 238, 234]; manual_color_1 = [97, 109, 108]; manual_color_2 = [197, 87, 101]; manual_color_3 = [180, 139, 116]; manual_color_4 = [122, 135, 135]; manual_color_5 = [55, 74, 82]; manual_color_6 = [214, 109, 139];
    ```
    
    
    <img src="https://github.com/user-attachments/assets/0e421ee0-2ed6-4ee5-80fd-6071533748a6">
    
    <img src="https://github.com/user-attachments/assets/61aced6e-f6c9-4c35-ab6f-a3aa2ec8ec81">

- Put TCP image into folder `fscs/myInput`, placing its mask in folder `fscs/myInput/mask`. Paste the primary colors into  `fscs/src/inference.py`.  
mask 的檔名需要做相對應的更改
    ```python
    mask_path = '../myInput/mask/'+ os.path.splitext(img_name)[0] + '_mask.png'
    ```
    
    <img src="https://github.com/user-attachments/assets/81c7e042-7a97-4c29-ac58-ed0209c2836c">
    
    <img src="https://github.com/user-attachments/assets/23aa9187-9b9d-4a39-a389-e4f18cde024f">


- Run the following command to get the **RGBA and alpha layers** of TCP image.

    ```powershell
    python inference.py
    ```
    
    <img src="https://github.com/user-attachments/assets/15f3506b-1f6e-41ac-a171-b6a7dd8a4357">

## Depth Map Generation

- Set the config file to connect to `lab server 83`.
    
    ```powershell
    Host LabServer83-cgvsl
      HostName 140.116.247.83
      User cgvsl
    ```
    
- There is a virtual environment in `lab server 83`.
    
    ```powershell
    conda activate pj_TCP_project
    cd /home/cgvsl/p76111351/TCP_project
    ```
    
    <img src="https://github.com/user-attachments/assets/723adc83-4cd8-4889-8b72-b4d8e47c5f0f">
    
    <img src="https://github.com/user-attachments/assets/f2f2e5e7-1a22-495a-a065-3f7589c71ef0">
    

### server內 TCP_project/configs/hyperparameter.yaml

- Line 2  
    enable_xl: False `使用SD1.5`  
    enable_xl: True `使用SDXL`

- Line 6  
    enable_other_base: True `使用外部model`  
    other_base_model: `SD1.5 外部model路徑`  
    other_xl_base_model: `SDXL 外部model路徑`

- Line 20  
    original_image == addition_image：`text to image`  
    original_image != addition_image：`image to image`

- Line 94  
    batch_size `要生幾張圖`

- Line 95  
    sampling_steps `跟圖的品質、時間成正比，預設40`  


### Image Resizing (opt.)

Resize your TCP into square will get better result of ITM. 512 x 512 is suggested.

### Image Transform Mechanism

Set the `hyperparameter.yaml`(region **Image, ControlNet** and **Prompt**), comment the function `depth_estimation` in main function and run the following command to get the **reconstructed image** in folder `output`. 

### 跑第一次 generate_image.py
將 TCP 放入`TCP_PROJECT/images/[圖片名稱]/`
修改```TCP_project/configs/hyperparameter.yaml```  
Line20 choice2底下 original_image與addition_image  
Line80 使用marigold  
Line115底下，使用正確的text prompt  
修改```generate_image.py```  
使用Line 688 inference(config)

```powershell
python generate_image.py --config "./configs/hyperparameter.yaml"
```

<img src="https://github.com/user-attachments/assets/2f75ab2c-fc9f-4d0c-b00a-142003659b84">

<img src="https://github.com/user-attachments/assets/24bcb8cc-bd8c-4140-87db-3263a707393f">

<img src="https://github.com/user-attachments/assets/c9b00ce2-29e5-44d8-bb4e-a56994d6e1a3">

<img src="https://github.com/user-attachments/assets/c1764b3b-45e6-44bd-9535-98c14338a145">

<img src="https://github.com/user-attachments/assets/87971791-ad4d-48ca-bf41-cab88617f5d8">

<img src="https://github.com/user-attachments/assets/51165a52-2807-44b5-b56a-c4cf6b87ed23">

## Depth Estimation

Set the `hyperparameter.yaml`(region **ControlNet**), source and destination directory, comment the function `inferencee` in main function and run the following command to get the **refined depth map** in destination directory.

### 跑第二次generate_image.py
修改```TCP_project/configs/hyperparameter.yaml```  
Line78 使用depth_anything  
修改```generate_image.py```  
使用Line 699 depth_estimation(config)  
function depth_estimation(config)內  
改input_dir與output_dir  

```powershell
python generate_image.py --config "./configs/hyperparameter.yaml"
```
<img src="https://github.com/user-attachments/assets/61c4a3ff-1785-4b2f-856f-3835f57e5afc">

## Layer Ordering

- `Local` Enter the folder `depth_segment`.
```powershell
cd depth_segment
```

上一步跑出來的結果`output_dir`內選一張深度圖下載到local  
放入`TCP_project/depth-segment/myInput/depth  
改名為圖片本身的名字

創建資料夾`TCP_project/depth-segment/myInput/alpha/圖片名稱`  
創建資料夾`TCP_project/depth-segment/myInput/layer/圖片名稱`  

把`fscs/src/results/sample/圖片名稱.png`資料夾內的alpha(七個圖片檔)  
複製至`TCP_project/depth-segment/myInput/alpha/圖片名稱`  
proc alpha 是較為平滑的alpha，論文就是用proc  

把`fscs/src/results/sample/圖片名稱.png`資料夾內的img_layer(七個圖片檔)  
複製至`TCP_project/depth-segment/myInput/layer/圖片名稱`

調整`depth_segment/depth_segment.py`  
調整threshold  
更改dir_name  
跑depth_segment.py

檢查`TCP_project\depth_segment\myOutput\mask\test`分割後的深度理想程度

調整`depth_segment/patch_segment.py`  
更改dir_name  
跑patch_segment.py

- Rename the refined depth map and put it into `myInput/depth`, run the following command to get **mask group** in folder `myOutput/mask`.

```powershell
python depth_segment.py \
	--base_dir_in './myInput/depth/' \
	--base_dir_out './myOutput/mask/' \
	--num_groups 10 \
	--threshold 10000 \
	--input_name 'human'
```    
- Put the RGBA and alpha layers into `myInput/layer` and  `myInput/alpha`, and run the following command to get **RGBA and alpha patches** in `myOutput/patch` and `myOutput/patch_alpha`.
```powershell
python patch_segment.py \
	--layer_dir './myInput/layer/' \
	--mask_dir './myInput/mask/' \
	--alpha_dir './myInput/alpha/' \
	--base_dir_out './myOutput/'\
	--input_name 'human'
```
<img src="https://github.com/user-attachments/assets/8aafa4ea-9bd1-4de1-b6cb-b44e6dfd0e15">

## 2.5D Model Generation

- Modify **base_path**, **layer_num**, **layer_offset** and **displace_strength** in `template.py`.
    <img src="https://github.com/user-attachments/assets/e47eb528-c357-47c5-8f00-b0ea51d39317">
    

- Copy RGBA and alpha patches to **blender directory**.
    
    ```powershell
    mkdir /home/gliamanti/下載/human_new
    mkdir /home/gliamanti/下載/human_new/layer
    mkdir /home/gliamanti/下載/human_new/alpha
    mkdir /home/gliamanti/下載/human_new/background
    ```
    
    ```powershell
    rm -r /home/gliamanti/下載/human_new/alpha/*
    rm -r /home/gliamanti/下載/human_new/layer/*
    rm -r /home/gliamanti/下載/human_new/background/*
    
    cp /home/gliamanti/myApps/depth_segment/myOutput/patch_alpha/human/* /home/gliamanti/下載/human_new/alpha
    cp /home/gliamanti/myApps/depth_segment/myOutput/patch/human/* /home/gliamanti/下載/human_new/layer
    cp /home/gliamanti/下載/scripting/overcast_soil_puresky_4k.exr /home/gliamanti/下載/human_new/background
    ```
    

- Open a `Blender` window, run the blender script `template.py`.

<img src="https://github.com/user-attachments/assets/4e995219-2a9e-41f7-bbb2-d1c0bc406164">
