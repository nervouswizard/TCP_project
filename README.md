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

* Matte Anything (following by [Matte Anything](https://github.com/hustvl/Matte-Anything))
    * conda
    * python 3.10.16 
    * cuda 12.4 
    * torch 2.5.1
    * delete all specific version in requirements.txt
    * but keep gradio==3.46.1
* TCPdepth
    * conda
    * python 3.10.16 
    * cuda 12.4 
    * torch 2.5.1

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

- Any matting method which can get the **mask of object** is fine. I use [Matte Anything](https://github.com/hustvl/Matte-Anything) in my thesis. And there is a virtual environment of it in `lab server 83`.
    
    ```powershell
    ssh P76134537@140.116.247.83
    conda activate MatteAnything
    ```
    

- Run the following command to activate the GUI.
    
    ```powershell
    cd /sean/ncku/sean/p76111351/Matte-Anything
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
    git clone https://github.com/nervouswizard/TCP_project.git
    ```
    
- Create a virtual environment and install packages for project `fscs`.
    
    ```powershell
    python -m venv virtual
    virtual\Scripts\activate
    pip install -r requirements.txt
    ```

### [Color Extraction]

- [Install deno package](https://nugine.github.io/deno-manual-cn/getting_started/installation.html)

- Put TCP image into folder `data/1_color_extraction_input`,  and run the following command to get the **primary colors** of the image in `txt` format.
    
    ```powershell
    cd src/1_color_extraction
    
    python dump2JSON.py # output the json file first
    deno run -A octree.ts
    ```

    💡 The content of output file will be like this:
    
    ```
    img_name = 'squirrel.png'; manual_color_0 = [239, 238, 234]; manual_color_1 = [97, 109, 108]; manual_color_2 = [197, 87, 101]; manual_color_3 = [180, 139, 116]; manual_color_4 = [122, 135, 135]; manual_color_5 = [55, 74, 82]; manual_color_6 = [214, 109, 139];
    ```


- 1. Run `transfer_file.py` to put TCP image into folder `data/2_fscs_input`
- 2. placing its mask in folder `data/2_fscs_input/mask`
- 3. Paste the primary colors into  `src/2_fscs/inference.py`.  
mask 的檔名需要做相對應的更改
    ```python
    mask_path = '../myInput/mask/'+ os.path.splitext(img_name)[0] + '_mask.png'
    ```
    
    <img src="https://github.com/user-attachments/assets/23aa9187-9b9d-4a39-a389-e4f18cde024f">


- Run the following command to get the **RGBA and alpha layers** of TCP image.

    ```powershell
    python inference.py
    ```

- Some model data is in `need_data.zip`.

## Depth Map Generation

- Set the config file to connect to `lab server 83`.
    
- There is a virtual environment in `lab server 83`.
    
    ```powershell
    conda activate TCPdepth
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

 Run at `Local`

上一步跑出來的結果`output_dir`內選一張深度圖下載到local  
放入`TCP_project/3_depth_segment_input/depth  
改名為圖片本身的名字

執行 transfer_file.py

transfer_file.py會把`data/2_fscs_output/圖片名稱.png/`資料夾內的alpha(七個圖片檔)  
複製至`data/3_depth_segment_input/`  
proc alpha 是較為平滑的alpha，論文就是用proc  

把`data/2_fscs_output/圖片名稱.png/`資料夾內的img_layer(七個圖片檔)  
複製至`data/3_depth_segment_input/`


```powershell
python src/transfer_file.py
```

- 調整`depth_segment/depth_segment.py`  
調整threshold  
更改dir_name  
跑depth_segment.py

檢查`data\3_depth_segment_output\mask\[圖片名稱]\`分割後的深度理想程度

- 調整`3_depth_segment/patch_segment.py`  
更改dir_name  
跑patch_segment.py

- run the following command to get **mask group** in folder `data/3_depth_segment_output/mask`.

```powershell
python depth_segment.py
```    
- run the following command to get **RGBA and alpha patches** in `data/3_depth_segment_output/patch` and `data/3_depth_segment_output/patch_alpha`.
```powershell
python patch_segment.py 
```

## 2.5D Model Generation

- Modify **base_path**, **layer_num**, **layer_offset** and **displace_strength** in `template.py`.
    <img src="https://github.com/user-attachments/assets/e47eb528-c357-47c5-8f00-b0ea51d39317">
    

- 執行transfer_file.py
    
    ```powershell
    python transfer_file.py
    ```
    

- Open a `Blender` window, run the blender script `template.py`.

<img src="https://github.com/user-attachments/assets/4e995219-2a9e-41f7-bbb2-d1c0bc406164">

- 將產生出的Collection放複製至camera_x.blend

- 修改output的輸出路徑