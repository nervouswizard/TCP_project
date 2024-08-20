# 基於擴散模型方法生成 2.5 維國畫視覺藝術

From [PJ_TCP_project](https://github.com/GliAmanti/TCP_project)

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

<img src="https://file.notion.so/f/f/8891a441-ac43-4fbc-b2a7-ea59d48c0ce6/b133d017-5a20-4ec2-ae9a-36dbee79b094/squirrel.png?table=block&id=97804160-9ad4-42e5-957e-bc6d96fb707e&spaceId=8891a441-ac43-4fbc-b2a7-ea59d48c0ce6&expirationTimestamp=1724220000000&signature=ZT5xmhZLcedJK2klQ9Yju2pCpVfYaX1LJ8MvgLw5wHo&downloadName=squirrel.png" width = "49%">
<img src="https://file.notion.so/f/f/8891a441-ac43-4fbc-b2a7-ea59d48c0ce6/3eee8b57-4dd8-494c-b8c5-bfa4a683742f/buffalo.png?table=block&id=31690458-95a8-4b6c-943a-422f5b28974c&spaceId=8891a441-ac43-4fbc-b2a7-ea59d48c0ce6&expirationTimestamp=1724220000000&signature=9N19Yhp8pJ6MwkvpNda2A6MsAxw3JLG2rYlAs4b2E3s&downloadName=buffalo.png" width = "49%">

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
<img src="https://file.notion.so/f/f/8891a441-ac43-4fbc-b2a7-ea59d48c0ce6/403dd748-435d-4015-99e6-6c17fbfcf366/image.png?table=block&id=64c72116-6467-46e5-baa8-84cf17fdb5a2&spaceId=8891a441-ac43-4fbc-b2a7-ea59d48c0ce6&expirationTimestamp=1724220000000&signature=FytgPg7mnr53JXcdbrFOu-fCF9dtbr0xU-7RzI3VKY8&downloadName=image.png" width = "100%">

### Image Inpainting

Any inpainting method which can remove the signature and stamp is fine. I use [iopaint-lama](https://huggingface.co/spaces/Sanster/iopaint-lama) in my thesis.

<img src="https://file.notion.so/f/f/8891a441-ac43-4fbc-b2a7-ea59d48c0ce6/4a14008a-bf06-4f4c-bf91-307d244d6627/image.png?table=block&id=1f464eba-469c-4e75-8a6d-1e0f2fa32683&spaceId=8891a441-ac43-4fbc-b2a7-ea59d48c0ce6&expirationTimestamp=1724220000000&signature=feXwTxoauPcXOr1PYEEo-xajvaPIpcGKKTjTScYFmUc&downloadName=image.png" width = "100%">

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
    <img src="https://file.notion.so/f/f/8891a441-ac43-4fbc-b2a7-ea59d48c0ce6/06f7fb85-49be-489b-859a-f611bd50505b/image.png?table=block&id=378f2934-e56b-42ae-8a67-6f6a40f11192&spaceId=8891a441-ac43-4fbc-b2a7-ea59d48c0ce6&expirationTimestamp=1724220000000&signature=689_l8QCwitQ0K1dHMm-mQXf65wAqvTvCUAO1A5jC30&downloadName=image.png">

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
    

    
    <img src="https://file.notion.so/f/f/8891a441-ac43-4fbc-b2a7-ea59d48c0ce6/a474e730-8c58-4c9c-b5d2-a1aae42526ee/image.png?table=block&id=06e04ab4-b603-46ba-99c9-4d138260178e&spaceId=8891a441-ac43-4fbc-b2a7-ea59d48c0ce6&expirationTimestamp=1724220000000&signature=POP54TMN2HmcaSG0GxXLi9BVCQ8gPWViQMDTnyOSPTk&downloadName=image.png">
    
    <img src="https://file.notion.so/f/f/8891a441-ac43-4fbc-b2a7-ea59d48c0ce6/1aa02027-137d-4f4d-aa20-1ffdf605fcb4/image.png?table=block&id=a85588ca-3fa8-458e-a660-cf2fcf4f04fe&spaceId=8891a441-ac43-4fbc-b2a7-ea59d48c0ce6&expirationTimestamp=1724220000000&signature=ha0kzwfWaAYgg8RoFYIzMFD2IbaH8O-zhSd1_G6B3KQ&downloadName=image.png">

- Put TCP image into folder `fscs/myInput`, placing its mask in folder `fscs/myInput/mask`. Paste the primary colors into  `fscs/src/inference.py`.  
mask 的檔名需要做相對應的更改
    ```python
    mask_path = '../myInput/mask/'+ os.path.splitext(img_name)[0] + '_mask.png'
    ```

    <img src="https://file.notion.so/f/f/8891a441-ac43-4fbc-b2a7-ea59d48c0ce6/680d21d2-c66e-4c55-a27f-5f2f7f1c114f/image.png?table=block&id=ab2683a5-f449-4c90-a5af-de00dca41801&spaceId=8891a441-ac43-4fbc-b2a7-ea59d48c0ce6&expirationTimestamp=1724220000000&signature=yWPTcYzmsWjJqwbWaLKzHbkyE1FMiijxHO2nW6Ri7r4&downloadName=image.png">

    <img src="https://file.notion.so/f/f/8891a441-ac43-4fbc-b2a7-ea59d48c0ce6/cf816ac8-a6c7-456f-b06e-48c65e5eb5c2/image.png?table=block&id=f0800b7b-513b-4f1f-9726-6608a821d0cb&spaceId=8891a441-ac43-4fbc-b2a7-ea59d48c0ce6&expirationTimestamp=1724220000000&signature=gpMb-iavexZx8rFv0Zg7otXeVKEqOEgFRWJXNlqWbyc&downloadName=image.png">


- Run the following command to get the **RGBA and alpha layers** of TCP image.

    ```powershell
    python inference.py
    ```

    <img src="https://file.notion.so/f/f/8891a441-ac43-4fbc-b2a7-ea59d48c0ce6/d52f2313-b0d2-4107-8b59-e654c1331553/image.png?table=block&id=76d47e9d-041a-43cf-b7eb-de956fefa788&spaceId=8891a441-ac43-4fbc-b2a7-ea59d48c0ce6&expirationTimestamp=1724220000000&signature=ALffVf5E-uRQuPPrayKDlUHLQT6-NgpxqlCQpRzcrLU&downloadName=image.png">

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
    
    <img src="https://file.notion.so/f/f/8891a441-ac43-4fbc-b2a7-ea59d48c0ce6/da5c854e-1ed3-4170-9db3-650872daa435/image.png?table=block&id=eea241fc-d50b-40c9-8b6f-2954852ebc17&spaceId=8891a441-ac43-4fbc-b2a7-ea59d48c0ce6&expirationTimestamp=1724227200000&signature=6roDkRef9G50C4A6PsNTmcQbqvBHty-TrsxfZ2lghIo&downloadName=image.png">
    
    <img src="https://file.notion.so/f/f/8891a441-ac43-4fbc-b2a7-ea59d48c0ce6/517df6cf-a8c8-40ac-9462-e45d85d92518/image.png?table=block&id=d6e7b917-20d1-4856-9faf-9c11671aa41c&spaceId=8891a441-ac43-4fbc-b2a7-ea59d48c0ce6&expirationTimestamp=1724227200000&signature=LO0N9zVTiqC7H2r5LDfUcKzFtOZ9X-fs4sBlwwf86jo&downloadName=image.png">
    

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

<img src="https://file.notion.so/f/f/8891a441-ac43-4fbc-b2a7-ea59d48c0ce6/16ab1231-3e67-47e1-aae8-05824182b5b1/image.png?table=block&id=55de6601-f298-4ad2-9d4e-c5bc8e3dfd1c&spaceId=8891a441-ac43-4fbc-b2a7-ea59d48c0ce6&expirationTimestamp=1724227200000&signature=ogWsF622-Oy2f7mmIezZOk7AOftigmut4D6_KXamAVU&downloadName=image.png">


<img src="https://file.notion.so/f/f/8891a441-ac43-4fbc-b2a7-ea59d48c0ce6/e6b8b34e-ff09-423d-bcf8-61e918a1aea6/image.png?table=block&id=06e55037-434d-4546-b285-95922043fe4f&spaceId=8891a441-ac43-4fbc-b2a7-ea59d48c0ce6&expirationTimestamp=1724227200000&signature=PZYNQ_LBAXs0-LgGu_Fpby29Xo8hwWZFTuBA_p4BsFs&downloadName=image.png">

<img src="https://file.notion.so/f/f/8891a441-ac43-4fbc-b2a7-ea59d48c0ce6/a564fb28-f33f-4570-8b73-d0d4ba92ef78/1eb7bf91-17f7-4c7b-9897-c677d8bc3b90.png?table=block&id=3b6320c3-9f9f-48fe-b12a-48230559183f&spaceId=8891a441-ac43-4fbc-b2a7-ea59d48c0ce6&expirationTimestamp=1724227200000&signature=pCgut7G5Ub8R22ulpxUrkzQv0JVWnYZzL2VF9RTjBwU&downloadName=image.png" width=260>

<img src="https://file.notion.so/f/f/8891a441-ac43-4fbc-b2a7-ea59d48c0ce6/8038c132-4dec-4570-ae52-d64a083a0f7d/image.png?table=block&id=355b9268-a6ed-42ac-9ae3-abcc29b2c33d&spaceId=8891a441-ac43-4fbc-b2a7-ea59d48c0ce6&expirationTimestamp=1724227200000&signature=9zd1P9x2GcLg_TAA0Xi5spXo80B-QV-y_sD10tYv9fw&downloadName=image.png">

<img src="https://file.notion.so/f/f/8891a441-ac43-4fbc-b2a7-ea59d48c0ce6/f3d299b0-1f1d-4019-aa09-dfff4c50db37/image.png?table=block&id=cac763c9-89ab-442c-8b06-a5b82a11252d&spaceId=8891a441-ac43-4fbc-b2a7-ea59d48c0ce6&expirationTimestamp=1724227200000&signature=FMXWoZWQ1hxE8eXA-o77YCb6TneyHfKVrLdge7S0Bjo&downloadName=image.png">

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
<img src="https://file.notion.so/f/f/8891a441-ac43-4fbc-b2a7-ea59d48c0ce6/46dfd268-ac80-434a-9d83-1f8050a94168/image.png?table=block&id=b737a4d1-f6ee-4bc4-b81a-86200323c45e&spaceId=8891a441-ac43-4fbc-b2a7-ea59d48c0ce6&expirationTimestamp=1724227200000&signature=k-FhmT0YzbB2aF5ue0xVA4D3o6Dc2X5jpMJSfMbkd64&downloadName=image.png">

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
<img src="https://file.notion.so/f/f/8891a441-ac43-4fbc-b2a7-ea59d48c0ce6/d466f9e1-ba64-43f0-9a93-a5a7ae53228d/image.png?table=block&id=f14431b7-7f4a-455c-9a0f-0250d268b7a4&spaceId=8891a441-ac43-4fbc-b2a7-ea59d48c0ce6&expirationTimestamp=1724227200000&signature=W6FYKaXuqM7xEu1-fZTmwhOg824NdDxQBYiVcYBW2Bs&downloadName=image.png">

## 2.5D Model Generation

- Modify **base_path**, **layer_num**, **layer_offset** and **displace_strength** in `template.py`.
    
    <img src="https://file.notion.so/f/f/8891a441-ac43-4fbc-b2a7-ea59d48c0ce6/598f5d02-fa41-4e59-be73-41d4466b123a/image.png?table=block&id=4dc577f3-911a-4610-bf09-8e4bb403b75e&spaceId=8891a441-ac43-4fbc-b2a7-ea59d48c0ce6&expirationTimestamp=1724227200000&signature=D8rQHeuO1vMzeiZbjtVDczRNkhctJXYTcr-j7dygI-c&downloadName=image.png">
    

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
<img src="https://file.notion.so/f/f/8891a441-ac43-4fbc-b2a7-ea59d48c0ce6/8ab5771a-d3b9-46b0-b200-cb8ad1481a2e/image.png?table=block&id=88583fd8-47f2-4d1b-af4d-45e0a612ce48&spaceId=8891a441-ac43-4fbc-b2a7-ea59d48c0ce6&expirationTimestamp=1724227200000&signature=Lb5cxtQBgB99_xK4Id_V6h1Gjmv0QROvqBICPoVGD0g&downloadName=image.png">