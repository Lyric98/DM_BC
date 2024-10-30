# A Diffusion-Wavelet Approach for  Enhancing Low-quality Mammography Images
This work presents a novel Diffusion-Wavelet (DiWa) approach for Single-Image Super-Resolution (SISR). It leverages the strengths of Denoising Diffusion Probabilistic Models (DDPMs) and Discrete Wavelet Transformation (DWT). By enabling DDPMs to operate in the DWT domain, our DDPM models effectively hallucinate high-frequency information for super-resolved images on the wavelet spectrum, resulting in high-quality and detailed reconstructions in image space. 

## reference
Original codebase: https://github.com/brian-moser/diwa \
**Waving Goodbye to Low-Res: A Diffusion-Wavelet Approach for Image Super-Resolution** ([arXiv paper](https://arxiv.org/abs/2304.01994))\
It complements the inofficial implementation of **SR3** ([GitHub](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)).


## Usage

### Environment

```python
pip install -r requirement.txt
```


### Data Preparation

Pre-process the dataset (select the save the full images from the original dataset):
```python
python CBIS_dataset.py
```

Central crop the dataset and put them into hr, lr subfolders:

```python
# here is the example of crop the dataset in `dataset/CBIS_full/full_image_RGB` and reshape them to hr 512x512 and lr 64x64, and save them to a new folder called `CBIS_test_64_512`
python data/prepare_data.py --path dataset/CBIS_full/full_image_RGB --out CBIS_test --size 64,512           
```


<!-- ```python
# Resize to get 16×16 LR_IMGS and 128×128 HR_IMGS, then prepare 128×128 Fake SR_IMGS by bicubic interpolation
python data/prepare_data.py  --path [dataset root]  --out [output root] --size 16,128 -l
``` -->

then you need to **change the datasets config** to your data path and image resolution, the example config for CBIS dataset is at `config/sr_wave_64_512CBIS.json`: 

<!-- ```json
"datasets": {
    "train": {
        "dataroot": "dataset/ffhq_16_128", // [output root] in prepare.py script
        "l_resolution": 16, // low resolution need to super_resolution
        "r_resolution": 128, // high resolution
        "datatype": "lmdb", //lmdb or img, path of img files
    },
    "val": {
        "dataroot": "dataset/celebahq_16_128", // [output root] in prepare.py script
    }
},
``` -->


### Training/Resume Training

simple version
```python
# Use sr.py and sample.py to train the super resolution task and unconditional generation task, respectively.
# Edit json files to adjust network structure and hyperparameters
python sr.py -p train -c config/sr_wave_64_512CBIS.json
```
turn on the wandb logging and eval logging:
```python
python sr.py -p train -c config/sr_wave_64_512CBIS.json -enable_wandb -log_wandb_ckpt -log_eval       
```

### Continue Training

```python
# Download the pretrained model and edit [sr|sample]_[ddpm|sr3]_[resolution option].json about "resume_state":
"resume_state": [your pretrained model's path]
```

<!-- #### Configurations for Training


| Tasks                             | Config File                                              | 
|-----------------------------------|----------------------------------------------------------|
| 16×16 -> 128×128 on FFHQ-CelebaHQ | [config/sr_wave_16_128.json](config/sr_wave_16_128.json) |  
| 64×64 -> 512×512 on FFHQ-CelebaHQ | [config/sr_wave_64_512.json](config/sr_wave_64_512.json) |   
| 48×48 -> 192×192 on DIV2K         | [config/sr_wave_48_192.json](config/sr_wave_48_192.json) |
| Ablation - baseline               | [config/sr_wave_48_192_abl_baseline.json](config/sr_wave_48_192_abl_baseline.json) |
| Ablation - Init. Pred. only       | [config/sr_wave_48_192_abl_pred_only.json](config/sr_wave_48_192_abl_pred_only.json) |
| Ablation - DWT only               | [config/sr_wave_48_192_abl_wave_only.json](config/sr_wave_48_192_abl_wave_only.json) |
| Ablation - DiWa                   | [config/sr_wave_48_192_abl_wave+pred.json](config/sr_wave_48_192_abl_wave+pred.json) | -->

### Test/Evaluation

```python
# Edit json to add pretrain model path and run the evaluation 
python sr.py -p val -c config/sr_sr3.json

# Quantitative evaluation alone using SSIM/PSNR/LPIPS metrics on given result root
python eval.py -p [result root]
```

### Inference Alone

Set the image path, then run the script:

```python
# run the script
python infer.py -c [config file]
```
