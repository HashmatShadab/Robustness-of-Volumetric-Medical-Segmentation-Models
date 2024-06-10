The complete code with pretrained weights will be released soon. Stay Tuned!

# **On Evaluating Adversarial Robustness of Volumetric Medical Segmentation Models**

[Hashmat Shadab Malik](https://github.com/HashmatShadab), 
[ Numan Saeed](https://github.com/numanai),
[ Asif Hanif](https://github.com/asif-hanif),
[Muzammal Naseer](https://scholar.google.ch/citations?user=tM9xKA8AAAAJ&hl=en),
[Mohammad Yaqub](https://scholar.google.co.uk/citations?user=9dfn5GkAAAAJ&hl=en),
[Salman Khan](https://salman-h-khan.github.io),
and [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()

[//]: # ([![Video]&#40;https://img.shields.io/badge/Video-Presentation-F9D371&#41;]&#40;https://drive.google.com/file/d/1ECkp_lbMj5Pz7RX_GgEvWWDHf5PUXlFd/view?usp=share_link&#41;)

[//]: # ([![slides]&#40;https://img.shields.io/badge/Poster-PDF-87CEEB&#41;]&#40;https://drive.google.com/file/d/1neYZca0KRIBCu5R6P78aQMYJa7R2aTFs/view?usp=share_link&#41;)
[//]: # ([![slides]&#40;https://img.shields.io/badge/Presentation-Slides-B762C1&#41;]&#40;https://drive.google.com/file/d/1wRgCs2uBO0p10FC75BKDUEdggz_GO9Oq/view?usp=share_link&#41;)

<hr />



> **Abstract:** Volumetric medical segmentation models have achieved significant success on organ and tumor-based segmentation tasks in recent years.
> However, their vulnerability to adversarial attacks remains largely unexplored, raising serious concerns regarding the real-world deployment of tools
> employing such models in the healthcare sector. This underscores the importance of investigating the robustness of existing models. In this context,
> our work aims to empirically examine the adversarial robustness across current volumetric segmentation architectures, encompassing Convolutional, 
> Transformer, and Mamba-based models. We extend this investigation across four volumetric segmentation datasets, evaluating robustness under both
> white box and black box adversarial attacks. Overall, we observe that while both pixel and frequency-based attacks perform reasonably well under 
> white box setting, the latter performs significantly better under transfer-based black box attacks. Across our experiments, we observe transformer-based 
> models show higher robustness than convolution-based models with Mamba-based models being the most vulnerable. Additionally, we show that large-scale 
> training of volumetric segmentation models improves the model's robustness against adversarial attacks.  


## Contents

1) [Installation](#Installation)
2) [Available Models and Datasets](#Available-Models-and-Datasets)
3) [Training of Volumetric Segmentation Models](#Training-of-Volumetric-Segmentation-Models)
4) [Robustness against White-Box Attacks](#Robustness-against-White-Box-Attacks)
5) [Robustness against Transfer-Based Black-Box Attacks](#Robustness-against-Transfer-Based-Black-Box-Attacks)
6) [BibTeX](#bibtex)

<hr>
<hr>



<a name="Installation"/>

## 💿 Installation

```python

conda create -n med3d python=3.10
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install packaging
pip install causal-conv1d==1.1.1
pip install mamba-ssm==1.1.4
pip install torchinfo timm numba
pip install monai==1.3.0

# Others
pip install -r req.txt
```

<a name="Available-Models-and-Datasets"/>

## 🏁 Available Models and Datasets

### Models available:
1. UNet : `unet`
2. UNETR : `unetr`
3. Swin-UNETR : `swin_unetr`
4. SegResNet : `seg_resnet`
5. UMamba-B : `umamba_bot`
6. UMamba-E : `umamba_enc`


### Datasets available:
1. BTCV: `btcv` 
2. Hecktor: `hecktor`
3. ACDC: `acdc`
4. Abdomen-CT: `abdomen`

Copy the json files from `json_files` folder to the respective dataset folder.


<a name="Training-of-Volumetric-Segmentation-Models"/>

## 🚀 Training of Volumetric Segmentation Models

```python
# Training  on BTCV dataset
 python training.py  --model_name <MODEL_NAME> --in_channels 1 --out_channel 14  --dataset btcv --data_dir=<DATA_PATH> --json_list dataset_synapse_18_12.json 
--batch_size=3 --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 --val_every 15 --save_model_dir="./Results"

# Training  on Hecktor dataset
 python training.py  --model_name <MODEL_NAME> --in_channels 1 --out_channel 3  --dataset hecktor --data_dir=<DATA_PATH> --json_list dataset_hecktor.json 
--batch_size=3 --max_epochs 500 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 --val_every 15 --save_model_dir="./Results"

# Training  on ACDC dataset
 python training.py  --model_name <MODEL_NAME> --in_channels 1 --out_channel 4  --dataset acdc --data_dir=<DATA_PATH> --json_list dataset_acdc_140_20_.json 
--batch_size=3 --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 --val_every 15 --save_model_dir="./Results"

# Training  on Abdomen-CT dataset
 python training.py  --model_name <MODEL_NAME> --in_channels 1 --out_channel 14  --dataset abdomen --data_dir=<DATA_PATH> --json_list dataset_abdomen.json 
--batch_size=3 --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 --val_every 15 --save_model_dir="./Results"
```
Follwing arguments can be passed for `--model_name`: `unet, unetr, swin_unetr, seg_resnet, umamba_bot, umamba_enc`


To run training across all models and datasets, run the following scripts:
```python
# Training on BTCV dataset for all models
bash scripts/btcv/training.sh

# Training on Hecktor dataset for all models
bash scripts/hecktor/training.sh

# Training on ACDC dataset for all models
bash scripts/acdc/training.sh

# Training on Abdomen-CT dataset for all models
bash scripts/abdomen/training.sh
```
The logs and trained models will be saved in the `Results` folder with the following structure: `Results/{model_name}/data_{dataset_name}/natural/`


<a name="Robustness-against-White-Box-Attacks"/>

## 🛡️ Robustness against White-Box Attacks

### 1. White box Attacks

```python
# Pixel-based PGD attack on Volumetric Segmentation models
python wb_attack.py  --model_name <MODEL_NAME>   --in_channels 1 --out_channel <NUM_CLASSES>  --checkpoint_path <MODEL_CKPT_PATH> --dataset <DATASET_NAME> 
--data_dir=<DATA_PATH> --json_list <DATA_JSON_FILE>  --attack_name pgd --eps 8  --steps 20

# Pixel-based CosPGD attack on Volumetric Segmentation models
python wb_attack.py  --model_name <MODEL_NAME>   --in_channels 1 --out_channel <NUM_CLASSES>  --checkpoint_path <MODEL_CKPT_PATH> --dataset <DATASET_NAME> 
--data_dir=<DATA_PATH> --json_list <DATA_JSON_FILE>  --attack_name cospgd --eps 8  --steps 20

# Pixel-based FGSM attack on Volumetric Segmentation models
python wb_attack.py  --model_name <MODEL_NAME>   --in_channels 1 --out_channel <NUM_CLASSES>  --checkpoint_path <MODEL_CKPT_PATH> --dataset <DATASET_NAME> 
--data_dir=<DATA_PATH> --json_list <DATA_JSON_FILE>  --attack_name fgsm --eps 8  

# Pixel-based GN attack on Volumetric Segmentation models
python wb_attack.py  --model_name <MODEL_NAME>   --in_channels 1 --out_channel <NUM_CLASSES>  --checkpoint_path <MODEL_CKPT_PATH> --dataset <DATASET_NAME> 
--data_dir=<DATA_PATH> --json_list <DATA_JSON_FILE>  --attack_name fgsm --std 8  

# Frequency-based VAFA attack on Volumetric Segmentation models
python wb_attack.py  --model_name <MODEL_NAME>   --in_channels 1 --out_channel <NUM_CLASSES>  --checkpoint_path <MODEL_CKPT_PATH> --dataset <DATASET_NAME> 
--data_dir=<DATA_PATH> --json_list <DATA_JSON_FILE>  --attack_name vafa-3d - --q_max 30 --steps 20 --block_size 32 32 32 --use_ssim_loss True
```

Available attacks: `fgsm, gn, pgd, vafa, cospgd, vmifgsm`

`--eps`: Perturbation budget for Pixel-based adversarial attacks
    
`--std`: Perturbation budget for Gaussian Noise attack
    
`--q_max`: Maximum quantization level for VAFA attack
    
`--block_size`: Block size for VAFA attack
    
`--use_ssim_loss`: Use SSIM loss for VAFA attack
    
`--steps`: Number of attack steps for PGD-based attacks

To run the above attacks across all models and datasets, run the following scripts:
```python
# Pixel and Frequency-based attacks on Volumetric Segmentation models trained on BTCV dataset
bash scripts/btcv/attacks.sh

# Pixel and Frequency-based attacks on Volumetric Segmentation models trained on Hecktor dataset
bash scripts/hecktor/attacks.sh

# Pixel and Frequency-based attacks on Volumetric Segmentation models trained on ACDC dataset
bash scripts/acdc/attacks.sh

# Pixel and Frequency-based attacks on Volumetric Segmentation models trained on Abdomen-CT dataset
bash scripts/abdomen/attacks.sh
```

In the above scripts replace the following arguments:

`<DATA_DIR>`: Path to the dataset

`<model_names>`: name of the models and their corresponding checkpoints in `<ckpt_paths>`



The generated adversarial images and logs will be saved in the same folder as from where the model checkpoint was loaded.

### 2. White box Frequency Attacks

After generating adversarial examples using the above scripts, frequency analysis  can be performed on them using Low-Pass and High-Pass filters. For evaluating the robustness of volumetric segmentation models against
low and high frequency attacks, the following scripts can be used:

```python
# Frequency Analysis on Volumetric Segmentation models trained on BTCV dataset
bash scripts/btcv/attack_freq.sh

# Frequency Analysis on Volumetric Segmentation models trained on Hecktor dataset
bash scripts/hecktor/attack_freq.sh

# Frequency Analysis on Volumetric Segmentation models trained on ACDC dataset
bash scripts/acdc/attack_freq.sh

# Frequency Analysis on Volumetric Segmentation models trained on Abdomen-CT dataset
bash scripts/abdomen/attack_freq.sh
```
In the above scripts replace the following arguments:

`<DATA_DIR>`: Path to the dataset

`<model_names>`: name of the models and their corresponding checkpoints in `<ckpt_paths>`

The evaluation logs will be saved in the same folder as from where the adversarial examples were loaded.


<a name="Robustness-against-Transfer-Based-Black-Box-Attacks"/>

## 🛡️ Robustness against Transfer-Based Black-Box Attacks

After generating adversarial examples using the above scripts, the transferability of adversarial examples can be reported by evaluating them on any other model trained on the same dataset.
To evaluate any model on the adversarial examples, run the following script:

```python
# Transferability on BTCV adversarial examples
python inference_on_adv_images.py --model_name <MODEL_NAME>  --in_channels 1 --out_channel 14  --checkpoint_path <BTCV_MODEL_CKPT_PATH> --dataset btcv 
--data_dir=<ORIG_BTCV_DATA_PATH> --json_list dataset_synapse_18_12.json  --adv_imgs_dir <PATH_TO_BTCV_ADVERSARIAL_IMAGES>

# Transferability on Hecktor adversarial examples
python inference_on_adv_images.py --model_name <MODEL_NAME>  --in_channels 1 --out_channel 3  --checkpoint_path <HECKTOR_MODEL_CKPT_PATH> --dataset hecktor
--data_dir=<ORIG_HECKTOR_DATA_PATH> --json_list dataset_hecktor.json  --adv_imgs_dir <PATH_TO_HECKTOR_ADVERSARIAL_IMAGES>

# Transferability on ACDC adversarial examples
python inference_on_adv_images.py --model_name <MODEL_NAME>  --in_channels 1 --out_channel 4  --checkpoint_path <ACDC_MODEL_CKPT_PATH> --dataset acdc
--data_dir=<ORIG_ACDC_DATA_PATH> --json_list dataset_acdc_140_20_.json  --adv_imgs_dir <PATH_TO_ACDC_ADVERSARIAL_IMAGES>

# Transferability on Abdomen-CT adversarial examples
python inference_on_adv_images.py --model_name <MODEL_NAME>  --in_channels 1 --out_channel 14  --checkpoint_path <ABDOMEN_MODEL_CKPT_PATH> --dataset abdomen
--data_dir=<ORIG_ABDOMEN_DATA_PATH> --json_list dataset_abdomen.json  --adv_imgs_dir <PATH_TO_ABDOMEN_ADVERSARIAL_IMAGES>
```



Furthermore, bash scripts are provided to evaluate transferability of adversarial examples across different models(given the adversarial examples are generated first across all models and datasets):
```python
# Transferability of BTCV adversarial examples across all models
bash scripts/btcv/transferability.sh

# Transferability of Hecktor adversarial examples across all models
bash scripts/hecktor/transferability.sh

# Transferability of ACDC adversarial examples across all models
bash scripts/acdc/transferability.sh

# Transferability of Abdomen-CT adversarial examples across all models
bash scripts/abdomen/transferability.sh
```





<a name="bibtex"/>

## 📚 BibTeX

```bibtex

```
<a name="license"/>
<hr />

## Contact
Should you have any question, please create an issue on this repository or contact at hashmat.malik@mbzuai.ac.ae

<hr />

## References
Our code is based on [VAFA](https://github.com/asif-hanif/vafa?tab=readme-ov-file), [On the Adversarial Robustness of Visual Transformer](https://github.com/RulinShao/on-the-adversarial-robustness-of-visual-transformer) and [monai](https://github.com/Project-MONAI/MONAI) libray. We thank them for open-sourcing their codebase.



