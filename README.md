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



## Installation
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

## **Available  Volumetric Segmentation models:**
1. UNet : `unet`
2. UNETR : `unetr`
3. Swin-UNETR : `swin_unetr`
4. SegResNet : `seg_resnet`
5. UMamba-B : `umamba_bot`
6. UMamba-E : `umamba_enc`


## Datasets available
1. BTCV: `btcv` 
2. Hecktor: `hecktor`
3. ACDC: `acdc`
4. Abdomen-CT: `abdomen`


## A. Training of Volumetric Segmentation Models


## B. Robustness against Adversarial attacks

### 1. White box Attacks

For crafting adversarial examples using Fast Gradient Sign Method (FGSM) at perturbation budget of 8/255, run:
```python
python generate_adv_images.py --data_dir <path to dataset> --attack_name fgsm  --source_model_name <model_name> --epsilon 8  
```
For crafting adversarial examples using Projected Gradient Descent (PGD) at perturbation budget of 8/255 with number of attacks steps equal to 20, run:
```python
python generate_adv_images.py --data_dir <path to dataset> --attack_name pgd  --source_model_name <model_name> --epsilon 8 --attack_steps 20 
```
Other available attacks: `fgsm, gn, pgd, vafa, cospgd, vmifgsm`


The results will be saved in  `AdvExamples_results` folder with the following structure: `AdvExamples_results/pgd_eps_{eps}_steps_{step}/{source_model_name}/accuracy.txt`


### 2. White box Frequency Attacks

#### Low-Pass Frequency Attack
For crafting adversarial examples using Projected Gradient Descent (PGD) at perturbation budget of 8/255 with number of attacks steps equal to 20, run:
```python
cd  classification/
python generate_adv_images.py --data_dir <path to dataset> --attack_name pgd  --source_model_name <model_name> --epsilon 8 --attack_steps 20 --filter True --filter_preserve low 
```
#### High-Pass Frequency Attack
For crafting adversarial examples using Projected Gradient Descent (PGD) at perturbation budget of 8/255 with number of attacks steps equal to 20, run:
```python
cd  classification/
python generate_adv_images.py --data_dir <path to dataset> --attack_name pgd  --source_model_name <model_name> --epsilon 8 --attack_steps 20 --filter True --filter_preserve high
```
The results will be saved in  `AdvExamples_freq_results` folder.

Run the below script to evaluate the robustness across different models against low and high frequency attacks at various perturbation budgets:
```python
cd  classification/
bash scripts/get_adv_freq_results.sh <DATA_PATH> <ATTACK_NAME> <BATCH_SIZE>
```


### 3. Transfer-based Black box Attacks

For evaluating transferability of adversarial examples, first save the generated adversarial examples by running:

```python
cd  classification/
python generate_adv_images.py --data_dir <path to dataset> --attack_name fgsm  --source_model_name <model_name> --epsilon 8 --save_results_only False  
```

The adversarial examples will be saved in  `AdvExamples` folder with the following structure: `AdvExamples/{attack_name}_eps_{eps}_steps_{step}/{source_model_name}/images_labels.pt`

Then run the below script to evaluate transferability of the generated adversarial examples across different models:

```python
cd  classification/
python inference.py --dataset imagenet_adv --data_dir <path to adversarial dataset> --batch_size <> --source_model_name <model name>
```
`--source_model_name`: name of the model on which the adversarial examples will be evaluated



Furthermore, bash scripts are provided to evaluate transferability of adversarial examples across different models:
```python
cd  classification/
# Generate adversarial examples
bash scripts/gen_adv_examples.sh <DATA_PATH> <EPSILON> <ATTACK_NAME> <BATCH_SIZE>
# Evaluate transferability of adversarial examples saved in AdvExamples folder
bash scripts/evaluate_transferability.sh <DATA_PATH> <EPSILON> <ATTACK_NAME> <BATCH_SIZE>
```






## Citation
If you use our work, please consider citing:
```bibtex 

```

<hr />

## Contact
Should you have any question, please create an issue on this repository or contact at hashmat.malik@mbzuai.ac.ae

<hr />

## References
Our code is based on [VAFA](https://github.com/asif-hanif/vafa?tab=readme-ov-file), [On the Adversarial Robustness of Visual Transformer](https://github.com/RulinShao/on-the-adversarial-robustness-of-visual-transformer) and [monai](https://github.com/Project-MONAI/MONAI) libray. We thank them for open-sourcing their codebase.



