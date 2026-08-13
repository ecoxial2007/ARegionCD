<div align="center">
  
# 
</div>

## 💡Overview
Anatomical Region-Guided Contrastive Decoding: A Plug-and-Play Strategy for Mitigating Hallucinations in Medical VLMs
## 🔨Setup
### 🔨Installation
```
conda create -n aregion python=3.10 -y
conda activate aregion
pip install -r requirements.txt
```

### 🔨Pre-trained weights

#### Baseline Model:
Download these weights locally.
+ Phi-3.5V: [HuggingFace](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)
+ Phi-3.5V-Med (LoRA): [HuggingFace](https://huggingface.co/ecoxial2007/Phi-3.5V-Med)
+ Phi-3.5V-Med-MIMIC: [HuggingFace](https://huggingface.co/ecoxial2007/Phi-3.5-Vision-Instruct-MIMIC)
+ Phi-3.5V-Med-SLAKE+OBScan (LoRA): [HuggingFace](https://huggingface.co/ecoxial2007/phi35_slake_obscan_with_pubmed)

Note: Phi-3.5V-Med was trained on PubMedVision. MIMIC, OBScan, and SLAKE were trained on data rewritten and augmented with explanations using GPT-4o.


### 📑Data Preparation
We test our model on:
+ [MIMIC-Annotations](https://physionet.org/content/mimic-ext-mimic-cxr-vqa/)
+ [SLAKE-Annotations+Images+Masks](https://www.med-vqa.com/slake/)
+ OBScan: Images cannot be provided due to ethical restrictions.
+ [Sampled Test Annotations](https://drive.google.com/file/d/1HbIYO0-5E-yJuumYmxDCJGj4CsBadjwr/view?usp=drive_link)

Download Data & Annotation:

| Dataset      | GPT-4o Rewrite Answer  & Original Question & Images                                    |
|--------------|------------------------------------------------------------------------------------------|
| MIMIC        | [GoogleDrive](https://drive.google.com/file/d/1kLb7j0Vx8hSISmatbiNZMkjLBhWMfih4/view) |
| SLAKE        | [GoogleDrive](https://drive.google.com/file/d/1tQrrVtSt9XSB-dHLiHAOQhHAP0wWbYk9/view)|
| PubMedVision | [HuggingFace](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision)        | 


### 📝 Demo

#### Preparation

1. Download Baseline Model and dataset
2. Use Custom Processor `checkpoints/Phi-3.5-vision-instruct/processing_phi3_v.py`
3. Run
```
torchrun --master_port 23333 evaluate_ARegion.py \
    --bf16 \
    --use_flash_attention \                                     # Faster
    --model_name_or_path 'checkpoints/Phi-3.5-vision-instruct' \  # Download from HF
    --use_lora \                                                # If use lora
    --lora_path ./path_to_lora/checkpoint-666 \                 # Add path here
    --input_path ./sample_data/test.json \                      # Question & Image path & Bbox(option) & Mask value(option)
    --save_path ./results/output.json \                         # Save Response
    --img_root ./sample_data/Name_of_Dataset \                  # Dataset root (Image & Mask)
    --max_new_tokens 256 \
    --num_crops 16                                              # Default setting
```

---

**num_crops=16** means the 336x336 image is upscaled to 1344x1344 for higher accuracy. If you're low on GPU memory, you can reduce this value.

### 📝 Citation

```
@inproceedings{liang2026anatomical,
  title={Anatomical Region-Guided Contrastive Decoding: A Plug-and-Play Strategy for Mitigating Hallucinations in Medical VLMs},
  author={Liang, Xiao and Liu, Chenxi and Ma, Zhi and Wang, Di and Jing, Bin and Wang, Quan and Shi, Yuanyuan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={9},
  pages={6871--6879},
  year={2026}
}
```

