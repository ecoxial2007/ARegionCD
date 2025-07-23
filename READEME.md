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
+ Phi-3.5V-Med-MIMIC: [HuggingFace]
+ Phi-3.5V-Med-OBScan (LoRA): [HuggingFace]
+ Phi-3.5V-Med-SLAKE (LoRA): [HuggingFace]

Note: Phi-3.5V-Med was trained on PubMedVision. MIMIC, OBScan, and SLAKE were trained on data rewritten and augmented with explanations using GPT-4o.


### 📑Data Preparation
We test our model on:
+ [MIMIC](https://osf.io/89kps/)
+ [SLAKE](https://www.med-vqa.com/slake/)
+ [OBScan](https://github.com/UCSD-AI4H/PathVQA)

Download Data & Annotation:

| Dataset      | GPT-4o Rewrite Answer  & Original Question & Images                                    |
|--------------|----------------------------------------------------------------------------------------|
| MIMIC        | [GoogleDrive](https://drive.google.com/drive/folders/1ZQU4Qsq1CFqVAj4fWIiU9wE6G_LCCdx8) |
| SLAKE        | [GoogleDrive]                    |
| OBScan       | [GoogleDrive]                                                                          |
| PubMedVision | [HuggingFace](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision)        | 

* Remove4anonymous

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
