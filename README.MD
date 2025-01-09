all the train and evlation used by Longlora and MS-SWIFT
# CO2RRChat 
An intelligent chat system for CO2 reduction reaction (CO2RR) analysis and prediction.

## Overview
CO2RRChat is an intelligent dialogue system based on multiple large language models, focusing on CO2 Reduction Reaction (CO2RR) analysis and prediction. This project utilizes techniques from [LongLoRA](https://github.com/dvlab-research/LongLoRA) and [MS-SWIFT](https://github.com/modelscope/ms-swift) for model optimization.

## Model Architecture
CO2RRChat employs a Mixture of Experts (MoE) architecture, integrating multiple specialized models to achieve three core functionalities:
### CO2RR Product Prediction Model
Based on optimized BERT architecture
Specifically trained for CO2RR reaction product prediction
High-accuracy product prediction capabilities
### Computational Plan Generation Model
Customized Large Language Model
Generates detailed computational pathways and plans
Supports various computational conditions and scenarios
### Adsorption Energy Code Generation Model
Specialized code generation language model
Automatically generates code for adsorption energy calculations
Supports multiple computational software packages and methods

## System Requirements
GPU: NVIDIA A100 (80GB)
RAM: 128GB+ recommended
Storage: 100GB+ available space

## Installation
Note: We did not choose to merge all models to generate a mistral 8*7b model because BERT performs best in product prediction. For large language models with different frameworks, merging them into a single model might degrade their performance. Therefore, we chose the following method for inference.

### Clone the Repository
```bash
git clone https://github.com/waleyW/CO2RRChat
cd CO2RRChat
```

### Download All the Models
```bash
git clone https://huggingface.co/waleyWang/bert_rounter
git clone https://huggingface.co/waleyWang/product_predict
git clone https://huggingface.co/waleyWang/computational_plan
git clone https://huggingface.co/waleyWang/code_generate
```

### Install Dependencies
Before these steps below, please install dependencies by [LongLoRA](https://github.com/dvlab-research/LongLoRA) and [MS-SWIFT](https://github.com/modelscope/ms-swift)

```bash
pip install -r requirements.txt
```

## Inference

### MoE

Note: please replace the path of all the models
```bash
python moe_inference_gradio.py
```
