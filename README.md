# FACE-PLM
**F**urther **A**ssessing **C**urrent **E**ndeavors in **PLM**s

<img width="350" alt="image" src="https://github.com/user-attachments/assets/1493dc74-8eed-49b2-8792-d79dc870d008" />


[![](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Installation

### Clone the repository

    git clone git@github.com:keiserlab/face-plm.git

### Navigate to the repository, create virtual conda environment  

    cd face-plm

    conda create -n face_plm python=3.10 -y

    conda activate face_plm

    pip install -e . 

    pip install pytorch-lightning

# Getting Started

In order to get started with generating embeddings and training models you need to sign into HuggingFace and WandB.

```bash
huggingface-cli login
```
```bash
wandb login
```

Additionally, you need to change the wandb config file at: training_probe/config/wandb_config/base_wandb.yaml
The entity and project need to be updated to properly log to the desired WandB location.

```bash
_target_: face_plm.probes.utils.WandbRunConfig
run_name: base_run+name
entity: temp_entity  # CHANGEME
project: temp_project  # CHANGEME
```

# Generating the PLM Embeddings

### Setup embedding generation env
```bash
bash scripts/setup_embed_env.sh
```

### Generating final layer embeddings for all PLMs
With ESM (requires ESMC/3 access)
```bash
bash scripts/get_all_plm_embedding.sh
```
Without ESM
```bash
bash scripts/get_all_plm_embedding_no_esm.sh
```

### Generating all layer embeddings for all Ankh-base
```bash
bash scripts/get_all_layer_ankh_embedding.sh
```

# Training Probes
### Training a single model (single probe type, single aggeregation type, final layer)
```bash
bash scripts/train_single_model.sh CONFIG_NAME
```
Example config: esmc_600m-agg_mlp

### Training multiple models for cross-validation (single probe type, single aggregation, final layer)
```bash
bash scripts/train_cross_val_model.sh CONFIG_NAME
```
Example config: esmc_600m-agg_mlp

### Training models on multiple layers (single probe type, single aggregation, all layers)
```bash
bash scripts/train_cross_val_model_ankh_multilayer.sh CONFIG_NAME
```
Example config: ankh_base_layer_specific_0-12


# Masked Language Model Fine-tuning
### EC 2.7.* Dataset Fine-tuning
```bash
bash scripts/finetune_mlm.sh ankh_large_ft_ec27
```
### ADK Dataset Fine-tuning
```bash
bash scripts/finetune_mlm.sh ankh_base_ft_kcat
```

# Direct Regression Fine-tuning
```bash
bash scripts/train_cross_val_direct_finetune.sh CONFIG_NAME
```
Example config: ankh_base_ft_kcat

# No Torch Linear and Non-linear Probing
```bash
bash no_torch_probing.sh OUTPUT_DIR
```
example output_dir: ./probe_outputs/

