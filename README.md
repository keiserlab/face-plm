# FACE-PLM
**F**urther **A**ssessing **C**urrent **E**ndeavors in **PLM**s

<img width="350" alt="image" src="https://github.com/user-attachments/assets/1493dc74-8eed-49b2-8792-d79dc870d008" />


[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Installation

### Install uv

Curl:

    curl -LsSf https://astral.sh/uv/install.sh | sh

Homebrew:

    brew install uv

### Clone the repository

    git clone git@github.com:keiserlab/face-plm.git

### Navigate to the repository, create virtual environment

    cd face-plm

    uv sync

    uv pip install . --python .venv

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
### Generating final layer embeddings for all PLMs
```bash

```
### Generating all layer embeddings for all Ankh-base
```bash

```

# Training Probes
### Training a single model (single probe type, single aggeregation type, final layer)
```bash

```
### Training multiple models for cross-validation (single probe type, single aggregation, final layer)
```bash

```
### Training models on multiple layers (single probe type, single aggregation, all layers)
```bash

```
### Training models for all aggregation (single probe type, all aggregations)
```bash

```

# Masked Language Model Fine-tuning
### EC 2.7.* Dataset Fine-tuning
```bash

```
### ADK Dataset Fine-tuning
```bash

```

# Direct Regression Fine-tuning
```bash

```

# No Torch Linear and Non-linear Probing
```bash

```

