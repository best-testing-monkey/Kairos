# Kronos Fine-tuning Training with Custom Dataset

Supports fine-tuning training with custom CSV data using configuration files

## Quick Start

### 1. Configuration Setup

First edit the `config.yaml` file to set the correct paths and parameters:

```yaml
# Data configuration
data:
  data_path: "/path/to/your/data.csv"
  lookback_window: 512
  predict_window: 48
  # ... other parameters

# Model path configuration
model_paths:
  pretrained_tokenizer: "/path/to/pretrained/tokenizer"
  pretrained_predictor: "/path/to/pretrained/predictor"
  base_save_path: "/path/to/save/models"
  # ... other paths
```

### 2. Run Training

Using train_sequential

```bash
# Complete training
python train_sequential.py --config configs/config_ali09988_candle-5min.yaml

# Skip existing models
python train_sequential.py --config configs/config_ali09988_candle-5min.yaml --skip-existing

# Only train tokenizer
python train_sequential.py --config configs/config_ali09988_candle-5min.yaml --skip-basemodel

# Only train basemodel
python train_sequential.py --config configs/config_ali09988_candle-5min.yaml --skip-tokenizer
```

Run each stage separately

```bash
# Only train tokenizer
python finetune_tokenizer.py --config configs/config_ali09988_candle-5min.yaml 

# Only train basemodel (requires fine-tuned tokenizer first)
python finetune_base_model.py --config configs/config_ali09988_candle-5min.yaml 
```

DDP Training
```bash
# Choose communication protocol yourself, nccl can be replaced with gloo
DIST_BACKEND=nccl \
torchrun --standalone --nproc_per_node=8 train_sequential.py --config configs/config_ali09988_candle-5min.yaml
```

## Configuration Description

### Main Configuration Items

- **data**: Data-related configuration
  - `data_path`: CSV data file path
  - `lookback_window`: Lookback window size
  - `predict_window`: Prediction window size
  - `train_ratio/val_ratio/test_ratio`: Dataset split ratios

- **training**: Training-related configuration
  - `epochs`: Number of training epochs
  - `batch_size`: Batch size
  - `tokenizer_learning_rate`: Tokenizer learning rate
  - `predictor_learning_rate`: Predictor learning rate

- **model_paths**: Model path configuration
  - `pretrained_tokenizer`: Pre-trained tokenizer path
  - `pretrained_predictor`: Pre-trained predictor path
  - `base_save_path`: Model save root directory
  - `finetuned_tokenizer`: Fine-tuned tokenizer path (for basemodel training)

- **experiment**: Experiment control
  - `train_tokenizer`: Whether to train tokenizer
  - `train_basemodel`: Whether to train basemodel
  - `skip_existing`: Whether to skip existing models

## Training Process

1. **Tokenizer Fine-tuning Stage**
   - Load pre-trained tokenizer
   - Fine-tune on custom data
   - Save fine-tuned tokenizer to `{base_save_path}/tokenizer/best_model/`

2. **Basemodel Fine-tuning Stage**
   - Load fine-tuned tokenizer and pre-trained predictor
   - Fine-tune on custom data
   - Save fine-tuned basemodel to `{base_save_path}/basemodel/best_model/`

**Data Format**: Ensure CSV file contains the following columns: `timestamps`, `open`, `high`, `low`, `close`, `volume`, `amount`
