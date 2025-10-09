# Kronos Finetuning on Your Custom csv Dataset

Supports fine-tuning training with custom CSV data using configuration files

## 1. Quick Start

### Configuration Setup

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

### Run Training

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
## 2. Training Results

![HK_ali_09988_kline_5min_all_historical_20250919_073929](examples/HK_ali_09988_kline_5min_all_historical_20250919_073929.png)

![HK_ali_09988_kline_5min_all_historical_20250919_073944](examples/HK_ali_09988_kline_5min_all_historical_20250919_073944.png)

![HK_ali_09988_kline_5min_all_historical_20250919_074012](examples/HK_ali_09988_kline_5min_all_historical_20250919_074012.png)

![HK_ali_09988_kline_5min_all_historical_20250919_074042](examples/HK_ali_09988_kline_5min_all_historical_20250919_074042.png)

![HK_ali_09988_kline_5min_all_historical_20250919_074251](examples/HK_ali_09988_kline_5min_all_historical_20250919_074251.png)

**Data Format**: Ensure CSV file contains the following columns: `timestamps`, `open`, `high`, `low`, `close`, `volume`, `amount`
