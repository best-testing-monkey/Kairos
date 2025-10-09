# 自定义数据集的Kronos微调训练

支持使用配置文件进行自定义csv数据的微调训练

## 快速开始

### 1. 配置设置

首先编辑 `config.yaml` 文件，设置正确的路径和参数：

```yaml
# 数据配置
data:
  data_path: "/path/to/your/data.csv"  
  lookback_window: 512
  predict_window: 48
  # ... 其他参数

# 模型路径配置
model_paths:
  pretrained_tokenizer: "/path/to/pretrained/tokenizer"
  pretrained_predictor: "/path/to/pretrained/predictor"
  base_save_path: "/path/to/save/models"
  # ... 其他路径
```

### 2. 运行训练


使用train_sequential

```bash
# 完整训练
python train_sequential.py --config configs/config_ali09988_candle-5min.yaml

# 跳过已存在的模型
python train_sequential.py --config configs/config_ali09988_candle-5min.yaml --skip-existing

# 只训练tokenizer
python train_sequential.py --config configs/config_ali09988_candle-5min.yaml --skip-basemodel

# 只训练basemodel
python train_sequential.py --config configs/config_ali09988_candle-5min.yaml --skip-tokenizer
```

单独运行各个阶段

```bash
# 只训练tokenizer
python finetune_tokenizer.py --config configs/config_ali09988_candle-5min.yaml 

# 只训练basemodel（需要先有微调后的tokenizer）
python finetune_base_model.py --config configs/config_ali09988_candle-5min.yaml 
```

DDP训练
```bash
# 通信协议自行选择，nccl可替换gloo
DIST_BACKEND=nccl \
torchrun --standalone --nproc_per_node=8 train_sequential.py --config configs/config_ali09988_candle-5min.yaml
```

## 配置说明

### 主要配置项

- **data**: 数据相关配置
  - `data_path`: CSV数据文件路径
  - `lookback_window`: 回望窗口大小
  - `predict_window`: 预测窗口大小
  - `train_ratio/val_ratio/test_ratio`: 数据集分割比例

- **training**: 训练相关配置
  - `epochs`: 训练轮数
  - `batch_size`: 批次大小
  - `tokenizer_learning_rate`: Tokenizer学习率
  - `predictor_learning_rate`: Predictor学习率

- **model_paths**: 模型路径配置
  - `pretrained_tokenizer`: 预训练tokenizer路径
  - `pretrained_predictor`: 预训练predictor路径
  - `base_save_path`: 模型保存根目录
  - `finetuned_tokenizer`: 微调后tokenizer路径（用于basemodel训练）

- **experiment**: 实验控制
  - `train_tokenizer`: 是否训练tokenizer
  - `train_basemodel`: 是否训练basemodel
  - `skip_existing`: 是否跳过已存在的模型

## 训练流程

1. **Tokenizer微调阶段**
   - 加载预训练tokenizer
   - 在自定义数据上微调
   - 保存微调后的tokenizer到 `{base_save_path}/tokenizer/best_model/`

2. **Basemodel微调阶段**
   - 加载微调后的tokenizer和预训练predictor
   - 在自定义数据上微调
   - 保存微调后的basemodel到 `{base_save_path}/basemodel/best_model/`


 **数据格式**: 确保CSV文件包含以下列：`timestamps`, `open`, `high`, `low`, `close`, `volume`, `amount`


